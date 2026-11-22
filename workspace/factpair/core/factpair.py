from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100


NAME_RE = re.compile(r"\b([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3})\b")
SPECIFIC_NAME_PATTERNS = [
    re.compile(r"author'?s name is ([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3})"),
    re.compile(r"author in question is ([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3})"),
    re.compile(r"\b([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3}),\s+an?\s+(?:esteemed|renowned|celebrated)"),
    re.compile(
        r"\b([A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+){1,3})\s+"
        r"(?:primarily|was|has|is|specializes|wrote|published|imbues|played|works|contributes|received)"
    ),
]
NAME_STOPWORDS = {
    "The", "In", "What", "Who", "Which", "Where", "How", "Are", "Some", "Yes", "No",
    "International", "Real", "Estate", "Doctor", "Geology", "Granite", "Shale",
    "Manual", "Handbook", "Award", "Medal", "University", "Pakistan", "Chile", "Santiago",
}

DEFAULT_TEMPLATE_ARGS: Dict[str, Any] = {
    "apply_chat_template": False,
    "system_prompt_with_special_tokens": "",
    "user_start_tag": "User: ",
    "user_end_tag": "\n",
    "asst_start_tag": "Assistant: ",
}


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_name(name: str) -> str:
    name = name.strip().strip(".,;:!?\"'")
    while name and name.split()[0] in NAME_STOPWORDS:
        name = " ".join(name.split()[1:])
    return name.strip()


def extract_author_name(text: str) -> Optional[str]:
    for pattern in SPECIFIC_NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            name = _clean_name(match.group(1))
            if len(name.split()) >= 2:
                return name
    candidates = [_clean_name(name) for name in NAME_RE.findall(text)]
    candidates = [name for name in candidates if len(name.split()) >= 2]
    return candidates[0] if candidates else None


@dataclass
class EntityCluster:
    name: str
    aliases: List[str]
    rows: List[Dict[str, Any]]
    source: str


def _aliases(name: str) -> List[str]:
    parts = name.split()
    aliases = {name, name.lower()}
    if len(parts) >= 2:
        aliases.add(parts[0])
        aliases.add(parts[-1])
        aliases.add(f"{parts[0]} {parts[-1]}")
    return sorted(alias for alias in aliases if alias)


def group_rows_by_author(
    rows: Sequence[Dict[str, Any]],
    source: str,
    rows_per_author: int = 20,
) -> List[EntityCluster]:
    clusters: List[EntityCluster] = []
    for start in range(0, len(rows), rows_per_author):
        chunk = rows[start : start + rows_per_author]
        votes: Counter[str] = Counter()
        for row in chunk:
            text = f"{row.get('answer', '')} {row.get('question', '')}"
            name = extract_author_name(text)
            if name:
                votes[name] += 1
        if not votes:
            continue
        name = sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]
        cluster_rows = [
            {
                "index": start + offset,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
            }
            for offset, row in enumerate(chunk)
        ]
        clusters.append(
            EntityCluster(
                name=name,
                aliases=_aliases(name),
                rows=cluster_rows,
                source=source,
            )
        )
    return _merge_clusters(clusters)


def _merge_clusters(clusters: Sequence[EntityCluster]) -> List[EntityCluster]:
    merged: Dict[str, EntityCluster] = {}
    for cluster in clusters:
        key = cluster.name.lower()
        if key not in merged:
            merged[key] = EntityCluster(
                name=cluster.name, aliases=list(cluster.aliases), rows=[], source=cluster.source
            )
        merged[key].rows.extend(cluster.rows)
        merged[key].aliases = sorted(set(merged[key].aliases) | set(cluster.aliases))
    return sorted(merged.values(), key=lambda cluster: cluster.name.lower())


def _cluster_payload(clusters: Sequence[EntityCluster]) -> List[Dict[str, Any]]:
    return [
        {
            "name": cluster.name,
            "aliases": cluster.aliases,
            "row_indices": [row["index"] for row in cluster.rows],
            "source": cluster.source,
        }
        for cluster in clusters
    ]


def build_entity_pools(
    forget_path: str | Path,
    retain_path: str | Path,
    rows_per_author: int = 20,
    donor_size: int = 20,
    retain_train_size: int = 20,
    seed: int = 0,
) -> Dict[str, Any]:
    forget_rows = read_jsonl(forget_path)
    retain_rows = read_jsonl(retain_path)
    forget_clusters = group_rows_by_author(forget_rows, "forget05", rows_per_author)
    retain_clusters = group_rows_by_author(retain_rows, "retain95", rows_per_author)

    forget_names = {cluster.name.lower() for cluster in forget_clusters}
    retained = [
        cluster for cluster in retain_clusters if cluster.name.lower() not in forget_names
    ]

    forget_families = {
        relation_family(row["question"])
        for cluster in forget_clusters
        for row in cluster.rows
    }

    def coverage(cluster: EntityCluster) -> int:
        families = {relation_family(row["question"]) for row in cluster.rows}
        return len(families & forget_families)

    ranked = sorted(
        retained,
        key=lambda cluster: (-coverage(cluster), sha256_text(cluster.name.lower() + str(seed))),
    )
    donor_clusters = [cluster for cluster in ranked if coverage(cluster) > 0][:donor_size]
    donor_names = {cluster.name.lower() for cluster in donor_clusters}
    remaining = [cluster for cluster in ranked if cluster.name.lower() not in donor_names]
    train_clusters = remaining[:retain_train_size]
    eval_clusters = remaining[retain_train_size:]

    return {
        "protocol": "factpair-progressive-alignment-v0.2-draft",
        "seed": seed,
        "rows_per_author": rows_per_author,
        "D_f": _cluster_payload(forget_clusters),
        "D_donor": _cluster_payload(donor_clusters),
        "D_R_train": _cluster_payload(train_clusters),
        "D_eval": _cluster_payload(eval_clusters),
        "sources": {
            "forget05": {
                "path": str(Path(forget_path).resolve()),
                "rows": len(forget_rows),
                "sha256": sha256_text(Path(forget_path).read_text(encoding="utf-8")),
            },
            "retain95": {
                "path": str(Path(retain_path).resolve()),
                "rows": len(retain_rows),
                "sha256": sha256_text(Path(retain_path).read_text(encoding="utf-8")),
            },
        },
        "cluster_rows": {
            "forget05": forget_rows,
            "retain95": retain_rows,
        },
    }


class PlaceholderTokenizer:
    """Deterministic tokenizer used only for offline data-plumbing smoke tests."""

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2

    def __init__(self) -> None:
        self.name_or_path = "PLACEHOLDER_WHITESPACE_HASH_TOKENIZER"

    def __call__(self, text: str, add_special_tokens: bool = False, **_: Any) -> Dict[str, List[int]]:
        pieces = re.findall(r"\w+|[^\w\s]", text)
        ids = [3 + (int(hashlib.sha256(piece.encode("utf-8")).hexdigest()[:8], 16) % 30000) for piece in pieces]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return {"input_ids": ids}


def load_tokenizer(tokenizer_name_or_path: Optional[str]) -> Tuple[Any, bool]:
    if not tokenizer_name_or_path:
        return PlaceholderTokenizer(), False
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    return tokenizer, True


def tokenizer_hash(tokenizer: Any, real: bool) -> str:
    if not real:
        return "PLACEHOLDER"
    signature = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    return sha256_text(
        json.dumps(
            {"name": signature, "init_kwargs": init_kwargs},
            sort_keys=True,
            default=str,
        )
    )


def build_prompt(tokenizer: Any, template_args: Optional[Dict[str, Any]], question: str) -> str:
    template = dict(DEFAULT_TEMPLATE_ARGS)
    template.update(template_args or {})
    if template.get("apply_chat_template") and hasattr(tokenizer, "apply_chat_template"):
        chat: List[Dict[str, str]] = []
        if template.get("system_prompt"):
            chat.append({"role": "system", "content": template["system_prompt"]})
        chat.append({"role": "user", "content": question})
        date_info: Dict[str, Any] = {}
        if template.get("date_string") is not None:
            date_info["date_string"] = template["date_string"]
        return tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            **date_info,
        )
    return (
        str(template.get("system_prompt_with_special_tokens", ""))
        + str(template.get("user_start_tag", ""))
        + question
        + str(template.get("user_end_tag", ""))
        + str(template.get("asst_start_tag", ""))
    )


def encode_without_truncation(tokenizer: Any, text: str) -> List[int]:
    from collections.abc import Mapping

    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, Mapping) and "input_ids" in encoded:
        return [int(token) for token in encoded["input_ids"]]
    return [int(token) for token in encoded]


def normalize_template(question: str, subject: str) -> str:
    return re.sub(r"\s+", " ", question.replace(subject, "<SUBJECT>")).strip()


def relation_family(question: str) -> str:
    text = question.lower()
    rules = [
        ("identity", ("full name", "who is this", "name of the author", "identify")),
        ("parents", ("parent", "father", "mother")),
        ("awards", ("award", "honored", "prize", "medal")),
        ("education", ("university", "college", "attend", "study", "degree")),
        ("birthplace", ("born in", "birthplace", "city was", "hails from")),
        ("gender", ("gender", "identify with")),
        ("books", ("book", "novel", "writings", "published", "maiden")),
        ("genre", ("genre", "field", "specialize", "contributes to")),
        ("style", ("writing style", "unique", "influence")),
    ]
    for family, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return family
    return "other"


def replace_entity(text: str, source: str, target: str) -> str:
    return text.replace(source, target)


def _span_check(prompt: str, entity: str, tokenizer: Any) -> Dict[str, Any]:
    char_start = prompt.find(entity)
    if char_start < 0 or prompt.count(entity) != 1:
        return {
            "start": -1,
            "length": 0,
            "single_contiguous_span": False,
            "boundary_tokenization_stable": False,
        }
    char_end = char_start + len(entity)
    try:
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    except Exception:
        prefix_ids = encode_without_truncation(tokenizer, prompt[:char_start])
        entity_ids = encode_without_truncation(tokenizer, entity)
        suffix_ids = encode_without_truncation(tokenizer, prompt[char_end:])
        full_ids = encode_without_truncation(tokenizer, prompt)
        return {
            "start": len(prefix_ids),
            "length": len(entity_ids),
            "single_contiguous_span": len(entity_ids) > 0,
            "boundary_tokenization_stable": prefix_ids + entity_ids + suffix_ids == full_ids,
        }
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if start < char_end and end > char_start and end > start
    ]
    if not indices or indices != list(range(indices[0], indices[0] + len(indices))):
        return {
            "start": -1,
            "length": 0,
            "single_contiguous_span": False,
            "boundary_tokenization_stable": False,
        }
    if hasattr(tokenizer, "decode"):
        decoded = tokenizer.decode([ids[index] for index in indices], skip_special_tokens=False).strip()
        stable = decoded == entity
    else:
        stable = True
    return {
        "start": indices[0],
        "length": len(indices),
        "single_contiguous_span": True,
        "boundary_tokenization_stable": stable,
    }


def pair_checks(
    prompt_plus: str,
    prompt_minus: str,
    answer_plus: str,
    answer_minus: str,
    tokenizer: Any,
    real_tokenizer: bool,
    max_length: int,
) -> Dict[str, Any]:
    ids_plus = encode_without_truncation(tokenizer, prompt_plus)
    ids_minus = encode_without_truncation(tokenizer, prompt_minus)
    plus_span = _span_check(prompt_plus, answer_plus, tokenizer)
    return {
        "prompt_plus_ids": ids_plus,
        "prompt_minus_ids": ids_minus,
        "entity_span_plus": plus_span,
        "token_length_equal": len(ids_plus) == len(ids_minus),
        "no_truncation": len(ids_plus) <= max_length and len(ids_minus) <= max_length,
        "answer_different": answer_plus.strip() != answer_minus.strip(),
        "real_tokenizer": real_tokenizer,
    }


def build_paired_prompts(
    entity_pools_path: str | Path,
    tokenizer_name_or_path: Optional[str],
    output_path: str | Path,
    template_args: Optional[Dict[str, Any]] = None,
    max_length: int = 512,
) -> Dict[str, Any]:
    pools = json.loads(Path(entity_pools_path).read_text(encoding="utf-8"))
    tokenizer, real_tokenizer = load_tokenizer(tokenizer_name_or_path)
    forget_rows = pools["cluster_rows"]["forget05"]
    retain_rows = pools["cluster_rows"]["retain95"]

    author_by_forget_index: Dict[int, str] = {}
    for cluster in pools["D_f"]:
        for row_index in cluster["row_indices"]:
            author_by_forget_index[row_index] = cluster["name"]

    donor_names = {cluster["name"].lower() for cluster in pools["D_donor"]}
    author_by_retain_index: Dict[int, str] = {}
    for cluster in pools["D_donor"]:
        for row_index in cluster["row_indices"]:
            author_by_retain_index[row_index] = cluster["name"]

    retain_by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(retain_rows):
        donor = author_by_retain_index.get(row_index)
        if donor is None:
            continue
        retain_by_family[relation_family(row["question"])].append(
            {"row_index": row_index, "donor": donor, "row": row}
        )

    paired: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for row_index, row in enumerate(forget_rows):
        source = author_by_forget_index.get(row_index)
        if source is None:
            failures.append({"row_index": row_index, "reason": "missing_forget_author"})
            continue
        template = normalize_template(row["question"], source)
        family = relation_family(row["question"])
        candidates = retain_by_family.get(family, [])
        chosen = None
        for candidate in candidates:
            donor = candidate["donor"]
            if donor.lower() == source.lower() or donor.lower() not in donor_names:
                continue
            prompt_plus = build_prompt(tokenizer, template_args, row["question"])
            prompt_minus = build_prompt(
                tokenizer,
                template_args,
                replace_entity(row["question"], source, donor),
            )
            checks = pair_checks(
                prompt_plus=prompt_plus,
                prompt_minus=prompt_minus,
                answer_plus=source,
                answer_minus=donor,
                tokenizer=tokenizer,
                real_tokenizer=real_tokenizer,
                max_length=max_length,
            )
            span_plus = checks["entity_span_plus"]
            span_minus = _span_check(prompt_minus, donor, tokenizer)
            ids_plus = checks["prompt_plus_ids"]
            ids_minus = checks["prompt_minus_ids"]
            outside_equal = (
                span_plus["start"] >= 0
                and span_minus["start"] >= 0
                and span_plus["start"] == span_minus["start"]
                and span_plus["length"] == span_minus["length"]
                and ids_plus[: span_plus["start"]] == ids_minus[: span_minus["start"]]
                and ids_plus[span_plus["start"] + span_plus["length"] :]
                == ids_minus[span_minus["start"] + span_minus["length"] :]
            )
            if (
                checks["token_length_equal"]
                and checks["no_truncation"]
                and outside_equal
                and span_plus["single_contiguous_span"]
                and span_minus["single_contiguous_span"]
                and row["answer"].strip() != candidate["row"]["answer"].strip()
            ):
                chosen = {
                    "id": f"pair-{row_index:04d}",
                    "source_row_index": row_index,
                    "donor_row_index": candidate["row_index"],
                    "source_author": source,
                    "donor_author": donor,
                    "template_key": template,
                    "relation_family": family,
                    "question_plus": row["question"],
                    "question_minus": replace_entity(row["question"], source, donor),
                    "answer_plus": row["answer"],
                    "answer_minus": candidate["row"]["answer"],
                    "prompt_plus": prompt_plus,
                    "prompt_minus": prompt_minus,
                    "prompt_plus_ids": ids_plus,
                    "prompt_minus_ids": ids_minus,
                    "entity_span": [span_plus["start"], span_plus["length"]],
                    "slot_outside_token_ids_equal": outside_equal,
                    "single_contiguous_span": span_plus["single_contiguous_span"]
                    and span_minus["single_contiguous_span"],
                    "boundary_tokenization_stable": span_plus["boundary_tokenization_stable"]
                    and span_minus["boundary_tokenization_stable"],
                    "no_truncation": checks["no_truncation"],
                    "g0_status": "PASS" if real_tokenizer else "BLOCKED_TOKENIZER",
                }
                break
        if chosen is None:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "no_relation_family_donor_with_equal_token_span",
                    "template_key": template,
                    "relation_family": family,
                }
            )
            continue
        paired.append(chosen)

    write_jsonl(output_path, paired)
    summary = {
        "pairs_written": len(paired),
        "failures": failures,
        "tokenizer": tokenizer_hash(tokenizer, real_tokenizer),
        "real_tokenizer": real_tokenizer,
        "g0_status": "PASS" if real_tokenizer and paired else "BLOCKED_TOKENIZER",
    }
    return summary


class FactPairDataset(Dataset):
    """Paired forget/donor prompts with an independent retain-train batch."""

    def __init__(
        self,
        pairs_path: str | Path,
        retain_path: str | Path,
        tokenizer: Any,
        template_args: Optional[Dict[str, Any]] = None,
        max_length: int = 512,
    ) -> None:
        self.pairs = read_jsonl(pairs_path)
        self.retain_rows = read_jsonl(retain_path)
        self.tokenizer = tokenizer
        self.template_args = template_args
        self.max_length = max_length
        if not self.pairs:
            raise ValueError("FactPairDataset received an empty paired prompt file")
        if not self.retain_rows:
            raise ValueError("FactPairDataset received an empty retain-train file")

    def __len__(self) -> int:
        return len(self.pairs)

    def _encode_answer(self, answer: str) -> List[int]:
        ids = encode_without_truncation(self.tokenizer, answer)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is not None and (not ids or ids[-1] != eos):
            ids = ids + [int(eos)]
        return ids

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair = self.pairs[index]
        prompt_ids = list(pair["prompt_plus_ids"])
        donor_ids = list(pair["prompt_minus_ids"])
        answer_ids = self._encode_answer(pair["answer_plus"])
        labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
        target_span_mask = [0] * len(prompt_ids) + [1] * len(answer_ids)

        retain_index = (int(pair["source_row_index"]) * 7919 + index) % len(self.retain_rows)
        retain_row = self.retain_rows[retain_index]
        retain_prompt = build_prompt(self.tokenizer, self.template_args, retain_row["question"])
        retain_prompt_ids = encode_without_truncation(self.tokenizer, retain_prompt)
        retain_answer_ids = self._encode_answer(retain_row["answer"])
        retain_input_ids = retain_prompt_ids + retain_answer_ids
        retain_labels = [IGNORE_INDEX] * len(retain_prompt_ids) + retain_answer_ids

        return {
            "pair_id": str(pair["id"]),
            "forget": {
                "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(prompt_ids), dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            },
            "teacher_minus": {
                "input_ids": torch.tensor(donor_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(donor_ids), dtype=torch.long),
            },
            "retain": {
                "input_ids": torch.tensor(retain_input_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(retain_input_ids), dtype=torch.long),
                "labels": torch.tensor(retain_labels, dtype=torch.long),
            },
            "target_span_mask": torch.tensor(target_span_mask, dtype=torch.long),
            "entity_span": torch.tensor(pair["entity_span"], dtype=torch.long),
        }

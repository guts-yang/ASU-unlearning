#!/usr/bin/env python3
"""Read-only precheck for prompt-only activation donor candidates v0.3.

This script loads a tokenizer only. It does not load model weights, run a
forward pass, extract activations, apply an activation patch, compute K2, or
start Teacher/Student training.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/usr/local/ASU-unlearning")
CANDIDATE_DIR = REPO / "workspace/data/external_donor_prompt_activation_v0.3"
OUT_DIR = REPO / "workspace/data/external_donor_prompt_activation_v0.3_validation"
TOFU_JSONL_DIR = REPO / "workspace/data/tofu"
TOFU_JSON_DIR = REPO / "Right-to-be-forgotten/data/tofu"
FORGET05 = TOFU_JSONL_DIR / "forget05.jsonl"
LLAMA_FILE = REPO / "Right-to-be-forgotten/my_models/my_llama.py"
SPEC_FILE = REPO / "workspace/方向I_prompt_only_activation_donor构造规格_v0.3.md"
EXPECTED_FORGET05_SHA256 = (
    "8f721842a4741fda0e4bfed459c4510c6c00604d5c332a533011648236862e08"
)
EXPECTED_FORGET05_ROWS = 200

SNAPSHOT = Path(
    "/root/autodl-tmp/huggingface/hub/"
    "models--open-unlearning--tofu_Llama-2-7b-chat-hf_full/"
    "snapshots/cc5b31c69127d5da881608e640f2b453c446435b"
)
NOUS_SNAPSHOT = Path(
    "/root/autodl-tmp/huggingface/hub/"
    "models--NousResearch--Llama-2-7b-chat-hf/"
    "snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee"
)

QUESTION_START = "[INST] "
QUESTION_END = " [/INST]"
ANSWER_TAG = ""
MANUAL_TEMPLATE = QUESTION_START + "{question}" + QUESTION_END

CHECKPOINT_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
]
TOKENIZER_COMPARE_FILES = [
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
]
CODE_FILES = [
    REPO / "Right-to-be-forgotten/dataset/data_module.py",
    REPO / "Right-to-be-forgotten/config/model_config.yaml",
    REPO / "Right-to-be-forgotten/config/tofu.yaml",
    LLAMA_FILE,
]
CANDIDATE_FILES = [
    "candidate_pairs.jsonl",
    "clean_facts.jsonl",
    "donor_entities.jsonl",
    "manifest.json",
    "README.md",
    "REMOTE_VALIDATION_TODO.md",
]
INVENTORY_PREFIXES = ("forget", "retain", "world_facts", "real_authors", "holdout")
CONFIGURED_EVAL_NAMES = {
    "retain_perturbed.json",
    "real_authors_perturbed.json",
    "world_facts_perturbed.json",
    "forget05_perturbed.json",
    "retain_perturbed.jsonl",
    "real_authors_perturbed.jsonl",
    "world_facts_perturbed.jsonl",
    "forget05_perturbed.jsonl",
}
# Alias and neighbor strings already declared in the v0.1 offline scan, plus
# public name forms for the two donors that list changed. Not completed online.
DONOR_TERMS = {
    "donor-01": {
        "entity": ["Katherine Mansfield"],
        "aliases": ["Kathleen Mansfield Beauchamp", "Kathleen Beauchamp"],
        "neighbors": ["John Middleton Murry"],
        "places": ["Wellington"],
    },
    "donor-02": {
        "entity": ["Bill Gates"],
        "aliases": ["William Henry Gates III", "William Henry Gates", "William H. Gates"],
        "neighbors": ["Paul Allen", "Melinda Gates", "Melinda French Gates"],
        "places": ["Seattle"],
    },
    "donor-03": {
        "entity": ["Muhammad Ali"],
        "aliases": ["Cassius Marcellus Clay", "Cassius Clay"],
        "neighbors": [],
        "places": ["Louisville"],
    },
    "donor-04": {
        "entity": ["Michael Jackson"],
        "aliases": ["Michael Joseph Jackson"],
        "neighbors": ["Janet Jackson", "The Jackson 5"],
        "places": ["Gary", "Gary, Indiana"],
    },
    "donor-05": {
        "entity": ["Elvis Presley"],
        "aliases": ["Elvis Aaron Presley", "Elvis Aron Presley"],
        "neighbors": ["Priscilla Presley"],
        "places": ["Tupelo"],
    },
    "donor-06": {
        "entity": ["Ada Lovelace"],
        "aliases": ["Augusta Ada King", "Augusta Ada Byron", "Ada Byron"],
        "neighbors": ["Charles Babbage"],
        "places": ["London"],
    },
    "donor-07": {
        "entity": ["Malala Yousafzai"],
        "aliases": ["Malala"],
        "neighbors": ["Ziauddin Yousafzai"],
        "places": ["Mingora"],
    },
    "donor-08": {
        "entity": ["Paulo Coelho"],
        "aliases": [],
        "neighbors": [],
        "places": ["Rio de Janeiro"],
    },
    "donor-09": {
        "entity": ["Anne Rice"],
        "aliases": [
            "Howard Allen Frances O'Brien",
            "Anne Rampling",
            "A. N. Roquelaure",
        ],
        "neighbors": ["Stan Rice"],
        "places": ["New Orleans"],
    },
    "donor-10": {
        "entity": ["Lionel Messi"],
        "aliases": ["Lionel Andrés Messi", "Lionel Andres Messi", "Leo Messi"],
        "neighbors": [],
        "places": ["Rosario"],
    },
}
AMBIGUOUS_PLACES = {
    "Gary",
    "Gary, Indiana",
    "London",
    "Seattle",
    "Wellington",
    "Rio de Janeiro",
    "New Orleans",
}
PERSON_CATEGORIES = {"entity", "alias", "neighbor"}
EXAMPLES_PER_TERM = 5
CITY_P31 = {"Q515", "Q1549591", "Q3957", "Q486972"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    rows = []
    errors = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            errors.append(f"{path.name}:{line_no}: empty line")
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_no}: {exc}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"{path.name}:{line_no}: JSON value is not an object")
            continue
        rows.append(obj)
    return rows, errors


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'").replace("`", "'")
    return text.casefold()


def term_pattern(term: str) -> re.Pattern[str]:
    normalized = normalize_text(term)
    return re.compile(r"(?<![0-9a-z])" + re.escape(normalized) + r"(?![0-9a-z])")


def inventory_kind(name: str) -> str | None:
    stem = name.lower()
    for prefix in INVENTORY_PREFIXES:
        if stem.startswith(prefix):
            return prefix
    return None


def git_text(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def match_excerpt(text: str, normalized: str, term: str, pattern: re.Pattern[str]) -> str:
    raw = re.search(re.escape(term), text, flags=re.IGNORECASE)
    if raw:
        start, end = raw.start(), raw.end()
        return text[max(0, start - 70) : end + 70]
    found = pattern.search(normalized)
    if found is None:
        return text[:180]
    return normalized[max(0, found.start() - 70) : found.end() + 70]


def load_eval_lines() -> tuple[list[dict], list[str]]:
    files = sorted(TOFU_JSONL_DIR.glob("*.jsonl")) + sorted(TOFU_JSON_DIR.glob("*.json"))
    rows = []
    skipped = []
    for path in files:
        kind = inventory_kind(path.name)
        if kind is None:
            skipped.append(str(path.relative_to(REPO)))
            continue
        rel = str(path.relative_to(REPO))
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            rows.append(
                {
                    "path": rel,
                    "file_name": path.name,
                    "line": line_no,
                    "text": line,
                    "normalized": normalize_text(line),
                    "inventory": kind,
                    "configured_eval": path.name in CONFIGURED_EVAL_NAMES,
                    "holdout": kind == "holdout",
                }
            )
    return rows, skipped


def scan_overlap(eval_rows: list[dict]) -> dict[str, dict]:
    compiled = {}
    for donor_id, groups in DONOR_TERMS.items():
        compiled[donor_id] = []
        for category, terms in groups.items():
            category_name = {
                "entity": "entity",
                "aliases": "alias",
                "neighbors": "neighbor",
                "places": "place",
            }[category]
            for term in terms:
                compiled[donor_id].append((category_name, term, term_pattern(term)))

    found: dict[str, list[dict]] = {donor_id: [] for donor_id in DONOR_TERMS}
    fact_lines: dict[str, list[dict]] = {donor_id: [] for donor_id in DONOR_TERMS}
    for row in eval_rows:
        for donor_id, terms in compiled.items():
            line_hits = []
            for category, term, pattern in terms:
                if pattern.search(row["normalized"]):
                    hit = {
                        "category": category,
                        "term": term,
                        "file": row["path"],
                        "file_name": row["file_name"],
                        "line": row["line"],
                        "inventory": row["inventory"],
                        "configured_eval_file": row["configured_eval"],
                        "holdout_file": row["holdout"],
                        "snippet": match_excerpt(
                            row["text"], row["normalized"], term, pattern
                        ),
                    }
                    line_hits.append(hit)
                    found[donor_id].append(hit)
            categories = {hit["category"] for hit in line_hits}
            if categories & {"entity", "alias"} and "place" in categories:
                fact_lines[donor_id].append(
                    {
                        "file": row["path"],
                        "line": row["line"],
                        "inventory": row["inventory"],
                        "configured_eval_file": row["configured_eval"],
                        "terms": sorted({hit["term"] for hit in line_hits}),
                    }
                )

    summaries = {}
    for donor_id, hits in found.items():
        by_term = defaultdict(list)
        for hit in hits:
            by_term[(hit["category"], hit["term"])].append(hit)
        term_counts = []
        examples = []
        for (category, term), term_hits in sorted(by_term.items()):
            file_counts = defaultdict(int)
            configured = 0
            holdout = 0
            for hit in term_hits:
                file_counts[hit["file"]] += 1
                configured += int(hit["configured_eval_file"])
                holdout += int(hit["holdout_file"])
            term_counts.append(
                {
                    "category": category,
                    "term": term,
                    "count": len(term_hits),
                    "configured_eval_count": configured,
                    "holdout_count": holdout,
                    "ambiguous_place": term in AMBIGUOUS_PLACES,
                    "files": dict(sorted(file_counts.items())),
                }
            )
            examples.extend(term_hits[:EXAMPLES_PER_TERM])
        person_hits = [hit for hit in hits if hit["category"] in PERSON_CATEGORIES]
        reasons = []
        if person_hits:
            reasons.append("person_alias_or_neighbor_hit")
        if fact_lines[donor_id]:
            reasons.append("fact_cluster_hit")
        summaries[donor_id] = {
            "person_hit_count": len(person_hits),
            "place_hit_count": sum(1 for hit in hits if hit["category"] == "place"),
            "fact_cluster_line_count": len(fact_lines[donor_id]),
            "configured_eval_person_hit_count": sum(
                1 for hit in person_hits if hit["configured_eval_file"]
            ),
            "configured_eval_place_hit_count": sum(
                1 for hit in hits if hit["category"] == "place" and hit["configured_eval_file"]
            ),
            "holdout_person_hit_count": sum(1 for hit in person_hits if hit["holdout_file"]),
            "holdout_place_hit_count": sum(
                1 for hit in hits if hit["category"] == "place" and hit["holdout_file"]
            ),
            "excluded": bool(reasons),
            "exclusion_reasons": reasons,
            "term_counts": term_counts,
            "examples": examples,
            "fact_cluster_examples": fact_lines[donor_id][:EXAMPLES_PER_TERM],
            "alias_list_status": "offline_predeclared_not_live_completed",
        }
    return summaries


def beyonce_in_text(text: str) -> bool:
    return term_pattern("Beyoncé").search(normalize_text(text)) is not None


def scan_beyonce(eval_rows: list[dict]) -> dict:
    hits = []
    for row in eval_rows:
        if beyonce_in_text(row["text"]):
            hits.append(
                {
                    "file": row["path"],
                    "line": row["line"],
                    "inventory": row["inventory"],
                    "configured_eval_file": row["configured_eval"],
                    "snippet": match_excerpt(
                        row["text"], row["normalized"], "Beyonce", term_pattern("Beyoncé")
                    ),
                }
            )
    return {
        "eval_hit_count": len(hits),
        "configured_eval_hit_count": sum(1 for hit in hits if hit["configured_eval_file"]),
        "examples": hits[:EXAMPLES_PER_TERM],
        "note": (
            "Historical string collision only. Beyoncé is not a v0.3 donor; "
            "these hits do not exclude a current donor."
        ),
    }


def verify_clean_row(
    pair: dict, clean: dict | None, donor: dict | None, forget_lines: list[str]
) -> dict:
    issues = []
    if "y_minus" in pair:
        issues.append("y_minus_present")
    if clean is None:
        issues.append("missing_clean_fact")
    if donor is None:
        issues.append("missing_donor_record")
    line_no = pair.get("clean_source", {}).get("line_1based")
    source_obj = None
    if not isinstance(line_no, int) or not 1 <= line_no <= len(forget_lines):
        issues.append("source_line_out_of_range")
    else:
        try:
            source_obj = json.loads(forget_lines[line_no - 1])
        except json.JSONDecodeError as exc:
            issues.append(f"source_line_json_error: {exc}")
    if source_obj is not None:
        clean_source = pair.get("clean_source", {})
        if source_obj.get("question") != clean_source.get("question"):
            issues.append("pair_source_question_mismatch")
        if source_obj.get("answer") != clean_source.get("answer"):
            issues.append("pair_source_answer_mismatch")
        if pair.get("y_plus") not in source_obj.get("answer", ""):
            issues.append("y_plus_absent_from_source_answer")
        if clean is not None:
            if clean.get("source_question") != source_obj.get("question"):
                issues.append("clean_source_question_mismatch")
            if clean.get("source_answer") != source_obj.get("answer"):
                issues.append("clean_source_answer_mismatch")
            if clean.get("source_line_1based") != line_no:
                issues.append("clean_source_line_mismatch")
            if clean.get("normalized_value") != pair.get("y_plus"):
                issues.append("y_plus_differs_from_clean_normalized_value")
            if clean.get("entity") != pair.get("clean_entity"):
                issues.append("clean_entity_mismatch")
    if donor is not None:
        if donor.get("entity") != pair.get("donor_entity"):
            issues.append("donor_entity_mismatch")
        if donor.get("donor_fact_value") != pair.get("donor_fact_value"):
            issues.append("donor_fact_value_mismatch")
        if donor.get("wikidata_qid") != pair.get("donor_source", {}).get("qid"):
            issues.append("donor_qid_mismatch")
    if pair.get("donor_fact_value") and pair.get("donor_fact_value") in pair.get("q_minus", ""):
        issues.append("donor_fact_value_embedded_in_q_minus")
    return {
        "pair_id": pair.get("pair_id"),
        "status": "pass" if not issues else "fail",
        "issues": issues,
    }


def internal_source_check(pair: dict, donor: dict | None) -> dict:
    issues = []
    if donor is None:
        return {"internal_consistency": "fail", "issues": ["missing_donor_record"], "checks": {}}
    qid = str(donor.get("wikidata_qid", ""))
    url = str(donor.get("source_url", ""))
    value = donor.get("donor_fact_value", "")
    checks = {
        "property_is_p19": pair.get("donor_source", {}).get("property") == "P19",
        "relation_id_is_place_of_birth": donor.get("relation_id") == "place_of_birth",
        "qid_format_ok": bool(re.fullmatch(r"Q\d+", qid)),
        "url_contains_qid": qid in url,
        "pair_qid_matches_donor": pair.get("donor_source", {}).get("qid") == qid,
        "pair_value_matches_donor": pair.get("donor_fact_value") == value,
        "value_is_single_place_label": isinstance(value, str)
        and bool(value.strip())
        and "," not in value,
    }
    for name, ok in checks.items():
        if not ok:
            issues.append(name)
    granularity_note = "city_label_in_candidate_file_not_remotely_confirmed"
    note = donor.get("source_note", "").lower()
    if "granularity" in note or "city-level" in note:
        granularity_note = "candidate_note_asks_for_remote_granularity_check"
    return {
        "donor_id": donor.get("donor_id"),
        "name": donor.get("entity"),
        "wikidata_qid": qid,
        "source_url": url,
        "donor_fact_value": value,
        "checks": checks,
        "internal_consistency": "pass" if not issues else "fail",
        "issues": issues,
        "granularity_note": granularity_note,
    }


def wikidata_labels(entity: dict) -> list[str]:
    labels = []
    en = entity.get("labels", {}).get("en", {}).get("value")
    if en:
        labels.append(en)
    for alias in entity.get("aliases", {}).get("en", []):
        value = alias.get("value")
        if value:
            labels.append(value)
    return labels


def p31_ids(entity: dict) -> list[str]:
    found = []
    for claim in entity.get("claims", {}).get("P31", []):
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        if isinstance(value, dict) and value.get("id"):
            found.append(value["id"])
    return found


def fetch_entity(qid: str) -> dict:
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ASU-unlearning-validation/0.3 (research precheck)"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    entity = payload.get("entities", {}).get(qid)
    if not isinstance(entity, dict):
        raise ValueError(f"missing entity object for {qid}")
    return entity


def live_p19_status(donor: dict | None) -> dict:
    if donor is None:
        return {"status": "pending", "live_wikidata_p19": "blocked_missing_donor"}
    qid = str(donor.get("wikidata_qid", ""))
    try:
        entity = fetch_entity(qid)
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "pending",
            "live_wikidata_p19": "blocked_no_network",
            "error": f"{type(exc).__name__}: {exc}",
        }
    claims = entity.get("claims", {}).get("P19", [])
    place_ids = []
    for claim in claims:
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
        if isinstance(value, dict) and value.get("id"):
            place_ids.append(value["id"])
    if len(place_ids) != 1:
        return {
            "status": "pending",
            "live_wikidata_p19": "pending_p19_count",
            "p19_ids": place_ids,
        }
    try:
        place = fetch_entity(place_ids[0])
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "pending",
            "live_wikidata_p19": "blocked_no_network",
            "p19_ids": place_ids,
            "error": f"{type(exc).__name__}: {exc}",
        }
    labels = wikidata_labels(place)
    target = normalize_text(str(donor.get("donor_fact_value", "")))
    exact = [label for label in labels if normalize_text(label) == target]
    classes = p31_ids(place)
    if not exact:
        return {
            "status": "fail",
            "live_wikidata_p19": "label_mismatch",
            "p19_ids": place_ids,
            "labels": labels,
            "p31": classes,
        }
    if set(classes) & CITY_P31 and all("," not in label for label in exact):
        return {
            "status": "pass",
            "live_wikidata_p19": "city_label_match",
            "p19_ids": place_ids,
            "matched_labels": exact,
            "p31": classes,
        }
    return {
        "status": "pending",
        "live_wikidata_p19": "pending_granularity",
        "p19_ids": place_ids,
        "matched_labels": exact,
        "p31": classes,
    }


def source_gate(internal: dict, live: dict) -> str:
    if internal.get("internal_consistency") != "pass":
        return "fail"
    if live.get("status") == "pass":
        return "pass"
    if live.get("status") == "fail":
        return "fail"
    return "pending"


def prompt_text(question: str) -> str:
    return QUESTION_START + question + QUESTION_END


def continuation_span(tokenizer, prompt: str, continuation: str) -> dict:
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + ANSWER_TAG + continuation, add_special_tokens=True)["input_ids"]
    prefix_ok = full_ids[: len(prompt_ids)] == prompt_ids
    span_ids = full_ids[len(prompt_ids) :] if prefix_ok else []
    return {
        "continuation": continuation,
        "answer_tag": ANSWER_TAG,
        "token_ids": span_ids,
        "token_strings": tokenizer.convert_ids_to_tokens(span_ids),
        "token_count": len(span_ids),
        "span": {
            "start": len(prompt_ids),
            "end": len(full_ids),
            "convention": "half_open_index_in_prompt_plus_continuation",
        },
        "prefix_matches_prompt": prefix_ok,
        "decoded": tokenizer.decode(span_ids, skip_special_tokens=False) if span_ids else "",
        "padding_applied": False,
        "truncation_applied": False,
        "aggregation": "not_frozen_spec_missing",
        "score_computed": False,
    }


def entity_token_span(tokenizer, prompt: str, question: str, entity: str) -> dict:
    encoding = tokenizer(
        prompt,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    ids = list(encoding["input_ids"])
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    char_start = question.find(entity)
    if char_start < 0:
        return {
            "status": "fail",
            "issues": ["entity_not_in_question"],
            "prompt_token_ids": ids,
            "prompt_token_strings": tokenizer.convert_ids_to_tokens(ids),
            "prompt_length": len(ids),
        }
    prompt_start = len(QUESTION_START) + char_start
    prompt_end = prompt_start + len(entity)
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if not (start == 0 and end == 0) and end > prompt_start and start < prompt_end
    ]
    issues = []
    if not indices:
        issues.append("no_token_covers_entity")
    contiguous = indices == list(range(indices[0], indices[-1] + 1)) if indices else False
    if indices and not contiguous:
        issues.append("entity_token_span_not_contiguous")
    covered = ""
    if indices:
        covered_start = min(offsets[index][0] for index in indices)
        covered_end = max(offsets[index][1] for index in indices)
        covered = prompt[covered_start:covered_end]
        if covered.strip() != entity:
            issues.append("entity_token_span_not_isolated")
    return {
        "status": "pass" if not issues else "fail",
        "issues": issues,
        "entity": entity,
        "char_span_in_question": {"start": char_start, "end": char_start + len(entity)},
        "char_span_in_prompt": {"start": prompt_start, "end": prompt_end},
        "token_indices": indices,
        "token_span": {
            "start": indices[0] if indices else None,
            "end": (indices[-1] + 1) if indices else None,
            "convention": "half_open_index_in_prompt_token_ids",
        },
        "token_strings": tokenizer.convert_ids_to_tokens([ids[index] for index in indices]),
        "covered_text": covered,
        "contiguous": contiguous,
        "prompt_token_ids": ids,
        "prompt_token_strings": tokenizer.convert_ids_to_tokens(ids),
        "prompt_length": len(ids),
        "padding_applied": False,
        "truncation_applied": False,
    }


def encode_question(tokenizer, question: str, entity: str, continuation: str | None) -> dict:
    prompt = prompt_text(question)
    encoded = entity_token_span(tokenizer, prompt, question, entity)
    record = {
        "question": question,
        "prompt_text": prompt,
        "donor_fact_or_answer_appended": False,
        "entity_span": encoded,
    }
    if continuation is not None:
        record["continuation_span"] = continuation_span(tokenizer, prompt, continuation)
    return record


def hook_status() -> dict:
    text = LLAMA_FILE.read_text(encoding="utf-8")
    present = "activation_editor" in text
    returns_down_proj = "return down_proj" in text
    return {
        "file": str(LLAMA_FILE.relative_to(REPO)),
        "sha256": sha256_file(LLAMA_FILE),
        "activation_editor_present": present,
        "mlp_forward_returns_down_proj": returns_down_proj,
        "status": "blocked_no_activation_editor" if not present else "present_not_exercised",
    }


def cuda_status() -> dict:
    import torch

    smi = Path("/usr/bin/nvidia-smi")
    devices = sorted(path.name for path in Path("/dev").glob("nvidia*"))
    return {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count()),
        "nvidia_smi_bytes": smi.stat().st_size if smi.exists() else None,
        "dev_nvidia_nodes": devices,
        "weights_loaded": False,
        "dtype": "not_loaded",
        "device": "cuda_unavailable",
    }


def fmt_hash_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| 文件 | SHA256 |", "| --- | --- |"]
    for name, digest in rows:
        lines.append(f"| `{name}` | `{digest}` |")
    return "\n".join(lines)


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_summary(context: dict) -> str:
    counts = context["counts"]
    lines: list[str] = []
    add = lines.append
    add("# v0.3 prompt-only activation donor 预检")
    add("")
    add("## 结论")
    add("")
    add(f"**{context['decision']}。**")
    add("")
    add(
        f"候选 {counts['pairs']} 条，clean 簇 {counts['unique_clean']} 个，"
        f"donor 簇 {counts['unique_donors']} 个。"
        "这是数据与激活可行性预检，不是正式实验。"
    )
    add("")
    add("| 门 | 通过 | 失败 | 待审 | 阻断 | 说明 |")
    add("| --- | ---: | ---: | ---: | ---: | --- |")
    for gate in context["gate_rows"]:
        add(
            f"| {gate['name']} | {gate['pass']} | {gate['fail']} | {gate['pending']} | "
            f"{gate['blocked']} | {gate['note']} |"
        )
    add("")
    add("未通过因而暂停该臂的门：")
    add("")
    for reason in context["blocking_reasons"]:
        add(f"- {reason}")
    add("")
    add("不进入正式 activation localization，不开始 Teacher/Student 训练。")
    add("")
    add("## 1. 文件核对")
    add("")
    add("候选目录只读。三个 JSONL 的每一行都是 JSON 对象。配对里没有 `y_minus`。")
    add("")
    add("| 项目 | 计数 | manifest | 一致 |")
    add("| --- | ---: | ---: | --- |")
    add(
        f"| candidate_pairs | {counts['pairs']} | {context['manifest']['pair_count']} | "
        f"{'是' if counts['pairs'] == context['manifest']['pair_count'] else '否'} |"
    )
    add(
        f"| clean_facts | {counts['clean']} | — | "
        f"{'是' if counts['clean'] == counts['unique_clean'] else '否'} |"
    )
    add(
        f"| donor_entities | {counts['donors']} | — | "
        f"{'是' if counts['donors'] == counts['unique_donors'] else '否'} |"
    )
    add(
        f"| unique clean clusters | {counts['unique_clean']} | "
        f"{context['manifest']['unique_clean_clusters']} | "
        f"{'是' if counts['unique_clean'] == context['manifest']['unique_clean_clusters'] else '否'} |"
    )
    add(
        f"| unique donor clusters | {counts['unique_donors']} | "
        f"{context['manifest']['unique_donor_clusters']} | "
        f"{'是' if counts['unique_donors'] == context['manifest']['unique_donor_clusters'] else '否'} |"
    )
    add("")
    add(f"解析错误：{context['parse_errors'] or '无'}。")
    add(f"完整性问题：{context['integrity_issues'] or '无'}。")
    add(f"donor 复用行：{counts['reuse_rows']}。")
    add(
        f"候选含 Beyoncé：{'是' if context['beyonce_in_candidates'] else '否'}。"
        f"donor-03 是 {context['donor03_entity']}。"
        f"donor-10 是 {context['donor10_entity']}。"
    )
    add("")
    add(
        f"`forget05.jsonl` 行数为 {counts['forget05_rows']}，SHA256 为 `{context['forget05_sha256']}`。"
        f"与 manifest 的 `clean_rows={context['manifest']['clean_rows']}`、"
        f"`clean_source_sha256={context['manifest']['clean_source_sha256']}` "
        f"{'一致' if context['forget05_ok'] else '不一致'}。"
    )
    add("")
    add("clean 侧来源行核对：" + context["clean_source_summary"])
    add("")
    add(fmt_hash_table(context["candidate_hashes"]))
    add("")
    add("写完结果后重新计算了上述候选文件哈希，与运行开始时相同。")
    add("")
    add("## 2. Checkpoint、tokenizer 与 chat template")
    add("")
    checkpoint = context["checkpoint"]
    add(f"- Hugging Face repo：`{checkpoint['repo_id']}`")
    add(f"- snapshot commit：`{checkpoint['snapshot_commit']}`")
    add(f"- 本地快照：`{checkpoint['snapshot_path']}`")
    add(f"- `config.json` 的 `_name_or_path`：`{checkpoint['name_or_path']}`")
    add("- 权重未加载，没有调用 `AutoModelForCausalLM`。")
    device = context["device"]
    add(
        f"- CUDA：`available={device['torch_cuda_available']}`，"
        f"`device_count={device['torch_device_count']}`，"
        f"`nvidia-smi` 字节数 {device['nvidia_smi_bytes']}，"
        f"设备节点 `{', '.join(device['dev_nvidia_nodes']) or '无'}`。"
    )
    add(f"- 精度：`{device['dtype']}`。设备：`{device['device']}`。")
    add("")
    add(
        "`Right-to-be-forgotten/config/tofu.yaml` 的 `model_path` 是 "
        "`locuslab/tofu_ft_llama2-7b`。本次没有把它当作 tokenizer 来源。"
    )
    add("")
    add("Checkpoint 文件哈希：")
    add("")
    add(fmt_hash_table(context["checkpoint_hashes"]))
    add("")
    nous = context["nous_compare"]
    same = "、".join(f"`{name}`" for name in nous["same_files"]) or "无"
    different = "、".join(f"`{name}`" for name in nous["different_files"]) or "无"
    add(
        f"与本地 `NousResearch/Llama-2-7b-chat-hf` snapshot `{nous['snapshot']}` 的 tokenizer 文件比较："
        f"相同文件为 {same}；不同文件为 {different}。"
    )
    add("")
    tok = context["tokenizer"]
    add("实际加载的 tokenizer：")
    add("")
    add(f"- 类：`{tok['class_name']}`")
    add(f"- transformers：`{tok['transformers_version']}`")
    add(f"- vocab size：{tok['vocab_size']}；`len(tokenizer)`：{tok['len']}")
    add(
        f"- bos/eos/pad：`{tok['bos_token']}`={tok['bos_token_id']}，"
        f"`{tok['eos_token']}`={tok['eos_token_id']}，"
        f"`{tok['pad_token']}`={tok['pad_token_id']}"
    )
    add(f"- padding side：`{tok['padding_side']}`")
    add("- 编码没有 padding，也没有 truncation。")
    add("")
    add("推理代码：")
    add("")
    add(f"- git commit：`{context['git_commit']}`")
    add(f"- 分支：`{context['git_branch']}`")
    add(f"- 运行开始时 `git status --porcelain`：`{context['git_status'] or 'clean'}`")
    add("")
    add("手工模板来自 `model_config.yaml` 的 `llama2-7b`，不是 `tokenizer.apply_chat_template`：")
    add("")
    add("```text")
    add(MANUAL_TEMPLATE)
    add("```")
    add("")
    add(f"- 模板 SHA256：`{context['manual_template_sha256']}`")
    add(f"- `question_start_tag` = `{QUESTION_START!r}`")
    add(f"- `question_end_tag` = `{QUESTION_END!r}`")
    add(f"- `answer_tag` = `{ANSWER_TAG!r}`")
    add("- `add_special_tokens=True`，序列以 BOS `<s>` 开头。")
    add(f"- 存储 chat template SHA256：`{context['stored_chat_template_sha256']}`。")
    add("")
    add("`apply_chat_template` 对 promptdonor-01 的 `q_minus`（只含用户问题）得到：")
    add("")
    add("```text")
    add(context["apply_chat_template_example"])
    add("```")
    add("")
    add("实际 prompt-only 拼接得到：")
    add("")
    add("```text")
    add(context["manual_template_example"])
    add("```")
    add("")
    add("激活提取使用上面的实际拼接。`donor_fact_value` 没有拼进 `q_minus`。")
    add("")
    add("相关代码哈希：")
    add("")
    add(fmt_hash_table(context["code_hashes"]))
    add("")
    add("## 3. 评测去重")
    add("")
    inventory = context["inventory"]
    add(
        f"扫描了 forget、retain、world_facts、real_authors、holdout："
        f"jsonl {inventory['jsonl_files']} 个，json {inventory['json_files']} 个，"
        f"共 {inventory['lines']} 行。"
    )
    add(f"未纳入本次清单、因此跳过的文件：{inventory['skipped'] or '无'}。")
    add("")
    add(
        "`tofu.yaml` 当前评测清单是 `retain_perturbed`、`real_authors_perturbed`、"
        "`world_facts_perturbed`、`forget05_perturbed`。人物命中与城市字符串命中分开计数。"
    )
    add("别名表沿用 v0.1 已声明项；Muhammad Ali 与 Lionel Messi 只用公开姓名形式。网络不可用，别名表未在线补全。")
    add("只有人物、别名、近邻或同一行上的人物加出生地事实簇命中才排除 donor。城市字符串单独列出，不据此排除。")
    add("")
    add("| donor | 人物命中 | 配置评测 | holdout 人物 | 事实簇行 | 城市字符串 | 排除 |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for donor_id, summary in context["overlap"].items():
        add(
            f"| {donor_id} | {summary['person_hit_count']} | "
            f"{summary['configured_eval_person_hit_count']} | "
            f"{summary['holdout_person_hit_count']} | {summary['fact_cluster_line_count']} | "
            f"{summary['place_hit_count']} | {'是' if summary['excluded'] else '否'} |"
        )
    add("")
    excluded = [
        f"{donor_id}（{', '.join(summary['exclusion_reasons'])}）"
        for donor_id, summary in context["overlap"].items()
        if summary["excluded"]
    ]
    add(f"排除的 donor：{excluded or '无'}。")
    add("")
    hit_terms = context["hit_terms"]
    add("有命中的搜索词：")
    add("")
    if not hit_terms:
        add("无。")
    else:
        add("| donor | 类别 | 词 | 次数 | 配置评测 | holdout | 歧义城市 |")
        add("| --- | --- | --- | ---: | ---: | ---: | --- |")
        for row in hit_terms:
            add(
                f"| {row['donor_id']} | {row['category']} | {md_cell(row['term'])} | {row['count']} | "
                f"{row['configured_eval_count']} | {row['holdout_count']} | "
                f"{'是' if row['ambiguous_place'] else '否'} |"
            )
    add("")
    person_examples = context["person_examples"]
    if not person_examples:
        add("实体、别名和近邻人物没有命中这份评测清单。")
    else:
        add("人物命中样例：")
        add("")
        add("| donor | 类别 | 词 | 位置 | 上下文 |")
        add("| --- | --- | --- | --- | --- |")
        for donor_id, example in person_examples:
            add(
                f"| {donor_id} | {example['category']} | {md_cell(example['term'])} | "
                f"`{example['file']}:{example['line']}` | `{md_cell(example['snippet'])}` |"
            )
    add("")
    beyonce = context["beyonce_eval"]
    add(
        f"候选文件中的 Beyoncé 字符串：{context['beyonce_candidate_hits'] or '无'}。"
        f"评测清单里的 Beyoncé 字符串命中 {beyonce['eval_hit_count']} 行，"
        f"其中配置评测清单 {beyonce['configured_eval_hit_count']} 行。"
        "这些命中不属于当前 donor，不作为本版排除项。"
    )
    if beyonce["examples"]:
        add("")
        add("| 位置 | 上下文 |")
        add("| --- | --- |")
        for example in beyonce["examples"]:
            add(f"| `{example['file']}:{example['line']}` | `{md_cell(example['snippet'])}` |")
    add("")
    add(
        "城市字符串次数把 `workspace/data/tofu` 的 jsonl 和 "
        "`Right-to-be-forgotten/data/tofu` 的 json 各计一次，perturbed 改写也会重复计入。"
        "样例在 `validation.jsonl` 的 `overlap.examples`。"
    )
    add("")
    add("## 4. 来源审核")
    add("")
    add(
        "在线 Wikidata："
        f"`{context['wikidata_outcome']}`。"
        "内部字段核对不能代替在线 P19。不确定项保持待审，没有猜测出生地。"
    )
    add("")
    add("| donor | QID | 候选城市 | 内部一致性 | 在线 P19 | 来源门 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for record in context["records"]:
        source = record["source"]
        add(
            f"| {record['donor_id']} {md_cell(record['donor_entity'])} | {source['wikidata_qid']} | "
            f"{md_cell(source['donor_fact_value'])} | {source['internal_consistency']} | "
            f"{source['live_wikidata_p19']} | {source['gate']} |"
        )
    add("")
    pending_notes = [
        f"{item['donor_id']}（{item['granularity_note']}）"
        for item in context["source_checks"]
        if item["granularity_note"] != "city_label_in_candidate_file_not_remotely_confirmed"
    ]
    add(f"候选笔记要求远程确认粒度：{pending_notes or '无'}。")
    add(f"内部一致性失败：{context['consistency_failures'] or '无'}。")
    add("")
    add("规格文件 `workspace/方向I_prompt_only_activation_donor构造规格_v0.3.md`："
        f"{'存在' if context['spec_present'] else '不存在'}。"
        "`y_plus` span 的 sum/mean 聚合没有可冻结定义，本次不选定。")
    add("")
    add("## 5. Tokenizer 与位置")
    add("")
    add("`q_plus` 和 `q_minus` 都只编码问题 prompt。`y_plus` 的 token span 是城市串接在 `q_plus` prompt 之后的位置，没有计算分数。")
    add("")
    add("| pair | 人名 token 数 | 人名 span | 隔离 | y_plus | y_plus token 数 | y_plus span | 前缀一致 |")
    add("| --- | ---: | --- | --- | --- | ---: | --- | --- |")
    for record in context["records"]:
        plus = record["q_plus"]["entity_span"]
        city = record["q_plus"]["continuation_span"]
        span = plus["token_span"]
        city_span = city["span"]
        add(
            f"| {record['pair_id']} | {len(plus.get('token_indices') or [])} | "
            f"{span.get('start')}:{span.get('end')} | {plus['status']} | "
            f"{md_cell(record['y_plus'])} | {city['token_count']} | "
            f"{city_span['start']}:{city_span['end']} | "
            f"{'是' if city['prefix_matches_prompt'] else '否'} |"
        )
    add("")
    add(
        f"人名 span 失败：{context['entity_span_failures'] or '无'}。"
        f"`y_plus` 前缀不一致：{context['continuation_mismatches'] or '无'}。"
    )
    add("")
    hook = context["hook"]
    add(
        f"hook 坐标：`{hook['status']}`。"
        f"`activation_editor` 出现在 `{hook['file']}`："
        f"{'是' if hook['activation_editor_present'] else '否'}。"
        f"`LlamaMLP.forward` 返回 `down_proj`："
        f"{'是' if hook['mlp_forward_returns_down_proj'] else '否'}。"
        "没有可对照的补丁坐标，位置映射门记为阻断。"
    )
    add("")
    example = context["records"][0]
    minus = example["q_minus"]["entity_span"]
    add("promptdonor-01 `q_minus` prompt-only 样例：")
    add("")
    add(f"- prompt：`{example['q_minus']['prompt_text']}`")
    add(f"- token IDs：`{minus['prompt_token_ids']}`")
    add(f"- tokens：`{minus['prompt_token_strings']}`")
    add(
        f"- 人名 span：[{minus['token_span']['start']}, {minus['token_span']['end']})，"
        f"覆盖 `{minus['covered_text']}`"
    )
    add("")
    add("## 6–7. 激活与 patch")
    add("")
    add("没有加载权重，没有前向，没有提取 `a_plus`/`a_minus`。")
    add("激活缓存：`blocked_no_cuda`。没有 cache key、shape 或重复运行结果。")
    add("patch 对照：`not_run_blocked_no_cuda`。没有 no-op、匹配 donor、shuffle donor 或 matched-null。")
    add("没有计算 `logit(y_plus)-logit(y_minus)`，也没有运行 K2。")
    add("")
    add("## 8. Go/no-go")
    add("")
    add(f"- unique clean clusters：{counts['unique_clean']}")
    add(f"- unique donor clusters：{counts['unique_donors']}")
    add(f"- 评测去重排除：{counts['excluded_donors']}")
    add(f"- 建议：**{context['decision']}**")
    add("")
    add("恢复该臂之前需要同时满足：来源在线审核通过，评测去重没有人物或事实簇冲突，真实 hook 的坐标映射通过，激活缓存在目标设备上可复现，并且 matched-null 对照使用已冻结的 `y_plus` span score。")
    add("")
    add("## 本阶段没有做的事")
    add("")
    add("没有修改候选、TOFU 评测文件或 v0.1 验证结果。没有运行激活定位。没有运行 Teacher/Student 训练。")
    add("")
    add(f"结果目录：`{OUT_DIR.relative_to(REPO)}`。")
    add(f"生成时间：`{context['generated_at']}`。")
    add(f"`validation.jsonl` SHA256：`{context['validation_sha256']}`。")
    add(f"本脚本 SHA256：`{context['script_sha256']}`。")
    add("`summary.md` 与上述文件的哈希写在 `output_hashes.json`。")
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    if OUT_DIR.resolve() == CANDIDATE_DIR.resolve():
        raise RuntimeError("output directory must stay outside the candidate directory")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    git_commit = git_text(["rev-parse", "HEAD"])
    git_branch = git_text(["rev-parse", "--abbrev-ref", "HEAD"])
    git_status = git_text(["status", "--porcelain"])

    candidate_hashes = [(name, sha256_file(CANDIDATE_DIR / name)) for name in CANDIDATE_FILES]
    code_hashes = [(str(path.relative_to(REPO)), sha256_file(path)) for path in CODE_FILES]
    checkpoint_hashes = [(name, sha256_file(SNAPSHOT / name)) for name in CHECKPOINT_FILES]

    pairs, pair_errors = read_jsonl(CANDIDATE_DIR / "candidate_pairs.jsonl")
    clean_rows, clean_errors = read_jsonl(CANDIDATE_DIR / "clean_facts.jsonl")
    donors, donor_errors = read_jsonl(CANDIDATE_DIR / "donor_entities.jsonl")
    manifest = json.loads((CANDIDATE_DIR / "manifest.json").read_text(encoding="utf-8"))
    parse_errors = pair_errors + clean_errors + donor_errors

    forget_lines = [line for line in FORGET05.read_text(encoding="utf-8").splitlines() if line.strip()]
    forget05_sha = sha256_file(FORGET05)
    forget05_ok = (
        len(forget_lines) == EXPECTED_FORGET05_ROWS
        and forget05_sha == EXPECTED_FORGET05_SHA256
        and len(forget_lines) == manifest["clean_rows"]
        and forget05_sha == manifest["clean_source_sha256"]
    )

    clean_by_id = {row["clean_cluster_id"]: row for row in clean_rows}
    donors_by_id = {row["donor_id"]: row for row in donors}
    integrity_issues = list(parse_errors)
    if len(pairs) != 10 or len(clean_rows) != 10 or len(donors) != 10:
        integrity_issues.append("row_count_is_not_10")
    if manifest.get("pair_count") != len(pairs):
        integrity_issues.append("manifest_pair_count_mismatch")
    if manifest.get("protocol") != "prompt_only_activation_donor_v0.3":
        integrity_issues.append("manifest_protocol_mismatch")
    if not forget05_ok:
        integrity_issues.append("forget05_hash_or_row_count_mismatch")

    clean_results = []
    for pair in pairs:
        result = verify_clean_row(
            pair,
            clean_by_id.get(pair.get("clean_cluster_id")),
            donors_by_id.get(pair.get("donor_id")),
            forget_lines,
        )
        clean_results.append(result)
        integrity_issues.extend(
            f"{result['pair_id']}:{issue}" for issue in result["issues"]
        )
    unique_clean = {row.get("clean_cluster_id") for row in pairs}
    unique_donors = {row.get("donor_id") for row in pairs}
    reuse_rows = len(pairs) - len(unique_donors)
    if len(unique_clean) != 10:
        integrity_issues.append("unique_clean_clusters_not_10")
    if len(unique_donors) != 10:
        integrity_issues.append("unique_donor_clusters_not_10")
    if reuse_rows:
        integrity_issues.append("donor_reused")

    candidate_blob = "\n".join(
        (CANDIDATE_DIR / name).read_text(encoding="utf-8")
        for name in ("candidate_pairs.jsonl", "clean_facts.jsonl", "donor_entities.jsonl")
    )
    beyonce_in_candidates = beyonce_in_text(candidate_blob)
    if beyonce_in_candidates:
        integrity_issues.append("beyonce_present_in_candidates")
    donor03 = donors_by_id.get("donor-03", {}).get("entity")
    donor10 = donors_by_id.get("donor-10", {}).get("entity")
    if donor03 != "Muhammad Ali":
        integrity_issues.append("donor03_is_not_muhammad_ali")
    if donor10 != "Lionel Messi":
        integrity_issues.append("donor10_is_not_lionel_messi")

    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    stored_chat_template = tokenizer.chat_template or ""
    manual_example = prompt_text("Where was Katherine Mansfield born?")
    apply_example = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Where was Katherine Mansfield born?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    nous_same = []
    nous_different = []
    nous_missing = not NOUS_SNAPSHOT.exists()
    if not nous_missing:
        for name in TOKENIZER_COMPARE_FILES:
            local_hash = sha256_file(SNAPSHOT / name)
            nous_hash = sha256_file(NOUS_SNAPSHOT / name)
            (nous_same if local_hash == nous_hash else nous_different).append(name)

    eval_rows, skipped_files = load_eval_lines()
    overlap = scan_overlap(eval_rows)
    beyonce_eval = scan_beyonce(eval_rows)
    config = json.loads((SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    ref_path = SNAPSHOT.parent.parent / "refs" / "main"
    snapshot_commit = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else SNAPSHOT.name
    device = cuda_status()
    hook = hook_status()

    live_by_donor = {}
    wikidata_error = None
    for donor in donors:
        live = live_p19_status(donor)
        live_by_donor[donor["donor_id"]] = live
        if wikidata_error is None and live.get("error"):
            wikidata_error = live["error"]
            break
    if wikidata_error is not None:
        for donor in donors:
            if donor["donor_id"] not in live_by_donor or "error" not in live_by_donor[donor["donor_id"]]:
                live_by_donor[donor["donor_id"]] = {
                    "status": "pending",
                    "live_wikidata_p19": "blocked_no_network",
                    "error": wikidata_error,
                }

    records = []
    source_checks = []
    for pair in pairs:
        donor = donors_by_id.get(pair["donor_id"])
        internal = internal_source_check(pair, donor)
        live = live_by_donor.get(pair["donor_id"], {"status": "pending", "live_wikidata_p19": "blocked_no_network"})
        gate = source_gate(internal, live)
        source = {
            **internal,
            "live_wikidata_p19": live.get("live_wikidata_p19"),
            "live_detail": live,
            "gate": gate,
        }
        source_checks.append(source)
        donor_overlap = overlap[pair["donor_id"]]
        q_plus = encode_question(tokenizer, pair["q_plus"], pair["clean_entity"], pair["y_plus"])
        q_minus = encode_question(tokenizer, pair["q_minus"], pair["donor_entity"], None)
        entity_ok = (
            q_plus["entity_span"]["status"] == "pass" and q_minus["entity_span"]["status"] == "pass"
        )
        continuation_ok = q_plus["continuation_span"]["prefix_matches_prompt"]
        if donor_overlap["excluded"]:
            overlap_gate = "fail"
        else:
            overlap_gate = "pass"
        if hook["status"] != "blocked_no_activation_editor" and entity_ok and continuation_ok:
            position_gate = "pass"
        elif not entity_ok or not continuation_ok:
            position_gate = "fail"
        else:
            position_gate = "blocked"
        records.append(
            {
                "pair_id": pair["pair_id"],
                "protocol": pair.get("protocol"),
                "relation_id": pair.get("relation_id"),
                "clean_cluster_id": pair.get("clean_cluster_id"),
                "clean_entity": pair.get("clean_entity"),
                "donor_id": pair.get("donor_id"),
                "donor_entity": pair.get("donor_entity"),
                "y_plus": pair.get("y_plus"),
                "donor_fact_value": pair.get("donor_fact_value"),
                "y_minus_present": "y_minus" in pair,
                "gates": {
                    "integrity": "pass" if not integrity_issues else "fail",
                    "eval_overlap": overlap_gate,
                    "source": gate,
                    "prompt_encoding": "pass" if entity_ok and continuation_ok else "fail",
                    "hook_coordinate_mapping": hook["status"],
                    "position_mapping": position_gate,
                    "activation_cache": "blocked_no_cuda",
                    "patch_controls": "not_run_blocked_no_cuda",
                    "span_aggregation": "not_frozen_spec_missing",
                },
                "exclusion_reasons": donor_overlap["exclusion_reasons"],
                "chat_template": {
                    "id": "llama2-7b-manual-inst-prompt-only",
                    "question_start_tag": QUESTION_START,
                    "question_end_tag": QUESTION_END,
                    "answer_tag": ANSWER_TAG,
                    "template_sha256": sha256_text(MANUAL_TEMPLATE),
                    "apply_chat_template_used": False,
                    "padding_applied": False,
                    "truncation_applied": False,
                },
                "q_plus": q_plus,
                "q_minus": q_minus,
                "source": source,
                "overlap": donor_overlap,
                "activation": {
                    "forward_pass_run": False,
                    "a_plus": None,
                    "a_minus": None,
                    "hook": None,
                    "layer": None,
                    "position": None,
                    "shape": None,
                    "dtype": None,
                    "cache_key": None,
                    "status": "blocked_no_cuda",
                },
                "patch": {
                    "status": "not_run_blocked_no_cuda",
                    "conditions": [],
                    "y_plus_span_score": None,
                    "logit_difference_computed": False,
                    "k2_computed": False,
                },
            }
        )

    results_path = OUT_DIR / "validation.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    validation_sha = sha256_file(results_path)

    rehash = [(name, sha256_file(CANDIDATE_DIR / name)) for name in CANDIDATE_FILES]
    if rehash != candidate_hashes:
        raise RuntimeError("candidate files changed during validation")

    hit_terms = []
    person_examples = []
    for donor_id, summary in overlap.items():
        for row in summary["term_counts"]:
            if row["count"]:
                hit_terms.append({"donor_id": donor_id, **row})
        for example in summary["examples"]:
            if example["category"] in PERSON_CATEGORIES:
                person_examples.append((donor_id, example))

    entity_span_failures = [
        record["pair_id"]
        for record in records
        if record["gates"]["prompt_encoding"] != "pass"
    ]
    continuation_mismatches = [
        record["pair_id"]
        for record in records
        if not record["q_plus"]["continuation_span"]["prefix_matches_prompt"]
    ]
    consistency_failures = [
        item["donor_id"] for item in source_checks if item["internal_consistency"] != "pass"
    ]
    excluded_donors = [donor_id for donor_id, summary in overlap.items() if summary["excluded"]]
    source_pass = sum(1 for item in source_checks if item["gate"] == "pass")
    source_fail = sum(1 for item in source_checks if item["gate"] == "fail")
    source_pending = sum(1 for item in source_checks if item["gate"] == "pending")
    overlap_fail = len(excluded_donors)
    position_blocked = sum(1 for record in records if record["gates"]["position_mapping"] == "blocked")
    position_fail = sum(1 for record in records if record["gates"]["position_mapping"] == "fail")
    integrity_gate = "pass" if not integrity_issues else "fail"

    blocking_reasons = []
    if integrity_gate != "pass":
        blocking_reasons.append("文件完整性未通过：" + "；".join(integrity_issues))
    if source_pass != len(records):
        blocking_reasons.append(
            f"来源门未通过：通过 {source_pass}，失败 {source_fail}，待审 {source_pending}。"
            "在线 P19 为 "
            + (wikidata_error or "未完成")
            + "。"
        )
    if overlap_fail:
        blocking_reasons.append("评测去重排除 donor：" + ", ".join(excluded_donors) + "。")
    if position_blocked or position_fail:
        blocking_reasons.append(
            f"位置映射未通过：阻断 {position_blocked}，失败 {position_fail}。"
            "当前 `my_llama.py` 没有 `activation_editor`。"
        )
    blocking_reasons.append("激活缓存未复现：CUDA 不可用，权重未加载。")
    blocking_reasons.append("matched-null 对照未运行。")
    blocking_reasons.append("`y_plus` span 聚合未冻结：规格文件不在仓库中。")
    decision = "NO-GO"

    jsonl_files = sorted(path.name for path in TOFU_JSONL_DIR.glob("*.jsonl") if inventory_kind(path.name))
    json_files = sorted(path.name for path in TOFU_JSON_DIR.glob("*.json") if inventory_kind(path.name))
    clean_failures = [row for row in clean_results if row["status"] != "pass"]
    context = {
        "decision": decision,
        "blocking_reasons": blocking_reasons,
        "gate_rows": [
            {
                "name": "文件完整性",
                "pass": len(records) if integrity_gate == "pass" else 0,
                "fail": 0 if integrity_gate == "pass" else len(records),
                "pending": 0,
                "blocked": 0,
                "note": "行数、哈希、源行、无 y_minus、无 Beyoncé",
            },
            {
                "name": "评测去重",
                "pass": len(records) - overlap_fail,
                "fail": overlap_fail,
                "pending": 0,
                "blocked": 0,
                "note": "人物或事实簇命中才排除；城市字符串不计为排除",
            },
            {
                "name": "来源",
                "pass": source_pass,
                "fail": source_fail,
                "pending": source_pending,
                "blocked": 0,
                "note": "在线 P19 未完成，记为待审而不是通过",
            },
            {
                "name": "位置映射",
                "pass": 0,
                "fail": position_fail,
                "pending": 0,
                "blocked": position_blocked,
                "note": "token span 已记录；hook 坐标阻断",
            },
            {
                "name": "激活缓存",
                "pass": 0,
                "fail": 0,
                "pending": 0,
                "blocked": len(records),
                "note": "blocked_no_cuda",
            },
            {
                "name": "patch 对照",
                "pass": 0,
                "fail": 0,
                "pending": 0,
                "blocked": len(records),
                "note": "not_run_blocked_no_cuda",
            },
        ],
        "counts": {
            "pairs": len(pairs),
            "clean": len(clean_rows),
            "donors": len(donors),
            "unique_clean": len(unique_clean),
            "unique_donors": len(unique_donors),
            "reuse_rows": reuse_rows,
            "forget05_rows": len(forget_lines),
            "excluded_donors": len(excluded_donors),
        },
        "manifest": manifest,
        "records": records,
        "parse_errors": parse_errors,
        "integrity_issues": integrity_issues,
        "forget05_sha256": forget05_sha,
        "forget05_ok": forget05_ok,
        "clean_source_summary": (
            "10 条来源行的 question、answer 和 y_plus 都与 forget05.jsonl 一致。"
            if not clean_failures
            else json.dumps(clean_failures, ensure_ascii=False)
        ),
        "candidate_hashes": candidate_hashes,
        "code_hashes": code_hashes,
        "checkpoint_hashes": checkpoint_hashes,
        "checkpoint": {
            "repo_id": "open-unlearning/tofu_Llama-2-7b-chat-hf_full",
            "snapshot_commit": snapshot_commit,
            "snapshot_path": str(SNAPSHOT),
            "name_or_path": config.get("_name_or_path"),
        },
        "nous_compare": {
            "snapshot": str(NOUS_SNAPSHOT),
            "same_files": nous_same,
            "different_files": nous_different,
            "missing": nous_missing,
        },
        "tokenizer": {
            "class_name": type(tokenizer).__name__,
            "transformers_version": transformers.__version__,
            "vocab_size": tokenizer.vocab_size,
            "len": len(tokenizer),
            "bos_token": tokenizer.bos_token,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "padding_side": tokenizer.padding_side,
        },
        "device": device,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_status": git_status,
        "manual_template_sha256": sha256_text(MANUAL_TEMPLATE),
        "stored_chat_template_sha256": sha256_text(stored_chat_template),
        "apply_chat_template_example": apply_example,
        "manual_template_example": manual_example,
        "inventory": {
            "jsonl_files": len(jsonl_files),
            "json_files": len(json_files),
            "lines": len(eval_rows),
            "skipped": skipped_files,
        },
        "overlap": overlap,
        "hit_terms": hit_terms,
        "person_examples": person_examples,
        "beyonce_in_candidates": beyonce_in_candidates,
        "beyonce_candidate_hits": ["Beyoncé"] if beyonce_in_candidates else [],
        "beyonce_eval": beyonce_eval,
        "donor03_entity": donor03,
        "donor10_entity": donor10,
        "wikidata_outcome": wikidata_error or "completed",
        "source_checks": source_checks,
        "consistency_failures": consistency_failures,
        "spec_present": SPEC_FILE.exists(),
        "entity_span_failures": entity_span_failures,
        "continuation_mismatches": continuation_mismatches,
        "hook": hook,
        "generated_at": generated_at,
        "validation_sha256": validation_sha,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    summary = build_summary(context)
    summary_path = OUT_DIR / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    hashes = {
        "validation.jsonl": validation_sha,
        "summary.md": sha256_file(summary_path),
        "run_validation.py": context["script_sha256"],
    }
    (OUT_DIR / "output_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_rehash = [(name, sha256_file(CANDIDATE_DIR / name)) for name in CANDIDATE_FILES]
    if final_rehash != candidate_hashes:
        raise RuntimeError("candidate files changed while writing the report")


if __name__ == "__main__":
    main()

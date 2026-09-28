#!/usr/bin/env python3
"""Download is separate. This script writes the Books corpus, alignment audit, and probes.

Forget and retain text must already sit in corpora/books/, or pass --download.
Multi-hop questions are probes only and are never written into forget.txt.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
BOOKS_JSON = REPO / "workspace/leak-resistant-unlearning/data/books_augmented.json"
CORPUS = ROOT / "corpora/books"
OUT = ROOT / "books"

ARTICLES = ("the ", "a ", "an ")
STOP_ENTITIES = {
    "news",
    "time",
    "people",
    "person",
    "thing",
    "things",
    "something",
    "someone",
    "yes",
    "no",
}


def normalize(text: str) -> str:
    text = text.casefold()
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains(haystack: str, needle: str) -> bool:
    needle = normalize(needle)
    if not needle:
        return False
    return needle in haystack


def entity_key(text: str) -> str:
    key = normalize(text)
    for article in ARTICLES:
        if key.startswith(article):
            key = key[len(article) :]
            break
    return key


def keep_entity(text: str) -> bool:
    key = entity_key(text)
    if len(key) < 4 or key in STOP_ENTITIES:
        return False
    if key.isdigit():
        return False
    return True


def download_corpora() -> None:
    from datasets import load_dataset

    CORPUS.mkdir(parents=True, exist_ok=True)
    for split, name in (("forget", "forget.txt"), ("retain1", "retain1.txt")):
        ds = load_dataset("muse-bench/MUSE-Books", "raw", split=split)
        text = "\n\n".join(ds["text"])
        (CORPUS / name).write_text(text, encoding="utf-8")


def load_corpus() -> tuple[str, str]:
    forget_path = CORPUS / "forget.txt"
    retain_path = CORPUS / "retain1.txt"
    if not forget_path.exists() or not retain_path.exists():
        download_corpora()
    return forget_path.read_text(encoding="utf-8"), retain_path.read_text(encoding="utf-8")


def collect_entities(item: dict) -> list[str]:
    found: list[str] = []
    answer = item.get("rewritten_answer") or ""
    if keep_entity(answer):
        found.append(entity_key(answer))
    for fact in item.get("facts") or []:
        for field in ("subject", "object", "entity", "category", "name1", "name2"):
            value = fact.get(field) or ""
            if keep_entity(value):
                found.append(entity_key(value))
    for hop in item.get("single_hops") or []:
        if keep_entity(hop.get("answer") or ""):
            found.append(entity_key(hop["answer"]))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download_corpora()

    forget_text, retain_text = load_corpus()
    forget_norm = normalize(forget_text)
    books = json.loads(BOOKS_JSON.read_text(encoding="utf-8"))

    single_rows = []
    multi_rows = []
    entities: set[str] = set()
    eval_single = []
    eval_multi = []
    support_index: dict[tuple[str, str], str] = {}

    for index, item in enumerate(books["single_hop"]["questions"]):
        answer = item.get("rewritten_answer") or ""
        if not answer.strip():
            row = {"index": index, "kept": False, "reason": "empty_answer"}
        elif contains(forget_norm, answer):
            row = {"index": index, "kept": True, "reason": "answer_in_forget"}
            entities.update(collect_entities(item))
            record_id = f"books-sh-{index}"
            eval_single.append(
                {
                    "id": record_id,
                    "question": item.get("rewritten_question") or "",
                    "answer": answer,
                    "aliases": [],
                    "reasoning_type": "single_hop",
                    "support_ids": [],
                }
            )
            support_index[(normalize(item.get("rewritten_question") or ""), normalize(answer))] = record_id
        else:
            row = {"index": index, "kept": False, "reason": "answer_not_in_forget"}
        single_rows.append(row)

    for index, item in enumerate(books["multi_hop"]["questions"]):
        premise1 = item.get("rewritten_premise1") or ""
        premise2 = item.get("rewritten_premise2") or ""
        missing = []
        if not premise1.strip():
            missing.append("empty_premise1")
        elif not contains(forget_norm, premise1):
            missing.append("premise1_not_in_forget")
        if not premise2.strip():
            missing.append("empty_premise2")
        elif not contains(forget_norm, premise2):
            missing.append("premise2_not_in_forget")
        hops = item.get("single_hops") or []
        if not hops:
            missing.append("no_supports")
        for hop_index, hop in enumerate(hops):
            if isinstance(hop, str):
                missing.append(f"support_not_question:{hop_index}")
            elif not contains(forget_norm, hop.get("answer") or ""):
                missing.append(f"support_answer_not_in_forget:{hop_index}")
        if missing:
            multi_rows.append({"index": index, "kept": False, "reason": ",".join(missing), "reasoning_type": item.get("type")})
            continue

        entities.update(collect_entities(item))
        support_ids = []
        for hop_index, hop in enumerate(item.get("single_hops") or []):
            question = hop.get("question") or ""
            answer = hop.get("answer") or ""
            key = (normalize(question), normalize(answer))
            support_id = support_index.get(key)
            if support_id is None:
                support_id = f"books-sup-{index}-{hop_index}"
                support_index[key] = support_id
                eval_single.append(
                    {
                        "id": support_id,
                        "question": question,
                        "answer": answer,
                        "aliases": [],
                        "reasoning_type": "single_hop",
                        "support_ids": [],
                    }
                )
            support_ids.append(support_id)

        eval_multi.append(
            {
                "id": f"books-mh-{index}",
                "question": item.get("rewritten_question") or "",
                "answer": item.get("rewritten_answer") or "",
                "aliases": [],
                "reasoning_type": item.get("type") or "",
                "support_ids": support_ids,
            }
        )
        multi_rows.append(
            {
                "index": index,
                "kept": True,
                "reason": "premises_and_support_answers_in_forget",
                "reasoning_type": item.get("type"),
            }
        )

    passages = []
    dropped_passages = 0
    for block in re.split(r"\n\s*\n", retain_text):
        block = block.strip()
        if not block:
            continue
        folded = normalize(block)
        hit = next((entity for entity in entities if entity in folded), None)
        if hit:
            dropped_passages += 1
            continue
        passages.append({"text": block})

    OUT.mkdir(parents=True, exist_ok=True)
    alignment = {
        "forget_chars": len(forget_text),
        "retain_chars": len(retain_text),
        "match": "casefold substring after quote and whitespace normalization",
        "single_hop": single_rows,
        "multi_hop": multi_rows,
        "summary": {
            "single_hop_total": len(single_rows),
            "single_hop_kept": sum(row["kept"] for row in single_rows),
            "multi_hop_total": len(multi_rows),
            "multi_hop_kept": sum(row["kept"] for row in multi_rows),
            "support_questions_added": sum(1 for row in eval_single if row["id"].startswith("books-sup-")),
            "entities": len(entities),
            "retain_passages_kept": len(passages),
            "retain_passages_dropped": dropped_passages,
            "multi_hop_note": (
                "Kept only when both rewritten premises and every support answer "
                "occur in the forget text. Rewritten premises are paraphrases, so "
                "a premise miss drops the item."
            ),
            "retain_note": (
                "Passages are the MUSE retain1 documents. A document is dropped "
                "when it contains an entity taken from a kept single-hop answer."
            ),
        },
    }
    (OUT / "alignment.json").write_text(json.dumps(alignment, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "eval_singlehop.json").write_text(json.dumps(eval_single, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "eval_multihop.json").write_text(json.dumps(eval_multi, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "retain_passages.jsonl").open("w", encoding="utf-8") as handle:
        for passage in passages:
            handle.write(json.dumps(passage, ensure_ascii=False) + "\n")
    print(json.dumps(alignment["summary"], indent=2))


if __name__ == "__main__":
    main()

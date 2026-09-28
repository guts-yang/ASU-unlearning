#!/usr/bin/env python3
"""Build MQuAKE forget text, retain text, and probes.

Forget sentences come only from multi-hop single_hops. Multi-hop questions
are written to the probe file and never to forget.txt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SOURCE = REPO / "workspace/leak-resistant-unlearning/data/mquake_augmented.json"
OUT = ROOT / "mquake"


def normalize(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def forget_sentence(hop: dict) -> str:
    cloze = (hop.get("cloze") or "").strip()
    answer = (hop.get("answer") or "").strip()
    if cloze and answer:
        sentence = cloze.rstrip(" .")
        return f"{sentence} {answer}."
    question = (hop.get("question") or "").strip()
    if question and answer:
        return f"{question} {answer}."
    return ""


def subject_of(hop: dict) -> str:
    cloze = (hop.get("cloze") or "").strip()
    if " is " in cloze:
        return cloze.split(" is ")[0].strip()
    # "X plays the position of" / "The official language of Y is" handled above.
    # Remaining cloze templates put the subject after " of " at the end, or first.
    match = re.match(r"^(.*?) (?:plays|was|were)\b", cloze)
    if match:
        return match.group(1).strip()
    return ""


def entities_of_hop(hop: dict) -> set[str]:
    found = set()
    subject = subject_of(hop)
    if len(subject) >= 3:
        found.add(normalize(subject))
    answer = hop.get("answer") or ""
    if len(answer.strip()) >= 3:
        found.add(normalize(answer))
    for alias in hop.get("answer_alias") or []:
        if len(alias.strip()) >= 3:
            found.add(normalize(alias))
    return {item for item in found if item}


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    sentences: list[str] = []
    seen_sentences: set[str] = set()
    forget_entities: set[str] = set()
    eval_single_by_key: dict[tuple[str, str], dict] = {}
    eval_multi = []

    single_by_case = {item["case_id"]: item for item in data["single_hop"]["questions"]}

    for item in data["multi_hop"]["questions"]:
        support_ids = []
        for hop_index, hop in enumerate(item.get("single_hops") or []):
            sentence = forget_sentence(hop)
            key = normalize(sentence)
            if sentence and key not in seen_sentences:
                seen_sentences.add(key)
                sentences.append(sentence)
            forget_entities.update(entities_of_hop(hop))

            question = hop.get("question") or ""
            answer = hop.get("answer") or ""
            probe_key = (normalize(question), normalize(answer))
            record = eval_single_by_key.get(probe_key)
            if record is None:
                record = {
                    "id": f"mq-sh-{item['case_id']}-{hop_index}",
                    "question": question,
                    "answer": answer,
                    "aliases": list(hop.get("answer_alias") or []),
                    "reasoning_type": "single_hop",
                    "support_ids": [],
                }
                eval_single_by_key[probe_key] = record
            support_ids.append(record["id"])

        paraphrases = item.get("questions") or []
        question = paraphrases[0] if paraphrases else ""
        eval_multi.append(
            {
                "id": f"mq-mh-{item['case_id']}",
                "question": question,
                "answer": item.get("answer") or "",
                "aliases": list(item.get("answer_alias") or []),
                "reasoning_type": item.get("reasoning_type") or "",
                "support_ids": support_ids,
            }
        )

    # Top-level single-hop questions are the same cases. Attach their ids when
    # the question text matches a support, and keep any that do not.
    for item in data["single_hop"]["questions"]:
        question = item.get("question") or ""
        answer = item.get("answer") or ""
        probe_key = (normalize(question), normalize(answer))
        if probe_key in eval_single_by_key:
            record = eval_single_by_key[probe_key]
            aliases = list(dict.fromkeys(record["aliases"] + list(item.get("answer_alias") or [])))
            record["aliases"] = aliases
            continue
        eval_single_by_key[probe_key] = {
            "id": f"mq-sh-{item['case_id']}",
            "question": question,
            "answer": answer,
            "aliases": list(item.get("answer_alias") or []),
            "reasoning_type": "single_hop",
            "support_ids": [],
        }

    retain_lines = []
    overlap = 0
    for item in data["single_hop"]["questions"]:
        question = item.get("question") or ""
        answer = item.get("answer") or ""
        blob = normalize(f"{question} {answer}")
        if any(entity in blob for entity in forget_entities):
            overlap += 1
            continue
        retain_lines.append(f"{question} {answer}".strip())

    forget_blob = "\n".join(sentences)
    for row in eval_multi:
        if row["question"] and row["question"] in forget_blob:
            raise SystemExit(f"multi-hop question leaked into forget.txt: {row['id']}")
    (OUT / "forget.txt").write_text(forget_blob + ("\n" if sentences else ""), encoding="utf-8")
    (OUT / "retain.txt").write_text("\n".join(retain_lines) + ("\n" if retain_lines else ""), encoding="utf-8")
    eval_single = list(eval_single_by_key.values())
    (OUT / "eval_singlehop.json").write_text(json.dumps(eval_single, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "eval_multihop.json").write_text(json.dumps(eval_multi, ensure_ascii=False, indent=2), encoding="utf-8")

    type_counts: dict[str, int] = {}
    for row in eval_multi:
        type_counts[row["reasoning_type"]] = type_counts.get(row["reasoning_type"], 0) + 1

    report = {
        "forget_sentences": len(sentences),
        "forget_entities": len(forget_entities),
        "single_hop_cases": len(single_by_case),
        "single_hop_overlapping_forget_entities": overlap,
        "retain_sentences": len(retain_lines),
        "eval_singlehop": len(eval_single),
        "eval_multihop": len(eval_multi),
        "multi_hop_by_type": type_counts,
        "subset_file": None,
        "comparable_to_2026_09_27_muse_or_tofu": False,
        "retain_note": (
            "All 3000 single-hop rows overlap the forget entity set, because each "
            "row is itself a multi-hop support and its answer is one of the forget "
            "entities. This JSON does not contain an entity-disjoint retain set. "
            "Do not compare this line with the 2026-09-27 MUSE or TOFU runs."
            if len(retain_lines) == 0
            else "Retain sentences are single-hop items whose question and answer contain no forget entity."
        ),
    }
    (OUT / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

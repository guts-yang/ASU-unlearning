"""Build the proposed sidecar from exact substrings. Does not edit source data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from proposals import PREANNOTATION_NOTES, PROPOSALS, UNRESOLVED
from span_match import locate_unique, ranges_overlap

SOURCE = Path(
    "/Users/chenyliao/Repositories/negative-preference-optimization/TOFU/TOFU_data/forget05.json"
)
EXPECTED_SHA = "88542818f98fd6f880ecaea71bc3f3e7a34f959944e5ff1cdef7ea8c5c137173"
CANONICAL_SHA = "a8c207a905af73ea4797b1af92a14cf458f42df567c941f7284f87d930096f4b"
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows() -> list[dict]:
    if sha256(SOURCE) != EXPECTED_SHA:
        raise SystemExit("candidate source SHA changed; refusing to annotate a different file")
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    if len(rows) != 200 or set(PROPOSALS) | set(UNRESOLVED) != set(range(200)):
        raise SystemExit("candidate pilot requires exactly the 200 source rows")
    return rows


def build_record(index: int, row: dict) -> dict:
    answer = row["answer"]
    spans = []
    unresolved_reasons = []
    if index in UNRESOLVED:
        unresolved_reasons.append(UNRESOLVED[index])
    for layer, text, fact_type, reason in PROPOSALS.get(index, []):
        located = locate_unique(answer, text)
        if isinstance(located, str):
            unresolved_reasons.append(f"{layer}:{text!r}: {located}")
            continue
        start, end = located
        spans.append(
            {
                "start_char": start,
                "end_char": end,
                "text": answer[start:end],
                "layer": layer,
                "fact_type": fact_type,
                "fact_id": "",
                "reason": reason,
                "status": "proposed",
            }
        )
    if ranges_overlap(spans):
        unresolved_reasons.append("resolved spans overlap")
        spans = []
    if unresolved_reasons:
        status = "unresolved"
    elif any(span["layer"] == "fact_value" for span in spans):
        status = "proposed"
    else:
        status = "no_target"
    return {
        "dataset_status": "local_candidate_pilot",
        "canonical_forget05_task_id_1": False,
        "canonical_sha256_not_met": CANONICAL_SHA,
        "candidate_source_sha256": EXPECTED_SHA,
        "candidate_row_index": index,
        "author_or_entity_id": None,
        "task_id": None,
        "split": None,
        "forget_scope": "question_fact",
        "question": row["question"],
        "answer": answer,
        "target_spans": spans,
        "annotation_status": status,
        "unresolved_reasons": unresolved_reasons,
        "annotator": "ai_preannotation_v1",
        "review_status": "pending_human_review",
        "preannotation_note": PREANNOTATION_NOTES.get(index, ""),
        "review_note": "",
    }


def main() -> None:
    records = [build_record(index, row) for index, row in enumerate(load_rows())]
    output = HERE / "proposed.jsonl"
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    counts = {}
    for record in records:
        counts[record["annotation_status"]] = counts.get(record["annotation_status"], 0) + 1
    summary = {
        "records": len(records),
        "status_counts": counts,
        "fact_value_spans": sum(span["layer"] == "fact_value" for record in records for span in record["target_spans"]),
        "relation_spans": sum(span["layer"] == "relation" for record in records for span in record["target_spans"]),
        "main_m_tar_layer": "fact_value",
        "relation_excluded_from_main_m_tar": True,
        "human_review_complete": False,
    }
    (HERE / "proposed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    accidental = [
        record for record in records
        if record["candidate_row_index"] not in UNRESOLVED and record["annotation_status"] != "proposed"
    ]
    for record in accidental:
        print(record["candidate_row_index"], record["annotation_status"], record["unresolved_reasons"])
    if accidental:
        raise SystemExit(f"{len(accidental)} rows failed exact matching")


if __name__ == "__main__":
    main()

"""Write final_spans from AI consensus plus a blind adjudication file.

This is not a human review. Disagreement rows are unresolved unless the blind
annotation confirms one existing fact_value set. The script never writes m_tar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_annotations import classify_row, layer_set, load_jsonl, normalize_status
from span_match import locate_unique, ranges_overlap

HERE = Path(__file__).resolve().parent
PROPOSED = HERE / "proposed.jsonl"
CROSSCHECK = HERE / "independent_ai_crosscheck.jsonl"
FINAL = HERE / "final_spans.jsonl"
STATS = HERE / "agreement_stats.json"
QUEUE = HERE / "blind_queue.jsonl"
CONSENSUS_REVIEWER = "ai_consensus"
DEFAULT_BLIND_REVIEWER = "blind_adjudicator:kimi-k3-max"
DIRECT_CATEGORIES = {"exact_match", "relation_only"}
FACT_TYPES = {"person", "attribute", "date", "location", "work", "other"}
LAYERS = {"fact_value", "relation"}


def needs_blind(category: str) -> bool:
    return category not in DIRECT_CATEGORIES


def copy_spans(spans: list[dict], layer: str | None = None) -> list[dict]:
    copied = []
    for span in spans or []:
        if layer is not None and span.get("layer") != layer:
            continue
        copied.append(
            {
                "start_char": span["start_char"],
                "end_char": span["end_char"],
                "text": span["text"],
                "layer": span["layer"],
                "fact_type": span.get("fact_type"),
                "reason": span.get("reason") or "",
            }
        )
    return sorted(copied, key=lambda span: (span["start_char"], span["end_char"], span["layer"]))


def span_problems(answer: str, spans: list[dict], *, strict_fact_type: bool = False) -> list[str]:
    problems = []
    for span in spans:
        if span.get("layer") not in LAYERS:
            problems.append("invalid layer")
            continue
        fact_type = span.get("fact_type")
        if strict_fact_type and fact_type is not None and fact_type not in FACT_TYPES:
            problems.append("invalid fact_type")
        located = locate_unique(answer, span.get("text"))
        if isinstance(located, str):
            problems.append(located)
            continue
        start, end = located
        if span.get("start_char") != start or span.get("end_char") != end:
            problems.append("offset does not match the unique substring")
        elif answer[span["start_char"] : span["end_char"]] != span.get("text"):
            problems.append("offset does not reproduce substring")
    if ranges_overlap(spans):
        problems.append("spans overlap")
    if not any(span.get("layer") == "fact_value" for span in spans):
        problems.append("annotated row has no fact_value")
    return problems


def terminal_record(
    source: dict,
    *,
    status: str,
    spans: list[dict],
    reviewer: str,
    source_of_spans: str,
    disagreement: bool,
    note: str,
) -> dict:
    return {
        "dataset_status": "local_candidate_pilot",
        "canonical_forget05_task_id_1": False,
        "candidate_row_index": source["candidate_row_index"],
        "question": source["question"],
        "answer": source["answer"],
        "review_status": status,
        "reviewer": reviewer,
        "source_of_spans": source_of_spans,
        "target_spans": spans,
        "disagreement_with_crosscheck": disagreement,
        "review_note": note,
    }


def empty_status_record(source: dict, status: str, reviewer: str, source_of_spans: str, note: str, cross: dict) -> dict:
    return terminal_record(
        source,
        status=status,
        spans=[],
        reviewer=reviewer,
        source_of_spans=source_of_spans,
        disagreement=layer_set(cross.get("target_spans"), "fact_value") != set()
        or layer_set(cross.get("target_spans"), "relation") != set(),
        note=note,
    )


def consensus_record(proposed: dict, cross: dict, category: str) -> dict:
    if proposed["question"] != cross["question"] or proposed["answer"] != cross["answer"]:
        raise ValueError(f"question or answer mismatch at {proposed['candidate_row_index']}")
    status = normalize_status(proposed)
    if status != normalize_status(cross):
        raise ValueError(f"consensus status mismatch at {proposed['candidate_row_index']}")
    if category == "exact_match" and status in {"unresolved", "no_target"}:
        return empty_status_record(
            proposed,
            status,
            CONSENSUS_REVIEWER,
            "ai_consensus",
            f"AI consensus {status}. Not a human adjudication.",
            cross,
        )
    if category == "exact_match":
        spans = copy_spans(proposed.get("target_spans"))
        if layer_set(spans, "fact_value") != layer_set(cross.get("target_spans"), "fact_value"):
            raise ValueError(f"exact_match fact mismatch at {proposed['candidate_row_index']}")
        if layer_set(spans, "relation") != layer_set(cross.get("target_spans"), "relation"):
            raise ValueError(f"exact_match relation mismatch at {proposed['candidate_row_index']}")
        problems = span_problems(proposed["answer"], spans)
        if problems:
            raise ValueError(f"consensus span check failed at {proposed['candidate_row_index']}: {problems}")
        return terminal_record(
            proposed,
            status="adjudicated",
            spans=spans,
            reviewer=CONSENSUS_REVIEWER,
            source_of_spans="ai_consensus",
            disagreement=False,
            note="AI consensus exact_match. Not a human adjudication.",
        )
    if category != "relation_only":
        raise ValueError(f"category {category} is not a consensus row")
    spans = copy_spans(proposed.get("target_spans"), "fact_value")
    if layer_set(spans, "fact_value") != layer_set(cross.get("target_spans"), "fact_value"):
        raise ValueError(f"relation_only fact mismatch at {proposed['candidate_row_index']}")
    problems = span_problems(proposed["answer"], spans)
    if problems:
        raise ValueError(f"relation_only span check failed at {proposed['candidate_row_index']}: {problems}")
    return terminal_record(
        proposed,
        status="adjudicated",
        spans=spans,
        reviewer=CONSENSUS_REVIEWER,
        source_of_spans="ai_consensus",
        disagreement=True,
        note="AI consensus on fact_value only. Differing relations were omitted.",
    )


def blind_problems(answer: str, blind: dict) -> list[str]:
    if blind.get("answer") != answer:
        return ["blind answer does not match the candidate answer"]
    status = blind.get("annotation_status")
    spans = blind.get("target_spans") or []
    if status in {"unresolved", "no_target"}:
        if spans:
            return ["empty status still contains spans"]
        return []
    if status not in {"annotated", "proposed"}:
        return ["blind status is not annotated, unresolved, or no_target"]
    return span_problems(answer, spans, strict_fact_type=True)


def relations_to_copy(side: dict, blind: dict) -> list[dict]:
    side_relations = layer_set(side.get("target_spans"), "relation")
    blind_relations = layer_set(blind.get("target_spans"), "relation")
    if side_relations != blind_relations:
        return []
    return copy_spans(side.get("target_spans"), "relation")


def apply_blind(proposed: dict, cross: dict, blind: dict, reviewer: str) -> dict:
    answer = proposed["answer"]
    problems = blind_problems(answer, blind)
    if problems:
        return empty_status_record(
            proposed,
            "unresolved",
            reviewer,
            "ai_consensus",
            "Blind row failed the character check and was left unresolved.",
            cross,
        )
    status = blind.get("annotation_status")
    if status in {"unresolved", "no_target"}:
        return empty_status_record(
            proposed,
            status,
            reviewer,
            "ai_consensus",
            f"Blind adjudication returned {status}.",
            cross,
        )
    blind_facts = layer_set(blind.get("target_spans"), "fact_value")
    proposed_match = blind_facts == layer_set(proposed.get("target_spans"), "fact_value")
    cross_match = blind_facts == layer_set(cross.get("target_spans"), "fact_value")
    if not proposed_match and not cross_match:
        return empty_status_record(
            proposed,
            "unresolved",
            reviewer,
            "ai_consensus",
            "Blind fact_value matched neither side and was left unresolved.",
            cross,
        )
    if proposed_match and cross_match:
        spans = copy_spans(proposed.get("target_spans"), "fact_value")
        source_name = "ai_consensus"
        note = "Blind adjudication matched both fact_value sets."
        if layer_set(proposed.get("target_spans"), "relation") == layer_set(blind.get("target_spans"), "relation"):
            spans.extend(relations_to_copy(proposed, blind))
        elif layer_set(cross.get("target_spans"), "relation") == layer_set(blind.get("target_spans"), "relation"):
            spans.extend(relations_to_copy(cross, blind))
            note += " Relation copied from crosscheck."
        else:
            note += " Relations differed and were omitted."
    elif proposed_match:
        spans = copy_spans(proposed.get("target_spans"), "fact_value")
        spans.extend(relations_to_copy(proposed, blind))
        source_name = "proposed"
        note = "Blind adjudication confirmed the proposed fact_value."
        if not relations_to_copy(proposed, blind):
            note += " Relation was omitted."
    else:
        spans = copy_spans(cross.get("target_spans"), "fact_value")
        spans.extend(relations_to_copy(cross, blind))
        source_name = "crosscheck"
        note = "Blind adjudication confirmed the crosscheck fact_value."
        if not relations_to_copy(cross, blind):
            note += " Relation was omitted."
    spans = sorted(spans, key=lambda span: (span["start_char"], span["end_char"], span["layer"]))
    problems = span_problems(answer, spans)
    if problems:
        return empty_status_record(
            proposed,
            "unresolved",
            reviewer,
            "ai_consensus",
            "Confirmed spans failed the character check and were left unresolved.",
            cross,
        )
    final_facts = layer_set(spans, "fact_value")
    final_relations = layer_set(spans, "relation")
    disagreement = final_facts != layer_set(cross.get("target_spans"), "fact_value") or final_relations != layer_set(
        cross.get("target_spans"), "relation"
    )
    return terminal_record(
        proposed,
        status="adjudicated",
        spans=spans,
        reviewer=reviewer,
        source_of_spans=source_name,
        disagreement=disagreement,
        note=note + " Not a human adjudication.",
    )


def indexed(rows: list[dict]) -> dict[int, dict]:
    return {int(row["candidate_row_index"]): row for row in rows}


def merge_records(
    proposed_rows: list[dict],
    cross_rows: list[dict],
    blind_rows: list[dict] | None,
    reviewer: str,
) -> list[dict]:
    proposed_by_index = indexed(proposed_rows)
    cross_by_index = indexed(cross_rows)
    if set(proposed_by_index) != set(range(200)) or set(cross_by_index) != set(range(200)):
        raise ValueError("annotations must cover candidate_row_index 0..199 exactly")
    blind_by_index = indexed(blind_rows or [])
    expected_blind = set()
    records = []
    for index in range(200):
        proposed = proposed_by_index[index]
        cross = cross_by_index[index]
        category = classify_row(proposed, cross)
        if needs_blind(category):
            expected_blind.add(index)
            continue
        records.append(consensus_record(proposed, cross, category))
    if blind_rows is None:
        return records
    if set(blind_by_index) != expected_blind:
        missing = sorted(expected_blind - set(blind_by_index))
        extra = sorted(set(blind_by_index) - expected_blind)
        raise ValueError(f"blind index set mismatch missing={missing[:10]} extra={extra[:10]}")
    for index in sorted(expected_blind):
        records.append(apply_blind(proposed_by_index[index], cross_by_index[index], blind_by_index[index], reviewer))
    records.sort(key=lambda row: row["candidate_row_index"])
    return records


def queue_records(proposed_rows: list[dict], cross_rows: list[dict]) -> list[dict]:
    proposed_by_index = indexed(proposed_rows)
    cross_by_index = indexed(cross_rows)
    queue = []
    for index in range(200):
        proposed = proposed_by_index[index]
        if needs_blind(classify_row(proposed, cross_by_index[index])):
            queue.append(
                {
                    "candidate_row_index": index,
                    "question": proposed["question"],
                    "answer": proposed["answer"],
                }
            )
    return queue


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def outcome_name(record: dict) -> str:
    note = record["review_note"]
    if note.startswith("AI consensus exact_match"):
        return "consensus_exact"
    if note.startswith("AI consensus unresolved"):
        return "consensus_unresolved"
    if note.startswith("AI consensus no_target"):
        return "consensus_no_target"
    if note.startswith("AI consensus on fact_value"):
        return "consensus_fact_only"
    if "character check" in note:
        return "blind_character_failure"
    if "matched neither" in note:
        return "blind_unconfirmed"
    if "returned unresolved" in note:
        return "blind_unresolved"
    if "returned no_target" in note:
        return "blind_no_target"
    if "confirmed the proposed" in note:
        return "blind_confirmed_proposed"
    if "confirmed the crosscheck" in note:
        return "blind_confirmed_crosscheck"
    if "matched both" in note:
        return "blind_confirmed_both"
    return "other"


def update_stats(stats_path: Path, records: list[dict]) -> None:
    stats = json.loads(stats_path.read_text())
    status_counts: dict[str, int] = {}
    reviewer_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    for record in records:
        status_counts[record["review_status"]] = status_counts.get(record["review_status"], 0) + 1
        reviewer_counts[record["reviewer"]] = reviewer_counts.get(record["reviewer"], 0) + 1
        source_counts[record["source_of_spans"]] = source_counts.get(record["source_of_spans"], 0) + 1
        name = outcome_name(record)
        outcomes[name] = outcomes.get(name, 0) + 1
    stats["reviewed_count"] = 0
    stats["adjudicated_count"] = 0
    stats["unresolved_count_after_review"] = None
    stats["second_annotator"] = None
    stats["thresholds_confirmed_before_results"] = False
    stats["span_boundary_agreement_threshold"] = None
    stats["token_level_agreement_threshold"] = None
    stats["span_boundary_agreement"] = None
    stats["token_level_agreement"] = None
    stats["disagreement_count"] = None
    stats["human_final_spans"] = {
        "path": "final_spans.jsonl",
        "pending_count": 0,
        "adjudicated_with_spans_count": 0,
        "unresolved_count": 0,
        "no_target_count": 0,
        "note": "No human reviewer. AI final counts are stored separately.",
        "deprecated_review_adjudication": "invalid JSON placeholder; do not use",
    }
    stats["ai_final_spans"] = {
        "path": "final_spans.jsonl",
        "is_human_adjudication": False,
        "pass_fail_threshold": None,
        "review_status_counts": status_counts,
        "reviewer_counts": reviewer_counts,
        "source_of_spans_counts": source_counts,
        "outcome_counts": outcomes,
        "fact_value_span_count": sum(
            1 for record in records for span in record["target_spans"] if span["layer"] == "fact_value"
        ),
        "relation_span_count": sum(
            1 for record in records for span in record["target_spans"] if span["layer"] == "relation"
        ),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge AI consensus and blind adjudication")
    parser.add_argument("--write-queue", type=Path, default=None)
    parser.add_argument("--part-size", type=int, default=40)
    parser.add_argument("--blind", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=FINAL)
    parser.add_argument("--stats", type=Path, default=STATS)
    parser.add_argument("--reviewer", default=DEFAULT_BLIND_REVIEWER)
    args = parser.parse_args()
    proposed_rows = load_jsonl(PROPOSED)
    cross_rows = load_jsonl(CROSSCHECK)
    if args.write_queue:
        queue = queue_records(proposed_rows, cross_rows)
        write_jsonl(args.write_queue, queue)
        part_dir = args.write_queue.parent
        for part_index, start in enumerate(range(0, len(queue), args.part_size)):
            part = queue[start : start + args.part_size]
            write_jsonl(part_dir / f"blind_queue_part{part_index}.jsonl", part)
        print(f"wrote {args.write_queue.name}: {len(queue)} disagreement rows")
        return
    if args.blind is None:
        raise SystemExit("Pass --blind to write final_spans, or --write-queue to export questions.")
    blind_rows = load_jsonl(args.blind)
    records = merge_records(proposed_rows, cross_rows, blind_rows, args.reviewer)
    if [row["candidate_row_index"] for row in records] != list(range(200)):
        raise SystemExit("merged records do not cover 0..199")
    write_jsonl(args.output, records)
    if args.stats:
        update_stats(args.stats, records)
    print(f"wrote {args.output.name}: {len(records)} terminal rows")


if __name__ == "__main__":
    main()

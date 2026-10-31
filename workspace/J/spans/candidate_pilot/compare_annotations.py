"""Compare proposed vs independent AI crosscheck. Descriptive only; not a human agreement study."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROPOSED = HERE / "proposed.jsonl"
CROSSCHECK = HERE / "independent_ai_crosscheck.jsonl"
REPORT = HERE / "crosscheck_disagreement_report.json"


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda row: row["candidate_row_index"])
    return rows


def span_tuple(span: dict) -> tuple:
    return (
        span.get("layer"),
        span.get("start_char"),
        span.get("end_char"),
        span.get("text"),
    )


def layer_set(spans: list[dict] | None, layer: str) -> set[tuple]:
    return {span_tuple(span) for span in (spans or []) if span.get("layer") == layer}


def layer_texts(spans: list[dict] | None, layer: str) -> list[str]:
    return sorted(span["text"] for span in (spans or []) if span.get("layer") == layer)


def normalize_status(row: dict) -> str:
    status = row.get("annotation_status")
    if status in {"proposed", "annotated"}:
        return "active"
    if status in {"unresolved", "no_target"}:
        return status
    return str(status)


def _nested_or_overlap(a: list[str], b: list[str]) -> bool:
    if not a or not b:
        return False
    return all(any(x in y or y in x for y in b) for x in a) and all(
        any(x in y or y in x for x in a) for y in b
    )


def classify_row(proposed: dict, cross: dict) -> str:
    p_status = normalize_status(proposed)
    c_status = normalize_status(cross)
    p_fact = layer_set(proposed.get("target_spans"), "fact_value")
    c_fact = layer_set(cross.get("target_spans"), "fact_value")
    p_rel = layer_set(proposed.get("target_spans"), "relation")
    c_rel = layer_set(cross.get("target_spans"), "relation")

    if p_status != c_status and (p_status != "active" or c_status != "active"):
        return "status_diff"
    if p_fact == c_fact:
        if p_rel == c_rel:
            return "exact_match"
        return "relation_only"
    p_texts = layer_texts(proposed.get("target_spans"), "fact_value")
    c_texts = layer_texts(cross.get("target_spans"), "fact_value")
    if p_texts == c_texts:
        return "content_diff"
    if _nested_or_overlap(p_texts, c_texts):
        p_len = sum(map(len, p_texts))
        c_len = sum(map(len, c_texts))
        if len(c_texts) > len(p_texts) or c_len < p_len:
            return "granularity"
        if len(p_texts) > len(c_texts) or p_len < c_len:
            return "granularity"
        return "span_expansion"
    return "content_diff"


def priority_for(category: str, cross: dict) -> str:
    if normalize_status(cross) in {"unresolved", "no_target"}:
        return "high"
    if category in {"exact_match"}:
        return "none"
    if category == "relation_only":
        return "low"
    return "high"


def compare(proposed_rows: list[dict], cross_rows: list[dict]) -> dict:
    if len(proposed_rows) != 200 or len(cross_rows) != 200:
        raise SystemExit("expected 200 rows in both annotation files")
    categories: Counter[str] = Counter()
    rows_out = []
    high_queue = []
    low_queue = []
    fact_exact = 0
    relation_exact = 0
    for proposed, cross in zip(proposed_rows, cross_rows):
        if proposed["candidate_row_index"] != cross["candidate_row_index"]:
            raise SystemExit("row index mismatch")
        category = classify_row(proposed, cross)
        categories[category] += 1
        p_fact = layer_set(proposed.get("target_spans"), "fact_value")
        c_fact = layer_set(cross.get("target_spans"), "fact_value")
        p_rel = layer_set(proposed.get("target_spans"), "relation")
        c_rel = layer_set(cross.get("target_spans"), "relation")
        if p_fact == c_fact:
            fact_exact += 1
        if p_rel == c_rel:
            relation_exact += 1
        priority = priority_for(category, cross)
        entry = {
            "candidate_row_index": proposed["candidate_row_index"],
            "category": category,
            "priority": priority,
            "proposed_status": proposed.get("annotation_status"),
            "crosscheck_status": cross.get("annotation_status"),
            "proposed_fact_values": layer_texts(proposed.get("target_spans"), "fact_value"),
            "crosscheck_fact_values": layer_texts(cross.get("target_spans"), "fact_value"),
            "proposed_relations": layer_texts(proposed.get("target_spans"), "relation"),
            "crosscheck_relations": layer_texts(cross.get("target_spans"), "relation"),
            "crosscheck_unresolved_reasons": cross.get("unresolved_reasons") or [],
        }
        rows_out.append(entry)
        if priority == "high":
            high_queue.append(proposed["candidate_row_index"])
        elif priority == "low":
            low_queue.append(proposed["candidate_row_index"])

    cross_status = Counter(row.get("annotation_status") for row in cross_rows)
    return {
        "dataset_status": "local_candidate_pilot",
        "identity": {
            "independent_ai_crosscheck_is_human_second_annotator": False,
            "purpose": "disagreement_mining_and_adjudication_aid",
            "not_a_dual_human_agreement_study": True,
        },
        "counts": {
            "rows": 200,
            "fact_value_exact_agree": fact_exact,
            "relation_exact_agree": relation_exact,
            "exact_match_rows": categories["exact_match"],
            "categories": dict(categories),
            "crosscheck_status_counts": dict(cross_status),
        },
        "human_review_queues": {
            "high_priority_fact_or_status": high_queue,
            "low_priority_relation_only": low_queue,
            "note": (
                "High priority covers all fact_value disagreements and crosscheck "
                "unresolved/no_target rows. Relation-only disagreements are low priority "
                "because the first main m_tar uses fact_value only."
            ),
        },
        "rows": rows_out,
        "pass_fail_threshold": None,
        "threshold_note": (
            "No post-hoc pass/fail threshold is set on this AI-AI disagreement rate. "
            "Future human-review thresholds must be confirmed before seeing that round's results."
        ),
    }


def main() -> None:
    report = compare(load_jsonl(PROPOSED), load_jsonl(CROSSCHECK))
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    counts = report["counts"]
    print(
        f"wrote {REPORT.name}: fact_exact={counts['fact_value_exact_agree']} "
        f"exact_match={counts['exact_match_rows']} "
        f"high_queue={len(report['human_review_queues']['high_priority_fact_or_status'])}"
    )


if __name__ == "__main__":
    main()

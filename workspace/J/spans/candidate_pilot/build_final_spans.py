"""Create the empty legal final_spans.jsonl shell. Does not copy span decisions."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROPOSED = HERE / "proposed.jsonl"
CROSSCHECK = HERE / "independent_ai_crosscheck.jsonl"
REPORT = HERE / "crosscheck_disagreement_report.json"
OUTPUT = HERE / "final_spans.jsonl"
DEPRECATED = HERE / "review_adjudication.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def disagreement_flags(report: dict | None) -> dict[int, bool]:
    if not report:
        return {}
    return {
        int(row["candidate_row_index"]): row["category"] != "exact_match"
        for row in report.get("rows", [])
    }


def build_shell(proposed_rows: list[dict], flags: dict[int, bool]) -> list[dict]:
    records = []
    for row in proposed_rows:
        index = int(row["candidate_row_index"])
        records.append(
            {
                "dataset_status": "local_candidate_pilot",
                "canonical_forget05_task_id_1": False,
                "candidate_row_index": index,
                "question": row["question"],
                "answer": row["answer"],
                "review_status": "pending_human_review",
                "reviewer": None,
                "source_of_spans": None,
                "target_spans": [],
                "disagreement_with_crosscheck": bool(flags.get(index, True)),
                "review_note": (
                    "Shell only. Fill target_spans after human adjudication, "
                    "or set review_status to unresolved/no_target. "
                    "Do not treat review_adjudication.jsonl as final."
                ),
            }
        )
    return records


def existing_final_is_locked(path: Path) -> bool:
    """Refuse to rebuild once a row has left the empty pending shell."""
    if not path.is_file():
        return False
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("review_status") != "pending_human_review" or row.get("target_spans"):
            return True
    return False


def main() -> None:
    if existing_final_is_locked(OUTPUT):
        raise SystemExit(
            "Refusing to overwrite final_spans.jsonl: it is no longer an empty shell. "
            "AI final rows must stay in place."
        )
    proposed = load_jsonl(PROPOSED)
    if len(proposed) != 200:
        raise SystemExit("proposed.jsonl must have 200 rows")
    if not CROSSCHECK.is_file():
        raise SystemExit("independent_ai_crosscheck.jsonl is required")
    cross = load_jsonl(CROSSCHECK)
    if len(cross) != 200:
        raise SystemExit("independent_ai_crosscheck.jsonl must have 200 rows")
    report = json.loads(REPORT.read_text()) if REPORT.is_file() else None
    records = build_shell(proposed, disagreement_flags(report))
    OUTPUT.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    note = {
        "status": "deprecated_broken_placeholder",
        "reason": (
            "Invalid JSON and empty adjudication fields. "
            "Use final_spans.jsonl as the only human-final sidecar."
        ),
        "replacement": str(OUTPUT.name),
    }
    print(f"wrote {OUTPUT.name}: {len(records)} pending rows")
    print(f"deprecated {DEPRECATED.name}: {json.dumps(note)}")


if __name__ == "__main__":
    main()

"""Refuse token alignment until human review and the ASU tokenizer are available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHIFT = "logits[:, :-1] predicts labels[:, 1:], so m_tar_logits[j-1] = m_tar_label[j]"


def blocking_reason(review_records: list[dict], tokenizer_path: str | None) -> str:
    if len(review_records) != 200:
        return "review queue does not cover all 200 candidate rows"
    pending = [
        record["candidate_row_index"]
        for record in review_records
        if record.get("review_status") != "adjudicated"
    ]
    if pending:
        return f"human review is incomplete for {len(pending)} rows"
    if not tokenizer_path or not Path(tokenizer_path).is_dir():
        return "ASU Llama-2 tokenizer path is missing"
    return "candidate pilot is not the verified canonical forget05/task_id=1 split"


def main() -> None:
    parser = argparse.ArgumentParser(description="Blocked causal-shift m_tar builder")
    parser.add_argument("--review", default=str(HERE / "review_queue.jsonl"))
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.review).read_text().splitlines() if line.strip()]
    reason = blocking_reason(records, args.tokenizer)
    raise SystemExit(f"Refusing to build m_tar: {reason}. Shift remains documented only: {SHIFT}")


if __name__ == "__main__":
    main()

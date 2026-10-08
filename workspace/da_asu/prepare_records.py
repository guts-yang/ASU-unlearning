"""Write the JSONL consumed by the tau scan and by D_reason training."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "workspace"))

from da_asu.spans import label_answer  # noqa: E402


def records_from_rows(rows, split):
    prepared = []
    for index, row in enumerate(rows):
        answer = row["answer"]
        factual, function = label_answer(answer)
        prepared.append(
            {
                "id": row.get("id", f"{split}-{index}"),
                "question": row["question"],
                "answer": answer,
                "factual_spans": [list(span) for span in factual],
                "function_spans": [list(span) for span in function],
            }
        )
    return prepared


def load_rows(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_gsm8k(split):
    from datasets import load_dataset

    data = load_dataset("gsm8k", "main", split=split)
    return [{"question": row["question"], "answer": row["answer"]} for row in data]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare DA-ASU GSM8K JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--input", default=None, help="Local JSONL with question and answer. Overrides --split download.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.input:
        rows = load_rows(args.input)
        split = args.split
    else:
        rows = load_gsm8k(args.split)
        split = args.split
    if args.limit is not None:
        rows = rows[: args.limit]
    records = records_from_rows(rows, split)
    write_jsonl(args.output, records)
    print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

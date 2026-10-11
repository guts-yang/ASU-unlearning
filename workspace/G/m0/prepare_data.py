"""Join sangyon/R-TOFU forget05 to local TOFU forget05 task_id==1."""
from __future__ import annotations

import json

import numpy as np
from datasets import load_dataset

from common import DATA, TOFU, write_json, write_jsonl


def load_local(split: str) -> list[dict]:
    rows = []
    with (TOFU / f"{split}.json").open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    local = [row for row in load_local("forget05") if int(row["task_id"]) == 1]
    by_question = {row["question"].strip(): row for row in local}
    remote = load_dataset("sangyon/R-TOFU", split="forget05")
    joined = []
    answer_mismatch = 0
    for index, row in enumerate(remote):
        question = row["question"].strip()
        local_row = by_question.get(question)
        if local_row is None:
            continue
        if local_row["answer"].strip() != row["answer"].strip():
            answer_mismatch += 1
        joined.append({
            "index": len(joined),
            "question": local_row["question"],
            "answer": local_row["answer"],
            "cot": row["cot"],
            "task_id": 1,
            "split": "forget05",
        })
    retain = load_local("retain90")
    rng = np.random.default_rng(1001)
    picked = np.sort(rng.choice(len(retain), size=50, replace=False))
    retain_rows = [
        {
            "index": int(position),
            "source_index": int(source),
            "question": retain[int(source)]["question"],
            "answer": retain[int(source)]["answer"],
            "split": "retain90",
        }
        for position, source in enumerate(picked)
    ]
    write_jsonl(DATA / "forget05_task1.jsonl", joined)
    write_jsonl(DATA / "retain90_n50.jsonl", retain_rows)
    write_json(DATA / "join_report.json", {
        "rtofu": "sangyon/R-TOFU forget05",
        "local": "Right-to-be-forgotten/data/tofu/forget05.json task_id==1",
        "local_n": len(local),
        "rtofu_n": len(remote),
        "join_n": len(joined),
        "answer_mismatch": answer_mismatch,
        "cot_fallback_used": False,
        "retain_source": "local retain90.json",
        "retain_n": len(retain_rows),
        "retain_sample_seed": 1001,
    })
    print(f"joined {len(joined)} answer_mismatch {answer_mismatch} retain {len(retain_rows)}")


if __name__ == "__main__":
    main()

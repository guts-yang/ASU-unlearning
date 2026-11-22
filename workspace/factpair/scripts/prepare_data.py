#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CORE = BASE / "core"
sys.path.insert(0, str(CORE))

from factpair import build_entity_pools, build_paired_prompts, write_json, write_jsonl, read_jsonl  # noqa: E402

WORKSPACE = BASE.parent
DATA_DIR = WORKSPACE / "data" / "factpair"
TOFU_DIR = Path("/jizhicfs/chenyliao/data/datasets/locuslab-TOFU")
MODEL = Path("/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full")


def _rows_for_clusters(clusters, rows):
    wanted = set()
    for cluster in clusters:
        wanted.update(cluster["row_indices"])
    return [row for index, row in enumerate(rows) if index in wanted]


def main():
    pools = build_entity_pools(
        forget_path=TOFU_DIR / "forget05.json",
        retain_path=TOFU_DIR / "retain95.json",
        donor_size=20,
        retain_train_size=20,
        seed=0,
    )
    write_json(DATA_DIR / "entity_pools.json", pools)
    retain_rows = pools["cluster_rows"]["retain95"]
    write_jsonl(DATA_DIR / "retain_train.jsonl", _rows_for_clusters(pools["D_R_train"], retain_rows))
    write_jsonl(DATA_DIR / "retain_eval.jsonl", _rows_for_clusters(pools["D_eval"], retain_rows))
    write_jsonl(DATA_DIR / "donor_rows.jsonl", _rows_for_clusters(pools["D_donor"], retain_rows))

    template_args = {"apply_chat_template": True, "system_prompt": None}
    summary = build_paired_prompts(
        entity_pools_path=DATA_DIR / "entity_pools.json",
        tokenizer_name_or_path=str(MODEL),
        output_path=DATA_DIR / "paired_prompts.jsonl",
        template_args=template_args,
        max_length=512,
    )
    write_json(DATA_DIR / "pairing_summary.json", summary)
    print(json.dumps({"pairs": summary["pairs_written"], "real_tokenizer": summary["real_tokenizer"], "g0_status": summary["g0_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

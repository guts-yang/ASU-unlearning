#!/usr/bin/env python3
"""Deterministically build entity_splits.yaml from TOFU forget05.

No downloads, no random split. Author blocks are the 10 consecutive 20-row
blocks in forget05.json; split assignment is by sorted canonical name so it
can be reproduced and frozen before any confirmation access.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

import yaml

from recodeloop.utils.hashing import sha256_file, sha256_json

DEFAULT_SOURCE = "/dockerdata/asu-unlearning/datasets/locuslab-TOFU/forget05.json"
BLOCK_SIZE = 20
SPLIT_RULE = (
    "sorted canonical_id -> [forget_train_edit, forget_train_edit, "
    "forget_train_reader, forget_train_reader, calibration, calibration, "
    "confirmation, confirmation, confirmation, confirmation]"
)


def extract_name(answer: str) -> str:
    capitalized = r"([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)+)"
    patterns = (
        rf"author's name is {capitalized}",
        rf"author's full name is {capitalized}",
        rf"is named {capitalized}",
        rf"is called {capitalized}",
        rf"is {capitalized}, an ",
        rf"is {capitalized}",
    )
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            return match.group(1).strip(" .,")
    raise ValueError(f"cannot extract canonical name from: {answer[:160]!r}")


def aliases_for(name: str) -> list[str]:
    tokens = name.split()
    short = " ".join([tokens[0], tokens[-1]]) if len(tokens) > 2 else name
    candidates = [name, name.lower(), short, short.lower(), name.replace("-", " ")]
    return list(dict.fromkeys(c for c in candidates if c))


def build(source: Path) -> dict:
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 10 * BLOCK_SIZE:
        raise ValueError(f"expected 200 rows, got {len(rows)}")

    clusters = []
    for block_idx in range(10):
        start = block_idx * BLOCK_SIZE
        rows_in_block = rows[start:start + BLOCK_SIZE]
        first_answer = rows_in_block[0]["answer"]
        name = extract_name(first_answer)
        split = [
            "forget_train_edit", "forget_train_edit",
            "forget_train_reader", "forget_train_reader",
            "calibration", "calibration",
            "confirmation", "confirmation", "confirmation", "confirmation",
        ][block_idx]
        clusters.append({
            "canonical_id": name,
            "aliases": aliases_for(name),
            "source_rows": [f"{start}-{start + BLOCK_SIZE - 1}"],
            "split": split,
            "fact_relation": "full_name",
            "fact_values": [name],
            "provenance": (
                f"source={source}; sha256={sha256_file(source)}; "
                f"rows={start}-{start + BLOCK_SIZE - 1}; "
                "name extracted from first row answer; split by canonical_id sort"
            ),
        })

    clusters.sort(key=lambda c: c["canonical_id"])
    assignment = [
        "forget_train_edit", "forget_train_edit",
        "forget_train_reader", "forget_train_reader",
        "calibration", "calibration",
        "confirmation", "confirmation", "confirmation", "confirmation",
    ]
    for cluster, split in zip(clusters, assignment):
        cluster["split"] = split
        cluster["source_rows"] = [
            str(int(cluster["source_rows"][0].split("-")[0]))
            + "-" + str(int(cluster["source_rows"][0].split("-")[1]))
        ]

    owners: dict[str, str] = {}
    for cluster in clusters:
        for key in [cluster["canonical_id"], *cluster["aliases"]]:
            if key in owners and owners[key] != cluster["canonical_id"]:
                raise ValueError(f"alias collision: {key!r}")
            owners[key] = cluster["canonical_id"]

    return {
        "schema_version": "recodeloop-m0.1",
        "experiment_id": "P08-M2",
        "note": (
            "实体级派生划分；author canonical ID/别名/实体簇为唯一隔离单位，"
            "不按 QA 行随机拆分。confirmation 由 AccessGuard 封存。"
        ),
        "canonical_id_field": "canonical_author_id",
        "alias_fields": ["author", "author_aliases"],
        "source_data": str(source),
        "source_sha256": sha256_file(source),
        "construction": {
            "rule": "10 consecutive 20-row author blocks; first-row answer gives canonical name",
            "split_rule": SPLIT_RULE,
            "fact_relation": "full_name",
            "fact_value_rule": "canonical author name decoded from the first-row answer",
            "canonical_hash": sha256_json(clusters),
        },
        "split_counts": {
            "forget_train_edit": 2,
            "forget_train_reader": 2,
            "calibration": 2,
            "confirmation": 4,
        },
        "clusters": clusters,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build frozen entity splits")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--out", default=str(WORKSPACE / "manifests" / "entity_splits.yaml"))
    args = parser.parse_args()

    manifest = build(Path(args.source))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    print(f"source_sha256={manifest['source_sha256']}")
    print(f"canonical_hash={manifest['construction']['canonical_hash']}")
    for cluster in manifest["clusters"]:
        print(f"  {cluster['split']:20s} {cluster['canonical_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

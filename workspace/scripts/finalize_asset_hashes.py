#!/usr/bin/env python3
"""Compute and record canonical bundle hashes after the .bin exclusion rule."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from safetensors import safe_open

from recodeloop.utils.hashing import sha256_bundle

BASE = (
    "/dockerdata/asu-unlearning/cache/huggingface/models--NousResearch--Llama-2-7b-chat-hf"
    "/snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee"
)
PATHS = {
    "base": BASE,
    "target": "/dockerdata/asu-unlearning/models/locuslab/tofu_ft_llama2-7b",
    "asu_student": str(WORKSPACE.parent / "open-unlearning/checkpoints/asu-forget05-asu-gd"),
}


def check_shards(path: str) -> tuple[int, int]:
    index = json.loads((Path(path) / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    total = 0
    for shard in shards:
        with safe_open(str(Path(path) / shard), framework="pt", device="cpu") as handle:
            total += len(list(handle.keys()))
    return len(index["weight_map"]), total


def main() -> int:
    result = {"hashes": {}, "safetensors": {}}
    for role, path in PATHS.items():
        result["hashes"][role] = sha256_bundle(path)
        result["safetensors"][role] = check_shards(path)
        print(role, result["hashes"][role], flush=True)
    out = WORKSPACE / "outputs" / "hash_finalize.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

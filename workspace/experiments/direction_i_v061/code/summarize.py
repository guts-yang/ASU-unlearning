"""Merge diagnostic + training results into SUMMARY.json."""
from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"


def load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    diag = load("diag.json")
    out = {
        "engine": "proxy-v061",
        "pair_stats": diag.get("pair_stats", {}),
        "forward": diag.get("results", {}),
        "train_align": load("train_all.json"),
        "train_neutral": load("train_all_neutral.json"),
        "caveats": [
            "relation-matched donor rate is low (template fingerprint only)",
            "D2 certificate needs a trained student checkpoint; base-model patch is detectability only",
            "D8 null donor uses a phrasing change, not strict subword/frequency matching",
            "training variants use 60 steps, lr 5e-6, single seed",
        ],
    }
    (R / "SUMMARY.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote SUMMARY.json")


if __name__ == "__main__":
    main()

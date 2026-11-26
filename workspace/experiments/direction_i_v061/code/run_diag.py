"""Run forward-only diagnostics for D2-D6, D8, D9 and write results/diag.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diag
from engine import RETAIN, load_jsonl, load_model
from pairs import build_pairs

OUT = Path(__file__).resolve().parent.parent / "results"


def main():
    t0 = time.time()
    model, tok = load_model()
    pairs, pstats = build_pairs(n=16)
    retain_pool = load_jsonl(RETAIN)[:64]
    res = diag.run_all(model, tok, pairs, retain_pool)
    out = {
        "engine": "proxy-v061",
        "samples": len(pairs),
        "pair_stats": pstats,
        "results": res,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "diag.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

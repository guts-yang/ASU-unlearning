"""Re-run forward diagnostics on v2 (synthetic same-template) pairs."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diag
from engine import RETAIN, load_jsonl, load_model
from pairs_v2 import build_pairs_v2

OUT = Path(__file__).resolve().parent.parent / "results"


def main():
    t0 = time.time()
    model, tok = load_model()
    pairs, pstats = build_pairs_v2(n=16)
    retain_pool = load_jsonl(RETAIN)[:64]
    res = diag.run_all(model, tok, pairs, retain_pool)
    out = {
        "engine": "proxy-v061",
        "pairs": "v2-synth-entity-swap",
        "samples": len(pairs),
        "pair_stats": pstats,
        "results": res,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "diag_v2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in res.items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk != "example_profile"}, ensure_ascii=False))
    print("pairs", pstats, flush=True)


if __name__ == "__main__":
    main()

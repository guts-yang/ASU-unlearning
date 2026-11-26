"""3-seed sweep for D3 (causal gate) and D9 (readout space) on v2 pairs."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diag
from engine import RETAIN, load_jsonl, load_model
from pairs_v2 import build_pairs_v2

OUT = Path(__file__).resolve().parent.parent / "results"
SEEDS = (0, 1, 2)


def main():
    t0 = time.time()
    model, tok = load_model()
    diag.model_tok = tok
    retain_pool = load_jsonl(RETAIN)[:64]
    res = {}
    for seed in SEEDS:
        pairs, st = build_pairs_v2(n=16, seed=seed)
        d3 = diag.d3_causal_gate(model, pairs)
        d9 = diag.d9_readout_space(model, pairs)
        res[f"s{seed}"] = {"pairs": st, "D3": d3, "D9": d9}
        print(
            f"seed={seed} synth={st['synth_rate']} best_layer={d3['best_layer']} "
            f"share={d3['top_layer_squared_share']:.4f} pearson={d9['pearson_readout_vs_hidden']:.4f}",
            flush=True,
        )
    shares = [res[f"s{s}"]["D3"]["top_layer_squared_share"] for s in SEEDS]
    pears = [res[f"s{s}"]["D9"]["pearson_readout_vs_hidden"] for s in SEEDS]
    out = {
        "engine": "proxy-v061",
        "pairs": "v2-synth-entity-swap",
        "seeds": res,
        "D3_best_layers": [res[f"s{s}"]["D3"]["best_layer"] for s in SEEDS],
        "D3_top_share_mean": round(statistics.mean(shares), 4),
        "D3_top_share_std": round(statistics.stdev(shares), 4),
        "D9_pearson_mean": round(statistics.mean(pears), 4),
        "D9_pearson_std": round(statistics.stdev(pears), 4),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "seed_sweep_D3_D9.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("MEAN D3 share", out["D3_top_share_mean"], "D9 pearson", out["D9_pearson_mean"])


if __name__ == "__main__":
    main()

"""Run D1/D7/D10 training variants.

Usage:
  python run_train.py align   [seeds...]
  python run_train.py neutral [seeds...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_v061 import train_variant

OUT = Path(__file__).resolve().parent.parent / "results"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "align"
    if mode in ("align", "neutral"):
        seeds = [int(x) for x in sys.argv[2:]] or [0]
    else:
        mode = "align"
        seeds = [int(x) for x in sys.argv[1:]] or [0]

    all_res = {}
    for variant in ("D1", "D7", "D10"):
        for seed in seeds:
            t0 = time.time()
            r = train_variant(variant, seed=seed, target_mode=mode)
            r["elapsed_s"] = round(time.time() - t0, 1)
            key = f"{variant}_{mode}_s{seed}"
            all_res[key] = r
            (OUT / f"train_{key}.json").write_text(
                json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(variant, mode, seed, r["delta"], r["behavior"]["mean_target_overlap"], flush=True)
    (OUT / f"train_all_{mode}.json").write_text(
        json.dumps(all_res, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

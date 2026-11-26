"""Run D7/D10 at the calibrated neutral strength (lambda=0.2)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_v061 import train_variant

OUT = Path(__file__).resolve().parent.parent / "results"
LAM = 0.2


def main():
    res = {}
    for variant in ("D7", "D10"):
        t0 = time.time()
        r = train_variant(variant, seed=0, target_mode="neutral", neutral_lambda=LAM)
        r["elapsed_s"] = round(time.time() - t0, 1)
        key = f"{variant}_lam{LAM}"
        res[key] = r
        (OUT / f"train_{key}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        b = r["behavior"]
        print(
            f"{variant} lam={LAM} dF={r['delta']['forget']:.4f} dR={r['delta']['retain']:.4f} "
            f"eos={b['eos_rate']:.3f} gen={b['mean_gen_tokens']:.1f} overlap={b['mean_target_overlap']:.3f}",
            flush=True,
        )
    (OUT / "train_calibrated.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

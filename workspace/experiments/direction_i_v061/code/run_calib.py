"""Calibrate the neutral-term strength (proxy for c/d_j/e).

Scans lambda over a grid on D1 and reports the trade-off between dF and output
integrity (eos_rate). Chooses the largest lambda that reaches dF>=0.3 with
eos_rate>=0.8.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_v061 import train_variant

OUT = Path(__file__).resolve().parent.parent / "results"
LAMBDAS = [0.02, 0.05, 0.1, 0.2, 0.5]


def main():
    res = {}
    for lam in LAMBDAS:
        t0 = time.time()
        r = train_variant("D1", seed=0, target_mode="neutral", neutral_lambda=lam)
        r["elapsed_s"] = round(time.time() - t0, 1)
        key = f"lam{lam}"
        res[key] = r
        (OUT / f"calib_{key}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        b = r["behavior"]
        print(
            f"lam={lam} dF={r['delta']['forget']:.4f} dR={r['delta']['retain']:.4f} "
            f"eos={b['eos_rate']:.3f} gen={b['mean_gen_tokens']:.1f} "
            f"short={b['short_rate']:.3f} overlap={b['mean_target_overlap']:.3f}",
            flush=True,
        )
    ok = [
        (k, v)
        for k, v in res.items()
        if v["delta"]["forget"] >= 0.3 and v["behavior"]["eos_rate"] >= 0.8
    ]
    ok.sort(key=lambda kv: kv[1]["delta"]["forget"], reverse=True)
    res["_choice"] = ok[0][0] if ok else None
    (OUT / "calibration.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("CHOICE", res["_choice"], flush=True)


if __name__ == "__main__":
    main()

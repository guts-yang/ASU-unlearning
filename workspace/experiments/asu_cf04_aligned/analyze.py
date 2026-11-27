#!/usr/bin/env python3
"""Paired difference analysis for the aligned CF-04 vs ASU experiment.

Per-seed paired deltas (CF-04 - ASU), mean +/- population std, sign consistency,
Cohen's dz, and a fixed-seed bootstrap 95% percentile CI (n=3, descriptive only).
No p-values and no significance claims.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from aligned_core import save_json

METRICS = {
    "dF": "forget NLL delta (higher better)",
    "dR": "retain NLL delta (lower better)",
    "dH": "holdout NLL delta (lower better)",
    "eos_rate": "EOS rate (availability)",
    "mean_gen_tokens": "mean generated tokens",
    "mean_target_overlap": "target overlap (availability)",
    "mean_recovery": "P2/P3 recovery (lower = deeper)",
    "probe_auc_post": "independent probe AUC after relearn (lower better)",
    "P2_gain": "few-shot recovery gain",
    "P3_gain": "relearning recovery gain",
    "INT8_shift": "INT8 fake-quant forget shift",
}

THRESHOLDS = {
    "retain_safe_dF": 0.30, "strong_dF": 0.90, "retain_max_dR": 0.05,
    "holdout_soft_dH": 0.10, "depth_recovery_max": 0.30,
    "eos_collapse_floor": 0.50, "overlap_collapse_floor": 0.20,
}


def metric_value(row, key):
    if key in ("dF", "dR", "dH"):
        d = row.get("train", {}).get("delta", {})
        return d.get({"dF": "forget", "dR": "retain", "dH": "holdout"}[key])
    b = row.get("behavior", {})
    if key in ("eos_rate", "mean_gen_tokens", "mean_target_overlap"):
        return b.get(key)
    a = row.get("audit", {})
    ds = a.get("depth_score", {}).get("mean_recovery", {})
    if key == "mean_recovery":
        return ds.get("mean")
    if key == "probe_auc_post":
        return a.get("probe_auc_post", {}).get("mean")
    if key in ("P2_gain", "P3_gain", "INT8_shift"):
        return a.get(key, {}).get("mean")
    return None


def bootstrap_ci(deltas, n_boot=10000, seed=20261127):
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return [lo, hi]


def paired_stats(values_a, values_c, seeds=SEEDS):
    pairs = [(s, a, c) for s, a, c in zip(seeds, values_a, values_c) if a is not None and c is not None]
    if not pairs:
        return {"n": 0}
    used_seeds = [p[0] for p in pairs]
    values = [(p[1], p[2]) for p in pairs]
    deltas = [c - a for a, c in values]
    n = len(deltas)
    mean = sum(deltas) / n
    std = math.sqrt(sum((d - mean) ** 2 for d in deltas) / n)
    return {
        "n": n,
        "per_seed": [{"seed": s, "asu": a, "cf04": c, "delta": c - a}
                     for s, (a, c) in zip(used_seeds, values)],
        "mean_delta": mean,
        "std_delta": std,
        "sign_consistent": sum(1 for d in deltas if d > 0) == n or sum(1 for d in deltas if d < 0) == n,
        "positive_count": sum(1 for d in deltas if d > 0),
        "cohens_dz": (mean / std) if std > 1e-12 else None,
        "bootstrap95_ci": bootstrap_ci(deltas),
    }


SEEDS = [1001, 1002, 1003]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evaluation", default="results/evaluation.json")
    ap.add_argument("--output", default="results/paired_analysis.json")
    args = ap.parse_args(argv)
    ev = json.loads(Path(args.evaluation).read_text())
    models = ev["models"]
    rows = {s: {"asu": models.get(f"asu_{s}", {}), "cf04": models.get(f"cf04_{s}", {})} for s in SEEDS}

    analysis = {}
    for key, desc in METRICS.items():
        a = [metric_value(rows[s]["asu"], key) for s in SEEDS]
        c = [metric_value(rows[s]["cf04"], key) for s in SEEDS]
        analysis[key] = {"description": desc, **paired_stats(a, c)}

    # usability / validity per seed
    per_seed = {}
    for s in SEEDS:
        r = rows[s]
        checks = {}
        for arm in ("asu", "cf04"):
            row = r[arm]
            d = row.get("train", {}).get("delta", {})
            b = row.get("behavior", {})
            aud = row.get("audit", {})
            rec = aud.get("depth_score", {}).get("mean_recovery", {}).get("mean")
            finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in d.values()) if d else False
            checks[arm] = {
                "no_nan_oom": finite and not row.get("eval_error") and not row.get("train", {}).get("error"),
                "checkpoint": bool(row.get("path")),
                "eos_not_collapsed": (b.get("eos_rate") or 0) >= THRESHOLDS["eos_collapse_floor"],
                "overlap_not_collapsed": (b.get("mean_target_overlap") or 0) >= THRESHOLDS["overlap_collapse_floor"],
                "retain_safe": (d.get("forget") or 0) >= THRESHOLDS["retain_safe_dF"] and (d.get("retain") or 0) <= THRESHOLDS["retain_max_dR"],
                "strong": (d.get("forget") or 0) >= THRESHOLDS["strong_dF"] and (d.get("retain") or 0) <= THRESHOLDS["retain_max_dR"],
                "holdout_soft": (d.get("holdout") or 0) <= THRESHOLDS["holdout_soft_dH"],
                "depth_ok": (rec is not None and rec < THRESHOLDS["depth_recovery_max"]),
            }
            checks[arm]["usable"] = all(checks[arm][k] for k in ("no_nan_oom", "checkpoint", "eos_not_collapsed", "overlap_not_collapsed"))
        per_seed[str(s)] = checks

    dF = analysis["dF"]
    dR = analysis["dR"]
    dH = analysis["dH"]
    all_usable = all(per_seed[str(s)][arm]["usable"] for s in SEEDS for arm in ("asu", "cf04"))
    conditions = {
        "direction_consistent_dF": dF.get("sign_consistent") is True and dF.get("mean_delta", 0) > 0,
        "direction_consistent_dR": dR.get("mean_delta", 0) <= 0,
        "holdout_not_worse": dH.get("mean_delta", 0) <= 0,
        "both_arms_usable": all_usable,
        "depth_not_worse": (analysis["mean_recovery"].get("mean_delta") or 0) <= 0,
    }
    verdict = "CONDITIONAL_ADVANTAGE" if all(conditions.values()) else "INCONCLUSIVE"
    if not conditions["both_arms_usable"]:
        verdict = "NEGATIVE_OR_UNAVAILABLE"

    out = {
        "claim_level": False, "gate_status": "NOT_APPLIED",
        "seeds": SEEDS, "thresholds": THRESHOLDS,
        "metrics": analysis, "per_seed_checks": per_seed,
        "decision_conditions": conditions, "verdict": verdict,
        "interpretation": "n=3 paired design; bootstrap CI and Cohen's dz are descriptive only. No statistical significance, no claim of superiority over official 7B ASU.",
        "limitations": [
            "1B / forget10 / single GPU instead of official 7B / forget05 / 2 GPU torchrun.",
            "ASU+GD replaced by a temperature-scaled ref + answer-KL + retain-NLL proxy.",
            "ASU arm uses alpha=1.0 (harness ASU_BASE definition); official forget_coeff=0.1 is not mapped.",
            "attention_temp yaml=2.0 vs script=2.3; tau=2.3 sensitivity is a single-seed side check.",
            "Per-method forward counts differ (ASU has an extra frozen ref forward); equal optimizer steps / data exposure, NOT equal FLOPs.",
        ],
    }
    save_json(out, args.output)
    print("VERDICT", verdict)
    for k in ("dF", "dR", "dH", "eos_rate", "mean_target_overlap", "mean_recovery", "probe_auc_post"):
        m = analysis[k]
        print(f"{k:22s} mean_delta={m.get('mean_delta')} ci={m.get('bootstrap95_ci')} dz={m.get('cohens_dz')} pos={m.get('positive_count')}/{m.get('n')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

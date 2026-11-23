"""Fixed environment, evaluation protocol, single metric and TERMINATION GATE.

DO NOT EDIT during the loop. Equivalent to autoresearch's prepare.py/evaluate_bpb:
the target definition is frozen here so the loop cannot move the goalposts.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

LOOP = Path(__file__).resolve().parent
REPO = LOOP.parents[1]
AUTO = REPO / "workspace" / "auto_experiments"
MODEL = "/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
FORGET = "/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json"
RETAIN = "/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json"
AUDIT = AUTO / "results" / "audit_depth.json"

# ---- frozen budget ----
STEPS = 60
TAU = 2.25
ALPHA = 0.1
LR = 5e-6
SEED = 0
P3_STEPS = 20
P3_LR = 1e-5

# ---- frozen termination target ----
RHO_TARGET = 0.30
DF_MIN = 1.00
DR_MAX = -0.20
DF_BAND = 1.1416280269622803   # Round-1 equal-budget anchor, measured
DF_TOL = 0.15
SEED_STD_MAX = 0.05
REFUSAL_MAX = 0.10

# ---- single metric coefficients ----
LAMBDA_R = 1.5
LAMBDA_DF = 1.5
GAMMA_K = 0.01

MAX_ROUNDS = 8
MAX_GPU_SECONDS = 900
MAX_CONSECUTIVE_CRASH = 3


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def base_forget_nll() -> float:
    return float(load_json(AUDIT)["models"]["base"]["P1_direct"]["forget"])


def rho_p3(f1: float, f3: float, f_base: float) -> float:
    return (f1 - f3) / max(f1 - f_base, 1e-6)


def val_bpb(rho: float, dF: float, dR: float, complexity: int) -> float:
    """Fixed single metric, lower is better."""
    return (
        rho
        + LAMBDA_R * max(0.0, dR - DR_MAX)
        + LAMBDA_DF * abs(dF - DF_BAND)
        + GAMMA_K * complexity
    )


def is_good(m: dict):
    """Frozen 'good enough' gate. Returns (bool, per-check dict).

    The generation-quality gate is NOT implemented in this environment, so this
    function structurally cannot return True. That is intentional and must be
    reported as BUDGET_EXHAUSTED rather than silently relaxing the target.
    """
    checks = {
        "rho_le_target": m["rho"] <= RHO_TARGET,
        "df_ge_min": m["dF"] >= DF_MIN,
        "dr_le_max": m["dR"] <= DR_MAX,
        "df_in_band": abs(m["dF"] - DF_BAND) <= DF_TOL,
        "seed_confirmed": bool(m.get("seed_confirmed", False)),
        "generation_gate": False,  # NOT_IMPLEMENTED: refusal-rate / answer quality
    }
    return all(checks.values()), checks


REQUIRED_ROUND_KEYS = ("round", "idea_id", "val_bpb", "rho", "dF", "dR",
                       "before", "after", "p3", "peak_vram_mb")


def verify_round(d: dict):
    missing = [k for k in REQUIRED_ROUND_KEYS if k not in d]
    if missing:
        return False, f"missing keys: {missing}"
    for k in ("val_bpb", "rho", "dF", "dR", "peak_vram_mb"):
        v = d[k]
        if not isinstance(v, (int, float)) or v != v or abs(v) == float("inf"):
            return False, f"non-finite {k}"
    return True, "ok"


def preflight():
    issues = []
    if not torch.cuda.is_available():
        issues.append("CUDA unavailable")
    for p in (MODEL, FORGET, RETAIN, AUDIT, AUTO / "common.py"):
        if not Path(p).exists():
            issues.append(f"missing: {p}")
    return issues

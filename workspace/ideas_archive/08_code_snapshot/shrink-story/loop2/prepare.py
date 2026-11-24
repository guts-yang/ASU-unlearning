"""Fixed evaluation environment for shrink-story Round 2 (depth / anti-recovery).

DO NOT EDIT during the loop. Holds the fixed single metric, P3 protocol,
base forget NLL (read from measured data, never hardcoded) and anchor I/O.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

LOOP = Path(__file__).resolve().parent
REPO = LOOP.parents[2]
AUTO = REPO / "workspace" / "auto_experiments"
MODEL = "/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
FORGET = "/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json"
RETAIN = "/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json"
AUDIT = AUTO / "results" / "audit_depth.json"
ANCHOR = LOOP / "anchors2" / "depth.json"
ROUNDS = LOOP / "rounds2"

TAU = 2.25
STEPS = 60
ALPHA = 0.1
LR = 5e-6
SEED = 0
P3_STEPS = 20
P3_LR = 1e-5
RANK = 8
TARGET_LAYER = 8

LAMBDA_R = 1.5
LAMBDA_F = 1.5
GAMMA_K = 0.02


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def base_forget_nll() -> float:
    """Read the measured base-model forget NLL; never hardcode a numeric constant."""
    return float(load_json(AUDIT)["models"]["base"]["P1_direct"]["forget"])


def rho_p3(f1: float, f3: float, f_base: float) -> float:
    """Normalized P3 recovery ratio: fraction of the forgetting gain that a
    20-step SGD relearning attack can undo. Lower = deeper forgetting."""
    denom = max(f1 - f_base, 1e-6)
    return (f1 - f3) / denom


def compute_regret(rho: float, dF: float, dR: float, anchor: dict,
                   complexity: int, kind: str) -> float:
    """Fixed single metric (lower is better).

    r2(c) = [rho_c - rho_a]
            + LAMBDA_R * max(0, dR_c - dR_a)
            + LAMBDA_F * max(0, dF_a - dF_c)
            + GAMMA_K * kappa
    The third term blocks the degenerate path of 'looking deeper' by simply
    forgetting less. Anchor rounds are the control, hence r2 = 0 by definition.
    """
    if kind == "anchor":
        return 0.0
    return (
        (rho - anchor["rho"])
        + LAMBDA_R * max(0.0, dR - anchor["dR"])
        + LAMBDA_F * max(0.0, anchor["dF"] - dF)
        + GAMMA_K * complexity
    )


def preflight():
    issues = []
    if not torch.cuda.is_available():
        issues.append("CUDA unavailable")
    for p in (MODEL, FORGET, RETAIN, AUDIT, AUTO / "common.py"):
        if not Path(p).exists():
            issues.append(f"missing: {p}")
    return issues

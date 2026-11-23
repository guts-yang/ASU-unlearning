"""Fixed evaluation environment for the shrink-story autoresearch loop.

DO NOT EDIT during the loop. Holds paths, the fixed evaluation metric
(equal-budget shrink regret), and anchor I/O. Equivalent to autoresearch's
prepare.py / evaluate_bpb.
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
SENS_FILE = AUTO / "results" / "gate_sensitemp.json"
ANCHORS = LOOP / "anchors"
ROUNDS = LOOP / "rounds"

LAMBDA_R = 2.0
GAMMA_K = 0.02
STEPS = 60
ALPHA = 0.1
LR = 5e-6
SEED = 0


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def anchor_path(mu: float) -> Path:
    return ANCHORS / f"{mu:.4f}.json"


def write_anchor(mu: float, dF: float, dR: float, detail: dict):
    dump_json({"mean_tau": mu, "dF": dF, "dR": dR, **detail}, anchor_path(mu))


def compute_regret(dF: float, dR: float, anchor: dict, complexity: int, kind: str) -> float:
    """Fixed single metric (lower is better).

    r(c) = [dF_ctrl - dF_c] + LAMBDA_R * max(0, dR_c - dR_ctrl) + GAMMA_K * kappa
    For anchor rounds the candidate IS the control, so r = 0 by construction.
    """
    if kind == "anchor":
        return 0.0
    return (
        (anchor["dF"] - dF)
        + LAMBDA_R * max(0.0, dR - anchor["dR"])
        + GAMMA_K * complexity
    )


def preflight():
    issues = []
    if not torch.cuda.is_available():
        issues.append("CUDA unavailable")
    for p in (MODEL, FORGET, RETAIN, SENS_FILE):
        if not Path(p).exists():
            issues.append(f"missing: {p}")
    if not (AUTO / "common.py").exists():
        issues.append(f"missing upstream common.py at {AUTO}")
    return issues

"""Fixed environment, evaluation protocol, single metric and FROZEN termination gate.

DO NOT EDIT during the loop. v2 implements the generation-quality gate and the
3-seed confirmation protocol so TARGET_MET is empirically decidable (unlike v1).
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

# frozen training budget
STEPS = 60
TAU = 2.25
ALPHA = 0.1
LR = 5e-6
P3_STEPS = 20
P3_LR = 1e-5

# frozen generation-quality gate
GEN_SAMPLES = 32
GEN_MAX_NEW_TOKENS = 64
REFUSAL_TEMPLATES = ("i don't know", "i cannot", "i'm sorry", "as an ai",
                     "no information", "i do not know", "i can't", "unknown")

# frozen "good enough" target (NOT relaxed from v1's intent; now fully measurable)
RHO_TARGET = 0.38
DF_BAND = 1.1416280269622803
DF_TOL = 0.15
DR_MAX = -0.20
REFUSAL_MAX = 0.10
MEAN_LEN_MIN = 8
SEED_STD_MAX = 0.05
CONFIRM_SEEDS = (0, 1, 2)

# single metric coefficients
LAMBDA_R = 1.5
LAMBDA_DF = 1.5
GAMMA_K = 0.01
LAMBDA_GEN = 2.0

MAX_ROUNDS = 10
MAX_GPU_SECONDS = 1200
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


def degenerate(text: str, n_tokens: int) -> bool:
    t = text.strip().lower()
    if n_tokens < 3 or not t:
        return True
    if any(k in t for k in REFUSAL_TEMPLATES):
        return True
    if n_tokens > 200:
        return True
    words = t.split()
    if len(words) >= 5 and len(set(words)) / len(words) < 0.3:
        return True
    return False


def val_bpb(rho, dF, dR, complexity, refusal_rate) -> float:
    return (
        rho
        + LAMBDA_R * max(0.0, dR - DR_MAX)
        + LAMBDA_DF * abs(dF - DF_BAND)
        + GAMMA_K * complexity
        + LAMBDA_GEN * max(0.0, refusal_rate - REFUSAL_MAX)
    )


def initial_checks(m: dict):
    """First-stage screen (no 3-seed yet)."""
    return {
        "rho_le_target": m["rho"] <= RHO_TARGET,
        "df_in_band": abs(m["dF"] - DF_BAND) <= DF_TOL,
        "dr_le_max": m["dR"] <= DR_MAX,
        "generation_ok": (m["refusal_rate"] <= REFUSAL_MAX and m["mean_len"] >= MEAN_LEN_MIN),
    }


def is_good(m: dict):
    """Full gate. Returns (bool, per-check dict). All dimensions are now measured."""
    checks = initial_checks(m)
    checks["seed_confirmed"] = bool(m.get("seed_confirmed", False))
    return all(checks.values()), checks


REQUIRED_ROUND_KEYS = (
    "round", "idea_id", "val_bpb", "rho", "dF", "dR", "before", "after", "p3",
    "generation", "refusal_rate", "mean_len", "peak_vram_mb", "initial_checks",
)


def verify_round(d: dict):
    missing = [k for k in REQUIRED_ROUND_KEYS if k not in d]
    if missing:
        return False, f"missing keys: {missing}"
    for k in ("val_bpb", "rho", "dF", "dR", "refusal_rate", "mean_len", "peak_vram_mb"):
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

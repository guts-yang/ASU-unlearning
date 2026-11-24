"""Fixed environment + INDEPENDENT replication metric + frozen termination gate (v5).

Seed idea = v4 R3 (the best idea whose metric was INDEPENDENTLY replicated:
alpha=0.04, hinge_beta=0.15, rho_mean=0.2570, rho_std=0.1019, val_bpb=1.0746).
v5 adds order-independent stabilizers (EMA / tail-averaging / retain mixing).
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

# the ALREADY-PASSED idea (source: auto_train_loop_v3/replications/R2.json)
SEED_IDEA = {"kind": "alpha_hinge", "alpha": 0.04, "hinge_beta": 0.15,
             "hinge_margin": 1.2, "lr": 5e-6, "retain_weight": 1.0,
             "grad_clip": 1.0, "stabilizer": {"type": "none"}}
SEED_METRIC = {"rho_mean": 0.2570, "rho_std": 0.1019, "val_bpb": 1.074645,
               "good": False, "source": "auto_train_loop_v4 R3 (3 independent data orders)"}

STEPS = 60
TAU = 2.25
ALPHA = 0.1
LR = 5e-6
P3_STEPS = 20
P3_LR = 1e-5
INIT_SEED = 0                 # fixed: pretrained weights, dropout=0
REPL_ORDERS = (0, 1, 2)      # INDEPENDENT replication axis

GEN_SAMPLES = 32
GEN_MAX_NEW_TOKENS = 64
REFUSAL_TEMPLATES = ("i don't know", "i cannot", "i'm sorry", "as an ai",
                     "no information", "i do not know", "i can't", "unknown")

RHO_MEAN_TARGET = 0.38
RHO_STD_MAX = 0.05
DF_BAND = 1.1416280269622803
DF_TOL = 0.15
DR_MAX = -0.20
REFUSAL_MAX = 0.10
MEAN_LEN_MIN = 8

W_RHO_STD = 2.0
LAMBDA_R = 1.5
LAMBDA_DF = 1.5
GAMMA_K = 0.01
LAMBDA_GEN = 2.0

MAX_ROUNDS = 5
MAX_GPU_SECONDS = 1800
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


def rho_p3(f1, f3, f_base):
    return (f1 - f3) / max(f1 - f_base, 1e-6)


def degenerate(text, n_tokens):
    t = text.strip().lower()
    if n_tokens < 3 or not t:
        return True
    if any(k in t for k in REFUSAL_TEMPLATES):
        return True
    if n_tokens > 200:
        return True
    w = t.split()
    if len(w) >= 5 and len(set(w)) / len(w) < 0.3:
        return True
    return False


def aggregate(order_results, complexity):
    """Reproduced metric over INDEPENDENT data-order perturbations."""
    rhos = [r["rho"] for r in order_results]
    dFs = [r["dF"] for r in order_results]
    dRs = [r["dR"] for r in order_results]
    refs = [r["refusal_rate"] for r in order_results]
    lens = [r["mean_len"] for r in order_results]
    mean = sum(rhos) / len(rhos)
    std = (sum((x - mean) ** 2 for x in rhos) / len(rhos)) ** 0.5
    worst_df = max(abs(d - DF_BAND) for d in dFs)
    worst_dr = max(dRs)
    worst_ref = max(refs)
    val = (mean + W_RHO_STD * std
           + LAMBDA_R * max(0.0, worst_dr - DR_MAX)
           + LAMBDA_DF * worst_df
           + GAMMA_K * complexity
           + LAMBDA_GEN * max(0.0, worst_ref - REFUSAL_MAX))
    checks = {
        "rho_mean_le_target": mean <= RHO_MEAN_TARGET,
        "rho_std_le_max": std <= RHO_STD_MAX,
        "all_df_in_band": all(abs(d - DF_BAND) <= DF_TOL for d in dFs),
        "all_dr_le_max": all(d <= DR_MAX for d in dRs),
        "all_generation_ok": all(r <= REFUSAL_MAX and l >= MEAN_LEN_MIN
                                 for r, l in zip(refs, lens)),
        "replicated": len(order_results) == len(REPL_ORDERS),
    }
    return {"rho_mean": mean, "rho_std": std, "rhos": rhos,
            "dFs": dFs, "dRs": dRs, "refusals": refs, "mean_lens": lens,
            "worst_df": worst_df, "worst_dr": worst_dr, "worst_ref": worst_ref,
            "val_bpb": val, "checks": checks, "good": all(checks.values())}


def verify_run(d):
    req = ("data_order", "rho", "dF", "dR", "refusal_rate", "mean_len", "val_bpb")
    missing = [k for k in req if k not in d]
    if missing:
        return False, f"missing {missing}"
    for k in ("rho", "dF", "dR", "refusal_rate", "mean_len", "val_bpb"):
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

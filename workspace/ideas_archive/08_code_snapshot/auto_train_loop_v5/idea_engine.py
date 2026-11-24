"""Deterministic Idea Engine v5 (no LLM backend on this host).

Starts from v4 R3 (best independently replicated idea) and searches a NEW lever
family: order-independent stabilizers (EMA weight averaging, tail-step parameter
averaging, retain mixing). These are the mechanisms v4's conclusion called for;
eta/alpha tuning is kept only as a fallback.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os

import prepare as P


def _p(**over):
    d = copy.deepcopy(P.SEED_IDEA)
    d.update(over)
    return d


def _ema(decay):
    return _p(stabilizer={"type": "ema", "decay": decay})


def _tail(k):
    return _p(stabilizer={"type": "tail_avg", "k": k})


def _mix(n):
    return _p(stabilizer={"type": "retain_mix", "n": n})


LEVERS = {
    "seed": {"values": [_p()], "kappa": 2,
             "kill": "re-validate v4 R3 on the independent axis before adding stabilizers"},
    "ema": {"values": [_ema(0.90), _ema(0.95), _ema(0.99)], "kappa": 1,
            "kill": "discard if rho_std not reduced or rho_mean worsens"},
    "tail_avg": {"values": [_tail(10), _tail(20)], "kappa": 1,
                 "kill": "discard if rho_std not reduced"},
    "retain_mix": {"values": [_mix(8), _mix(12)], "kappa": 1,
                   "kill": "discard if dF leaves band in any order"},
    "alpha_tune": {"values": [_p(alpha=0.05), _p(alpha=0.06)], "kappa": 0,
                   "kill": "discard if dF leaves band in any order"},
    "beta_tune": {"values": [_p(hinge_beta=0.10), _p(hinge_beta=0.20)], "kappa": 0,
                  "kill": "discard if rho_mean worsens"},
}

FORBIDDEN_LEVERS = {"tau_field", "tau_schedule", "projection", "counterfactual", "gradient_surgery"}
LEVER_ORDER = ["seed", "ema", "tail_avg", "retain_mix", "alpha_tune", "beta_tune"]
STABILITY_FAMILY = ["ema", "tail_avg", "retain_mix"]


def llm_backend_status():
    ep = os.environ.get("IDEA_LLM_ENDPOINT")
    if ep:
        return {"status": "NOT_IMPLEMENTED", "endpoint": ep,
                "note": "adapter declared but not wired; deterministic fallback used"}
    return {"status": "REVIEW_UNAVAILABLE",
            "note": "no .aris/reviewer.json; codex/gemini/grok/knot CLIs all missing"}


def idea_hash(lever, params):
    return hashlib.sha256((lever + "|" + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:12]


def diagnose(history, last):
    if not history:
        return "seed", "no_history: re-validate v4 R3 before adding stabilizers"
    if last["rho_std"] > P.RHO_STD_MAX:
        return "ema", f"unstable rho_std={last['rho_std']:.4f}>{P.RHO_STD_MAX}; try order-independent EMA averaging"
    if last["worst_df"] > P.DF_TOL:
        return "retain_mix", f"df_band_violation worst={last['worst_df']:.4f}; stabilize gradients with more retain"
    if last["rho_mean"] > P.RHO_MEAN_TARGET:
        return "beta_tune", f"rho_mean_high {last['rho_mean']:.4f}>{P.RHO_MEAN_TARGET}"
    return "alpha_tune", "all gates met: conservative alpha probe"


def _pick(lever, seen):
    for params in LEVERS[lever]["values"]:
        h = idea_hash(lever, params)
        if h not in seen:
            return lever, params, h
    return None


def next_idea(state):
    history, seen = state["history"], set(state["seen_ideas"])
    last = history[-1] if history else None
    lever, why = diagnose(history, last)

    picked = _pick(lever, seen)
    if picked is None and lever in STABILITY_FAMILY:
        for lv in STABILITY_FAMILY:
            if lv == lever:
                continue
            picked = _pick(lv, seen)
            if picked:
                break
    if picked is None:
        for lv in LEVER_ORDER:
            if lv == lever or lv in FORBIDDEN_LEVERS:
                continue
            picked = _pick(lv, seen)
            if picked:
                break
    if picked is None:
        return None
    lever, params, h = picked

    return {
        "id": f"idea_R{state['next_round']}_{lever}_{h}", "round": state["next_round"],
        "hypothesis": f"{why}; apply {lever} {json.dumps(params['stabilizer'])}",
        "lever": lever, "params": params, "seed_idea": P.SEED_IDEA,
        "expected_effect": "reproduced rho_mean<=0.38 with rho_std<=0.05 and all orders in the dF band",
        "kill_criterion": LEVERS[lever]["kill"],
        "source": "deterministic_rule:" + why.split(":")[0],
        "llm_backend": llm_backend_status()["status"],
        "complexity": LEVERS[lever]["kappa"],
        "budget": {"mean_tau": P.TAU, "steps": P.STEPS,
                   "replication_orders": list(P.REPL_ORDERS), "init_seed": P.INIT_SEED},
        "hash": h,
    }

"""Deterministic Idea Engine v3 (no LLM backend available on this host).

Prioritizes the config that was best in v2 at seed 0 (alpha=0.05,
hinge_beta=0.15) so the loop can test whether it REPLICATES under 3 seeds.
Reproduction status (rho_std) is part of the diagnosis input.
"""

from __future__ import annotations

import hashlib
import json
import os

import prepare as P

LEVERS = {
    "baseline": {"values": [{"kind": "baseline"}], "kappa": 0,
                 "kill": "anchor: establishes the reproduced baseline metric"},
    "alpha_hinge": {"values": [
        {"kind": "alpha_hinge", "alpha": 0.05, "hinge_beta": 0.15, "hinge_margin": 1.2},
        {"kind": "alpha_hinge", "alpha": 0.05, "hinge_beta": 0.10, "hinge_margin": 1.2},
        {"kind": "alpha_hinge", "alpha": 0.07, "hinge_beta": 0.15, "hinge_margin": 1.2},
        {"kind": "alpha_hinge", "alpha": 0.07, "hinge_beta": 0.10, "hinge_margin": 1.2},
        {"kind": "alpha_hinge", "alpha": 0.05, "hinge_beta": 0.20, "hinge_margin": 1.2},
        {"kind": "alpha_hinge", "alpha": 0.10, "hinge_beta": 0.10, "hinge_margin": 1.2},
    ], "kappa": 2, "kill": "discard if rho_std>0.05 or dF leaves band in any seed"},
    "retain_weight": {"values": [
        {"kind": "retain_weight", "retain_weight": 1.5},
        {"kind": "retain_weight", "retain_weight": 2.0},
    ], "kappa": 0, "kill": "discard if rho_mean worsens"},
    "hinge_beta": {"values": [
        {"kind": "hinge", "hinge_beta": 0.05, "hinge_margin": 1.2},
        {"kind": "hinge", "hinge_beta": 0.10, "hinge_margin": 1.2},
    ], "kappa": 2, "kill": "discard if dF inflation not offset"},
    "alpha": {"values": [{"kind": "alpha", "alpha": 0.07}, {"kind": "alpha", "alpha": 0.15}],
              "kappa": 0, "kill": "discard if retain degrades"},
}

FORBIDDEN_LEVERS = {"tau_field", "tau_schedule", "projection", "counterfactual", "gradient_surgery"}
LEVER_ORDER = ["baseline", "alpha_hinge", "retain_weight", "hinge_beta", "alpha"]


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
        return "baseline", "no_history: establish the 3-seed reproduced baseline"
    if last["rho_std"] > P.RHO_STD_MAX:
        return "retain_weight", f"unstable: rho_std={last['rho_std']:.4f}>{P.RHO_STD_MAX}; damp initialization sensitivity"
    if last["worst_df"] > P.DF_TOL:
        return "alpha_hinge", f"df_band_violation worst={last['worst_df']:.4f}"
    if last["rho_mean"] > P.RHO_MEAN_TARGET:
        return "alpha_hinge", f"rho_mean_high: {last['rho_mean']:.4f}>{P.RHO_MEAN_TARGET}"
    return "alpha", "all gates met: conservative probe"


def next_idea(state):
    history, seen = state["history"], set(state["seen_ideas"])
    last = history[-1] if history else None
    lever, why = diagnose(history, last)
    picked = None
    for params in LEVERS[lever]["values"]:
        h = idea_hash(lever, params)
        if h not in seen:
            picked = (lever, params, h)
            break
    if picked is None:
        for lv in LEVER_ORDER:
            if lv == lever or lv in FORBIDDEN_LEVERS:
                continue
            for params in LEVERS[lv]["values"]:
                h = idea_hash(lv, params)
                if h not in seen:
                    picked = (lv, params, h)
                    break
            if picked:
                break
    if picked is None:
        return None
    lever, params, h = picked
    return {
        "id": f"idea_R{state['next_round']}_{lever}_{h}", "round": state["next_round"],
        "hypothesis": f"{why}; apply {lever} {json.dumps(params)}",
        "lever": lever, "params": params,
        "expected_effect": "reproducible rho_mean<=0.38 across 3 seeds, all seeds in the dF band",
        "kill_criterion": LEVERS[lever]["kill"],
        "source": "deterministic_rule:" + why.split(":")[0],
        "llm_backend": llm_backend_status()["status"],
        "complexity": LEVERS[lever]["kappa"],
        "budget": {"mean_tau": P.TAU, "steps": P.STEPS,
                   "replication_seeds": list(P.REPL_SEEDS),
                   "data_order_seed": P.DATA_ORDER_SEED},
        "hash": h,
    }

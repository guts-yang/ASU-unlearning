"""Deterministic Idea Engine v4 (no LLM backend on this host).

Starts from the ALREADY-PASSED idea (v3 R2, alpha_hinge 0.05/0.15) and searches
only levers that can reduce data-order sensitivity or keep the improvement
inside the equal-budget band under independent replication.
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


LEVERS = {
    "seed": {"values": [_p()], "kappa": 2,
             "kill": "re-validate the passed idea on the INDEPENDENT replication axis"},
    "stability_lr": {"values": [_p(lr=2e-6), _p(lr=1e-6)], "kappa": 0,
                     "kill": "discard if rho_std not reduced or rho_mean worsens"},
    "stability_retain": {"values": [_p(retain_weight=1.5), _p(retain_weight=2.0)], "kappa": 0,
                         "kill": "discard if rho_mean worsens"},
    "stability_clip": {"values": [_p(grad_clip=0.5), _p(grad_clip=0.2)], "kappa": 0,
                       "kill": "discard if rho_std not reduced"},
    "alpha_tune": {"values": [_p(alpha=0.04), _p(alpha=0.06), _p(alpha=0.07)], "kappa": 0,
                   "kill": "discard if dF leaves band in any order"},
    "beta_tune": {"values": [_p(hinge_beta=0.10), _p(hinge_beta=0.20)], "kappa": 0,
                  "kill": "discard if rho_mean worsens"},
    "margin_tune": {"values": [_p(hinge_margin=1.0), _p(hinge_margin=1.5)], "kappa": 0,
                    "kill": "discard if dF band violation persists"},
}

FORBIDDEN_LEVERS = {"tau_field", "tau_schedule", "projection", "counterfactual", "gradient_surgery"}
LEVER_ORDER = ["seed", "stability_lr", "stability_retain", "stability_clip",
               "alpha_tune", "beta_tune", "margin_tune"]
STABILITY_FAMILY = ["stability_lr", "stability_retain", "stability_clip"]


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
        return "seed", "no_history: re-validate the PASSED idea under independent data-order replication"
    if last["rho_std"] > P.RHO_STD_MAX:
        return "stability_lr", f"unstable: rho_std={last['rho_std']:.4f}>{P.RHO_STD_MAX}; reduce lr / damp sensitivity"
    if last["worst_df"] > P.DF_TOL:
        return "alpha_tune", f"df_band_violation worst={last['worst_df']:.4f}"
    if last["rho_mean"] > P.RHO_MEAN_TARGET:
        return "beta_tune", f"rho_mean_high {last['rho_mean']:.4f}>{P.RHO_MEAN_TARGET}"
    return "margin_tune", "all gates met: conservative margin probe"


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
        "id": f"idea_R{state['next_round']}_{lever}_{h}",
        "round": state["next_round"],
        "hypothesis": f"{why}; apply {lever} {json.dumps(params)}",
        "lever": lever, "params": params,
        "seed_idea": P.SEED_IDEA,
        "expected_effect": "reproduced rho_mean<=0.38 with rho_std<=0.05 across 3 independent data orders",
        "kill_criterion": LEVERS[lever]["kill"],
        "source": "deterministic_rule:" + why.split(":")[0],
        "llm_backend": llm_backend_status()["status"],
        "complexity": LEVERS[lever]["kappa"],
        "budget": {"mean_tau": P.TAU, "steps": P.STEPS,
                   "replication_orders": list(P.REPL_ORDERS), "init_seed": P.INIT_SEED},
        "hash": h,
    }

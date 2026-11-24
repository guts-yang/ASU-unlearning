"""Deterministic Idea Engine v2: diagnose -> emit structured idea_spec.

No LLM / cross-model reviewer backend on this host. Idea generation is a
deterministic rule engine over a FINITE, pre-declared lever space; the LLM path
is a pluggable adapter currently REVIEW_UNAVAILABLE. Never claim a model "thought".
"""

from __future__ import annotations

import hashlib
import json
import os

import prepare as P

LEVERS = {
    "baseline": {"values": [{"kind": "baseline"}], "kappa": 0,
                 "kill": "anchor round, no claim"},
    "alpha_hinge": {"values": [
        {"kind": "alpha_hinge", "alpha": a, "hinge_beta": b, "hinge_margin": 1.2}
        for a in (0.05, 0.07) for b in (0.05, 0.10, 0.15)
    ], "kappa": 2, "kill": "discard if dF leaves band or rho does not improve"},
    "hinge_beta": {"values": [
        {"kind": "hinge", "hinge_beta": b, "hinge_margin": 1.2}
        for b in (0.05, 0.10, 0.20)
    ], "kappa": 2, "kill": "discard if dF inflation is not offset"},
    "hinge_margin": {"values": [
        {"kind": "hinge", "hinge_beta": 0.10, "hinge_margin": m}
        for m in (1.0, 1.2, 1.5)
    ], "kappa": 2, "kill": "discard if band violation persists"},
    "alpha": {"values": [{"kind": "alpha", "alpha": a} for a in (0.07, 0.15, 0.20)],
              "kappa": 0, "kill": "discard if retain degrades"},
    "retain_weight": {"values": [{"kind": "retain_weight", "retain_weight": w}
                                 for w in (1.5, 2.0)],
                      "kappa": 0, "kill": "discard if rho worsens"},
}

FORBIDDEN_LEVERS = {"tau_field", "tau_schedule", "projection", "counterfactual", "gradient_surgery"}
LEVER_ORDER = ["baseline", "alpha_hinge", "hinge_beta", "hinge_margin", "alpha", "retain_weight"]


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
        return "baseline", "no_history: establish equal-budget anchor"
    if abs(last["dF"] - P.DF_BAND) > P.DF_TOL:
        return "alpha_hinge", f"df_band_violation: |dF-{P.DF_BAND:.4f}|>{P.DF_TOL}; lower alpha to offset hinge"
    if last["dR"] > P.DR_MAX:
        return "retain_weight", f"retain_damage: dR={last['dR']:+.4f}"
    if last["refusal_rate"] > P.REFUSAL_MAX:
        return "hinge_margin", f"degenerate_output: refusal={last['refusal_rate']:.3f}"
    if last["rho"] > P.RHO_TARGET:
        return "alpha_hinge", f"rho_high: {last['rho']:.4f}>{P.RHO_TARGET}; keep dF in band while adding hinge"
    return "alpha", "screening passed: conservative alpha probe before 3-seed confirmation"


def pivot(seen, history):
    if len(history) < 2:
        return None
    v = [h["val_bpb"] for h in history[-2:]]
    if len(v) == 2 and v[-1] >= v[-2]:
        for lever in LEVER_ORDER:
            for params in LEVERS[lever]["values"]:
                h = idea_hash(lever, params)
                if h not in seen:
                    return lever, params, h
    return None


def next_idea(state):
    history, seen = state["history"], set(state["seen_ideas"])
    last = history[-1] if history else None
    pv = pivot(seen, history)
    if pv:
        lever, params, h = pv
        why = "stall_pivot"
    else:
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
        "id": f"idea_R{state['next_round']}_{lever}_{h}",
        "round": state["next_round"], "hypothesis": f"{why}; apply {lever} {json.dumps(params)}",
        "lever": lever, "params": params,
        "expected_effect": "lower rho_P3 within the equal-budget dF band, retain intact, no degenerate output",
        "kill_criterion": LEVERS[lever]["kill"],
        "source": "deterministic_rule:" + why.split(":")[0],
        "llm_backend": llm_backend_status()["status"],
        "complexity": LEVERS[lever]["kappa"],
        "budget": {"mean_tau": P.TAU, "steps": P.STEPS, "seed": state["seed"]},
        "hash": h,
    }

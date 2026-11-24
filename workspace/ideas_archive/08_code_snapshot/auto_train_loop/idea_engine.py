"""Deterministic Idea Engine: diagnose last round -> emit a structured idea_spec.

IMPORTANT: this host has no LLM / cross-model reviewer backend. Idea generation is
a deterministic rule engine over a FINITE, pre-declared lever space. The LLM path
is a pluggable adapter that is currently REVIEW_UNAVAILABLE; we never pretend a
model "thought up" an idea.
"""

from __future__ import annotations

import hashlib
import json
import os

import prepare as P

# Finite, auditable search space. The engine may ONLY pick from these levers.
LEVERS = {
    "baseline": {"values": [{"kind": "baseline"}], "kappa": 0,
                 "kill": "anchor round, no claim"},
    "hinge_beta": {"values": [
        {"kind": "hinge", "hinge_beta": 0.05, "hinge_margin": 1.2},
        {"kind": "hinge", "hinge_beta": 0.10, "hinge_margin": 1.2},
        {"kind": "hinge", "hinge_beta": 0.20, "hinge_margin": 1.2},
        {"kind": "hinge", "hinge_beta": 0.30, "hinge_margin": 1.2},
    ], "kappa": 2, "kill": "discard if dF leaves the equal-budget band or rho does not drop"},
    "hinge_margin": {"values": [
        {"kind": "hinge", "hinge_beta": 0.10, "hinge_margin": 1.0},
        {"kind": "hinge", "hinge_beta": 0.10, "hinge_margin": 1.5},
        {"kind": "hinge", "hinge_beta": 0.10, "hinge_margin": 2.0},
    ], "kappa": 2, "kill": "discard if dF band violation persists"},
    "alpha": {"values": [
        {"kind": "alpha", "alpha": 0.05},
        {"kind": "alpha", "alpha": 0.15},
        {"kind": "alpha", "alpha": 0.20},
    ], "kappa": 0, "kill": "discard if retain degrades or rho does not improve"},
    "retain_weight": {"values": [
        {"kind": "retain_weight", "retain_weight": 1.5},
        {"kind": "retain_weight", "retain_weight": 2.0},
    ], "kappa": 0, "kill": "discard if rho worsens"},
}

# Directions already falsified by prior rounds; hard-blocked (idea-discovery
# anti-duplication memory).
FORBIDDEN_LEVERS = {"tau_field", "tau_schedule", "projection", "counterfactual", "gradient_surgery"}

LEVER_ORDER = ["baseline", "hinge_beta", "hinge_margin", "alpha", "retain_weight"]


def llm_backend_status():
    ep = os.environ.get("IDEA_LLM_ENDPOINT")
    if ep:
        return {"status": "NOT_IMPLEMENTED", "endpoint": ep,
                "note": "adapter declared but not wired; deterministic fallback used"}
    return {"status": "REVIEW_UNAVAILABLE",
            "note": "no .aris/reviewer.json; codex/gemini/grok/knot CLIs all missing"}


def idea_hash(lever, params):
    key = lever + "|" + json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def diagnose(history, last):
    """Deterministic mapping from measured state to a lever family."""
    if not history:
        return "baseline", "no_history: establish equal-budget anchor"
    if abs(last["dF"] - P.DF_BAND) > P.DF_TOL:
        return "hinge_margin", f"df_band_violation: |dF-{P.DF_BAND:.4f}|>{P.DF_TOL}"
    if last["dR"] > P.DR_MAX:
        return "retain_weight", f"retain_damage: dR={last['dR']:+.4f}>{P.DR_MAX}"
    if last["rho"] > P.RHO_TARGET:
        return "hinge_beta", f"rho_high: rho={last['rho']:.4f}>{P.RHO_TARGET}"
    return "alpha", "all_gates_met_except_generation: conservative alpha probe"


def pivot(seen, history):
    """Forced structural pivot after 2 non-improving rounds."""
    if len(history) < 2:
        return None
    v = [h["val_bpb"] for h in history[-2:]]
    if len(v) == 2 and v[-1] >= v[-2]:
        for lever in LEVER_ORDER:
            for params in LEVERS[lever]["values"]:
                h = idea_hash(lever, params)
                if h not in seen:
                    return lever, params, h, "stall_pivot"
    return None


def next_idea(state):
    """Return a schema-valid, non-duplicate idea_spec, or None if space exhausted."""
    history = state["history"]
    seen = set(state["seen_ideas"])
    last = history[-1] if history else None

    pv = pivot(seen, history)
    if pv:
        lever, params, h, why = pv
    else:
        lever, why = diagnose(history, last)
        candidates = LEVERS[lever]["values"]
        chosen = None
        for params in candidates:
            h = idea_hash(lever, params)
            if h not in seen:
                chosen, params = h, params
                break
        if chosen is None:
            # exhausted lever family -> scan remaining families (anti-repetition)
            chosen = None
            for lv in LEVER_ORDER:
                if lv == lever or lv in FORBIDDEN_LEVERS:
                    continue
                for params in LEVERS[lv]["values"]:
                    h = idea_hash(lv, params)
                    if h not in seen:
                        lever, chosen = lv, h
                        break
                if chosen:
                    break
        if chosen is None:
            return None
        h = chosen

    spec = {
        "id": f"idea_R{state['next_round']}_{lever}_{h}",
        "round": state["next_round"],
        "hypothesis": f"{why}; apply {lever} with {json.dumps(params, ensure_ascii=False)}",
        "lever": lever,
        "params": params,
        "expected_effect": "lower rho_P3 while keeping dF in the equal-budget band and retain intact",
        "kill_criterion": LEVERS[lever]["kill"],
        "source": "deterministic_rule:" + why.split(":")[0],
        "llm_backend": llm_backend_status()["status"],
        "complexity": LEVERS[lever]["kappa"],
        "budget": {"mean_tau": P.TAU, "steps": P.STEPS, "seed": state["seed"]},
        "hash": h,
    }
    return spec

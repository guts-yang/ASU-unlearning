"""Deterministic Idea Engine v6 (no LLM backend on this host).

Gate 0 (verdict=KILL, per-layer cos) is a DIAGNOSTIC INPUT, not a hard stop:
its only high-cos layer (L15) drives the first contraction hypothesis.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os

import prepare as P

BASE = {"layer_select": "cos_high", "alpha_B": 1.0,
        "kl_weight": 0.1, "lf_form": "mse", "lf_weight": 1.0}


def _p(**over):
    d = copy.deepcopy(BASE)
    d.update(over)
    return d


LEVERS = {
    "layer_select": {"values": [
        _p(layer_select="cos_high"),      # only L15 (Gate0 cos=0.566)
        _p(layer_select="all"),
        _p(layer_select="cos_weighted"),
    ], "kappa": 1, "kill": "Gate0 flagged layers with cos<0.30; discard if dF below ASU baseline"},
    "alpha_B": {"values": [_p(layer_select="cos_high", alpha_B=a) for a in (0.5, 2.0)],
                "kappa": 0, "kill": "discard if donor_leak rises or dF drops"},
    "kl_weight": {"values": [_p(layer_select="cos_high", kl_weight=w) for w in (0.05, 0.2)],
                  "kappa": 0, "kill": "discard if donor_leak exceeds 0.10"},
    "lf_form": {"values": [_p(layer_select="cos_high", lf_form="cosine")],
                "kappa": 0, "kill": "discard if truth_leak exceeds 0.30"},
}

FORBIDDEN_LEVERS = {"tau_field", "tau_schedule", "projection", "gradient_surgery"}
LEVER_ORDER = ["layer_select", "alpha_B", "kl_weight", "lf_form"]


def llm_backend_status():
    ep = os.environ.get("IDEA_LLM_ENDPOINT")
    if ep:
        return {"status": "NOT_IMPLEMENTED", "endpoint": ep,
                "note": "adapter declared but not wired; deterministic fallback used"}
    return {"status": "REVIEW_UNAVAILABLE",
            "note": "no .aris/reviewer.json; codex/gemini/grok/knot CLIs all missing"}


def idea_hash(lever, params):
    return hashlib.sha256((lever + "|" + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:12]


def gate0_diagnosis():
    p = P.LOOP / "gate0.json"
    if not p.exists():
        return {"available": False, "verdict": "MISSING"}
    d = P.load_json(p)
    ql = (d.get("layers") or {}).get("query_last", {})
    per = ql.get("per_layer", {})
    high = [int(k) for k, v in per.items() if v.get("cos_r_r0", 0) >= 0.30]
    return {"available": True, "verdict": d.get("verdict"),
            "cos_mean": ql.get("cos_mean"), "ratio_mean": ql.get("ratio_mean"),
            "high_cos_layers": sorted(high)}


def diagnose(history, last, g0):
    if not history:
        hi = g0.get("high_cos_layers") or [15]
        return "layer_select", f"gate0_{g0.get('verdict')}: restrict to cos-high layers {hi}; original premise unsupported"
    if last["worst_donor"] > P.DONOR_LEAK_MAX:
        return "kl_weight", f"donor_leak={last['worst_donor']:.3f}; weaken counterfactual output alignment"
    if last["worst_truth"] > P.TRUTH_LEAK_MAX:
        return "lf_form", f"truth_leak={last['worst_truth']:.3f}; change activation target form"
    if last["df_mean"] < P.DF_MIN:
        return "alpha_B", f"df={last['df_mean']:.4f}<{P.DF_MIN}; adjust finite-displacement strength"
    return "layer_select", "all gates met: conservative layer probe"


def _pick(lever, seen):
    for params in LEVERS[lever]["values"]:
        h = idea_hash(lever, params)
        if h not in seen:
            return lever, params, h
    return None


def next_idea(state):
    history, seen = state["history"], set(state["seen_ideas"])
    g0 = gate0_diagnosis()
    last = history[-1] if history else None
    lever, why = diagnose(history, last, g0)
    picked = _pick(lever, seen)
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
        "hypothesis": f"{why}; apply {lever} {json.dumps(params)}",
        "lever": lever, "params": params,
        "gate0_diagnosis": g0,
        "seed_idea": "RelResidual (original premise KILLED by Gate0; conditional contraction)",
        "expected_effect": "true fact forgetting >= ASU baseline, no donor substitution, no premature ending, replicated over 3 data orders",
        "kill_criterion": LEVERS[lever]["kill"],
        "source": "deterministic_rule:" + why.split(":")[0],
        "llm_backend": llm_backend_status()["status"],
        "complexity": LEVERS[lever]["kappa"],
        "budget": {"mean_tau": None, "steps": P.STEPS,
                   "replication_orders": list(P.REPL_ORDERS), "init_seed": P.INIT_SEED},
        "hash": h,
    }

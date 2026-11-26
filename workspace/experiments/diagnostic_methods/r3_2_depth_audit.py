"""R3-2: uniform shallow/deep recovery audit wrapper.

The heavy P1/P2/P3/INT8 implementation already lives in
``auto_experiments/audit_depth.py``.  This module only standardizes its output
and makes the Probe-channel gap explicit.
"""

from __future__ import annotations


def summarize_depth_audit(audit: dict) -> dict:
    models = audit.get("models", {})
    rows = {}
    for name, data in models.items():
        rows[name] = {
            "p1_forget": data.get("P1_direct", {}).get("forget"),
            "p2_gain": data.get("P2_fewshot", {}).get("gain"),
            "p3_gain": data.get("P3_relearn", {}).get("gain"),
            "int8_shift": data.get("INT8_fakequant", {}).get("forget_shift"),
            "mean_recovery": data.get("depth_score", {}).get("mean_recovery"),
        }
    values = [r["mean_recovery"] for r in rows.values() if r["mean_recovery"] is not None]
    return {
        "methods": rows,
        "method_separation": max(values) - min(values) if len(values) >= 2 else 0.0,
        "shallow_forgetting_detected": audit.get("gate", {}).get("shallow_forgetting_detected"),
        "probe_channel_present": False,
        "verdict": "POSITIVE_SCREEN" if audit.get("gate", {}).get("shallow_forgetting_detected") else "INCONCLUSIVE",
    }


def summarize_probe_audit(audit: dict) -> dict:
    """Summarize the 3-seed probe-audit JSON without re-running the audit."""
    rows = {}
    for name, data in audit.get("models", {}).items():
        def m(key):
            return data.get(key, {}).get("mean")
        rows[name] = {
            "p1_forget": m("P1_forget"),
            "p2_gain": m("P2_gain"),
            "p3_gain": m("P3_gain"),
            "int8_shift": m("INT8_shift"),
            "probe_auc_pre": m("probe_auc_pre"),
            "probe_auc_post": m("probe_auc_post"),
            "probe_recoverability_pre": m("probe_recoverability_pre"),
            "probe_recoverability_post": m("probe_recoverability_post"),
            "mean_recovery": data.get("depth_score", {}).get("mean_recovery", {}).get("mean"),
        }
    return {
        "methods": rows,
        "probe_channel_present": bool(audit.get("gate", {}).get("probe_channel_present")),
        "shallow_forgetting_detected": audit.get("gate", {}).get("shallow_forgetting_detected"),
        "verdict": audit.get("gate", {}).get("verdict"),
    }


def self_test() -> dict:
    audit = {
        "models": {
            "a": {"P1_direct": {"forget": 1.0}, "P2_fewshot": {"gain": 0.2}, "P3_relearn": {"gain": 0.1}, "INT8_fakequant": {"forget_shift": 0.2}, "depth_score": {"mean_recovery": 0.45}},
            "b": {"P1_direct": {"forget": 0.9}, "P2_fewshot": {"gain": 0.1}, "P3_relearn": {"gain": 0.05}, "INT8_fakequant": {"forget_shift": 0.1}, "depth_score": {"mean_recovery": 0.35}},
        },
        "gate": {"shallow_forgetting_detected": True},
    }
    out = summarize_depth_audit(audit)
    assert out["method_separation"] > 0
    assert out["probe_channel_present"] is False
    return {"status": "ok", "report": out}


__all__ = ["summarize_depth_audit", "summarize_probe_audit", "self_test"]

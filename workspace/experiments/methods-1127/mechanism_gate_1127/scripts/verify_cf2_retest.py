#!/usr/bin/env python3
"""Read-only R1-R3 verifier for the CF2 fix retest.

Thresholds are frozen in PROTOCOL.md §8.3 before the run. Only reads
result.json / diagnostics.json / console.log; never trains or writes weights.
The §7 S1-S4 verdict is deliberately NOT re-judged here.

Usage: python3 verify_cf2_retest.py <mechanism_gate_1127/results>
"""
from __future__ import annotations

import json
import math
import os
import sys

root = sys.argv[1]
status = json.load(open(os.path.join(root, "fix_retest_status.json")))
rows = status["rows"]
REF_SHARE, REF_GDIFF = 0.051575, 0.0082  # §7 S2 observed values
TOL = 0.20

checks = []


def rec(name, ok, detail):
    checks.append({"check": name, "pass": bool(ok), "detail": detail})


def path(tag, arm, fname):
    return os.path.join(root, "fix_retest", tag, arm, fname)


def load(tag, arm, fname="result.json"):
    p = path(tag, arm, fname)
    return json.load(open(p)) if os.path.exists(p) else None


def first(tag, arm):
    res = load(tag, arm)
    h = (res or {}).get("train", {}).get("history", [])
    return res, (h[0] if h else None)


def exit_code(tag, arm):
    for r in rows:
        if r["method"] == tag and r["arm"] == arm:
            return r["exit_code"]
    return None


def console_error(tag, arm):
    p = path(tag, arm, "console.log")
    if not os.path.exists(p):
        return False
    text = open(p, errors="replace").read()
    return "RuntimeError" in text or "Traceback" in text or "FloatingPointError" in text


def share(h, neutral_weight=0.1, retain_weight=0.0, method_weight=0.1):
    ml, neu, ret = h.get("method_loss", 0.0), h.get("neutral", 0.0), h.get("retain_nll", 0.0)
    total = method_weight * ml + neutral_weight * neu + retain_weight * ret
    return (method_weight * ml / total) if total > 0 else 0.0


def gdiff(g1, g2):
    return abs(g1 - g2) / max(abs(g2), 1e-8)


# ---- R1: CF2-06 shape defect fixed, default behaviour still valid ----
r_none, h_none = first("CF2-06", "none")
r_wrong, h_wrong = first("CF2-06", "wrong")
r1_ok = bool(h_none and h_wrong
             and exit_code("CF2-06", "none") == "0"
             and exit_code("CF2-06", "wrong") == "0"
             and h_none.get("control_changed") == 0.0
             and h_wrong.get("control_changed") == 1.0
             and h_wrong.get("method_loss") is not None
             and h_none.get("method_loss") is not None
             and abs(h_wrong["method_loss"] - h_none["method_loss"]) > 1e-12
             and not console_error("CF2-06", "none")
             and not console_error("CF2-06", "wrong"))
rec("R1", r1_ok, {
    "none": {"exit": exit_code("CF2-06", "none"), "control_changed": (h_none or {}).get("control_changed"),
             "method_loss": (h_none or {}).get("method_loss")},
    "wrong": {"exit": exit_code("CF2-06", "wrong"), "control_changed": (h_wrong or {}).get("control_changed"),
              "method_loss": (h_wrong or {}).get("method_loss")},
    "prior_wrong_exit": 1, "prior_error": "tensor a (27) vs tensor b (15)"})

# ---- R2: retest is comparable with §7 ----
_, h_fix = first("CF2-01", "fix")
_, h_nom = first("CF2-01", "fix_nomethod")
r2 = {"ok": False, "note": "missing result files"}
if h_fix and h_nom:
    s = share(h_fix)
    gd = gdiff(h_fix.get("grad_norm", 0.0), h_nom.get("grad_norm", 0.0))
    r2 = {"ok": (REF_SHARE * (1 - TOL) <= s <= REF_SHARE * (1 + TOL)
                 and REF_GDIFF * (1 - TOL) <= gd <= REF_GDIFF * (1 + TOL)),
          "method_share": round(s, 6), "grad_relative_diff": round(gd, 6),
          "grad_norm_fix": h_fix.get("grad_norm"),
          "grad_norm_fix_nomethod": h_nom.get("grad_norm"),
          "reference": {"method_share": REF_SHARE, "grad_relative_diff": REF_GDIFF,
                        "tolerance": f"±{TOL * 100:.0f}%"}}
rec("R2", r2["ok"], r2)

# ---- R3: falsifiable "neutral guards dominate gradient" hypothesis ----
_, h_nn = first("CF2-01", "fix_no_neutral")
_, h_nn_nom = first("CF2-01", "fix_no_neutral_nomethod")
r3 = {"ok": False, "note": "missing result files"}
if h_nn and h_nn_nom:
    s_nn = share(h_nn, neutral_weight=0.0, retain_weight=0.0)
    gd_nn = gdiff(h_nn.get("grad_norm", 0.0), h_nn_nom.get("grad_norm", 0.0))
    g_nom = h_nn_nom.get("grad_norm", 0.0)
    r3 = {"ok": s_nn > REF_SHARE * (1 + TOL) and gd_nn > 0.01,
          "hypothesis_supported": s_nn > REF_SHARE * (1 + TOL) and gd_nn > 0.01,
          "method_share_no_neutral": round(s_nn, 6),
          "grad_relative_diff_no_neutral": round(gd_nn, 6),
          "grad_norm_no_neutral": h_nn.get("grad_norm"),
          "grad_norm_no_neutral_nomethod": g_nom,
          "trivial_warning": bool(g_nom is not None and abs(g_nom) < 1e-6),
          "threshold": {"share_above": round(REF_SHARE * (1 + TOL), 6),
                        "grad_relative_diff": 0.01},
          "limitation": "PASS only proves the method term CAN produce gradient; "
                        "does NOT prove observability under the full loss and "
                        "does NOT re-judge §7 S2."}
rec("R3", r3["ok"], r3)

verdict = {"stage": "fix_retest_verdict", "claim_level": False,
           "pre_registered": "PROTOCOL.md §8.3", "checks": checks,
           "all_pass": all(c["pass"] for c in checks),
           "prior_s1_s4_preserved": True,
           "note": "retest / root-cause verification only; §7 S2-S4 verdicts unchanged; "
                   "retain_weight=0 / neutral_weight=0 are diagnostic arms, not a "
                   "training configuration; no forgetting claim"}
out = os.path.join(root, "fix_retest_verdict.json")
with open(out, "w") as f:
    json.dump(verdict, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps({"all_pass": verdict["all_pass"],
                  "checks": [{"check": c["check"], "pass": c["pass"]} for c in checks]},
                 ensure_ascii=False))
for c in checks:
    print(f"  {c['check']}: {'PASS' if c['pass'] else 'FAIL'}  "
          + json.dumps(c["detail"], ensure_ascii=False, default=str)[:700])
print("wrote", out)

#!/usr/bin/env python3
"""Read-only S1-S4 verifier for the CF2 fix smoke queue.

Thresholds are frozen in PROTOCOL.md §6.3 before the run. This script only
reads result.json / diagnostics.json / console.log; it never trains or writes
weights. Usage: python3 verify_cf2_fix.py <mechanism_gate_1127/results>
"""
from __future__ import annotations

import json
import math
import os
import sys

root = sys.argv[1]
status = json.load(open(os.path.join(root, "fix_smoke_status.json")))
rows = status["rows"]


def path(tag, arm, fname):
    return os.path.join(root, "fix_smoke", tag, arm, fname)


def load(tag, arm, fname="result.json"):
    p = path(tag, arm, fname)
    return json.load(open(p)) if os.path.exists(p) else None


def console_has_error(tag, arm):
    p = path(tag, arm, "console.log")
    if not os.path.exists(p):
        return False
    text = open(p, errors="replace").read()
    return "FloatingPointError" in text or "Traceback" in text


checks = []


def rec(name, ok, detail):
    checks.append({"check": name, "pass": bool(ok), "detail": detail})


# S1: 10/10 default arms run cleanly with finite non-zero gradients.
s1 = []
for r in [x for x in rows if x["arm"] == "none"]:
    tag = r["method"]
    res = load(tag, "none")
    hist = (res or {}).get("train", {}).get("history", [])
    grads = [h.get("grad_norm") for h in hist]
    ok = (r["exit_code"] == "0" and res is not None
          and res.get("status") == "COMPLETED" and bool(hist)
          and all(g is not None and math.isfinite(g) and g > 0 for g in grads)
          and not console_has_error(tag, "none"))
    s1.append({"tag": tag, "ok": ok, "exit_code": r["exit_code"], "history_len": len(hist)})
rec("S1", all(x["ok"] for x in s1) and len(s1) == 10, {"arms": s1})

# S2: the method-specific gradient channel opens on the repaired CF2-01 arm.
fix = load("CF2-01", "fix")
nom = load("CF2-01", "fix_nomethod")
s2 = {"ok": False, "note": "missing result files"}
if fix and nom and fix["train"]["history"] and nom["train"]["history"]:
    h = fix["train"]["history"][0]
    ml, neu = h.get("method_loss", 0.0), h.get("neutral", 0.0)
    total = 0.1 * ml + 0.1 * neu  # retain_weight=0 on this arm
    ratio = 0.1 * ml / total if total > 0 else 0.0
    g1 = h.get("grad_norm", 0.0)
    g2 = nom["train"]["history"][0].get("grad_norm", 0.0)
    gdiff = abs(g1 - g2) / max(abs(g2), 1e-8)
    s2 = {"ok": ratio >= 0.01 and gdiff > 0.01,
          "method_share": round(ratio, 6), "grad_norm_fix": g1,
          "grad_norm_nomethod": g2, "grad_relative_diff": round(gdiff, 6),
          "threshold": {"share": 0.01, "grad_relative_diff": 0.01}}
rec("S2", s2["ok"], s2)

# S3: CF2-06 numerical stability with distance_mode=median.
f6 = load("CF2-06", "fix")
s3 = {"ok": False, "note": "missing result file"}
if f6 and f6["train"]["history"]:
    hist = f6["train"]["history"]
    s3 = {"ok": all(math.isfinite(h.get("method_loss", float("nan")))
                    and h.get("grad_norm", float("inf")) < 100 for h in hist),
          "steps_logged": len(hist),
          "max_method_loss": max(h.get("method_loss", 0.0) for h in hist),
          "max_grad_norm": max(h.get("grad_norm", 0.0) for h in hist),
          "threshold": {"grad_norm": 100},
          "old_values": {"method_loss": 65796.344, "grad_norm": 35198.92}}
rec("S3", s3["ok"], s3)

# S4: semantic gate intercepts; gate weights no longer collapse; control arms differ.
g9 = load("CF2-09", "gate")
gate_ok = bool(g9) and g9.get("status") == "DIAGNOSTIC_KILL" \
    and g9.get("train", {}).get("steps_completed") == 0
d1 = load("CF2-01", "fix", "diagnostics.json")
weights = [v.get("weight", 0.0) for v in (d1 or {}).get("spectra", {}).values()]
weight_ok = bool(weights) and any(w > 1e-12 for w in weights)

wrong_detail = []
for r in [x for x in rows if x["arm"] == "wrong"]:
    tag = r["method"]
    w, n = load(tag, "wrong"), load(tag, "none")
    ok, cc, mlw, mln = False, None, None, None
    if w and n and w["train"]["history"] and n["train"]["history"]:
        cc = w["train"]["history"][0].get("control_changed")
        mlw = w["train"]["history"][0].get("method_loss")
        mln = n["train"]["history"][0].get("method_loss")
        ok = (cc == 1.0 and mlw is not None and mln is not None
              and abs(mlw - mln) > 1e-12)
    wrong_detail.append({"tag": tag, "ok": ok, "exit_code": r["exit_code"],
                         "control_changed": cc, "method_loss_wrong": mlw,
                         "method_loss_none": mln})

s4_ok = gate_ok and weight_ok and len(wrong_detail) == 10 \
    and all(x["ok"] for x in wrong_detail)
rec("S4", s4_ok, {
    "cf2_09_gate": {"ok": gate_ok, "status": (g9 or {}).get("status"),
                    "steps": (g9 or {}).get("train", {}).get("steps_completed")},
    "cf2_01_weight": {"ok": weight_ok,
                      "min": min(weights) if weights else None,
                      "max": max(weights) if weights else None},
    "wrong_arms": wrong_detail,
})

verdict = {"stage": "fix_smoke_verdict", "claim_level": False,
           "pre_registered": "PROTOCOL.md §6.3", "checks": checks,
           "all_pass": all(c["pass"] for c in checks),
           "note": "smoke checks device discriminability only; NOT forgetting effectiveness"}
out = os.path.join(root, "fix_smoke_verdict.json")
with open(out, "w") as f:
    json.dump(verdict, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps({"all_pass": verdict["all_pass"],
                  "checks": [{"check": c["check"], "pass": c["pass"]} for c in checks]},
                 ensure_ascii=False))
for c in checks:
    print(f"  {c['check']}: {'PASS' if c['pass'] else 'FAIL'}  "
          + json.dumps(c["detail"], ensure_ascii=False, default=str)[:700])
print("wrote", out)

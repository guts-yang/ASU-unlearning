#!/usr/bin/env python3
"""Read-only verifier for the CF2 per-term gradient decomposition diagnostic.

G1 is the only hard gate (engineering completeness, frozen in PROTOCOL §10).
G2/G3 are DESCRIPTIVE observations: no pass/fail threshold, no claim.

Usage: python3 verify_cf2_grad_diag.py <mechanism_gate_1127/results>
"""
from __future__ import annotations

import json
import math
import os
import sys

root = sys.argv[1]
status = json.load(open(os.path.join(root, "grad_diag_status.json")))
rows = status["rows"]

if status.get("gpu0_busy_timeout"):
    print(json.dumps({"stage": "grad_diag", "gpu0_busy_timeout": True,
                      "note": "no arms started; GPU0 busy, no preemption"}))
    raise SystemExit(0)


def path(tag, arm, fname="result.json"):
    return os.path.join(root, "grad_diag", tag, arm, fname)


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


# ---- G1: engineering completeness (the only hard gate) ----
g1_rows = []
for r in rows:
    tag, arm = r["method"], r["arm"]
    _, h = first(tag, arm)
    share = (h or {}).get("grad_part_share") or {}
    l2 = (h or {}).get("grad_part_l2") or {}
    param_l2 = (h or {}).get("grad_param_l2")
    total = sum(float(v) for v in share.values()) if share else 0.0
    ok = (r["exit_code"] == "0" and h is not None
          and set(share) >= {"method", "neutral", "retain"}
          and set(l2) >= {"method", "neutral", "retain"}
          and abs(total - 1.0) < 1e-3
          and param_l2 is not None and math.isfinite(param_l2) and param_l2 > 0)
    g1_rows.append({"tag": tag, "arm": arm, "ok": ok, "exit": r["exit_code"],
                    "share_sum": round(total, 6),
                    "grad_param_l2": round(param_l2, 6) if param_l2 is not None else None})
g1_ok = len(g1_rows) == 20 and all(x["ok"] for x in g1_rows)

# ---- G2/G3: descriptive, no threshold, no verdict ----
table = []
tags = sorted({r["method"] for r in rows})
for tag in tags:
    _, hm = first(tag, "main")
    _, hc = first(tag, "control")
    sm = (hm or {}).get("grad_part_share") or {}
    sc = (hc or {}).get("grad_part_share") or {}
    lm = (hm or {}).get("grad_part_l2") or {}
    lc = (hc or {}).get("grad_part_l2") or {}
    mm, cm = float(sm.get("method", 0.0)), float(sc.get("method", 0.0))
    rel = abs(mm - cm) / max(abs(cm), 1e-8)
    table.append({"tag": tag,
                  "main_share": {k: round(float(v), 6) for k, v in sm.items()},
                  "control_share": {k: round(float(v), 6) for k, v in sc.items()},
                  "main_method_l2": round(float(lm.get("method", 0.0)), 8),
                  "control_method_l2": round(float(lc.get("method", 0.0)), 8),
                  "method_share_delta": round(mm - cm, 6),
                  "method_share_relative_delta": round(rel, 6)})

# Default-off check: re-read the pure-CPU tiny comparison produced before this run.
unit = "/jizhicfs/chenyliao/llm-unlearning/results/cf2-mechanism-gate/weights/grad_diag_unit"
off_ok = False
try:
    a = json.load(open(os.path.join(unit, "off", "result.json")))
    b = json.load(open(os.path.join(unit, "on", "result.json")))
    ha, hb = a["train"]["history"], b["train"]["history"]
    off_ok = (not any("grad_part_share" in h for h in ha)
              and all("grad_part_share" in h for h in hb)
              and [round(h["loss"], 9) for h in ha] == [round(h["loss"], 9) for h in hb]
              and [round(h["grad_norm"], 9) for h in ha] == [round(h["grad_norm"], 9) for h in hb])
except (OSError, KeyError, ValueError):
    pass

verdict = {"stage": "grad_diag_verdict", "claim_level": False,
           "pre_registered": "PROTOCOL.md §10",
           "G1": {"pass": bool(g1_ok), "arms": g1_rows,
                  "default_off_no_fields": bool(off_ok)},
           "G2_G3": {"descriptive_only": True, "no_pass_fail_threshold": True,
                     "table": table,
                     "note": "grad_part_share = per-term L2 / sum of per-term L2; "
                             "an upper-bound contribution measure, NOT an exact "
                             "vector decomposition. No forgetting/validity claim."},
           "all_hard_gates_pass": bool(g1_ok),
           "note": "diagnostic only; §7 S1-S4 and §9 R1-R3 verdicts unchanged; "
                   "no pilot / Stage B / full started"}
out = os.path.join(root, "grad_diag_verdict.json")
with open(out, "w") as f:
    json.dump(verdict, f, indent=2, ensure_ascii=False, default=str)

print(json.dumps({"G1_pass": g1_ok, "arms": len(g1_rows)}, ensure_ascii=False))
print("tag | main method/neutral/retain share | control method/neutral/retain share | method_share_delta | rel_delta")
for t in table:
    m, c = t["main_share"], t["control_share"]
    print(f"  {t['tag']} | {m.get('method')}/{m.get('neutral')}/{m.get('retain')} | "
          f"{c.get('method')}/{c.get('neutral')}/{c.get('retain')} | "
          f"{t['method_share_delta']:+.6f} | {t['method_share_relative_delta']:.4f}")
print("wrote", out)

"""GATE 0 (fixed protocol, run once before the training loop).

Tests the premise of RelResidual: is the identity component shared across
relations, so that subtracting the r0 difference removes it from the raw
delta-A? Also reports the subject-last ablation, but -- per the idea text --
subject-last MUST NOT be used to declare the signal absent (if the relation
appears after the subject, subject-last cannot see it).
"""

from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import prepare as P

COS_THRESHOLD = 0.30      # premise: identity is shared across relations
RATIO_RANGE = (0.10, 0.90)  # B must be neither ~0 nor the whole delta-A


def stat(delta_r, delta_r0):
    cos = F.cosine_similarity(delta_r, delta_r0, dim=-1).mean().item()
    ratio = (delta_r - delta_r0).norm(dim=-1).mean().item() / max(
        delta_r.norm(dim=-1).mean().item(), 1e-9)
    return round(cos, 4), round(ratio, 4)


def main():
    issues = P.preflight()
    if issues:
        print("PREFLIGHT FAILED:", issues, flush=True)
        sys.exit(2)
    tok = AutoTokenizer.from_pretrained(P.MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(P.MODEL, torch_dtype=torch.float32).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    quads = P.build_quadruples(tok, n_pairs=P.PAIRS, seed=P.INIT_SEED)
    print(f"[G0] valid quadruples: {len(quads)} / {P.PAIRS}", flush=True)
    if len(quads) < 8:
        print("[G0] BLOCKED: too few token-equal counterfactual quadruples", flush=True)
        P.dump_json({"verdict": "BLOCKED", "n_quads": len(quads)},
                    P.LOOP / "gate0.json")
        return

    report = {"n_quads": len(quads), "layers": {}, "examples": []}
    for q in quads[:3]:
        report["examples"].append({k: q[k] for k in
                                  ("q_fr", "q_rr", "q_fr0", "q_rr0", "s_f", "s_r")})

    for position in ("query_last", "subject_last"):
        dr_all, dr0_all = {}, {}
        span = quads[0]["span"] if position == "subject_last" else None
        for l in P.TARGET_LAYERS:
            qs = [q["q_fr"] for q in quads] + [q["q_rr"] for q in quads]
            h = P.hidden_last(model, tok, qs, [l], position=position, span=span)[l]
            n = len(quads)
            dr = h[:n] - h[n:2 * n]
            qs0 = [q["q_fr0"] for q in quads] + [q["q_rr0"] for q in quads]
            h0 = P.hidden_last(model, tok, qs0, [l], position=position, span=span)[l]
            dr0 = h0[:n] - h0[n:2 * n]
            dr_all[l], dr0_all[l] = dr, dr0

        per_layer = {}
        for l in P.TARGET_LAYERS:
            cos, ratio = stat(dr_all[l], dr0_all[l])
            per_layer[str(l)] = {"cos_r_r0": cos, "residual_ratio": ratio}
        cos_mean = sum(v["cos_r_r0"] for v in per_layer.values()) / len(per_layer)
        ratio_mean = sum(v["residual_ratio"] for v in per_layer.values()) / len(per_layer)
        report["layers"][position] = {"per_layer": per_layer,
                                      "cos_mean": round(cos_mean, 4),
                                      "ratio_mean": round(ratio_mean, 4)}

    q = report["layers"]["query_last"]
    premise = (q["cos_mean"] >= COS_THRESHOLD
               and RATIO_RANGE[0] < q["ratio_mean"] < RATIO_RANGE[1])
    report["verdict"] = "PASS" if premise else "KILL"
    report["decision_rule"] = {
        "cos_threshold": COS_THRESHOLD, "ratio_range": list(RATIO_RANGE),
        "basis": "query_last ONLY; subject_last is diagnostic and cannot kill the signal",
    }
    report["premise_supported"] = premise
    P.dump_json(report, P.LOOP / "gate0.json")
    print(json.dumps({k: report[k] for k in ("n_quads", "verdict", "premise_supported")}, indent=2))
    print(json.dumps(report["layers"], indent=2))
    print(f"[G0] verdict={report['verdict']}", flush=True)


if __name__ == "__main__":
    main()

"""Forward-only diagnostics for D2-D6, D8, D9 of the 10-idea set.

All results are PROXY measurements on the frozen base model: they test whether the
hypothesis's *measurement* is non-degenerate, not whether a trained method works.
"""
from __future__ import annotations

import math
import statistics

import torch
import torch.nn.functional as F

from engine import encode, forward_hidden, nll_ids, patch_forward_nll, span_ids
from pairs import build_pairs, extract_entity

LAYERS = [3, 7, 11, 15]
model_tok = None


def _span_vec(h, span):
    idx = [p for p in span if p < h.size(0)]
    return h[idx]


def _cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def d3_causal_gate(model, pairs):
    """Per-layer causal effect of patching A+ with A- on the answer-span NLL."""
    effects = {l: [] for l in LAYERS}
    for pr in pairs:
        plus, minus = pr["plus"], pr["minus"]
        ip, sp = encode(model_tok, plus["question"], plus["answer"])
        im, sm = encode(model_tok, minus["question"], plus["answer"])
        base = nll_ids(model, ip, sp)
        hp = forward_hidden(model, ip, LAYERS)
        hm = forward_hidden(model, im, LAYERS)
        for l in LAYERS:
            newh = hp[l].clone()
            n = min(len(sp), len(sm))
            newh[sp[:n]] = hm[l][sm[:n]]
            patched = patch_forward_nll(model, ip, sp, l, newh.unsqueeze(0))
            effects[l].append(patched - base)  # negative = patching A- raises NLL => layer carries fact info
    mean = {l: statistics.mean(v) for l, v in effects.items()}
    mag = {l: abs(v) for l, v in mean.items()}
    vals = sorted(mag.values(), reverse=True)
    top_share = (vals[0] ** 2) / max(1e-9, sum(v * v for v in vals))
    best = int(max(mag, key=mag.get))
    return {
        "per_layer_delta_nll": {str(k): round(v, 5) for k, v in mean.items()},
        "magnitude": {str(k): round(v, 5) for k, v in mag.items()},
        "best_layer": best,
        "top_layer_squared_share": round(top_share, 4),
        "verdict": "deep-layer dominant" if best >= 11 else "shallow-layer dominant",
        "note": "negative delta means patching A- raises NLL, i.e. that layer carries fact information.",
    }


def d4_dual_probe(model, pairs):
    """Compare ΔA alignment with answer-token vs entity-token embeddings."""
    ans_cos, ent_cos = [], []
    for pr in pairs:
        plus, minus = pr["plus"], pr["minus"]
        ip, sp = encode(model_tok, plus["question"], plus["answer"])
        im, sm = encode(model_tok, minus["question"], plus["answer"])
        hp = forward_hidden(model, ip, LAYERS)
        hm = forward_hidden(model, im, LAYERS)
        l = LAYERS[len(LAYERS) // 2]
        n = min(len(sp), len(sm))
        dA = (_span_vec(hp[l], sp[:n]) - _span_vec(hm[l], sm[:n])).mean(0)
        emb = model.get_input_embeddings().weight
        with torch.no_grad():
            a_ids = span_ids(model_tok, plus["answer"])[:n]
            ans_vec = emb[torch.tensor(a_ids, device=dA.device)].mean(0)
            ent_p = extract_entity(plus["answer"]) or ""
            ent_m = extract_entity(minus["answer"]) or ""
            ent_ids = model_tok.encode(" " + ent_p, add_special_tokens=False) + model_tok.encode(
                " " + ent_m, add_special_tokens=False
            )
            ent_vec = emb[torch.tensor(ent_ids, device=dA.device)].mean(0)
        ans_cos.append(_cos(dA, ans_vec))
        ent_cos.append(_cos(dA, ent_vec))
    return {
        "cos_dA_answer_mean": round(statistics.mean(ans_cos), 4),
        "cos_dA_entity_mean": round(statistics.mean(ent_cos), 4),
        "answer_closer_fraction": round(
            sum(a > e for a, e in zip(ans_cos, ent_cos)) / max(1, len(ans_cos)), 4
        ),
        "verdict": "fact-leaning"
        if statistics.mean(ans_cos) > statistics.mean(ent_cos)
        else "identity-leaning",
        "note": "proxy probes: mean-pooled token embeddings, not trained linear probes.",
    }


def d5_trajectory(model, pairs):
    """Position-wise ΔA norm profile over the answer span."""
    top1, gini, profiles = [], [], []
    for pr in pairs:
        plus, minus = pr["plus"], pr["minus"]
        ip, sp = encode(model_tok, plus["question"], plus["answer"])
        im, sm = encode(model_tok, minus["question"], plus["answer"])
        hp = forward_hidden(model, ip, LAYERS)
        hm = forward_hidden(model, im, LAYERS)
        l = LAYERS[len(LAYERS) // 2]
        n = min(len(sp), len(sm))
        norms = (hp[l][sp[:n]] - hm[l][sm[:n]]).norm(dim=-1)
        v = norms / max(1e-9, float(norms.sum()))
        top1.append(float(v.max()))
        xs = sorted(v.tolist())
        s = max(1e-9, sum(xs))
        g = sum((2 * (i + 1) - n - 1) * xs[i] for i in range(n)) / (n * s) if n > 1 else 0.0
        gini.append(g)
        profiles.append([round(float(x), 4) for x in norms.tolist()])
    return {
        "mean_top1_position_share": round(statistics.mean(top1), 4),
        "mean_gini": round(statistics.mean(gini), 4),
        "verdict": "concentrated" if statistics.mean(gini) > 0.5 else "diffuse",
        "example_profile": profiles[0] if profiles else [],
        "note": "gini>0.5 means one answer position dominates ||dA||.",
    }


def d6_multi_donor(model, pairs, retain_pool, K=4):
    """Marginalized ΔA over K donors vs single-donor ΔA."""
    single_vs_marg, marg_vs_ans = [], []
    emb = model.get_input_embeddings().weight
    for pr in pairs:
        plus = pr["plus"]
        ip, sp = encode(model_tok, plus["question"], plus["answer"])
        hp = forward_hidden(model, ip, LAYERS)
        l = LAYERS[len(LAYERS) // 2]
        base_vec = _span_vec(hp[l], sp)
        diffs = []
        for d in retain_pool[:K]:
            if extract_entity(d["answer"]) == extract_entity(plus["answer"]):
                continue
            im, sm = encode(model_tok, d["question"], plus["answer"])
            hm = forward_hidden(model, im, LAYERS)
            n = min(len(sp), len(sm))
            diffs.append((base_vec[:n] - _span_vec(hm[l], sm[:n])).mean(0))
        if len(diffs) < 2:
            continue
        single = diffs[0]
        marg = torch.stack(diffs).mean(0)
        single_vs_marg.append(_cos(single, marg))
        a_ids = span_ids(model_tok, plus["answer"])
        ans_vec = emb[torch.tensor(a_ids, device=marg.device)].mean(0)
        marg_vs_ans.append(_cos(marg, ans_vec))
    return {
        "cos_single_vs_marginalized": round(statistics.mean(single_vs_marg), 4),
        "cos_marginalized_vs_answer": round(statistics.mean(marg_vs_ans), 4),
        "verdict": "marginalization changes direction"
        if statistics.mean(single_vs_marg) < 0.9
        else "marginalization is stable",
        "note": "K donors sampled from retain90 with same-template preference.",
    }


def d8_residualize(model, pairs, retain_pool):
    """Regress out a subword/length-matched null donor direction."""
    before, after = [], []
    emb = model.get_input_embeddings().weight
    for pr in pairs:
        plus = pr["plus"]
        ip, sp = encode(model_tok, plus["question"], plus["answer"])
        hp = forward_hidden(model, ip, LAYERS)
        l = LAYERS[len(LAYERS) // 2]
        base_vec = _span_vec(hp[l], sp)
        # null donor: same entity, different question phrasing (best available proxy)
        null_q = "Please answer briefly. " + plus["question"]
        im_null, sm_null = encode(model_tok, null_q, plus["answer"])
        hm_null = forward_hidden(model, im_null, LAYERS)
        im_minus, sm_minus = encode(model_tok, pr["minus"]["question"], plus["answer"])
        hm_minus = forward_hidden(model, im_minus, LAYERS)
        n = min(len(sp), len(sm_null), len(sm_minus))
        dA = (base_vec[:n] - _span_vec(hm_minus[l], sm_minus[:n])).mean(0)
        dnull = (base_vec[:n] - _span_vec(hm_null[l], sm_null[:n])).mean(0)
        resid = dA - (torch.dot(dA, dnull) / max(1e-9, torch.dot(dnull, dnull))) * dnull
        a_ids = span_ids(model_tok, plus["answer"])
        ans_vec = emb[torch.tensor(a_ids, device=dA.device)].mean(0)
        before.append(_cos(dA, ans_vec))
        after.append(_cos(resid, ans_vec))
    if not before:
        return {"verdict": "inconclusive", "note": "no same-entity null donor found."}
    return {
        "cos_before_residualization": round(statistics.mean(before), 4),
        "cos_after_residualization": round(statistics.mean(after), 4),
        "verdict": "residual keeps fact signal"
        if statistics.mean(after) >= 0.8 * statistics.mean(before)
        else "residual destroys fact signal",
        "note": "null donor uses the same entity (phrasing-only contrast), a weak proxy.",
    }


def d9_readout_space(model, pairs):
    """Readout-space Δ (scalar NLL delta) vs hidden-space ΔA norm."""
    ratios, corr = [], []
    for pr in pairs:
        plus, minus = pr["plus"], pr["minus"]
        ip, sp = encode(model_tok, plus["question"], plus["answer"])
        im, sm = encode(model_tok, minus["question"], plus["answer"])
        n1 = nll_ids(model, ip, sp)
        n2 = nll_ids(model, im, sm)
        hp = forward_hidden(model, ip, LAYERS)
        hm = forward_hidden(model, im, LAYERS)
        l = LAYERS[len(LAYERS) // 2]
        n = min(len(sp), len(sm))
        dnorm = float((hp[l][sp[:n]] - hm[l][sm[:n]]).norm())
        ratios.append(abs(n2 - n1) / max(1e-9, dnorm))
        corr.append((abs(n2 - n1), dnorm))
    xs = [c[0] for c in corr]
    ys = [c[1] for c in corr]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return {
        "mean_readout_delta_per_hidden_norm": round(statistics.mean(ratios), 5),
        "pearson_readout_vs_hidden": round(num / den, 4) if den > 0 else 0.0,
        "verdict": "readout and hidden co-linear"
        if den > 0 and num / den > 0.8
        else "readout adds independent information",
        "note": "scalar NLL delta vs ||dA||; Jacobian direction not estimated.",
    }


def d2_certificate_protocol(model, pairs):
    """Protocol feasibility: does an A- patch measurably restore the fact readout?"""
    deltas = []
    for pr in pairs:
        plus, minus = pr["plus"], pr["minus"]
        ip, sp = encode(model_tok, plus["question"], plus["answer"])
        im, sm = encode(model_tok, minus["question"], plus["answer"])
        nll_base = nll_ids(model, ip, sp)
        hp = forward_hidden(model, ip, LAYERS)
        hm = forward_hidden(model, im, LAYERS)
        l = LAYERS[len(LAYERS) // 2]
        newh = hp[l].clone()
        n = min(len(sp), len(sm))
        newh[sp[:n]] = hm[l][sm[:n]]
        nll_patch = patch_forward_nll(model, ip, sp, l, newh.unsqueeze(0))
        deltas.append(nll_patch - nll_base)
    mean = statistics.mean(deltas) if deltas else 0.0
    mag = statistics.mean([abs(d) for d in deltas]) if deltas else 0.0
    return {
        "mean_signed_nll_change_under_Aminus_patch": round(mean, 5),
        "mean_abs_nll_change": round(mag, 5),
        "sign_negative_fraction": round(sum(d < 0 for d in deltas) / max(1, len(deltas)), 4),
        "verdict": "deferred: needs trained student checkpoint",
        "note": "base-model patch only measures dA causal detectability; a certificate must patch a trained student back to A+.",
    }


def run_all(model, tok, pairs, retain_pool):
    global model_tok
    model_tok = tok
    return {
        "D2": d2_certificate_protocol(model, pairs),
        "D3": d3_causal_gate(model, pairs),
        "D4": d4_dual_probe(model, pairs),
        "D5": d5_trajectory(model, pairs),
        "D6": d6_multi_donor(model, pairs, retain_pool),
        "D8": d8_residualize(model, pairs, retain_pool),
        "D9": d9_readout_space(model, pairs),
    }

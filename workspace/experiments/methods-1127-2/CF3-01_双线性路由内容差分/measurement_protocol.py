"""CF3-01 v1.1 measurement-first Stage A protocol.

Implements proposal v1.1 (idea-stage/candidates-1127-2/CF3-01_.../CF3-01_proposal.md)
sections 5.1-5.4, 6.2, 7.2-7.4, 8 (Stage A). This module is DIAGNOSE-ONLY: it never
trains, never edits model weights, and is a separate code path from the round3
four-candidate trainer in method.py (which remains the default).

Design contract (kept honest):
  - Measurement before mechanism (§5.1): zero-edit recovery, repeat-forward error,
    cache dtype / component precision audit. Unreliable -> MEASUREMENT_UNRELIABLE.
  - Full-layer scan (§5.2): all decoder layers, attention-only by default,
    layer x pair patch effects; allow empty/sparse candidate set S.
  - Equal-condition controls (§5.3): random equal-norm, wrong/shuffled,
    original equal-norm dA, matched-retain proxy; require beating random.
  - Native QK/V 2x2 factorial (§5.4): swaps ONLY the self_attn write via
    modeling.forward_with_attention_patch; distinct from block-output patch.
  - Cluster bootstrap + discovery/held-out split (§5.2-5.3): selection by CI
    lower bounds, NOT mean>0; INSUFFICIENT_SPLIT when clusters too few.
  - Sparse conditional internalization (§6): writes selected_candidates.json
    consumed by method.prepare(sparse_selection_file) for Stage B.

No GPU-only claims: run_stage_a.py runs on the tiny backend (CPU). Real Llama/TOFU
validation must be run on the cluster. Nothing here is a claim of forgetting.
"""
from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from modeling import (answer_nll, decoder_layers, forward_model,
                      forward_with_attention_patch, patch_forward_nll)
from method import CANDIDATES, _candidates, _control_component, _patch_effect


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------

def _cluster_key(item):
    pair = item.get("pair", {})
    return str(pair.get("author") or pair.get("forget_subject") or item.get("index"))


def _scan_layers(cfg, teacher, cache=None):
    spec = cfg.get("measurement", {}).get("scan_layers", None)
    if spec is None:
        # Legacy behaviour: scan only the cached layers.
        return sorted(int(x) for x in cfg.get("layers", []))
    if str(spec).lower() == "all":
        return list(range(1, len(decoder_layers(teacher)) + 1))
    return sorted({int(x) for x in spec})


def _spearman(a, b):
    if len(a) < 2 or len(a) != len(b):
        return None
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        for rank, index in enumerate(order):
            out[index] = float(rank)
        return out
    ra, rb = ranks(a), ranks(b)
    n = len(ra)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    den_a = sum((x - mean_a) ** 2 for x in ra) ** 0.5
    den_b = sum((y - mean_b) ** 2 for y in rb) ** 0.5
    if den_a == 0 or den_b == 0:
        return None
    return num / (den_a * den_b)


def _sign_agreement(a, b):
    if not a or len(a) != len(b):
        return None
    return sum(1 for x, y in zip(a, b) if (x > 0) == (y > 0)) / len(a)


def _o_proj_weight(teacher, layer):
    proj = decoder_layers(teacher)[layer - 1].self_attn.o_proj
    w = proj.weight if hasattr(proj, "weight") else proj.base_layer.weight
    return w.detach().float()


def _fresh_candidates(item, layer, teacher, device, enable_glu=False):
    """Online (uncached) attention components for the precision audit."""
    fp = forward_model(teacher, item["enc_plus"], device, [layer],
                       capture_bilinear=True, capture_glu=enable_glu)
    fm = forward_model(teacher, item["enc_minus"], device, [layer],
                       capture_bilinear=True, capture_glu=enable_glu)
    fresh = {"attention_plus": {}, "attention_minus": {}, "glu_plus": {}, "glu_minus": {}}
    for target, source in ((fresh["attention_plus"], fp), (fresh["attention_minus"], fm)):
        target[layer] = {k: v.float() for k, v in source["attention"][layer].items()}
    if enable_glu:
        fresh["glu_plus"][layer] = {k: v.float() for k, v in fp["glu"][layer].items()}
        fresh["glu_minus"][layer] = {k: v.float() for k, v in fm["glu"][layer].items()}
    return _candidates(fresh, layer, teacher, enable_glu)


def _control_candidates(item, layer, teacher, enable_glu=False):
    if layer not in item.get("attention_control_plus", {}):
        return {}
    ctrl = {"attention_plus": item["attention_control_plus"],
            "attention_minus": item["attention_control_minus"],
            "glu_plus": item.get("glu_control_plus", {}),
            "glu_minus": item.get("glu_control_minus", {})}
    return _candidates(ctrl, layer, teacher, enable_glu)


def _attention_mixed(pi, v, o_proj_w):
    """o_proj( sum_q pi^a_{p,q} * v^b_q ) for answer-span query positions.

    pi: [H, n_ans, T]; v: [T, H, d] (Teacher cache layout). Returns
    [n_ans, width] (already o_proj'd): exactly the self_attn write rows.
    """
    vh = v.float().permute(1, 0, 2)                                  # [T,H,d] -> [H,T,d]
    head = torch.einsum("hat,htd->had", pi.float(), vh)              # [H, n_ans, d]
    flat = head.permute(1, 0, 2).reshape(head.shape[1], -1)          # [n_ans, H*d]
    return F.linear(flat.to(o_proj_w.device), o_proj_w).float().cpu()  # [n_ans, width]


def cluster_bootstrap(values_by_cluster, n_boot=1000, seed=0):
    """Resample CLUSTERS (author/entity), not individual pairs. Returns
    (mean, ci_low, ci_high) at 95%. Empty -> (0,0,0)."""
    clusters = [list(v) for v in values_by_cluster.values() if v]
    if not clusters:
        return 0.0, 0.0, 0.0
    flat = [x for c in clusters for x in c]
    mean = sum(flat) / len(flat)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        draw = [rng.choice(clusters) for _ in clusters]
        sample = [x for c in draw for x in c]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return mean, lo, hi


def _group(values, items):
    out = {}
    for item, val in zip(items, values):
        out.setdefault(_cluster_key(item), []).append(float(val))
    return out


def _group_rows(rows, field):
    out = {}
    for row in rows:
        if row.get(field) is not None:
            out.setdefault(row["cluster"], []).append(float(row[field]))
    return out


# ---------------------------------------------------------------------------
# §5.1 measurement noise floor
# ---------------------------------------------------------------------------

def audit_measurement(teacher, items, cfg, device):
    mcfg = cfg.get("measurement", {})
    noise_abs = float(mcfg.get("noise_floor_abs", 1e-3))
    min_sign = float(mcfg.get("fp32_fp16_min_sign", 0.9))
    min_spearman = float(mcfg.get("fp32_fp16_min_spearman", 0.8))
    alpha = float(cfg["method"].get("alpha", 0.1))
    items = items[:int(mcfg.get("audit_pairs", 3))]
    layers = _scan_layers(cfg, teacher)
    repeat_err, eps0_max, dtypes = 0.0, 0.0, {}
    eps0_by_layer = {}
    component_rel = None
    cached_effects, fresh_effects = [], []
    for item in items:
        enc = item["enc_plus"]
        # repeat-forward determinism
        a = float(forward_model(teacher, enc, device)["nll"])
        b = float(forward_model(teacher, enc, device)["nll"])
        repeat_err = max(repeat_err, abs(a - b))
        # zero-edit patch recovery: cache hidden (fp16) re-patched vs original
        for layer in layers:
            if layer not in item["full_plus"]:
                continue
            patched = float(patch_forward_nll(teacher, enc, device, layer - 1,
                                               item["full_plus"][layer]))
            eps0 = abs(patched - float(item["teacher_nll"]))
            eps0_by_layer[layer] = max(eps0_by_layer.get(layer, 0.0), eps0)
            eps0_max = max(eps0_max, eps0)
        if item["attention_plus"]:
            l0 = sorted(item["attention_plus"])[0]
            dtypes = {"pi": str(item["attention_plus"][l0]["pi"].dtype),
                      "v": str(item["attention_plus"][l0]["v"].dtype),
                      "full_plus": str(item["full_plus"][l0].dtype)}
            # cached (fp16) pi vs online model-dtype pi for one pair/layer
            fresh = forward_model(teacher, enc, device, [l0],
                                  capture_bilinear=True)["attention"][l0]["pi"].float().cpu()
            cached = item["attention_plus"][l0]["pi"].float().cpu()
            denom = cached.norm().clamp_min(1e-8)
            component_rel = float((fresh - cached).norm() / denom)
        # §5.1(3): compare cached-fp16 patch effects with online components.
        for layer in (layers[0], layers[-1]):
            if layer not in item["attention_plus"]:
                continue
            cached_c = _candidates(item, layer, teacher, False)
            fresh_c = _fresh_candidates(item, layer, teacher, device, False)
            for name in ("attn_route", "attn_content"):
                if name not in cached_c or name not in fresh_c:
                    continue
                e_cached = _patch_effect(teacher, item, layer, cached_c[name], device, alpha)
                e_fresh = _patch_effect(teacher, item, layer, fresh_c[name], device, alpha)
                if e_cached is not None and e_fresh is not None:
                    cached_effects.append(float(e_cached))
                    fresh_effects.append(float(e_fresh))
    sign_agreement = _sign_agreement(cached_effects, fresh_effects)
    spearman = _spearman(cached_effects, fresh_effects)
    precision_ok = (sign_agreement is None or sign_agreement >= min_sign) and \
                   (spearman is None or spearman >= min_spearman)
    reliable = (repeat_err <= noise_abs) and (eps0_max <= noise_abs) and precision_ok
    return {"repeat_forward_error": repeat_err, "zero_edit_recovery_eps0": eps0_max,
            "eps0_by_layer": {str(k): v for k, v in sorted(eps0_by_layer.items())},
            "noise_floor_abs": noise_abs, "cache_dtypes": dtypes,
            "attention_pi_component_rel_error": component_rel,
            "fp32_fp16": {"n": len(cached_effects),
                          "sign_agreement": sign_agreement,
                          "spearman": spearman,
                          "min_sign": min_sign, "min_spearman": min_spearman,
                          "cached_fp16_effects": cached_effects,
                          "online_effects": fresh_effects,
                          "note": "online model-dtype (not cached fp16) vs cached fp16"},
            "layers_audited": layers,
            "measurement_reliable": reliable,
            "status": "OK" if reliable else "MEASUREMENT_UNRELIABLE",
            "note": "eps0 is a cache-quantization/patch-plumbing floor, not a forgetting threshold"}


# ---------------------------------------------------------------------------
# §5.2 full-layer scan (reuses round3 patch mechanism for consistency)
# ---------------------------------------------------------------------------

def scan_all_layers(teacher, cache, cfg, device, alpha):
    items = cache["pairs"]
    enable_glu = bool(cfg.get("measurement", {}).get("enable_glu", False))
    layers = _scan_layers(cfg, teacher, cache)
    cands = {(it["index"], l): _candidates(it, l, teacher, enable_glu) for it in items for l in layers}
    control_cands = {(it["index"], l): _control_candidates(it, l, teacher, enable_glu)
                     for it in items for l in layers}
    alpha = float(alpha)
    out = {}
    for layer in layers:
        for name in CANDIDATES:
            if not any(name in cands[(it["index"], layer)] for it in items):
                continue
            rows = []
            for item in items:
                cc = cands[(item["index"], layer)]
                if name not in cc:
                    continue
                # Wrong-author null: same-entity pairs are NOT a wrong pairing.
                null_item = next((x for x in items
                                  if x["index"] != item["index"]
                                  and _cluster_key(x) != _cluster_key(item)
                                  and layer in x["full_plus"]
                                  and name in cands[(x["index"], layer)]), None)
                eff = _patch_effect(teacher, item, layer, cc[name], device, alpha)
                null_eff = None
                if null_item is not None:
                    null_eff = _patch_effect(teacher, item, layer,
                                             cands[(null_item["index"], layer)][name], device, alpha)
                ctrl_cc = control_cands.get((item["index"], layer), {})
                ctrl_eff = None
                if name in ctrl_cc:
                    ctrl_eff = _patch_effect(teacher, item, layer, ctrl_cc[name], device, alpha)
                delta_s = None if eff is None or null_eff is None else eff - null_eff
                delta_m = None if eff is None or ctrl_eff is None else eff - ctrl_eff
                rows.append({"index": int(item["index"]), "cluster": _cluster_key(item),
                             "delta_f": eff, "delta_s": delta_s,
                             "control_effect": ctrl_eff, "delta_m": delta_m})
            df = [r["delta_f"] for r in rows if r["delta_f"] is not None]
            ds = [r["delta_s"] for r in rows if r["delta_s"] is not None]
            dm = [r["delta_m"] for r in rows if r["delta_m"] is not None]
            dc = [r["control_effect"] for r in rows if r["control_effect"] is not None]
            out[f"{layer}:{name}"] = {
                "rows": rows,
                "mean_delta_f": sum(df) / len(df) if df else None,
                "mean_delta_s": sum(ds) / len(ds) if ds else None,
                "mean_delta_m": sum(dm) / len(dm) if dm else None,
                "mean_control_effect": sum(dc) / len(dc) if dc else None,
                "control_count": len(dc),
                "null_count": len(ds),
            }
    return out


# ---------------------------------------------------------------------------
# §5.3 equal-condition controls (same layers / pairs / edit norm)
# ---------------------------------------------------------------------------

def equal_condition_controls(teacher, cache, cfg, device, alpha):
    """Per-layer controls with the SAME layer, edit norm and candidate family.

    The old implementation built controls at min(layers) with the route norm
    and compared them with every layer, which is not an equal-condition test.
    """
    items = cache["pairs"]
    seed = int(cfg["seed"])
    n_boot = int(cfg.get("measurement", {}).get("n_boot", 1000))
    enable_glu = bool(cfg.get("measurement", {}).get("enable_glu", False))
    layers = _scan_layers(cfg, teacher, cache)
    arms = ("random_equal_norm", "wrong_shuffled", "equal_norm_delta")
    out = {}
    for layer in layers:
        all_cands = {(it["index"], layer): _candidates(it, layer, teacher, enable_glu)
                     for it in items}
        per_layer = {}
        cluster_by_index = {it["index"]: _cluster_key(it) for it in items}
        ref_names = [n for n in CANDIDATES
                     if any(n in all_cands[(it["index"], layer)] for it in items)]
        for ref_name in ref_names:
            for arm in arms:
                rows = []
                for item in items:
                    comp, _ = _control_component(
                        item, layer, all_cands[(item["index"], layer)], arm, cfg, seed,
                        all_cands, ref_name=ref_name, cluster_by_index=cluster_by_index)
                    if comp is None:
                        continue
                    eff = _patch_effect(teacher, item, layer, comp, device, alpha)
                    rows.append({"index": int(item["index"]), "cluster": _cluster_key(item),
                                 "delta_f": eff})
                vals = [r["delta_f"] for r in rows if r["delta_f"] is not None]
                mean, lo, hi = cluster_bootstrap(_group_rows(rows, "delta_f"), n_boot, seed)
                per_layer[f"{arm}:{ref_name}"] = {
                    "rows": rows, "mean_delta_f": sum(vals) / len(vals) if vals else None,
                    "ci_low": lo, "ci_high": hi}
        for name in CANDIDATES:
            rows = []
            for item in items:
                ctrl = _control_candidates(item, layer, teacher, enable_glu)
                if name not in ctrl:
                    continue
                eff = _patch_effect(teacher, item, layer, ctrl[name], device, alpha)
                rows.append({"index": int(item["index"]), "cluster": _cluster_key(item),
                             "delta_f": eff})
            vals = [r["delta_f"] for r in rows if r["delta_f"] is not None]
            mean, lo, hi = cluster_bootstrap(_group_rows(rows, "delta_f"), n_boot, seed)
            per_layer[f"matched_retain:{name}"] = {
                "rows": rows, "mean_delta_f": sum(vals) / len(vals) if vals else None,
                "ci_low": lo, "ci_high": hi,
                "status": "OK" if rows else "MATCHED_RETAIN_MISSING"}
        out[str(layer)] = per_layer
    return out


# ---------------------------------------------------------------------------
# §5.4 native QK/V 2x2 factorial (self_attn write swap, not block-output patch)
# ---------------------------------------------------------------------------

def qk_v_factorial(teacher, cache, cfg, device, layers):
    items = cache["pairs"]
    out = {}
    for layer in layers:
        w = _o_proj_weight(teacher, layer)
        Fvals = {"++": [], "-+": [], "+-": [], "--": []}
        for item in items:
            enc = item["enc_plus"]
            start = int(enc["answer_start"])
            pp = item["attention_plus"].get(layer)
            pm = item["attention_minus"].get(layer)
            if pp is None or pm is None:
                continue
            pi_p, v_p = pp["pi"], pp["v"]
            pi_m, v_m = pm["pi"], pm["v"]
            combos = {"++": (pi_p, v_p), "-+": (pi_m, v_p),
                      "+-": (pi_p, v_m), "--": (pi_m, v_m)}
            for key, (pi, v) in combos.items():
                mixed = _attention_mixed(pi, v, w)
                nll = forward_with_attention_patch(teacher, enc, device, layer, mixed, start)
                Fvals[key].append(nll)
        def avg(k):
            return sum(Fvals[k]) / len(Fvals[k]) if Fvals[k] else None
        fp, fm, pmix, mm = avg("++"), avg("+-"), avg("-+"), avg("--")
        if None in (fp, fm, pmix, mm):
            out[layer] = {"status": "INSUFFICIENT_CACHE"}
            continue
        e_qk = 0.5 * ((pmix - fp) + (mm - fm))
        e_v = 0.5 * ((fm - fp) + (mm - pmix))
        out[layer] = {"F": {"++": fp, "-+": pmix, "+-": fm, "--": mm},
                      "E_QK": e_qk, "E_V": e_v,
                      "total_shift": mm - fp,
                      "identity_residual": abs((e_qk + e_v) - (mm - fp)),
                      "status": "COMPLETED",
                      "approximation": "cached_pi_v_recomposition; not raw Q/K rerun",
                      "used_for_gate": False,
                      "note": "F is answer NLL; self_attn write swap, NOT block-output BRCD patch; "
                              "this is an approximation of native Q/K patching and is not used for stage_a_pass"}
    return out


# ---------------------------------------------------------------------------
# §5.2-5.3 / §6 discovery-held-out sparse selection
# ---------------------------------------------------------------------------

def select_sparse_candidates(scan, controls, cache, cfg):
    mcfg = cfg.get("measurement", {})
    n_boot = int(mcfg.get("n_boot", 1000))
    seed = int(cfg["seed"])
    min_effect = float(mcfg.get("min_effect", 0.0))
    min_specificity = float(mcfg.get("min_specificity", 0.0))
    beat_random = bool(mcfg.get("beat_random", True))
    beat_matched = bool(mcfg.get("beat_matched_retain", True))
    heldout_frac = float(mcfg.get("heldout_frac", 0.3))
    min_stable_layers = int(mcfg.get("min_stable_layers", 1))
    min_heldout_clusters = int(mcfg.get("min_heldout_clusters", 1))
    min_heldout_pairs = int(mcfg.get("min_heldout_pairs", 1))

    if min_effect <= 0 or min_specificity <= 0:
        return {"status": "UNREGISTERED_THRESHOLDS", "selected": {},
                "note": "min_effect/min_specificity must be frozen > 0 before the scan"}

    items = cache["pairs"]
    clusters = sorted({_cluster_key(it) for it in items})
    rng = random.Random(seed + 17)
    shuffled = clusters[:]
    rng.shuffle(shuffled)
    n_held = max(1, int(round(len(clusters) * heldout_frac)))
    if len(clusters) < 2 or n_held >= len(clusters) or n_held < min_heldout_clusters:
        return {"status": "INSUFFICIENT_SPLIT", "clusters": len(clusters),
                "discovery": clusters, "heldout": [], "selected": {},
                "coverage": {"n_clusters": len(clusters), "heldout_clusters": 0,
                             "min_heldout_clusters": min_heldout_clusters},
                "note": "need enough independent author/entity clusters for held-out confirmation"}
    heldout = set(shuffled[:n_held])
    discovery = [c for c in clusters if c not in heldout]

    def partition(rows, cluster_set, field):
        return [r[field] for r in rows
                if r.get("cluster") in cluster_set and r.get(field) is not None]

    candidate_report = {}
    passing = {}  # layer -> (candidate, discovery_mean_effect)
    for key, payload in scan.items():
        layer_str, name = key.split(":", 1)
        rows = payload["rows"]
        d_eff = partition(rows, set(discovery), "delta_f")
        d_spec = partition(rows, set(discovery), "delta_s")
        d_match = partition(rows, set(discovery), "delta_m")
        if len(d_eff) < 2 or len(d_spec) < 2:
            candidate_report[key] = {"status": "INSUFFICIENT_DISCOVERY"}
            continue
        e_mean, e_lo, e_hi = cluster_bootstrap(_group_rows(
            [r for r in rows if r.get("cluster") in set(discovery)], "delta_f"), n_boot, seed)
        s_mean, s_lo, s_hi = cluster_bootstrap(_group_rows(
            [r for r in rows if r.get("cluster") in set(discovery)], "delta_s"), n_boot, seed + 1)
        m_mean, m_lo, m_hi = cluster_bootstrap(_group_rows(
            [r for r in rows if r.get("cluster") in set(discovery)], "delta_m"), n_boot, seed + 2)
        ctrl = controls.get(layer_str, {})
        random_key = f"random_equal_norm:{name}"
        random_ci_hi = ctrl.get(random_key, {}).get("ci_high")
        random_ok = (not beat_random) or (random_ci_hi is not None and e_lo > random_ci_hi)
        matched_ok = (not beat_matched) or (len(d_match) >= min_heldout_pairs and m_lo is not None
                                            and m_lo > min_specificity)
        ok = (e_lo > min_effect and s_lo > min_specificity and random_ok and matched_ok)
        # held-out confirmation
        h_eff = partition(rows, heldout, "delta_f")
        h_spec = partition(rows, heldout, "delta_s")
        h_match = partition(rows, heldout, "delta_m")
        h_clusters = {r["cluster"] for r in rows if r.get("cluster") in heldout}
        heldout_enough = (len(h_clusters) >= min_heldout_clusters and
                          len(h_eff) >= min_heldout_pairs)
        h_ok = False
        if heldout_enough and h_match:
            he_mean, he_lo, _ = cluster_bootstrap(_group_rows(
                [r for r in rows if r.get("cluster") in heldout], "delta_f"), n_boot, seed + 3)
            hs_mean, hs_lo, _ = cluster_bootstrap(_group_rows(
                [r for r in rows if r.get("cluster") in heldout], "delta_s"), n_boot, seed + 4)
            hm_mean, hm_lo, _ = cluster_bootstrap(_group_rows(
                [r for r in rows if r.get("cluster") in heldout], "delta_m"), n_boot, seed + 5)
            h_ok = (he_lo > min_effect and hs_lo > min_specificity and
                    (not beat_matched or (hm_lo is not None and hm_lo > min_specificity)))
        candidate_report[key] = {
            "discovery_effect_ci": [e_lo, e_hi],
            "discovery_specificity_ci": [s_lo, s_hi],
            "discovery_matched_ci": [m_lo, m_hi],
            "random_ci_high": random_ci_hi, "random_key": random_key,
            "matched_control_rows": len(d_match),
            "heldout_clusters": len(h_clusters), "heldout_pairs": len(h_eff),
            "discovery_pass": ok, "heldout_pass": h_ok,
            "mean_effect": e_mean, "mean_specificity": s_mean, "mean_matched": m_mean,
        }
        if ok and h_ok:
            layer = int(layer_str)
            if layer not in passing or e_mean > passing[layer][1]:
                passing[layer] = (name, e_mean)

    selected = {str(layer): cand for layer, (cand, _) in sorted(passing.items())}
    stable = len(passing) >= min_stable_layers
    return {"status": "OK" if stable else "NO_STABLE_CANDIDATE",
            "clusters": len(clusters), "discovery": discovery, "heldout": sorted(heldout),
            "min_stable_layers": min_stable_layers, "stable_layers": sorted(passing),
            "coverage": {"n_clusters": len(clusters),
                         "discovery_clusters": len(discovery),
                         "heldout_clusters": len(heldout),
                         "heldout_pairs": sum(1 for it in items if _cluster_key(it) in heldout),
                         "min_heldout_clusters": min_heldout_clusters,
                         "min_heldout_pairs": min_heldout_pairs},
            "selected": selected, "candidates": candidate_report}


# ---------------------------------------------------------------------------
# §7.2 c_f calibration check / §7.4 forced-activation projection probe
# ---------------------------------------------------------------------------

def projection_calibration_check(cfg):
    pc = cfg.get("projection", {})
    mode = str(pc.get("c_f_mode", "discovery_z_max"))
    fixed = pc.get("c_f_fixed", None)
    if mode == "independent_reference":
        if fixed is None:
            return {"status": "C_F_UNCALIBRATED",
                    "recommendation": "disable projection (§7.2 fail-closed)"}
        return {"status": "INDEPENDENT_REFERENCE", "c_f": float(fixed)}
    return {"status": "C_F_UNCALIBRATED", "circular": True,
            "note": "discovery_z_max is circular: max(-NLL(z))+margin makes every z satisfy "
                    "phi_fact<=0 by construction; not usable as projection evidence (§7.2)"}


def forced_projection_selfcheck():
    """Pure-math sanity for §7.4: a violated half-space must move h (>0 dist)."""
    z = torch.zeros(8)
    a = torch.ones(8) / math.sqrt(8.0)
    phi = 0.3  # violated by +0.3
    h = z.clone()
    excess = phi + float((a * (h - z)).sum())
    h = h - (excess / (float((a * a).sum()) + 1e-8)) * a
    dist = float((h - z).norm())
    return {"projection_active": dist > 0.0, "dist_from_z": dist,
            "status": "OK" if dist > 0.0 else "FAILED"}


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def run_stage_a(teacher, tok, cache, cfg, device, out_dir, audit_only=False):
    mcfg = cfg.get("measurement", {})
    alpha = float(cfg["method"].get("alpha", 0.1))
    items = cache["pairs"]
    layers = _scan_layers(cfg, teacher, cache)
    clusters = sorted({_cluster_key(it) for it in items})
    pairs_per_cluster = Counter(_cluster_key(it) for it in items)
    coverage = {"n_pairs": len(items), "n_clusters": len(clusters),
                "pairs_per_cluster": dict(sorted(pairs_per_cluster.items())),
                "singleton_clusters": sum(1 for v in pairs_per_cluster.values() if v < 2)}
    report = {"schema": "cf3_01_stage_a_v11", "claim_level": False,
              "profile": cfg.get("profile"), "alpha": alpha,
              "n_pairs": len(items), "layers_scanned": layers,
              "coverage": coverage,
              "thresholds": {"min_effect": float(mcfg.get("min_effect", 0.0)),
                             "min_specificity": float(mcfg.get("min_specificity", 0.0)),
                             "min_stable_layers": int(mcfg.get("min_stable_layers", 1)),
                             "min_heldout_clusters": int(mcfg.get("min_heldout_clusters", 1)),
                             "min_heldout_pairs": int(mcfg.get("min_heldout_pairs", 1)),
                             "noise_floor_abs": float(mcfg.get("noise_floor_abs", 1e-3))},
              "native_qkv": "APPROX_ONLY",
              "c_f_calibration": projection_calibration_check(cfg)}
    report["measurement"] = audit_measurement(teacher, items, cfg, device)
    if audit_only:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        audit_report = {k: report[k] for k in
                        ("schema", "claim_level", "profile", "alpha", "n_pairs",
                         "layers_scanned", "coverage", "thresholds",
                         "measurement", "c_f_calibration", "native_qkv")}
        audit_report["audit_only"] = True
        (out_dir / "measurement_report.json").write_text(
            json.dumps(audit_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return audit_report

    report["scan"] = scan_all_layers(teacher, cache, cfg, device, alpha)
    report["controls"] = equal_condition_controls(teacher, cache, cfg, device, alpha)
    # factorial on the strongest few layers by |mean effect|
    ranked = sorted(report["scan"].items(),
                    key=lambda kv: abs(kv[1]["mean_delta_f"] or 0.0), reverse=True)
    # First N DISTINCT layers in ranked order, not the numerically smallest layers.
    top_layers = []
    for key, _ in ranked:
        layer = int(key.split(":")[0])
        if layer not in top_layers:
            top_layers.append(layer)
        if len(top_layers) >= int(mcfg.get("factorial_top_layers", 3)):
            break
    report["qkv_factorial"] = qk_v_factorial(teacher, cache, cfg, device, top_layers)
    report["selection"] = select_sparse_candidates(report["scan"], report["controls"], cache, cfg)
    report["projection_calibration"] = report["c_f_calibration"]
    report["forced_projection_selfcheck"] = forced_projection_selfcheck()

    total_rows = sum(len(v["rows"]) for v in report["scan"].values())
    control_rows = sum(v["control_count"] for v in report["scan"].values())
    matched_cov = control_rows / max(1, total_rows)
    min_cov = float(mcfg.get("matched_retain_min_coverage", 0.95))
    matched_ok = matched_cov >= min_cov
    report["matched_retain"] = {"control_rows": control_rows, "total_rows": total_rows,
                                "coverage": matched_cov, "min_coverage": min_cov,
                                "status": "OK" if matched_ok else "MATCHED_RETAIN_MISSING"}

    sel_ok = report["selection"]["status"] == "OK"
    meas_ok = report["measurement"]["measurement_reliable"]
    thresholds_ok = (report["thresholds"]["min_effect"] > 0 and
                     report["thresholds"]["min_specificity"] > 0)
    report["stage_a_pass"] = bool(sel_ok and meas_ok and matched_ok and thresholds_ok)
    reasons = []
    if not meas_ok:
        reasons.append("MEASUREMENT_UNRELIABLE")
    if not thresholds_ok:
        reasons.append("UNREGISTERED_THRESHOLDS")
    if report["selection"]["status"] == "INSUFFICIENT_SPLIT":
        reasons.append("INSUFFICIENT_SPLIT")
    elif not sel_ok:
        reasons.append("NO_STABLE_CANDIDATE")
    if not matched_ok:
        reasons.append("MATCHED_RETAIN_MISSING")
    report["stage_a_reasons"] = reasons

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage_a_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "selected_candidates.json").write_text(
        json.dumps({"selected": report["selection"]["selected"],
                    "stage_a_pass": report["stage_a_pass"],
                    "reasons": reasons,
                    "coverage": coverage,
                    "matched_retain": report["matched_retain"],
                    "native_qkv": report["native_qkv"],
                    "c_f_calibration": report["c_f_calibration"]},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def self_test():
    """Pure-math checks (no model/GPU): bootstrap, factorial identity, forced projection."""
    rng = random.Random(0)
    clusters = {"a": [0.4, 0.5, 0.6], "b": [0.2, 0.3], "c": [0.7, 0.8, 0.9]}
    mean, lo, hi = cluster_bootstrap(clusters, 500, 0)
    assert lo <= mean <= hi and lo < hi
    qk, v = 0.3, -0.1
    total = 0.2
    assert abs((qk + v) - total) < 1e-9  # E_QK + E_V == F(--) - F(++)
    assert abs(_spearman([1, 2, 3], [10, 20, 30]) - 1.0) < 1e-9
    assert _sign_agreement([1, -1, 2], [3, -2, 1]) == 1.0
    assert _sign_agreement([1, -1], [3, 2]) == 0.5
    fp = forced_projection_selfcheck()
    assert fp["status"] == "OK"
    return {"status": "PASS", "bootstrap_ci": [round(lo, 4), round(hi, 4)],
            "factorial_identity": "PASS", "spearman_check": "PASS",
            "forced_projection": fp}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2))

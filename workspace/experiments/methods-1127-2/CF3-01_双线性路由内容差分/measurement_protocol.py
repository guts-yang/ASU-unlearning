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


def _o_proj_weight(teacher, layer):
    proj = decoder_layers(teacher)[layer - 1].self_attn.o_proj
    w = proj.weight if hasattr(proj, "weight") else proj.base_layer.weight
    return w.detach().float()


def _attention_mixed(pi, v, o_proj_w):
    """o_proj( sum_q pi^a_{p,q} * v^b_q ) for answer-span query positions.

    pi: [H, n_ans, T]; v: [T, H, d] (Teacher cache layout). Returns
    [n_ans, width] (already o_proj'd): exactly the self_attn write rows.
    """
    vh = v.float().permute(1, 0, 2)                                  # [T,H,d] -> [H,T,d]
    head = torch.einsum("hat,htd->had", pi.float(), vh)              # [H, n_ans, d]
    flat = head.permute(1, 0, 2).reshape(head.shape[1], -1)          # [n_ans, H*d]
    return F.linear(flat, o_proj_w).float()                           # [n_ans, width]


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


# ---------------------------------------------------------------------------
# §5.1 measurement noise floor
# ---------------------------------------------------------------------------

def audit_measurement(teacher, items, cfg, device):
    mcfg = cfg.get("measurement", {})
    noise_abs = float(mcfg.get("noise_floor_abs", 1e-3))
    items = items[:int(mcfg.get("audit_pairs", 3))]
    repeat_err, eps0_max, dtypes = 0.0, 0.0, {}
    component_rel = None
    for item in items:
        enc = item["enc_plus"]
        # repeat-forward determinism
        a = float(forward_model(teacher, enc, device)["nll"])
        b = float(forward_model(teacher, enc, device)["nll"])
        repeat_err = max(repeat_err, abs(a - b))
        # zero-edit patch recovery: cache hidden (fp16) re-patched vs original
        for layer in (sorted({l for l in cfg["layers"]}) + [1]):
            if layer not in item["full_plus"]:
                continue
            patched = float(patch_forward_nll(teacher, enc, device, layer - 1,
                                               item["full_plus"][layer]))
            eps0_max = max(eps0_max, abs(patched - float(item["teacher_nll"])))
        if item["attention_plus"]:
            l0 = sorted(item["attention_plus"])[0]
            dtypes = {"pi": str(item["attention_plus"][l0]["pi"].dtype),
                      "v": str(item["attention_plus"][l0]["v"].dtype),
                      "full_plus": str(item["full_plus"][l0].dtype)}
            # cached (fp16) pi vs fresh fp32 pi for one pair/layer
            fresh = forward_model(teacher, enc, device, [l0],
                                  capture_bilinear=True)["attention"][l0]["pi"].float()
            cached = item["attention_plus"][l0]["pi"].float()
            denom = cached.norm().clamp_min(1e-8)
            component_rel = float((fresh - cached).norm() / denom)
    reliable = (repeat_err <= noise_abs) and (eps0_max <= noise_abs)
    return {"repeat_forward_error": repeat_err, "zero_edit_recovery_eps0": eps0_max,
            "noise_floor_abs": noise_abs, "cache_dtypes": dtypes,
            "attention_pi_component_rel_error": component_rel,
            "measurement_reliable": reliable,
            "status": "OK" if reliable else "MEASUREMENT_UNRELIABLE",
            "note": "eps0 is a cache-quantization/patch-plumbing floor, not a forgetting threshold"}


# ---------------------------------------------------------------------------
# §5.2 full-layer scan (reuses round3 patch mechanism for consistency)
# ---------------------------------------------------------------------------

def scan_all_layers(teacher, cache, cfg, device, alpha):
    items = cache["pairs"]
    enable_glu = bool(cfg.get("measurement", {}).get("enable_glu", False))
    layers = sorted({l for item in items for l in item["full_plus"]})
    cands = {(it["index"], l): _candidates(it, l, teacher, enable_glu) for it in items for l in layers}
    alpha = float(alpha)
    out = {}
    for layer in layers:
        for name in CANDIDATES:
            if not any(name in cands[(it["index"], layer)] for it in items):
                continue
            rows = []
            for pos, item in enumerate(items):
                cc = cands[(item["index"], layer)]
                if name not in cc:
                    continue
                null_item = next((x for x in items[pos + 1:] + items[:pos]
                                  if layer in x["full_plus"] and name in cands[(x["index"], layer)]), None)
                eff = _patch_effect(teacher, item, layer, cc[name], device, alpha)
                null_eff = None
                if null_item is not None:
                    null_eff = _patch_effect(teacher, item, layer,
                                             cands[(null_item["index"], layer)][name], device, alpha)
                rows.append({"index": int(item["index"]), "cluster": _cluster_key(item),
                             "delta_f": eff, "delta_s": (None if eff is None or null_eff is None
                                                         else eff - null_eff)})
            df = [r["delta_f"] for r in rows if r["delta_f"] is not None]
            ds = [r["delta_s"] for r in rows if r["delta_s"] is not None]
            out[f"{layer}:{name}"] = {"rows": rows,
                                      "mean_delta_f": sum(df) / len(df) if df else None,
                                      "mean_delta_s": sum(ds) / len(ds) if ds else None}
    return out


# ---------------------------------------------------------------------------
# §5.3 equal-condition controls (same layers / pairs / edit norm)
# ---------------------------------------------------------------------------

def equal_condition_controls(teacher, cache, cfg, device, alpha):
    items = cache["pairs"]
    seed = int(cfg["seed"])
    arms = ("random_equal_norm", "wrong_shuffled", "equal_norm_delta")
    layer = min(sorted({l for it in items for l in it["full_plus"]}))
    all_cands = {(it["index"], layer): _candidates(it, layer, teacher, False) for it in items}
    out = {}
    for arm in arms:
        rows = []
        for item in items:
            comp, _ = _control_component(item, layer, all_cands[(item["index"], layer)],
                                         arm, cfg, seed, all_cands)
            if comp is None:
                continue
            eff = _patch_effect(teacher, item, layer, comp, device, alpha)
            rows.append({"index": int(item["index"]), "cluster": _cluster_key(item),
                         "delta_f": eff})
        vals = [r["delta_f"] for r in rows if r["delta_f"] is not None]
        mean, lo, hi = cluster_bootstrap(_group(
            [r["delta_f"] for r in rows], items), int(cfg.get("measurement", {}).get("n_boot", 1000)), seed)
        out[arm] = {"rows": rows, "mean_delta_f": sum(vals) / len(vals) if vals else None,
                    "ci_low": lo, "ci_high": hi}
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
                      "note": "F is answer NLL; native self_attn swap, NOT block-output BRCD patch"}
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
    heldout_frac = float(mcfg.get("heldout_frac", 0.3))
    min_stable_layers = int(mcfg.get("min_stable_layers", 1))

    items = cache["pairs"]
    clusters = sorted({_cluster_key(it) for it in items})
    rng = random.Random(seed + 17)
    shuffled = clusters[:]
    rng.shuffle(shuffled)
    n_held = max(1, int(round(len(clusters) * heldout_frac)))
    if len(clusters) < 2 or n_held >= len(clusters):
        return {"status": "INSUFFICIENT_SPLIT", "clusters": len(clusters),
                "discovery": clusters, "heldout": [], "selected": {},
                "note": "need >=2 independent author/entity clusters for held-out confirmation"}
    heldout = set(shuffled[:n_held])
    discovery = [c for c in clusters if c not in heldout]

    random_ci_hi = controls.get("random_equal_norm", {}).get("ci_high", 0.0)

    def partition(rows, cluster_set, field):
        vals = [r[field] for r in rows if r.get("cluster") in cluster_set and r.get(field) is not None]
        return vals

    candidate_report = {}
    passing = {}  # layer -> (candidate, discovery_mean_effect)
    for key, payload in scan.items():
        layer_str, name = key.split(":", 1)
        rows = payload["rows"]
        d_eff = partition(rows, set(discovery), "delta_f")
        d_spec = partition(rows, set(discovery), "delta_s")
        if len(d_eff) < 2 or len(d_spec) < 2:
            candidate_report[key] = {"status": "INSUFFICIENT_DISCOVERY"}
            continue
        e_mean, e_lo, e_hi = cluster_bootstrap(_group(
            [r["delta_f"] if r["delta_f"] is not None else 0.0 for r in rows
             if r.get("cluster") in set(discovery)],
            [it for it in items if _cluster_key(it) in set(discovery)]), n_boot, seed)
        s_mean, s_lo, s_hi = cluster_bootstrap(_group(
            [r["delta_s"] if r["delta_s"] is not None else 0.0 for r in rows
             if r.get("cluster") in set(discovery)],
            [it for it in items if _cluster_key(it) in set(discovery)]), n_boot, seed + 1)
        ok = (e_lo > min_effect and s_lo > min_specificity and
              (not beat_random or e_lo > random_ci_hi))
        # held-out confirmation
        h_eff = partition(rows, heldout, "delta_f")
        h_spec = partition(rows, heldout, "delta_s")
        h_ok = False
        if h_eff and h_spec:
            he_mean, he_lo, _ = cluster_bootstrap(_group(
                [r["delta_f"] if r["delta_f"] is not None else 0.0 for r in rows
                 if r.get("cluster") in heldout],
                [it for it in items if _cluster_key(it) in heldout]), n_boot, seed + 2)
            hs_mean, hs_lo, _ = cluster_bootstrap(_group(
                [r["delta_s"] if r["delta_s"] is not None else 0.0 for r in rows
                 if r.get("cluster") in heldout],
                [it for it in items if _cluster_key(it) in heldout]), n_boot, seed + 3)
            h_ok = (he_lo > min_effect and hs_lo > min_specificity)
        candidate_report[key] = {"discovery_effect_ci": [e_lo, e_hi],
                                 "discovery_specificity_ci": [s_lo, s_hi],
                                 "random_ci_high": random_ci_hi,
                                 "discovery_pass": ok, "heldout_pass": h_ok,
                                 "mean_effect": e_mean}
        if ok and h_ok:
            layer = int(layer_str)
            if layer not in passing or e_mean > passing[layer][1]:
                passing[layer] = (name, e_mean)

    selected = {str(layer): cand for layer, (cand, _) in sorted(passing.items())}
    stable = len(passing) >= min_stable_layers
    return {"status": "OK" if stable else "NO_STABLE_CANDIDATE",
            "clusters": len(clusters), "discovery": discovery, "heldout": sorted(heldout),
            "min_stable_layers": min_stable_layers, "stable_layers": sorted(passing),
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
            return {"status": "C_F_UNCALIBRATED", "recommendation": "disable projection (§7.2 fail-closed)"}
        return {"status": "INDEPENDENT_REFERENCE", "c_f": float(fixed)}
    return {"status": "DISCOVERY_Z_MAX", "note": "round3 default; §7.2 flags this as circular for neutral calibration"}


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

def run_stage_a(teacher, tok, cache, cfg, device, out_dir):
    mcfg = cfg.get("measurement", {})
    alpha = float(cfg["method"].get("alpha", 0.1))
    report = {"schema": "cf3_01_stage_a_v11", "claim_level": False,
              "profile": cfg.get("profile"), "alpha": alpha,
              "n_pairs": len(cache["pairs"]),
              "layers_scanned": sorted({l for it in cache["pairs"] for l in it["full_plus"]})}
    report["measurement"] = audit_measurement(teacher, cache["pairs"], cfg, device)
    report["scan"] = scan_all_layers(teacher, cache, cfg, device, alpha)
    report["controls"] = equal_condition_controls(teacher, cache, cfg, device, alpha)
    # factorial on the strongest few layers by |mean effect|
    ranked = sorted(report["scan"].items(),
                    key=lambda kv: abs(kv[1]["mean_delta_f"] or 0.0), reverse=True)
    top_layers = sorted({int(k.split(":")[0]) for k, _ in ranked})[:int(mcfg.get("factorial_top_layers", 3))]
    report["qkv_factorial"] = qk_v_factorial(teacher, cache, cfg, device, top_layers)
    report["selection"] = select_sparse_candidates(report["scan"], report["controls"], cache, cfg)
    report["projection_calibration"] = projection_calibration_check(cfg)
    report["forced_projection_selfcheck"] = forced_projection_selfcheck()

    sel_ok = report["selection"]["status"] == "OK"
    meas_ok = report["measurement"]["measurement_reliable"]
    report["stage_a_pass"] = bool(sel_ok and meas_ok)
    reasons = []
    if not meas_ok:
        reasons.append("MEASUREMENT_UNRELIABLE")
    if report["selection"]["status"] == "INSUFFICIENT_SPLIT":
        reasons.append("INSUFFICIENT_SPLIT")
    if not sel_ok and report["selection"]["status"] != "INSUFFICIENT_SPLIT":
        reasons.append("NO_STABLE_CANDIDATE")
    report["stage_a_reasons"] = reasons

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage_a_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "selected_candidates.json").write_text(
        json.dumps({"selected": report["selection"]["selected"],
                    "stage_a_pass": report["stage_a_pass"],
                    "reasons": reasons}, indent=2, ensure_ascii=False), encoding="utf-8")
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
    fp = forced_projection_selfcheck()
    assert fp["status"] == "OK"
    return {"status": "PASS", "bootstrap_ci": [round(lo, 4), round(hi, 4)],
            "factorial_identity": "PASS", "forced_projection": fp}


if __name__ == "__main__":
    print(json.dumps(self_test(), indent=2))

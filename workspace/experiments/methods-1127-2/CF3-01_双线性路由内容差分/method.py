"""CF3-01 BRCD: four independent route/content/gate/value candidates + local
affine half-space projection with real frozen-downstream re-checking.

Implements the derivation package (pasted-text-1795838967747.txt):

  1. SwiGLU branch uses the EXACT symmetric identity on (SiLU(g), u), mapped
     back through mlp.down_proj. Its gate/value candidates stay INDEPENDENT from
     attention route/content (no summation); family-internal uniqueness and
     cross-family ambiguity are diagnosed separately.
  2. The original K involves frozen nonlinear readouts, so it is NOT convex.
     The computable proxy is a local affine half-space system + spherical trust
     region, followed by REAL frozen-downstream re-checking. POCS gives a feasible
     point, not the exact Pi_K projection; infeasibility fails closed.

Default behaviour remains attention-only unless explicitly enabled by config
(method.enable_glu_family / method.enable_affine_projection), so the default
algorithm is unchanged and the new branches can be ablated independently.
See cf3_01_replication/PROTOCOL.md and DERIVATION_IMPL.md.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling import (answer_nll, decoder_layers, encode_qa, forward_model,
                      patch_forward_nll, patch_forward_output_multi, span_hidden)

METHOD_ID = "CF3-01"
NEEDS_TEACHER_BILINEAR = True
NEEDS_TEACHER_GLU = True  # cache always captured; use is config-gated
NEEDS_STUDENT_MINUS = False
NEEDS_STUDENT_GLU = False
REQUIRES_CONTROL = False

ATTN = ("attn_route", "attn_content")
GLU = ("glu_gate", "glu_value")
CANDIDATES = ATTN + GLU
CONTROL_ARMS = ("brcd", "route_only", "content_only", "equal_norm_delta",
                "random_equal_norm", "wrong_shuffled")

_WEIGHT_CACHE = {}


def bilinear_difference(a_plus, a_minus, b_plus, b_minus):
    """Exact symmetric identity: a+*b+ - a-*b- = route + content."""
    route = 0.5 * (a_plus - a_minus) * (b_plus + b_minus)
    content = 0.5 * (a_plus + a_minus) * (b_plus - b_minus)
    return route, content


def _linear_map(component, module):
    """Map a component back to residual width in fp32 (bf16 F.linear SIGFPEs here)."""
    key = id(module)
    if key not in _WEIGHT_CACHE:
        w = module.weight if hasattr(module, "weight") else module.base_layer.weight
        _WEIGHT_CACHE[key] = w.detach().float()
    w = _WEIGHT_CACHE[key]
    return F.linear(component.to(w.device).float(), w).float().cpu()


def _attention_candidates(item, layer, teacher):
    pp, pm = item["attention_plus"][layer], item["attention_minus"][layer]
    vp = pp["v"].float().permute(1, 0, 2)
    vm = pm["v"].float().permute(1, 0, 2)
    route = 0.5 * torch.einsum("hat,htd->had", pp["pi"].float() - pm["pi"].float(), vp + vm)
    content = 0.5 * torch.einsum("hat,htd->had", pp["pi"].float() + pm["pi"].float(), vp - vm)
    route = route.permute(1, 0, 2).reshape(route.shape[1], -1)
    content = content.permute(1, 0, 2).reshape(content.shape[1], -1)
    proj = decoder_layers(teacher)[layer - 1].self_attn.o_proj
    return {"attn_route": _linear_map(route, proj), "attn_content": _linear_map(content, proj)}


def _glu_candidates(item, layer, teacher):
    """D_gate/D_value from (SiLU(g), u); their sum reconstructs m+ - m- exactly."""
    gp = item["glu_plus"][layer].get("g")
    up = item["glu_plus"][layer].get("u")
    gm = item["glu_minus"][layer].get("g")
    um = item["glu_minus"][layer].get("u")
    if gp is None or up is None or gm is None or um is None:
        raise RuntimeError("GLU family requires glu_plus/glu_minus Teacher cache")
    ap, am = F.silu(gp.float()), F.silu(gm.float())
    bp, bm = up.float(), um.float()
    d_gate, d_value = bilinear_difference(ap, am, bp, bm)
    down = decoder_layers(teacher)[layer - 1].mlp.down_proj
    return {"glu_gate": _linear_map(d_gate, down), "glu_value": _linear_map(d_value, down)}


def _candidates(item, layer, teacher, enable_glu):
    out = _attention_candidates(item, layer, teacher)
    if enable_glu:
        out.update(_glu_candidates(item, layer, teacher))
    return out


def _patch_effect(teacher, item, layer, component, device, alpha):
    full = item["full_plus"][layer].clone()
    start = item["enc_plus"]["answer_start"]
    n = min(len(component), len(full) - start)
    if n <= 0:
        return None
    full[start:start + n] -= alpha * component[:n]
    return float(patch_forward_nll(teacher, item["enc_plus"], device, layer - 1, full)) - float(item["teacher_nll"])


# ---------------------------------------------------------------------------
# Local affine half-space proxy with real frozen-downstream re-checking
# ---------------------------------------------------------------------------

def _encode_donor(tok, pair, max_len):
    """X+ with donor answer yj- for the one-sided donor-support measurement."""
    return encode_qa(tok, pair["x_plus"], pair["answer_minus"] or pair["answer_plus"], max_len)


def calibrate_c_f(teacher, gate_items, layers, selected, all_cands, cfg, device, alpha):
    """Freeze the r_f upper bound c on the DISCOVERY set (gate_pairs' z).

    Derivation §2: `r_f <= c` means "original-fact support is low enough"; it is
    a DIFFERENT constraint from runtime's L_fact NLL upper bound
    (`runtime.py` loss.fact_delta).  Reusing teacher_nll + fact_delta here made
    every z violate phi_fact by construction and forced a projection correction
    of the same magnitude as the BRCD edit itself.  c is instead measured on
    discovery z's and then held fixed for every projection (no per-item re-fit).
    """
    margin = float(cfg.get("projection", {}).get("c_margin", 0.05))
    arm = str(cfg["method"].get("control_arm", "brcd"))
    seed = int(cfg["seed"])
    vals = []
    for item in gate_items:
        for layer in layers:
            name = selected[layer]
            if name is None or name not in all_cands[(item["index"], layer)]:
                continue
            hp, hm = item["h_plus"][layer].float(), item["h_minus"][layer].float()
            cands = all_cands[(item["index"], layer)]
            comp, _ = _control_component(item, layer, cands, arm, cfg, seed, all_cands)
            if comp is None:
                comp = cands[name]
            n = min(hp.shape[0], hm.shape[0], comp.shape[0])
            z = hp[:n] - alpha * comp[:n]
            start = item["enc_plus"]["answer_start"]
            full = item["full_plus"][layer].clone().float().to(device)
            full[start:start + n] = z.to(device)
            with torch.no_grad():
                out = patch_forward_output_multi(teacher, item["enc_plus"], device,
                                                 {layer - 1: full}, layers=None)
                nll = float(answer_nll(out["logits"], item["enc_plus"]["labels"].to(device)))
            vals.append(-nll)  # r_f(z)
    c_f = max(vals) + margin if vals else None
    return {"c_f": c_f, "r_f_z_min": min(vals) if vals else None,
            "r_f_z_max": max(vals) if vals else None, "margin": margin,
            "n_discovery": len(vals),
            "source": "gate_pairs (discovery) z, fixed before projection",
            "note": "c frozen on discovery z; NOT derived from teacher_nll / L_fact.fact_delta"}


def affine_project_to_k(teacher, item, layer, tok, cfg, device, z, component,
                        alpha, c_f):
    """POCS over local affine half-spaces + spherical trust region, with real
    downstream re-checking. Returns (T, report); infeasible -> PROJECTION_INFEASIBLE.

    Only the answer span is the variable; the full sequence is patched at layer-1.
    phi_fact  = -NLL(h; X+, y+) - c_f           (c_f frozen on discovery z)
    phi_donor = -NLL(h; X+, yj-) - d_donor      (donor support does not rise)
    phi_eos+  =  (margin(h) - margin(z)) - e_eos
    phi_eos-  = -(margin(h) - margin(z)) - e_eos

    c_f is REQUIRED and must come from calibrate_c_f(): runtime's L_fact NLL
    upper bound is a different constraint and is deliberately not accepted here.
    """
    if c_f is None:
        raise RuntimeError("projection needs a frozen discovery-calibrated c_f; "
                           "L_fact.fact_delta / teacher_nll is NOT a valid substitute")
    pc = cfg.get("projection", {})
    steps = int(pc.get("steps", 12))
    outer = int(pc.get("outer_steps", 2))
    rho_max = float(pc.get("rho_max", 0.5))
    kappa = float(pc.get("kappa", 1.0))
    d_donor = float(pc.get("d_donor_rel", 0.05))
    e_eos = float(pc.get("eos_deadzone", 1.0))
    tol = float(pc.get("tolerance", 0.02))
    max_len = int(cfg["data"]["max_len"])
    layer_index = layer - 1
    enc = item["enc_plus"]
    enc_donor = _encode_donor(tok, item["pair"], max_len)
    labels_donor = enc_donor["labels"].to(device)
    labels_fact = enc["labels"].to(device)
    full_template = item["full_plus"][layer].clone().float().to(device)
    start = enc["answer_start"]
    n = z.shape[0]
    eos_id = int(cfg["model"].get("eos_token_id", 2))

    # s_dj needs a SEPARATE downstream forward on (X+, yj-): the donor answer
    # has a different token count, so reusing the fact logits is invalid.
    with torch.no_grad():
        donor_hidden = forward_model(teacher, enc_donor, device, [layer])["hidden"][layer].float().to(device)
    n_donor_span = max(1, donor_hidden.shape[0] - start)
    n_use = min(n, n_donor_span)

    def readouts(h, want_grad):
        h = h.detach().clone().to(device).requires_grad_(want_grad)
        full_fact = full_template.clone()
        full_fact[start:start + n] = h
        full_donor = donor_hidden.clone()
        full_donor[start:start + n_use] = h[:n_use]
        ctx = torch.enable_grad() if want_grad else torch.no_grad()
        with ctx:
            out_f = patch_forward_output_multi(teacher, enc, device,
                                               {layer_index: full_fact}, layers=None)
            out_d = patch_forward_output_multi(teacher, enc_donor, device,
                                               {layer_index: full_donor}, layers=None)
            logits = out_f["logits"]
            nll_fact = answer_nll(logits, labels_fact)
            nll_donor = answer_nll(out_d["logits"], labels_donor)
            end = min(start + n, logits.shape[0])
            span_logits = logits[start:max(start, end - 1)].float()
            if span_logits.numel() == 0:
                margin = logits.new_zeros(())
            else:
                without = span_logits.clone()
                without[:, eos_id] = -torch.inf
                margin = (span_logits[:, eos_id] - without.max(dim=-1).values).mean()
        return h, nll_fact, nll_donor, margin

    z_dev = z.to(device).float()
    h0, nll_f0, nll_d0, m0 = readouts(z_dev, True)
    # r_f <= c  <=>  -NLL_fact <= c_f, with c_f FROZEN on the discovery set.
    # No teacher_nll / loss.fact_delta here: that is runtime's L_fact, a
    # DIFFERENT constraint (derivation §2), not the proposal's r_f <= c.
    d_d = -float(nll_d0) + d_donor     # s_dj <= s_dj(z) + eps
    m_ref = m0.detach()
    phis = [
        ("fact", -nll_f0 - c_f),
        ("donor", -nll_d0 - d_d),
        ("eos_pos", m0 - m_ref - e_eos),
        ("eos_neg", -(m0 - m_ref) - e_eos),
    ]
    # Derivation §4: if z already satisfies the real constraints, the projection
    # must be the identity (it is a constraint corrector, not a forgetting source).
    _, nll_fz, nll_dz, mz = readouts(z_dev, False)
    phi_z_real = [
        float(-nll_fz - c_f), float(-nll_dz - d_d),
        float(mz - m_ref - e_eos), float(-(mz - m_ref) - e_eos)
    ]
    if all(v <= tol for v in phi_z_real):
        return z.detach(), {"status": "identity", "projection_active": False,
                            "c_f_frozen": round(float(c_f), 6),
                            "phi_z_real": [round(v, 6) for v in phi_z_real],
                            "dist_from_z": 0.0, "dist_over_edit": 0.0}

    grads, phi_vals = [], []
    zero_jacobian_violated = False
    for name, phi in phis:
        (g,) = torch.autograd.grad(phi, h0, retain_graph=True, allow_unused=True)
        if g is None:
            g = torch.zeros_like(h0)
        g = g.detach()
        # Derivation: a near-zero Jacobian with a violated constraint cannot give
        # a correction direction; fail closed instead of dividing by a tiny norm.
        if float((g * g).sum()) <= 1e-12 and float(phi.detach()) > 0:
            zero_jacobian_violated = True
        grads.append(g)
        phi_vals.append(float(phi.detach()))

    edit_norm = (alpha * component.to(device)).norm().item()
    rho = min(rho_max, max(1e-6, kappa * edit_norm))
    report = {"steps": steps, "outer": outer, "rho": rho, "edit_norm": edit_norm,
              "c_f_frozen": round(float(c_f), 6),
              "phi_z": [round(v, 6) for v in phi_vals],
              "jacobian_norms": [round(float(g.norm()), 6) for g in grads],
              "status": "ok"}
    if zero_jacobian_violated:
        report["status"] = "PROJECTION_INFEASIBLE"
        report["reason"] = "zero_jacobian_violated_constraint"
        return z.detach(), report
    accepted = None
    for attempt in range(outer):
        h = z_dev.clone()
        for _ in range(steps):
            for a_i, phi_i in zip(grads, phi_vals):
                excess = phi_i + float((a_i * (h - z_dev)).sum())
                if excess > 0:
                    denom = float((a_i * a_i).sum()) + 1e-8
                    if denom <= 1e-8:
                        report["status"] = "PROJECTION_INFEASIBLE"
                        report["reason"] = "degenerate_halfspace"
                        return z.detach(), report
                    h = h - (excess / denom) * a_i
                delta = h - z_dev
                dn = delta.norm()
                if dn > rho:
                    h = z_dev + rho * delta / dn
        # REAL frozen downstream re-check
        _, nll_f1, nll_d1, m1 = readouts(h, False)
        real_phis = [
            float(-nll_f1 - c_f), float(-nll_d1 - d_d),
            float(m1 - m_ref - e_eos), float(-(m1 - m_ref) - e_eos)
        ]
        dist = float((h - z_dev).norm())
        report.update({"attempt": attempt,
                       "real_phis": [round(v, 6) for v in real_phis],
                       "dist_from_z": round(dist, 6),
                       "dist_over_edit": round(dist / max(edit_norm, 1e-8), 6)})
        if all(v <= tol for v in real_phis):
            accepted = h.detach().cpu()
            report["status"] = "accepted"
            break
        report["status"] = "PROJECTION_INFEASIBLE"
        rho *= 0.5
        if rho < 1e-4:
            break
    if accepted is None:
        return z.detach(), report
    return accepted, report


def _control_component(item, layer, candidates, arm, cfg, seed, all_candidates,
                       ref_name="attn_route", cluster_by_index=None):
    hp, hm = item["h_plus"][layer].float(), item["h_minus"][layer].float()
    ref = candidates.get(ref_name)
    if ref is None:
        ref = candidates.get("attn_route")
    if ref is None:
        return None, {"arm": arm, "reason": "reference_candidate_missing"}
    if arm in ("route_only", "content_only"):
        name = "attn_route" if arm == "route_only" else "attn_content"
        return candidates[name], {"arm": arm, "forced": name}
    if arm == "equal_norm_delta":
        delta = hp - hm
        scale = ref.norm().item() / delta.norm().clamp_min(1e-8).item()
        return delta * scale, {"arm": arm}
    if arm == "random_equal_norm":
        g = torch.Generator().manual_seed(int(seed) + int(item["index"]) * 131 + layer)
        rnd = torch.randn(ref.shape, generator=g)
        scale = ref.norm().item() / rnd.norm().clamp_min(1e-8).item()
        return rnd * scale, {"arm": arm}
    if arm == "wrong_shuffled":
        # Same candidate family, ideally a different author cluster.
        keys = [k for k in sorted(all_candidates) if k[1] == layer]
        if cluster_by_index is not None:
            here = cluster_by_index.get(item["index"])
            other_keys = [k for k in keys if cluster_by_index.get(k[0]) != here]
            if other_keys:
                keys = other_keys
        if not keys:
            return None, {"arm": arm, "reason": "no_shuffled_candidate"}
        other_c = all_candidates[keys[(int(item["index"]) + layer) % len(keys)]]
        other = other_c.get(ref_name)
        if other is None:
            other = other_c.get("attn_route")
        if other is None:
            return None, {"arm": arm, "reason": "shuffled_candidate_missing"}
        scale = ref.norm().item() / other.norm().clamp_min(1e-8).item()
        return other * scale, {"arm": arm}
    return None, {"arm": "brcd"}


def prepare(cache, teacher, tok, cfg, device, student=None, out_dir=None):
    del student, out_dir
    if not cache["pairs"][0]["full_plus"]:
        raise RuntimeError("CF3-01 requires data.cache_full_sequence=true for patch gates")
    alpha = float(cfg["method"].get("alpha", 0.1))
    seed = int(cfg["seed"])
    arm = str(cfg["method"].get("control_arm", "brcd"))
    if arm not in CONTROL_ARMS:
        raise ValueError(f"unknown control_arm: {arm}")
    enable_glu = bool(cfg["method"].get("enable_glu_family", False))
    enable_proj = bool(cfg["method"].get("enable_affine_projection", False))
    gate_n = min(int(cfg["method"].get("gate_pairs", 4)), len(cache["pairs"]))
    gate_items = cache["pairs"][:gate_n]
    all_cands = {(item["index"], layer): _candidates(item, layer, teacher, enable_glu)
                 for item in cache["pairs"] for layer in cfg["layers"]}

    cfg_layers = list(cfg["layers"])
    sel_file = cfg["method"].get("sparse_selection_file", None)
    if sel_file:
        # v1.1 §6: Stage A has already done measurement audit + full-layer scan +
        # equal-condition controls + cluster-bootstrap held-out selection. Here we
        # only materialize its SPARSE layer set into fixed Student targets.
        import json as _json
        from pathlib import Path as _Path
        sel_payload = _json.loads(_Path(sel_file).read_text(encoding="utf-8"))
        if not sel_payload.get("stage_a_pass", False):
            return {"enabled": False, "selected": {},
                    "reports": {"stage_a_gate": {"status": sel_payload.get("reasons")}},
                    "ambiguous_layers": [], "cross_family_ambiguity": [], "alpha": alpha,
                    "targets": {}, "control_arm": arm,
                    "enable_glu_family": enable_glu, "enable_affine_projection": False,
                    "projection_c_f_calibration": None, "projection_stats": [],
                    "projection_infeasible": [],
                    "kill_reasons": [{"type": "stage_a_gate_failed",
                                      "reasons": sel_payload.get("reasons")}],
                    "gate_failed_layers": [], "stage_a_selected": {}}
        stage_sel = {int(k): v for k, v in sel_payload["selected"].items()}
        stage_layers = sorted(stage_sel)
        targets = {}
        for item in cache["pairs"]:
            for layer in stage_layers:
                cands = all_cands.get((item["index"], layer))
                name = stage_sel[layer]
                if not cands or name not in cands:
                    continue
                hp = item["h_plus"][layer].float()
                comp = cands[name]
                n = min(hp.shape[0], comp.shape[0])
                # §6.2 optional normalized MSE handled in loss(); target is fixed.
                targets[(item["index"], layer)] = (hp[:n] - alpha * comp[:n]).detach()
        if not targets:
            return {"enabled": False, "selected": {str(l): stage_sel[l] for l in stage_layers},
                    "reports": {}, "ambiguous_layers": [], "cross_family_ambiguity": [],
                    "alpha": alpha, "targets": {}, "control_arm": arm,
                    "enable_glu_family": enable_glu, "enable_affine_projection": False,
                    "projection_c_f_calibration": None, "projection_stats": [],
                    "projection_infeasible": [],
                    "kill_reasons": [{"type": "stage_a_no_targets"}],
                    "gate_failed_layers": [], "stage_a_selected": {str(l): stage_sel[l] for l in stage_layers}}
        return {"enabled": True, "selected": {str(l): stage_sel[l] for l in stage_layers},
                "reports": {"stage_a": {"layers": stage_layers, "source": str(sel_file)}},
                "ambiguous_layers": [], "cross_family_ambiguity": [], "alpha": alpha,
                "targets": targets, "control_arm": arm,
                "enable_glu_family": enable_glu, "enable_affine_projection": False,
                "projection_c_f_calibration": None, "projection_stats": [],
                "projection_infeasible": [], "kill_reasons": [],
                "gate_failed_layers": [], "stage_a_selected": {str(l): stage_sel[l] for l in stage_layers}}

    families = ("attn", "glu") if enable_glu else ("attn",)
    reports, pass_map, selected, ambiguity, cross_family = {}, {}, {}, [], []
    if arm == "brcd":
        for layer in cfg["layers"]:
            stats = {k: [] for k in CANDIDATES}
            for pos, item in enumerate(gate_items):
                null_item = next((x for x in gate_items[pos + 1:] + gate_items[:pos]
                                  if x["h_plus"][layer].shape == item["h_plus"][layer].shape), None)
                for name in CANDIDATES:
                    if name not in all_cands[(item["index"], layer)]:
                        continue
                    effect = _patch_effect(teacher, item, layer, all_cands[(item["index"], layer)][name], device, alpha)
                    null_effect = None
                    if null_item is not None:
                        null_effect = _patch_effect(teacher, item, layer,
                                                    all_cands[(null_item["index"], layer)][name], device, alpha)
                    stats[name].append({"pair": item["index"], "nll_increase": effect,
                                        "null_nll_increase": null_effect,
                                        "specificity": None if effect is None or null_effect is None else effect - null_effect})
            for name in CANDIDATES:
                effects = [x["nll_increase"] for x in stats[name] if x["nll_increase"] is not None]
                if not effects:
                    continue
                spec = [x["specificity"] for x in stats[name] if x["specificity"] is not None]
                mean_e = sum(effects) / len(effects)
                mean_s = sum(spec) / max(1, len(spec))
                passed = (mean_e > float(cfg["method"].get("min_nll_increase", 0.0)) and
                          mean_s > float(cfg["method"].get("specificity_margin", 0.0)))
                pass_map[(layer, name)] = passed
                reports[f"{layer}:{name}"] = {"mean_nll_increase": mean_e,
                                              "mean_specificity": mean_s, "passed": passed,
                                              "pairs": stats[name]}
        for layer in cfg["layers"]:
            uniq = {}
            for fam in families:
                names = [n for n in CANDIDATES if n.startswith(fam) and (layer, n) in pass_map]
                passed = [n for n in names if pass_map[(layer, n)]]
                if len(passed) == 1:
                    uniq[fam] = passed[0]
                elif len(passed) > 1:
                    ambiguity.append({"layer": layer, "family": fam, "candidates": passed})
            if len(uniq) > 1:
                cross_family.append(layer)
                selected[layer] = None
            elif len(uniq) == 1:
                selected[layer] = next(iter(uniq.values()))
            else:
                selected[layer] = None
        strict = bool(cfg["method"].get("strict_gate", True))
        if not strict:
            for layer in cfg["layers"]:
                if selected[layer] is None:
                    selected[layer] = "attn_content"
        enabled = all(selected[layer] is not None for layer in cfg["layers"])
    else:
        forced = "attn_route" if arm != "content_only" else "attn_content"
        selected = {layer: forced for layer in cfg["layers"]}
        enabled = True
        reports = {"control_arm": {"arm": arm, "gate_skipped": True}}

    # c_f is frozen ONCE on the discovery (gate_pairs) z's before any projection
    # and is never re-fitted per item.  It is NOT runtime's L_fact.fact_delta.
    any_selected = any(selected[layer] is not None for layer in cfg["layers"])
    c_calibration = None
    if enable_proj and any_selected:
        c_calibration = calibrate_c_f(teacher, gate_items, cfg["layers"], selected,
                                      all_cands, cfg, device, alpha)
    c_f = None if c_calibration is None else c_calibration["c_f"]

    targets, proj_stats, infeasible = {}, [], []
    for item in cache["pairs"]:
        for layer in cfg["layers"]:
            name = selected[layer]
            if name is None:
                continue
            hp, hm = item["h_plus"][layer].float(), item["h_minus"][layer].float()
            component, info = _control_component(item, layer, all_cands[(item["index"], layer)],
                                                 arm, cfg, seed, all_cands)
            if component is None:
                component = all_cands[(item["index"], layer)][name]
            n = min(hp.shape[0], hm.shape[0], component.shape[0])
            raw = hp[:n] - alpha * component[:n]
            if enable_proj:
                if c_f is None:
                    t, rep = raw.detach(), {"status": "PROJECTION_INFEASIBLE",
                                           "reason": "c_f_calibration_empty",
                                           "dist_over_edit": 0.0}
                else:
                    t, rep = affine_project_to_k(teacher, item, layer, tok, cfg, device,
                                                 raw, component[:n], alpha, c_f)
                proj_stats.append({"layer": layer, "index": int(item["index"]), **rep})
                if rep["status"] not in ("accepted", "identity"):
                    infeasible.append((int(item["index"]), layer))
                targets[(item["index"], layer)] = t
            else:
                targets[(item["index"], layer)] = raw.detach()
    gate_failed_layers = sorted(layer for layer in cfg["layers"] if selected[layer] is None)
    kill_reasons = []
    if gate_failed_layers:
        kill_reasons.append({"type": "gate_no_unique_candidate",
                             "layers": gate_failed_layers})
    if cross_family:
        kill_reasons.append({"type": "cross_family_ambiguity", "layers": cross_family})
    if c_calibration is not None and c_calibration["n_discovery"] == 0:
        kill_reasons.append({"type": "projection_c_f_uncalibrated", "count": 0})
    if infeasible:
        kill_reasons.append({"type": "projection_infeasible", "count": len(infeasible)})
    if (gate_failed_layers or cross_family or infeasible) and cfg["method"].get("strict_gate", True):
        enabled = False
    return {"enabled": enabled, "selected": selected, "candidates": all_cands,
            "reports": reports, "ambiguous_layers": ambiguity,
            "cross_family_ambiguity": cross_family, "alpha": alpha,
            "targets": targets, "control_arm": arm, "enable_glu_family": enable_glu,
            "enable_affine_projection": enable_proj,
            "projection_c_f_calibration": c_calibration,
            "projection_stats": proj_stats, "projection_infeasible": infeasible,
            "kill_reasons": kill_reasons, "gate_failed_layers": gate_failed_layers}


def loss(batch, prepared, cfg):
    terms = []
    normalized = bool(cfg.get("method", {}).get("normalized_mse", False))
    eps = float(cfg.get("method", {}).get("normalized_mse_eps", 1e-8))
    for row in batch:
        item, plus = row["item"], row["plus"]
        for key, target_full in prepared["targets"].items():
            if not (isinstance(key, tuple) and key[0] == item["index"]):
                continue
            layer = key[1]
            target = target_full.to(plus["nll"].device)
            sh = span_hidden(plus, layer)
            n = min(len(sh), target.shape[0])
            diff2 = (sh[:n].float() - target[:n].detach()).pow(2).sum(-1)
            if normalized:
                hp = item["h_plus"][layer].to(plus["nll"].device).float()
                denom = (hp[:n] - target[:n].detach()).pow(2).sum(-1).clamp_min(eps)
                diff2 = diff2 / denom
            terms.append(diff2.mean())
    if not terms:
        raise RuntimeError("no BRCD candidate selected")
    return {"method_loss": torch.stack(terms).mean(),
            "selected_terms": torch.tensor(float(len(terms))),
            "normalized_mse": normalized}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    ps = prepared["projection_stats"]
    return {"operator": "four_candidate_bilinear_difference + local_affine_pocs_proxy",
            "selected": prepared["selected"], "ambiguous_layers": prepared["ambiguous_layers"],
            "cross_family_ambiguity": prepared["cross_family_ambiguity"],
            "gates": prepared["reports"], "enabled": prepared["enabled"],
            "kill_reasons": prepared["kill_reasons"],
            "gate_failed_layers": prepared["gate_failed_layers"],
            "donor_target": False,
            "donor_constraint_only": prepared["enable_affine_projection"],
            "note_on_donor": "donor answer is used ONLY as a one-sided support measurement (donor_constraint_only); never an MSE/KL target or soft label.",
            "repatch_after_projection": "NOT_DONE (see DERIVATION_IMPL.md open items)",
            "control_arm": prepared["control_arm"],
            "enable_glu_family": prepared["enable_glu_family"],
            "enable_affine_projection": prepared["enable_affine_projection"],
            "c_f_calibration": prepared["projection_c_f_calibration"],
            "projection_proxy": {
                "name": "local affine half-spaces + spherical trust region + real downstream re-check",
                "projected_targets": len(ps),
                "accepted": sum(1 for x in ps if x["status"] == "accepted"),
                "identity": sum(1 for x in ps if x["status"] == "identity"),
                "infeasible": sum(1 for x in ps if x["status"] not in ("accepted", "identity")),
                "max_dist_over_edit": max((x.get("dist_over_edit", 0) for x in ps), default=0.0),
                "note": "POCS feasible point, NOT exact Pi_K projection; real frozen-downstream re-check decides acceptance."},
            "projection_stats": ps,
            "scope": "derivation-package implementation; default attention-only when not explicitly enabled"}


def self_test():
    gen = torch.Generator().manual_seed(7)
    ap, am, bp, bm = [torch.randn(3, 5, generator=gen) for _ in range(4)]
    r, c = bilinear_difference(ap, am, bp, bm)
    assert float((r + c - (ap * bp - am * bm)).abs().max()) < 1e-5
    gp, gm = [torch.randn(3, 5, generator=gen) for _ in range(2)]
    up, um = [torch.randn(3, 5, generator=gen) for _ in range(2)]
    d_gate, d_value = bilinear_difference(F.silu(gp), F.silu(gm), up, um)
    exact = F.silu(gp) * up - F.silu(gm) * um
    rel = float((d_gate + d_value - exact).norm() / exact.norm().clamp_min(1e-8))
    assert rel < 1e-5
    return {"method_id": METHOD_ID, "status": "PASS",
            "attention_identity_max_error": float((r + c - (ap * bp - am * bm)).abs().max()),
            "swiglu_reconstruction_rel_frobenius": rel}

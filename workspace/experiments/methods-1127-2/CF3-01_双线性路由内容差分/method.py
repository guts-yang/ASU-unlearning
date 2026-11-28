"""CF3-01 BRCD: exact attention + SwiGLU route/content decomposition and internalization.

Original idea (proposal.md) keeps its hypothesis, decisive experiment and failure
criteria unchanged. This file adds the two branches that the first implementation
omitted (MECHANISM_GUIDE.md §12):

  1. SwiGLU bilinear branch: (Swish(g), u) uses the SAME symmetric identity and
     is mapped back to the residual stream through mlp.down_proj.
  2. A computable Pi_K feasible-set proxy: POCS alternating projection onto a
     fact/deadzone ball and a donor half-space, with thresholds taken from
     config.json's existing soft-constraint values.

The projection is explicitly a *computable proxy*, NOT the mathematical convex
projection of the original proposal (which only defines K, not an algorithm or
thresholds). See cf3_01_replication/PROTOCOL.md for the preregistered definition.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling import decoder_layers, patch_forward_nll, span_hidden

METHOD_ID = "CF3-01"
NEEDS_TEACHER_BILINEAR = True
NEEDS_TEACHER_GLU = True
NEEDS_STUDENT_MINUS = False
NEEDS_STUDENT_GLU = False
REQUIRES_CONTROL = False

CONTROL_ARMS = ("brcd", "route_only", "content_only", "equal_norm_delta",
                "random_equal_norm", "wrong_shuffled")


def bilinear_difference(a_plus, a_minus, b_plus, b_minus):
    """Symmetric identity: a+b- - a-b- -> route + content (exact, not 1st order)."""
    route = 0.5 * (a_plus - a_minus) * (b_plus + b_minus)
    content = 0.5 * (a_plus + a_minus) * (b_plus - b_minus)
    return route, content


def _linear_map(component, module, device):
    """Map a bilinear component back to residual width with a fp32 projection.

    bf16 F.linear with these shapes aborts with SIGFPE on this torch/CUDA build.
    """
    weight = module.weight if hasattr(module, "weight") else module.base_layer.weight
    w = weight.detach().float()
    return F.linear(component.to(w.device).float(), w).float().cpu()


def _attention_components(item, layer, teacher):
    pp, pm = item["attention_plus"][layer], item["attention_minus"][layer]
    vp = pp["v"].float().permute(1, 0, 2)
    vm = pm["v"].float().permute(1, 0, 2)
    route = 0.5 * torch.einsum("hat,htd->had", pp["pi"].float() - pm["pi"].float(), vp + vm)
    content = 0.5 * torch.einsum("hat,htd->had", pp["pi"].float() + pm["pi"].float(), vp - vm)
    route = route.permute(1, 0, 2).reshape(route.shape[1], -1)
    content = content.permute(1, 0, 2).reshape(content.shape[1], -1)
    proj = decoder_layers(teacher)[layer - 1].self_attn.o_proj
    return {"route": _linear_map(route, proj, None), "content": _linear_map(content, proj, None)}


def _glu_components(item, layer, teacher):
    """Swish(g) x u bilinear difference, mapped back through mlp.down_proj."""
    gp = item["glu_plus"][layer].get("g")
    up = item["glu_plus"][layer].get("u")
    gm = item["glu_minus"][layer].get("g")
    um = item["glu_minus"][layer].get("u")
    if gp is None or up is None or gm is None or um is None:
        raise RuntimeError("CF3-01 SwiGLU branch requires glu_plus/glu_minus cache")
    a_plus, a_minus = F.silu(gp.float()), F.silu(gm.float())
    b_plus, b_minus = up.float(), um.float()
    route, content = bilinear_difference(a_plus, a_minus, b_plus, b_minus)
    down = decoder_layers(teacher)[layer - 1].mlp.down_proj
    return {"route": _linear_map(route, down, None), "content": _linear_map(content, down, None)}


def _components(item, layer, teacher, use_glu=True):
    attn = _attention_components(item, layer, teacher)
    if not use_glu:
        return attn
    glu = _glu_components(item, layer, teacher)
    n = min(attn["route"].shape[0], glu["route"].shape[0])
    return {kind: (attn[kind][:n] + glu[kind][:n]) for kind in ("route", "content")}


def _patch_effect(teacher, item, layer, component, device, alpha):
    full = item["full_plus"][layer].clone()
    start = item["enc_plus"]["answer_start"]
    n = min(len(component), len(full) - start)
    if n <= 0:
        return None
    full[start:start + n] -= alpha * component[:n]
    return float(patch_forward_nll(teacher, item["enc_plus"], device, layer - 1, full)) - float(item["teacher_nll"])


def _project_ball(h, anchor, radius):
    """Euclidean ball ||h - anchor||_F <= radius, per-row analytic projection."""
    delta = h - anchor
    norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (radius / norm).clamp(max=1.0)
    return anchor + delta * scale


def _project_halfspace(h, anchor, direction, threshold):
    """Half-space <h - anchor, direction> <= threshold, analytic projection."""
    delta = h - anchor
    excess = (delta * direction).sum(dim=-1, keepdim=True) - threshold
    return h - F.relu(excess) * direction


def project_to_k(raw, anchor, donor_anchor, cfg, kind="brcd"):
    """Computable Pi_K proxy (POCS alternating projection), stop-grad target.

    K_proxy = { h : ||h - A+||_F <= radius } ∩ { h : <h - A+, d> <= deadzone }
    where d = normalize(A- - A+) and the thresholds are derived from config.json's
    existing soft-constraint values (fact_delta / donor_guard_weight / eos_margin).
    This is NOT the exact convex projection of proposal.md §3.3; see PROTOCOL.md.
    """
    method = dict(cfg.get("method", {}))
    proj_cfg = dict(cfg.get("projection", {}))
    steps = int(proj_cfg.get("steps", 20))
    alpha = float(method.get("alpha", 0.1))
    fact_delta = float(cfg.get("loss", {}).get("fact_delta", 0.35))
    donor_weight = float(cfg.get("loss", {}).get("donor_guard_weight", 0.1))
    eos_margin = float(cfg.get("loss", {}).get("eos_margin", 1.0))
    component_norm = (raw - anchor).norm().item()
    radius = max(1e-6, alpha * component_norm * (1.0 + fact_delta))
    direction = F.normalize((donor_anchor - anchor).mean(dim=0, keepdim=True).float(), dim=-1, eps=1e-8)
    deadzone = donor_weight * radius / max(eos_margin, 1e-6)
    n = min(raw.shape[0], anchor.shape[0], donor_anchor.shape[0])
    h, a, d = raw[:n].float(), anchor[:n].float(), donor_anchor[:n].float()
    with torch.no_grad():
        for _ in range(steps):
            h = _project_ball(h, a, radius)
            h = _project_halfspace(h, a, direction, deadzone)
        residual = (h - a).norm(dim=-1).max().item() - radius
    return h.detach(), {"radius": radius, "deadzone": deadzone, "steps": steps,
                        "ball_residual": max(0.0, residual), "component_norm": component_norm,
                        "kind": kind}


def _control_component(item, layer, components, arm, cfg, seed, all_components=None):
    """Build the target component for a decisive-experiment control arm."""
    hp, hm = item["h_plus"][layer].float(), item["h_minus"][layer].float()
    if arm in ("route_only", "content_only"):
        kind = arm.split("_")[0]
        return components[kind], {"arm": arm, "forced_kind": kind}
    if arm == "equal_norm_delta":
        delta = (hp - hm)
        ref = components["route"].norm().item()
        scale = ref / delta.norm().clamp_min(1e-8).item()
        return delta * scale, {"arm": arm, "equal_norm_to": ref}
    if arm == "random_equal_norm":
        g = torch.Generator().manual_seed(int(seed) + int(item["index"]) * 131 + layer)
        rnd = torch.randn(components["route"].shape, generator=g)
        ref = components["route"].norm().item()
        scale = ref / rnd.norm().clamp_min(1e-8).item()
        return rnd * scale, {"arm": arm, "equal_norm_to": ref}
    if arm == "wrong_shuffled":
        keys = sorted(all_components)
        k = (int(item["index"]) + layer) % len(keys)
        other = all_components[keys[k]]["route"]
        ref = components["route"].norm().item()
        scale = ref / other.norm().clamp_min(1e-8).item()
        return other * scale, {"arm": arm, "equal_norm_to": ref}
    return None, {"arm": "brcd"}


def prepare(cache, teacher, tok, cfg, device, student=None, out_dir=None):
    del tok, student, out_dir
    if not cache["pairs"][0]["full_plus"]:
        raise RuntimeError("CF3-01 requires data.cache_full_sequence=true for independent patch gates")
    alpha = float(cfg["method"].get("alpha", 0.1))
    seed = int(cfg["seed"])
    arm = str(cfg["method"].get("control_arm", "brcd"))
    if arm not in CONTROL_ARMS:
        raise ValueError(f"unknown control_arm: {arm}")
    use_glu = bool(cfg["method"].get("use_glu_branch", True))
    use_projection = bool(cfg["method"].get("use_pi_k_projection", True))
    gate_n = min(int(cfg["method"].get("gate_pairs", 4)), len(cache["pairs"]))
    gate_items = cache["pairs"][:gate_n]
    all_components = {(item["index"], layer): _components(item, layer, teacher, use_glu)
                      for item in cache["pairs"] for layer in cfg["layers"]}

    reports, pass_map = {}, {}
    if arm == "brcd":
        for layer in cfg["layers"]:
            stats = {"route": [], "content": []}
            for pos, item in enumerate(gate_items):
                null_item = next((x for x in gate_items[pos + 1:] + gate_items[:pos]
                                  if x["h_plus"][layer].shape == item["h_plus"][layer].shape), None)
                for kind in ("route", "content"):
                    component = all_components[(item["index"], layer)][kind]
                    effect = _patch_effect(teacher, item, layer, component, device, alpha)
                    null_effect = None
                    if null_item is not None:
                        null_component = all_components[(null_item["index"], layer)][kind]
                        null_effect = _patch_effect(teacher, item, layer, null_component, device, alpha)
                    stats[kind].append({"pair": item["index"], "nll_increase": effect,
                                        "null_nll_increase": null_effect,
                                        "specificity": None if effect is None or null_effect is None else effect - null_effect})
            for kind in ("route", "content"):
                effects = [x["nll_increase"] for x in stats[kind] if x["nll_increase"] is not None]
                specificity = [x["specificity"] for x in stats[kind] if x["specificity"] is not None]
                mean_effect = sum(effects) / max(1, len(effects))
                mean_specificity = sum(specificity) / max(1, len(specificity))
                passed = (mean_effect > float(cfg["method"].get("min_nll_increase", 0.0)) and
                          mean_specificity > float(cfg["method"].get("specificity_margin", 0.0)))
                pass_map[(layer, kind)] = passed
                reports[f"{layer}:{kind}"] = {"mean_nll_increase": mean_effect,
                                              "mean_specificity": mean_specificity, "passed": passed,
                                              "pairs": stats[kind]}
        selected, ambiguity = {}, []
        for layer in cfg["layers"]:
            route_ok, content_ok = pass_map[(layer, "route")], pass_map[(layer, "content")]
            if route_ok and content_ok:
                ambiguity.append(layer)
                selected[layer] = None
            elif route_ok:
                selected[layer] = "route"
            elif content_ok:
                selected[layer] = "content"
            else:
                selected[layer] = None
        strict = bool(cfg["method"].get("strict_gate", True))
        if not strict:
            for layer in cfg["layers"]:
                if selected[layer] is None:
                    selected[layer] = str(cfg["method"].get("fallback_component", "content"))
        enabled = all(selected[layer] is not None for layer in cfg["layers"])
    else:
        # Decisive-experiment control arms: skip the true/null gate by construction.
        forced = "route" if arm in ("route_only", "equal_norm_delta", "random_equal_norm", "wrong_shuffled") else "content"
        selected = {layer: forced for layer in cfg["layers"]}
        ambiguity, enabled = [], True
        reports = {"control_arm": {"arm": arm, "gate_skipped": True,
                                   "note": "control arm for proposal.md §5; not part of the original CF3-01 gate"}}

    # Precompute stop-grad training targets T = Pi_K(A+ - alpha * C*).
    targets, projection_stats = {}, []
    for item in cache["pairs"]:
        for layer in cfg["layers"]:
            if selected[layer] is None:
                continue
            hp, hm = item["h_plus"][layer].float(), item["h_minus"][layer].float()
            component, info = _control_component(item, layer, all_components[(item["index"], layer)],
                                                 arm, cfg, seed, all_components)
            if component is None:
                component = all_components[(item["index"], layer)][selected[layer]]
            n = min(hp.shape[0], hm.shape[0], component.shape[0])
            raw = hp[:n] - alpha * component[:n]
            if use_projection:
                target, pinfo = project_to_k(raw, hp[:n], hm[:n], cfg, arm)
                projection_stats.append({"layer": layer, "index": int(item["index"]), **pinfo})
            else:
                target = raw.detach()
            targets[(item["index"], layer)] = target
    return {"enabled": enabled, "selected": selected, "components": all_components,
            "reports": reports, "ambiguous_layers": ambiguity, "alpha": alpha,
            "targets": targets, "control_arm": arm, "use_glu_branch": use_glu,
            "use_pi_k_projection": use_projection, "projection_stats": projection_stats}


def loss(batch, prepared, cfg):
    terms = []
    for row in batch:
        item, plus = row["item"], row["plus"]
        for layer in cfg["layers"]:
            key = (item["index"], layer)
            if key not in prepared["targets"]:
                continue
            target = prepared["targets"][key].to(plus["nll"].device)
            sh = span_hidden(plus, layer)
            n = min(len(sh), target.shape[0])
            terms.append(F.mse_loss(sh[:n].float(), target[:n].detach()))
    if not terms:
        raise RuntimeError("no BRCD component selected")
    return {"method_loss": torch.stack(terms).mean(), "selected_terms": torch.tensor(float(len(terms)))}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    pstats = prepared["projection_stats"]
    return {"operator": "exact_attention_plus_swiglu_route_content_bilinear_difference",
            "selected": prepared["selected"], "ambiguous_layers": prepared["ambiguous_layers"],
            "gates": prepared["reports"], "enabled": prepared["enabled"], "donor_target": False,
            "control_arm": prepared["control_arm"], "use_glu_branch": prepared["use_glu_branch"],
            "use_pi_k_projection": prepared["use_pi_k_projection"],
            "projection_proxy": {
                "name": "POCS ball ∩ donor half-space (computable proxy, NOT exact Pi_K)",
                "projected_targets": len(pstats),
                "mean_radius": sum(x["radius"] for x in pstats) / max(1, len(pstats)),
                "max_ball_residual": max((x["ball_residual"] for x in pstats), default=0.0),
                "note": "Thresholds derived from config loss.fact_delta / donor_guard_weight / eos_margin; see PROTOCOL.md."},
            "scope": "proposal hypothesis and failure criteria unchanged; SwiGLU branch + Pi_K proxy added"}


def self_test():
    generator = torch.Generator().manual_seed(7)
    ap, am, bp, bm = [torch.randn(3, 5, generator=generator) for _ in range(4)]
    route, content = bilinear_difference(ap, am, bp, bm)
    error = (route + content - (ap * bp - am * bm)).abs().max()
    assert float(error) < 1e-5
    # SwiGLU branch uses the same identity on (Swish(g), u)
    gp, gm = [torch.randn(3, 5, generator=generator) for _ in range(2)]
    up, um = [torch.randn(3, 5, generator=generator) for _ in range(2)]
    gr, gc = bilinear_difference(F.silu(gp), F.silu(gm), up, um)
    glu_error = (gr + gc - (F.silu(gp) * up - F.silu(gm) * um)).abs().max()
    assert float(glu_error) < 1e-5
    # POCS proxy: finite, bounded by the configured radius
    cfg = {"method": {"alpha": 0.1}, "loss": {"fact_delta": 0.35, "donor_guard_weight": 0.1, "eos_margin": 1.0}}
    anchor = torch.zeros(4, 8)
    donor = torch.ones(4, 8)
    raw = torch.randn(4, 8, generator=generator) * 0.1
    projected, info = project_to_k(raw, anchor, donor, cfg)
    assert torch.isfinite(projected).all() and info["radius"] > 0
    assert (projected - anchor).norm(dim=-1).max().item() <= info["radius"] + 1e-4
    return {"method_id": METHOD_ID, "status": "PASS",
            "attention_identity_max_error": float(error),
            "swiglu_identity_max_error": float(glu_error),
            "projection_radius": info["radius"]}

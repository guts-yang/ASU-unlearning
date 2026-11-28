"""CF3-01 BRCD: exact attention route/content decomposition and internalization."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from modeling import decoder_layers, patch_forward_nll, span_hidden

METHOD_ID = "CF3-01"
NEEDS_TEACHER_BILINEAR = True
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def bilinear_difference(a_plus, a_minus, b_plus, b_minus):
    route = 0.5 * (a_plus - a_minus) * (b_plus + b_minus)
    content = 0.5 * (a_plus + a_minus) * (b_plus - b_minus)
    return route, content


def _raw_components(item, layer):
    pp, pm = item["attention_plus"][layer], item["attention_minus"][layer]
    vp = pp["v"].float().permute(1, 0, 2)
    vm = pm["v"].float().permute(1, 0, 2)
    route = 0.5 * torch.einsum("hat,htd->had", pp["pi"].float() - pm["pi"].float(), vp + vm)
    content = 0.5 * torch.einsum("hat,htd->had", pp["pi"].float() + pm["pi"].float(), vp - vm)
    return (route.permute(1, 0, 2).reshape(route.shape[1], -1),
            content.permute(1, 0, 2).reshape(content.shape[1], -1))


def _components(item, layer, teacher):
    route, content = _raw_components(item, layer)
    proj = decoder_layers(teacher)[layer - 1].self_attn.o_proj
    weight = proj.weight if hasattr(proj, "weight") else proj.base_layer.weight
    # bf16 F.linear with these shapes aborts with SIGFPE on this torch/CUDA build;
    # compute the identical projection in fp32 (same linear map, dtype only).
    w = weight.detach().float()
    route = F.linear(route.to(w.device).float(), w).float().cpu()
    content = F.linear(content.to(w.device).float(), w).float().cpu()
    return {"route": route, "content": content}


def _patch_effect(teacher, item, layer, component, device, alpha):
    full = item["full_plus"][layer].clone()
    start = item["enc_plus"]["answer_start"]
    n = min(len(component), len(full) - start)
    if n <= 0:
        return None
    full[start:start + n] -= alpha * component[:n]
    return float(patch_forward_nll(teacher, item["enc_plus"], device, layer - 1, full)) - float(item["teacher_nll"])


def prepare(cache, teacher, tok, cfg, device, student=None, out_dir=None):
    del tok, student, out_dir
    if not cache["pairs"][0]["full_plus"]:
        raise RuntimeError("CF3-01 requires data.cache_full_sequence=true for independent patch gates")
    alpha = float(cfg["method"].get("alpha", 0.1))
    gate_n = min(int(cfg["method"].get("gate_pairs", 4)), len(cache["pairs"]))
    gate_items = cache["pairs"][:gate_n]
    all_components = {(item["index"], layer): _components(item, layer, teacher)
                      for item in cache["pairs"] for layer in cfg["layers"]}
    reports, pass_map = {}, {}
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
    selected = {}
    ambiguity = []
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
    return {"enabled": enabled, "selected": selected, "components": all_components,
            "reports": reports, "ambiguous_layers": ambiguity, "alpha": alpha}


def loss(batch, prepared, cfg):
    terms = []
    for row in batch:
        item, plus = row["item"], row["plus"]
        for layer in cfg["layers"]:
            kind = prepared["selected"][layer]
            if kind is None:
                continue
            component = prepared["components"][(item["index"], layer)][kind].to(plus["nll"].device)
            hp = item["h_plus"][layer].to(plus["nll"].device)
            sh = span_hidden(plus, layer)
            n = min(len(sh), len(hp), len(component))
            target = hp[:n] - prepared["alpha"] * component[:n]
            terms.append(F.mse_loss(sh[:n], target.detach()))
    if not terms:
        raise RuntimeError("no BRCD component selected")
    return {"method_loss": torch.stack(terms).mean(), "selected_terms": torch.tensor(float(len(terms)))}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    return {"operator": "exact_attention_route_content_bilinear_difference", "selected": prepared["selected"],
            "ambiguous_layers": prepared["ambiguous_layers"], "gates": prepared["reports"],
            "enabled": prepared["enabled"], "donor_target": False}


def self_test():
    generator = torch.Generator().manual_seed(7)
    ap, am, bp, bm = [torch.randn(3, 5, generator=generator) for _ in range(4)]
    route, content = bilinear_difference(ap, am, bp, bm)
    error = (route + content - (ap * bp - am * bm)).abs().max()
    assert float(error) < 1e-5
    return {"method_id": METHOD_ID, "status": "PASS", "identity_max_error": float(error)}

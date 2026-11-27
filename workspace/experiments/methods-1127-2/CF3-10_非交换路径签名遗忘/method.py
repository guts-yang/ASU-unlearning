"""CF3-10 NPSU: noncommutative level-2 path-signature areas for Student unlearning."""
from __future__ import annotations

import torch
import torch.nn.functional as F

METHOD_ID = "CF3-10"
NEEDS_STUDENT_MINUS = True
NEEDS_STUDENT_CONTROL = True
REQUIRES_CONTROL = True


def level2_signature(path):
    increments = path[1:] - path[:-1]
    first = increments.sum(0)
    before = increments.cumsum(0) - increments
    second = before.T @ increments
    area = 0.5 * (second - second.T)
    return first, second, area


def make_path(deltas, projection):
    deltas = deltas.float()
    projection = projection.to(device=deltas.device, dtype=deltas.dtype)
    layers = deltas.shape[0]
    time = torch.linspace(0, 1, layers, dtype=deltas.dtype, device=deltas.device).unsqueeze(-1)
    return torch.cat([time, deltas @ projection], dim=-1)


def _teacher_path(item, layer_keys, projection, control=False):
    if control:
        plus = torch.stack([item["h_control_plus"][layer].mean(0) for layer in layer_keys])
        minus = torch.stack([item["h_control_minus"][layer].mean(0) for layer in layer_keys])
    else:
        plus = torch.stack([item["h_plus"][layer].mean(0) for layer in layer_keys])
        minus = torch.stack([item["h_minus"][layer].mean(0) for layer in layer_keys])
    return make_path(plus - minus, projection)


def _student_path(row, layer_keys, projection, control=False):
    if control:
        plus_forward, minus_forward = row["control_plus"], row["control_minus"]
    else:
        plus_forward, minus_forward = row["plus"], row["minus"]
    plus, minus = [], []
    for layer in layer_keys:
        for target, forward in ((plus, plus_forward), (minus, minus_forward)):
            start, end = forward["enc"]["answer_start"], forward["enc"]["answer_end"]
            target.append(forward["hidden"][layer][start:end].mean(0))
    return make_path(torch.stack(plus) - torch.stack(minus), projection)


def prepare(cache, teacher, tok, cfg, device, student=None, out_dir=None):
    del teacher, tok, student, out_dir
    layer_keys = list(cfg["layers"])
    width = cache["pairs"][0]["h_plus"][layer_keys[0]].shape[-1]
    dim = min(int(cfg["method"].get("projection_dim", 8)), width)
    projection = torch.linalg.qr(torch.randn(width, dim, generator=torch.Generator().manual_seed(int(cfg["seed"]))), mode="reduced").Q
    true_areas, null_areas = [], []
    for i, item in enumerate(cache["pairs"]):
        path = _teacher_path(item, layer_keys, projection)
        null_item = cache["pairs"][(i + 1) % len(cache["pairs"])]
        null_path = make_path(torch.stack([item["h_plus"][layer].mean(0) for layer in layer_keys]) -
                              torch.stack([null_item["h_minus"][layer].mean(0) for layer in layer_keys]), projection)
        true_areas.append(level2_signature(path)[2])
        null_areas.append(level2_signature(null_path)[2])
    true, null = torch.stack(true_areas), torch.stack(null_areas)
    threshold = float(cfg["method"].get("null_margin", 0.0))
    mask = true.mean(0).abs() > null.abs().median(0).values + threshold
    split_a, split_b = true[0::2].mean(0), true[1::2].mean(0)
    stable = (split_a * split_b) >= 0
    mask = mask & stable
    if not mask.any() and not bool(cfg["method"].get("strict_gate", True)):
        index = true.mean(0).abs().flatten().argmax()
        mask.view(-1)[index] = True
    control_targets = {item["index"]: _teacher_path(item, layer_keys, projection, control=True)
                       for item in cache["pairs"] if "h_control_plus" in item}
    return {"enabled": bool(mask.any()), "mask": mask.to(device), "projection": projection.to(device),
            "layers": layer_keys, "control_targets": control_targets,
            "true_area_mean": true.mean(0), "null_area_median": null.abs().median(0).values,
            "selected_areas": int(mask.sum())}


def loss(batch, prepared, cfg):
    signature_terms, anchor_terms = [], []
    for row in batch:
        item = row["item"]
        path = _student_path(row, prepared["layers"], prepared["projection"])
        area = level2_signature(path)[2]
        signature_terms.append((area * prepared["mask"]).pow(2).sum())
        if item["index"] in prepared["control_targets"]:
            control_path = _student_path(row, prepared["layers"], prepared["projection"], control=True)
            control_area = level2_signature(control_path)[2]
            target_path = prepared["control_targets"][item["index"]].to(control_area.device)
            target_area = level2_signature(target_path)[2].detach()
            anchor_terms.append((control_area - target_area).pow(2).mean())
    if not signature_terms:
        raise RuntimeError("no path-signature terms")
    anchor = torch.stack(anchor_terms).mean() if anchor_terms else signature_terms[0].new_zeros(())
    return {"method_loss": torch.stack(signature_terms).mean(),
            "extra_loss": float(cfg["method"].get("control_anchor_weight", 0.1)) * anchor,
            "control_anchor": anchor}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    return {"operator": "level2_noncommutative_levy_area", "selected_areas": prepared["selected_areas"],
            "enabled": prepared["enabled"], "path_scope": "answer-span pooled, ordered selected layers",
            "control_anchor": "same-entity non-target relation counterfactual signature",
            "true_area_mean": prepared["true_area_mean"].tolist()}


def self_test():
    path = torch.tensor([[0.0, 0.0, 0.0], [0.4, 1.0, 0.0], [1.0, 1.0, 1.0]])
    _, _, area = level2_signature(path)
    _, _, reversed_area = level2_signature(path.flip(0))
    assert torch.allclose(area, -reversed_area, atol=1e-6)
    straight = torch.stack([torch.zeros(3), torch.ones(3), 2 * torch.ones(3)])
    assert float(level2_signature(straight)[2].abs().max()) < 1e-6
    repeated = torch.cat([path[:1], path[:1], path[1:2], path[1:2], path[2:]])
    assert torch.allclose(area, level2_signature(repeated)[2], atol=1e-6)
    return {"method_id": METHOD_ID, "status": "PASS", "reversal": True, "straight_zero": True, "time_reparameterization": True}

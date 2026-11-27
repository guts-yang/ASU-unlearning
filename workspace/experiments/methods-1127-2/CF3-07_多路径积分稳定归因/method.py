"""CF3-07 MPIA: sequential subject-path integration and cross-path stable attribution."""
from __future__ import annotations

import torch
import torch.nn.functional as F

METHOD_ID = "CF3-07"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def stability(mean_z, paths_z, epsilon=1e-8):
    return mean_z.abs() / (paths_z.abs().mean(0) + epsilon)


@torch.no_grad()
def _answer_vectors(model, embeddings, enc, layers):
    mask = enc["attention_mask"].to(embeddings.device).unsqueeze(0)
    out = model(inputs_embeds=embeddings.unsqueeze(0), attention_mask=mask, output_hidden_states=True, use_cache=False)
    start, end = enc["answer_start"], enc["answer_end"]
    return {layer: out.hidden_states[layer][0, start:end].mean(0).float() for layer in layers}


@torch.no_grad()
def sequential_path_attribution(model, item, cfg, device):
    enc = item["enc_plus"]
    ids = enc["input_ids"].to(device)
    minus_ids = item["enc_minus"]["input_ids"].to(device)
    embedding = model.get_input_embeddings()
    base = embedding(ids).detach()
    minus = embedding(minus_ids).detach()
    subject = list(item["subject_indices"])
    layers = list(cfg["layers"])
    steps = int(cfg["method"].get("quadrature_steps", 4))
    epsilon = float(cfg["method"].get("derivative_epsilon", 0.01))
    orders = [tuple(subject), tuple(reversed(subject))]
    paths = []
    for order in orders:
        state = base.clone()
        state[subject] = minus[subject]
        attribution = {layer: torch.zeros(len(subject), state.shape[-1], device=device) for layer in layers}
        for position in order:
            slot = subject.index(position)
            start = state[position].clone()
            delta = base[position] - start
            for step in range(steps):
                t = (step + 0.5) / steps
                low_t, high_t = max(0.0, t - epsilon), min(1.0, t + epsilon)
                low, high = state.clone(), state.clone()
                low[position] = start + low_t * delta
                high[position] = start + high_t * delta
                low_h = _answer_vectors(model, low, enc, layers)
                high_h = _answer_vectors(model, high, enc, layers)
                for layer in layers:
                    derivative = (high_h[layer] - low_h[layer]) / max(1e-8, high_t - low_t)
                    attribution[layer][slot] += derivative.to(device) / steps
            state[position] = base[position]
        paths.append({layer: value.detach().float().cpu() for layer, value in attribution.items()})
    endpoints = _answer_vectors(model, base, enc, layers)
    minus_state = base.clone()
    minus_state[subject] = minus[subject]
    minus_end = _answer_vectors(model, minus_state, enc, layers)
    return paths, endpoints, minus_end


def prepare(cache, teacher, tok, cfg, device, student=None, out_dir=None):
    del tok, student, out_dir
    targets, reports = {}, {}
    alpha = float(cfg["method"].get("alpha", 0.1))
    strict = bool(cfg["method"].get("strict_gate", True))
    items = cache["pairs"]
    for item in items:
        paths, endpoint_plus, endpoint_minus = sequential_path_attribution(teacher, item, cfg, device)
        wrong_paths = None
        if strict:
            wrong = next((x for x in items if x["index"] != item["index"] and
                          x["enc_minus"]["input_ids"].numel() == item["enc_plus"]["input_ids"].numel()), None)
            if wrong is not None:
                wrong_item = dict(item)
                wrong_item["enc_minus"] = wrong["enc_minus"]
                wrong_paths, _, _ = sequential_path_attribution(teacher, wrong_item, cfg, device)
        for layer in cfg["layers"]:
            stacked = torch.stack([path[layer] for path in paths])
            mean_z = stacked.mean(0)
            stable = stability(mean_z, stacked)
            null_stability = None
            if wrong_paths is not None:
                wrong_stack = torch.stack([path[layer] for path in wrong_paths])
                null_stability = float(stability(wrong_stack.mean(0), wrong_stack).mean())
            passed = null_stability is None or float(stable.mean()) >= null_stability + float(cfg["method"].get("stability_margin", 0.0))
            filtered = (stable * mean_z).sum(0)
            completeness = (stacked[0].sum(0) - (endpoint_plus[layer].cpu() - endpoint_minus[layer].cpu())).norm()
            scale = (endpoint_plus[layer].cpu() - endpoint_minus[layer].cpu()).norm().clamp_min(1e-8)
            pooled_plus = item["h_plus"][layer].mean(0)
            if passed or not strict:
                targets[(item["index"], layer)] = pooled_plus - alpha * filtered
            reports[f"{item['index']}:{layer}"] = {"mean_path_stability": float(stable.mean()),
                                                    "wrong_pair_path_stability": null_stability,
                                                    "specificity_gate": passed,
                                                    "relative_completeness_error": float(completeness / scale)}
    errors = [x["relative_completeness_error"] for x in reports.values()]
    max_error = max(errors, default=0.0)
    tolerance = float(cfg["method"].get("completeness_tolerance", 0.25))
    layer_coverage = all(any(key[1] == layer for key in targets) for layer in cfg["layers"])
    enabled = max_error <= tolerance and layer_coverage
    if not strict:
        enabled = True
    return {"enabled": enabled, "targets": targets, "reports": reports, "alpha": alpha,
            "max_completeness_error": max_error, "completeness_tolerance": tolerance}


def loss(batch, prepared, cfg):
    terms = []
    for row in batch:
        item, plus = row["item"], row["plus"]
        for layer in cfg["layers"]:
            target = prepared["targets"][(item["index"], layer)].to(plus["nll"].device)
            start, end = plus["enc"]["answer_start"], plus["enc"]["answer_end"]
            terms.append(F.mse_loss(plus["hidden"][layer][start:end].mean(0), target.detach()))
    return {"method_loss": torch.stack(terms).mean()}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    return {"operator": "sequential_subject_path_directional_integration",
            "paths": ["forward_subject_order", "reverse_subject_order"],
            "max_relative_completeness_error": prepared["max_completeness_error"],
            "tolerance": prepared["completeness_tolerance"], "enabled": prepared["enabled"],
            "reports": prepared["reports"], "scope": "answer-span pooled vector attribution"}


def self_test():
    generator = torch.Generator().manual_seed(5)
    paths = torch.randn(2, 3, 4, generator=generator)
    same = paths[0:1].repeat(2, 1, 1)
    stable = stability(same.mean(0), same)
    opposite = torch.stack([paths[0], -paths[0]])
    unstable = stability(opposite.mean(0), opposite)
    assert float((stable - 1).abs().max()) < 1e-5
    assert float(unstable.abs().max()) < 1e-5
    return {"method_id": METHOD_ID, "status": "PASS", "stable": float(stable.mean()), "opposite": float(unstable.mean())}

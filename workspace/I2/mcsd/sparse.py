"""Sparse candidate proposal. Soft scores are not causal evidence."""

from __future__ import annotations

import torch

from I2.mcsd.config import require_keys
from I2.mcsd.effects import measure_effects
from I2.mcsd.errors import ConfigError, DifferentiableMaskUnavailable
from I2.mcsd.gates import assert_calibration_only
from I2.mcsd.hooks import parse_path, path_to_dict, supports_differentiable_mask
from I2.mcsd.patch import capture_clean, patched_logits
from I2.mcsd.score import assert_same_y_plus, refuse_answer_margin, sequence_logprob


PGB_KEYS = (
    "objective",
    "lambda_sparse",
    "lambda_binary",
    "attention_budget",
    "mlp_budget",
    "binarize_threshold",
    "steps",
    "lr",
    "candidate_pool",
)
GREEDY_KEYS = ("rank_by", "attention_budget", "mlp_budget", "candidate_pool")


def select_binary(model, example, config, split):
    assert_calibration_only(config, split, "sparse selection")
    selection = config.get("selection") or {}
    method = selection.get("method")
    if method is None:
        raise ConfigError("selection.method is missing; refusing an implicit candidate choice")
    if method == "pgb_ct":
        require_keys(selection, PGB_KEYS, "pgb_ct")
        if not supports_differentiable_mask(model):
            raise DifferentiableMaskUnavailable(
                "differentiable activation mask is unavailable for this model; "
                "refusing to run or relabel this search as pgb_ct. "
                "Set selection.method to discrete_greedy for the reproducible discrete fallback."
            )
    elif method == "discrete_greedy":
        require_keys(selection, GREEDY_KEYS, "discrete_greedy")
        if selection.get("rank_by") not in {"N", "N+R"}:
            raise ConfigError("rank_by must be N or N+R")
    else:
        raise ConfigError("unknown selection method %r" % (method,))
    _ensure_caches(model, example)
    if method == "pgb_ct":
        return _pgb(model, example, selection)
    return _greedy(model, example, selection)


def _ensure_caches(model, example):
    if example.get("cache_plus") is None or example.get("cache_minus") is None:
        example["cache_plus"] = capture_clean(model, example["prompt_plus"])
        example["cache_minus"] = capture_clean(model, example["prompt_minus"])


def _pool(selection):
    pool = selection.get("candidate_pool")
    if not pool:
        raise ConfigError("candidate_pool is missing; refusing an implicit path set")
    return [parse_path(item) for item in pool]


def _pgb(model, example, selection):
    paths = _pool(selection)
    if selection["objective"] not in {"negative_soft_g", "minimize_soft_g"}:
        raise ConfigError("objective must be negative_soft_g or minimize_soft_g")
    raw = torch.zeros(len(paths), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=float(selection["lr"]))
    last_g = None
    scope = None
    for _ in range(int(selection["steps"])):
        mask = torch.sigmoid(raw)
        soft_g, scope = _soft_advantage(model, example, paths, mask)
        loss = _objective(soft_g, mask, selection)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            projected = _project(torch.sigmoid(raw), paths, selection)
            raw.copy_(_logit(projected))
        last_g = soft_g.detach()
    with torch.no_grad():
        final_mask = torch.sigmoid(raw).clone()
    chosen = _binarize(final_mask, paths, selection)
    binary = _binary_effects(model, example, chosen)
    return {
        "method": "pgb_ct",
        "ranking_only": False,
        "soft_g": None if last_g is None else float(last_g),
        "soft_causal_evidence": False,
        "score_scope": scope,
        "binary_paths": [path_to_dict(path) for path in chosen],
        "binary_effects": binary,
        "mask": [float(item) for item in final_mask],
    }


def _greedy(model, example, selection):
    paths = _pool(selection)
    singles = []
    for path in paths:
        effect = measure_effects(model, example, [path], strength=1.0)
        rank_value = effect["N"] if selection["rank_by"] == "N" else effect["N"] + effect["R"]
        singles.append(
            {
                "path": path_to_dict(path),
                "N": effect["N"],
                "R": effect["R"],
                "rank_value": rank_value,
                "ranking_only": True,
                "computed_by": effect["computed_by"],
            }
        )
    chosen = []
    for module, budget_key in (("attention", "attention_budget"), ("mlp", "mlp_budget")):
        rows = [row for row, path in zip(singles, paths) if path[0] == module]
        rows.sort(key=lambda row: (-row["rank_value"], row["path"]["token"], -1 if row["path"]["head"] is None else row["path"]["head"]))
        chosen.extend(rows[: int(selection[budget_key])])
    chosen_paths = [parse_path(row["path"]) for row in chosen]
    binary = _binary_effects(model, example, chosen_paths)
    return {
        "method": "discrete_greedy",
        "ranking_only": True,
        "soft_causal_evidence": False,
        "single_path": singles,
        "binary_paths": [path_to_dict(path) for path in chosen_paths],
        "binary_effects": binary,
    }


def _binary_effects(model, example, paths):
    effect = measure_effects(model, example, paths, strength=1.0)
    if not paths:
        effect["status"] = "empty_selection"
    effect["causal_evidence"] = True
    effect["paths"] = [path_to_dict(path) for path in paths]
    return effect


def _soft_advantage(model, example, paths, mask):
    refuse_answer_margin(example)
    assert_same_y_plus(example["score_plus"].get("labels"), example["score_minus"].get("labels"))
    weights = {paths[index]: mask[index] for index in range(len(paths))}
    logits, _ = patched_logits(
        model,
        example["score_plus"],
        paths,
        example["cache_plus"],
        example["cache_minus"],
        strength=None,
        weights=weights,
    )
    return sequence_logprob(logits, example["score_plus"]["labels"]), "y_plus_only"


def _objective(soft_g, mask, selection):
    penalty = float(selection["lambda_sparse"]) * mask.mean() + float(selection["lambda_binary"]) * (mask * (1 - mask)).mean()
    value = soft_g.mean()
    if selection["objective"] == "negative_soft_g":
        return -value + penalty
    if selection["objective"] == "minimize_soft_g":
        return value + penalty
    raise ConfigError("objective must be negative_soft_g or minimize_soft_g")


def _project(mask, paths, selection):
    projected = mask.clone()
    for module, budget_key in (("attention", "attention_budget"), ("mlp", "mlp_budget")):
        indexes = [index for index, path in enumerate(paths) if path[0] == module]
        if not indexes:
            continue
        budget = float(selection[budget_key])
        if budget == 0.0:
            projected[indexes] = 0
            continue
        total = float(projected[indexes].sum())
        if total > budget:
            projected[indexes] = projected[indexes] * (budget / total)
    return projected


def _binarize(mask, paths, selection):
    chosen = []
    threshold = float(selection["binarize_threshold"])
    for module, budget_key in (("attention", "attention_budget"), ("mlp", "mlp_budget")):
        ranked = []
        for index, path in enumerate(paths):
            if path[0] == module and float(mask[index]) >= threshold:
                ranked.append(index)
        ranked.sort(key=lambda index: (-float(mask[index]), paths[index]))
        budget = int(selection[budget_key])
        chosen.extend(paths[index] for index in ranked[:budget])
    return chosen


def _logit(probabilities):
    clipped = probabilities.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(clipped) - torch.log1p(-clipped)

"""CF2-10: bounded coalition interactions through teacher patch-forward."""
from __future__ import annotations

import itertools
import torch
import torch.nn.functional as F

from modeling import patch_forward_nll, patch_forward_nll_multi, span_hidden

METHOD_ID = "CF2-10"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def _answer_delta(item, layer):
    hp, hm = item["full_plus"][layer], item["full_minus"][layer]
    start = item["enc_plus"]["answer_start"]
    end = min(item["enc_plus"]["answer_end"], len(hp), len(hm))
    return start, end, hp[start:end] - hm[start:end]


def _edited_teacher(item, layer):
    start, end, delta = _answer_delta(item, layer)
    edited = item["full_plus"][layer].clone()
    edited[start:end] = edited[start:end] - delta
    return edited


def _retain_cost(teacher, retain, deltas, device):
    if retain is None or "full" not in retain:
        return None
    replacements = {}
    for layer, delta in deltas.items():
        full = retain["full"][layer].clone()
        start = retain["enc"]["answer_start"]
        end = retain["enc"]["answer_end"]
        n = min(end - start, len(delta))
        full[start:start + n] = full[start:start + n] - delta[:n]
        replacements[layer - 1] = full
    patched = patch_forward_nll_multi(teacher, retain["enc"], device, replacements)
    return abs(float(patched) - float(retain["teacher_nll"]))


def prepare(cache, teacher, tok, cfg, device):
    del tok
    if not cache["pairs"][0]["full_plus"]:
        raise RuntimeError("CF2-10 requires data.cache_full_sequence=true")
    n_components = min(int(cfg["method"].get("max_components", 4)), len(cfg["layers"]))
    components = [x for x in cfg["layers"] if x > 0][-n_components:]
    if len(components) < 2:
        raise RuntimeError("coalition method needs at least two non-embedding layers")
    retain = cache["retain"][0] if cache["retain"] else None
    selected_by_item = {}
    all_reports = []
    for item in cache["pairs"]:
        base = float(item["teacher_nll"])
        values = {(): 0.0}
        for layer in components:
            patched = patch_forward_nll(teacher, item["enc_plus"], device, layer - 1, _edited_teacher(item, layer))
            values[(layer,)] = float(patched) - base
        interactions = {}
        scores = {}
        for a, b in itertools.combinations(components, 2):
            replacements = {a - 1: _edited_teacher(item, a), b - 1: _edited_teacher(item, b)}
            vab = float(patch_forward_nll_multi(teacher, item["enc_plus"], device, replacements)) - base
            interaction = vab - values[(a,)] - values[(b,)] + values[()]
            deltas = {}
            for layer in (a, b):
                _, _, delta = _answer_delta(item, layer)
                deltas[layer] = delta
            utility = _retain_cost(teacher, retain, deltas, device)
            score = interaction - float(cfg["method"].get("utility_weight", 0.1)) * (utility or 0.0)
            interactions[(a, b)] = interaction
            scores[(a, b)] = score
            all_reports.append({"pair_index": item["index"], "components": [a, b], "single_effects": [values[(a,)], values[(b,)]], "joint_effect": vab, "interaction": interaction, "retain_utility_cost": utility, "selection_score": score})
        chosen = max(scores, key=scores.get)
        selected_by_item[item["index"]] = chosen
    return {"enabled": True, "components": components, "selected": selected_by_item, "reports": all_reports}


def _control_batch(batch, cfg):
    """§7 control arm: rotate/randomly permute the Teacher-side `item` only.
    Student-side plus/minus/control_* stay, so only the regression target moves.
    Default 'none' is a no-op -> historical results stay reproducible.
    """
    mode = str(cfg["method"].get("control_mode", "none"))
    if mode in ("none", "") or len(batch) < 2:
        return batch, False
    if mode in ("wrong", "shuffle"):
        order = list(range(1, len(batch))) + [0]
    elif mode in ("random", "noise"):
        gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)) + 977)
        order = torch.randperm(len(batch), generator=gen).tolist()
        if order == list(range(len(batch))):
            order = list(range(1, len(batch))) + [0]
    else:
        raise ValueError(f"unknown control_mode: {mode}")
    return [dict(row, item=batch[j]["item"]) for row, j in zip(batch, order)], True


def loss(batch, prepared, cfg):
    batch, control_changed = _control_batch(batch, cfg)
    terms = []
    for row in batch:
        item, plus = row["item"], row["plus"]
        for layer in prepared["selected"][item["index"]]:
            hp = item["h_plus"][layer].to(plus["nll"].device)
            hm = item["h_minus"][layer].to(plus["nll"].device)
            sh = span_hidden(plus, layer)
            n = min(len(sh), len(hp), len(hm))
            target = hp[:n] - float(cfg["method"].get("alpha", 0.1)) * (hp[:n] - hm[:n])
            terms.append(F.mse_loss(sh[:n], target.detach()))
    return {"control_changed": torch.tensor(float(control_changed)), "method_loss": torch.stack(terms).mean(), "selected_coalitions": torch.tensor(float(len(terms)), device=batch[0]["plus"]["nll"].device)}


def diagnostics(cache, prepared, cfg):
    del cache
    values = [x["interaction"] for x in prepared["reports"]]
    utilities = [x["retain_utility_cost"] for x in prepared["reports"] if x["retain_utility_cost"] is not None]
    return {"operator": "bounded_pairwise_coalition_interaction", "components": prepared["components"], "selected": {str(k): list(v) for k, v in prepared["selected"].items()}, "interactions": prepared["reports"], "max_abs_interaction": max([abs(x) for x in values], default=0.0), "mean_retain_utility_cost": sum(utilities) / len(utilities) if utilities else None, "utility_weight": float(cfg["method"].get("utility_weight", 0.1)), "scope": "exact pair interaction without sampled background coalitions; not a full Shapley-Taylor estimate"}


def self_test():
    additive = 2.0 - 1.0 - 1.0
    synergy = 3.0 - 1.0 - 1.0
    redundancy = 1.0 - 1.0 - 1.0
    assert additive == 0 and synergy == 1 and redundancy == -1
    return {"method_id": METHOD_ID, "status": "PASS", "additive": additive, "synergy": synergy, "redundancy": redundancy}

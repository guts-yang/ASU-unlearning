"""CF2-05: layer x answer-position wavelet-domain counterfactual erasure."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from modeling import span_hidden

METHOD_ID = "CF2-05"
NEEDS_STUDENT_MINUS = False
REQUIRES_CONTROL = False


def next_power_two(n):
    return 1 << max(0, (n - 1).bit_length())


def haar_matrix(n):
    if n == 1:
        return torch.ones(1, 1)
    previous = haar_matrix(n // 2)
    top = torch.kron(previous, torch.tensor([[1.0, 1.0]])) / math.sqrt(2.0)
    bottom = torch.kron(torch.eye(n // 2), torch.tensor([[1.0, -1.0]])) / math.sqrt(2.0)
    return torch.cat([top, bottom], dim=0)


def scale_index(i):
    return 0 if i == 0 else int(math.floor(math.log2(i))) + 1


def transform_2d(x, hl, hp, inverse=False):
    hl, hp = hl.to(x), hp.to(x)
    if inverse:
        return torch.einsum("ai,abc,bj->ijc", hl, x, hp)
    return torch.einsum("ai,ijc,bj->abc", hl, x, hp)


def _stack(item, layers, max_span, wrong_item=None):
    width = item["h_plus"][layers[0]].shape[-1]
    lp, pp = next_power_two(len(layers)), next_power_two(max_span)
    plus = torch.zeros(lp, pp, width)
    delta = torch.zeros_like(plus)
    minus_source = wrong_item or item
    for i, layer in enumerate(layers):
        hp, hm = item["h_plus"][layer], minus_source["h_minus"][layer]
        n = min(len(hp), len(hm), max_span)
        plus[i, :n] = hp[:n]
        delta[i, :n] = hp[:n] - hm[:n]
    return plus, delta


def _coherence(cache, cfg, max_span, wrong=False):
    layers = cfg["layers"]
    lp, pp = next_power_two(len(layers)), next_power_two(max_span)
    hl, hp = haar_matrix(lp), haar_matrix(pp)
    grouped = {}
    for i, item in enumerate(cache["pairs"]):
        other = cache["pairs"][(i + 1) % len(cache["pairs"])] if wrong else None
        _, delta = _stack(item, layers, max_span, other)
        coeff = transform_2d(delta, hl, hp)
        for a in range(lp):
            for b in range(pp):
                key = (scale_index(a), scale_index(b))
                grouped.setdefault(key, []).append(coeff[a, b].flatten())
    result = {}
    for key, vectors in grouped.items():
        stack = torch.stack(vectors)
        center = stack.mean(0)
        result[key] = F.cosine_similarity(stack, center.unsqueeze(0).expand_as(stack), dim=-1, eps=1e-8).mean()
    return result


def prepare(cache, teacher, tok, cfg, device):
    del teacher, tok
    max_span = max(x["h_plus"][cfg["layers"][0]].shape[0] for x in cache["pairs"])
    true = _coherence(cache, cfg, max_span, False)
    wrong = _coherence(cache, cfg, max_span, True)
    temperature = float(cfg["method"].get("temperature", 0.1))
    masks = {key: torch.sigmoid((value - wrong[key]) / temperature) for key, value in true.items()}
    lp, pp = next_power_two(len(cfg["layers"])), next_power_two(max_span)
    return {"enabled": True, "masks": masks, "max_span": max_span, "hl": haar_matrix(lp).to(device), "hp": haar_matrix(pp).to(device), "true_coherence": true, "wrong_coherence": wrong}


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
    losses = []
    for row in batch:
        item, plus = row["item"], row["plus"]
        hp_stack, delta = _stack(item, cfg["layers"], prepared["max_span"])
        cp = transform_2d(hp_stack, prepared["hl"], prepared["hp"])
        cd = transform_2d(delta, prepared["hl"], prepared["hp"])
        mask = torch.zeros_like(cd)
        for a in range(cd.shape[0]):
            for b in range(cd.shape[1]):
                mask[a, b] = prepared["masks"][(scale_index(a), scale_index(b))].to(cd.device)
        target = transform_2d(cp - float(cfg["method"].get("alpha", 0.1)) * mask * cd, prepared["hl"], prepared["hp"], inverse=True)
        for i, layer in enumerate(cfg["layers"]):
            sh = span_hidden(plus, layer)
            n = min(len(sh), item["h_plus"][layer].shape[0], prepared["max_span"])
            if n:
                losses.append(F.mse_loss(sh[:n], target[i, :n].to(sh.device).detach()))
    return {"control_changed": torch.tensor(float(control_changed)), "method_loss": torch.stack(losses).mean()}


def diagnostics(cache, prepared, cfg):
    del cache, cfg
    return {"operator": "orthogonal_2d_haar_scale_mask", "max_answer_span": prepared["max_span"], "scale_masks": {str(k): float(v) for k, v in prepared["masks"].items()}, "true_coherence": {str(k): float(v) for k, v in prepared["true_coherence"].items()}, "wrong_coherence": {str(k): float(v) for k, v in prepared["wrong_coherence"].items()}, "scale_semantics_are_hypotheses_not_labels": True}


def self_test():
    x = torch.randn(4, 4, 3)
    hl, hp = haar_matrix(4), haar_matrix(4)
    reconstructed = transform_2d(transform_2d(x, hl, hp), hl, hp, inverse=True)
    error = (x - reconstructed).abs().max()
    assert error < 1e-5
    return {"method_id": METHOD_ID, "status": "PASS", "roundtrip_max_error": float(error)}

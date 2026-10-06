"""Hinge rank-shattering loss. Forces gold logits behind a competitor by margin m."""

import torch
import torch.nn.functional as F


def _competitor_ids(logits, gold_ids):
    """Second-place token if gold is top-1, else current top-1. Stop-grad ids."""
    blocked = logits.detach().clone()
    blocked.scatter_(-1, gold_ids.unsqueeze(-1), float("-inf"))
    return blocked.argmax(dim=-1)


def rank_hinge_loss(logits, gold_ids, margin=1.0, mask=None):
    """logits (..., V), gold_ids (...). Optional mask on the leading dims.

    L = mean max(0, m - (z_v* - z_a*)). Minimizing pushes z_v* >= z_a* + m.
    """
    if gold_ids.shape != logits.shape[:-1]:
        raise ValueError("gold_ids must match logits batch/time shape")
    v_star = _competitor_ids(logits, gold_ids)
    z_gold = logits.gather(-1, gold_ids.unsqueeze(-1)).squeeze(-1)
    z_comp = logits.gather(-1, v_star.unsqueeze(-1)).squeeze(-1)
    hinge = F.relu(margin - (z_comp - z_gold))
    # Active hinge = m - z_v* + z_gold, so descent lowers z_gold and raises z_v*.
    if mask is None:
        return hinge.mean()
    mask = mask.to(dtype=hinge.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (hinge * mask).sum() / denom

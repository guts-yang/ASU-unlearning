"""Gold-token rank profile from logits. CPU tensors only; no checkpoints."""

import torch
import torch.nn.functional as F


def profile_logits(logits, gold_ids):
    """logits: (..., V), gold_ids: (...) matching the leading dims."""
    if gold_ids.shape != logits.shape[:-1]:
        raise ValueError("gold_ids must match logits batch/time shape")
    gold = gold_ids.unsqueeze(-1)
    gold_logit = logits.gather(-1, gold).squeeze(-1)
    rank = (logits > gold_logit.unsqueeze(-1)).sum(dim=-1) + 1
    top2 = logits.topk(2, dim=-1)
    top1_logit = top2.values[..., 0]
    second_logit = top2.values[..., 1]
    is_top1 = rank == 1
    competitor = torch.where(is_top1, second_logit, top1_logit)
    margin = gold_logit - competitor
    probs = F.softmax(logits, dim=-1)
    gold_prob = probs.gather(-1, gold).squeeze(-1)
    entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
    return {
        "rank": rank,
        "margin": margin,
        "gold_prob": gold_prob,
        "entropy": entropy,
        "gold_logit": gold_logit,
    }


def temperature_scale(logits, tau):
    return logits / tau


def order_preserving_leak(profile_base, profile_tau):
    """True if tau flattening keeps gold as top-1 while entropy rises."""
    same_rank_one = (profile_base["rank"] == 1) & (profile_tau["rank"] == 1)
    entropy_up = profile_tau["entropy"] > profile_base["entropy"] + 1e-6
    return bool((same_rank_one & entropy_up).all())

"""Teacher-side gold-rank demotion, applied to logits before the forget KL.

Rank 1 is the highest logit. Equal logits break toward the smaller token id.
Only the gold logit is edited; the caller re-normalizes with softmax. This
module does not update the student, the optimizer, or the retain anchor.
"""

import torch
import torch.nn.functional as F


class GoldRankConfigError(ValueError):
    """Raised when a frozen gold-rank spec is incomplete or inconsistent."""


MODES = (
    "demote",
    "fixed_penalty",
    "matched_penalty",
    "random_position",
    "shuffle_k",
    "topk_flatten",
)


def validate_spec(spec):
    mode = spec.get("mode")
    if mode not in MODES:
        raise GoldRankConfigError(f"unknown gold_rank mode {mode}")
    if mode in {"demote", "random_position", "topk_flatten"}:
        if spec.get("k") is None or int(spec["k"]) < 1:
            raise GoldRankConfigError("gold_rank K must be frozen before training")
    if mode in {"demote", "random_position", "shuffle_k"}:
        if spec.get("delta") is None or float(spec["delta"]) < 0:
            raise GoldRankConfigError("gold_rank delta must be frozen and >= 0")
    if mode in {"fixed_penalty", "matched_penalty"}:
        if spec.get("penalty") is None or float(spec["penalty"]) < 0:
            raise GoldRankConfigError("gold-logit penalty must be frozen before training")
    if mode == "shuffle_k" and not list(spec.get("k_grid") or []):
        raise GoldRankConfigError("shuffle_k requires the preregistered K grid")
    return spec


def gold_ranks(logits, gold_ids):
    """Return the 1-based stable rank of each gold id. ``logits`` is ``[N, V]``."""
    gold_logit = logits.gather(1, gold_ids[:, None]).squeeze(1)
    token_ids = torch.arange(logits.shape[-1], device=logits.device)
    strictly_better = logits > gold_logit[:, None]
    tie_better = (logits == gold_logit[:, None]) & (token_ids[None, :] < gold_ids[:, None])
    return 1 + (strictly_better | tie_better).sum(dim=-1)


def _kth_nongold(logits, gold_ids, k):
    vocab = logits.shape[-1]
    if k < 1 or k >= vocab:
        missing = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        return logits.new_zeros(logits.shape[0]), missing
    hidden = logits.masked_fill(
        F.one_hot(gold_ids, num_classes=vocab).bool(),
        torch.finfo(logits.dtype).min,
    )
    kth = torch.topk(hidden, k, dim=-1).values[:, -1]
    return kth, torch.isfinite(kth)


def _stable_seed(seed, gold_ids):
    mixed = int(seed) & 0x7FFFFFFF
    for value in gold_ids.detach().cpu().tolist():
        mixed = (mixed * 1000003 + int(value)) & 0x7FFFFFFF
    return mixed


def _edit_valid_rows(logits, gold_ids, spec):
    """Edit ``[N, V]`` rows. Every row is a valid answer position."""
    mode = spec["mode"]
    out = logits.clone()
    skipped = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
    edited = skipped.clone()
    if logits.shape[0] == 0:
        return out, edited, skipped

    if mode in {"fixed_penalty", "matched_penalty"}:
        penalty = float(spec["penalty"])
        gold_logit = logits.gather(1, gold_ids[:, None]).squeeze(1)
        out[torch.arange(logits.shape[0], device=logits.device), gold_ids] = gold_logit - penalty
        return out, torch.ones_like(edited), skipped

    if mode == "shuffle_k":
        grid = [int(value) for value in spec["k_grid"]]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(int(spec.get("seed", 0)), gold_ids))
        picks = torch.randint(0, len(grid), (logits.shape[0],), generator=generator)
        for index, k in enumerate(grid):
            group = (picks == index).to(logits.device)
            if not bool(group.any()):
                continue
            group_spec = dict(spec)
            group_spec["mode"] = "demote"
            group_spec["k"] = k
            edited_group, edited_mask, skipped_mask = _edit_valid_rows(logits[group], gold_ids[group], group_spec)
            out[group] = edited_group
            edited[group] = edited_mask.to(edited.device)
            skipped[group] = skipped_mask.to(skipped.device)
        return out, edited, skipped

    k = int(spec["k"])
    delta = float(spec.get("delta") or 0.0)
    rank = gold_ranks(logits, gold_ids)
    kth, available = _kth_nongold(logits, gold_ids, k)
    skipped = ~available
    gold_logit = logits.gather(1, gold_ids[:, None]).squeeze(1)
    demoted = torch.minimum(gold_logit, kth - delta)
    if mode == "demote":
        edited = (rank <= k) & available
    elif mode == "random_position":
        would = (rank <= k) & available
        pool = available.nonzero(as_tuple=False).flatten()
        chosen = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        if pool.numel() and int(would.sum()) > 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_stable_seed(int(spec.get("seed", 0)), gold_ids))
            order = torch.randperm(pool.numel(), generator=generator)
            take = min(int(would.sum()), int(pool.numel()))
            picked = pool[order[:take].to(pool.device)]
            chosen[picked] = True
        edited = chosen & available
    elif mode == "topk_flatten":
        edited = (rank <= k) & available
        if bool(edited.any()):
            order = torch.argsort(logits, dim=-1, descending=True, stable=True)
            top = order[:, :k]
            means = logits.gather(1, top).mean(dim=-1, keepdim=True)
            flat = out[edited].clone()
            flat.scatter_(1, top[edited], means[edited].expand_as(top[edited]))
            out[edited] = flat
        return out, edited, skipped
    else:
        raise GoldRankConfigError(f"unknown gold_rank mode {mode}")

    if bool(edited.any()):
        rows = torch.arange(logits.shape[0], device=logits.device)[edited]
        out[rows, gold_ids[edited]] = demoted[edited]
    return out, edited, skipped


def apply_gold_rank_logits(logits, gold_ids, mask, spec):
    """Return edited ``[batch, positions, vocab]`` logits and optional stats.

    Invalid labels stay unchanged. Stats are omitted unless
    ``spec["collect_stats"]`` is true, so the training step does not synchronize.
    """
    if logits.ndim != 3:
        raise GoldRankConfigError("teacher logits must have shape [batch, positions, vocab]")
    gold_ids = gold_ids.long()
    valid = mask.bool() & (gold_ids >= 0) & (gold_ids < logits.shape[-1])
    if not bool(valid.any()):
        return logits, None
    edited = logits.float().clone()
    rows, edited_mask, skipped_mask = _edit_valid_rows(edited[valid], gold_ids[valid], spec)
    edited[valid] = rows
    stats = None
    if spec.get("collect_stats"):
        before = logits.float()[valid]
        stats = _row_stats(before, rows, gold_ids[valid], edited_mask, skipped_mask)
    return edited, stats


def _row_stats(before, after, gold_ids, edited, skipped):
    prob_before = torch.softmax(before, dim=-1).gather(1, gold_ids[:, None]).squeeze(1)
    prob_after = torch.softmax(after, dim=-1).gather(1, gold_ids[:, None]).squeeze(1)
    rank_before = gold_ranks(before, gold_ids).float()
    rank_after = gold_ranks(after, gold_ids).float()
    top1 = after.argmax(dim=-1)
    return {
        "n_valid": int(gold_ids.shape[0]),
        "n_edited": int(edited.sum().item()),
        "n_skipped": int(skipped.sum().item()),
        "gold_prob_before_sum": float(prob_before.sum().item()),
        "gold_prob_after_sum": float(prob_after.sum().item()),
        "rank_before_sum": float(rank_before.sum().item()),
        "rank_after_sum": float(rank_after.sum().item()),
        "top1_is_gold": int((top1 == gold_ids).sum().item()),
        "top1_changed": int((top1 != before.argmax(dim=-1)).sum().item()),
    }


def gold_components(logits, gold_ids):
    """Gold logit and log-mass of the other tokens, enough to match a penalty."""
    gold_ids = gold_ids.long()
    gold_logit = logits.gather(-1, gold_ids[:, None]).squeeze(-1)
    hidden = logits.masked_fill(
        F.one_hot(gold_ids, num_classes=logits.shape[-1]).bool(),
        torch.finfo(logits.dtype).min,
    )
    return gold_logit, torch.logsumexp(hidden, dim=-1)


def gold_probability(gold_logit, log_rest):
    return torch.exp(gold_logit - torch.logaddexp(gold_logit, log_rest))


def fit_logit_penalty(gold_logit, log_rest, target_drop, tolerance=1e-3, max_penalty=100.0):
    """Find a uniform gold-logit subtraction with the requested mean probability drop."""
    baseline = gold_probability(gold_logit, log_rest)
    target = float(target_drop)
    if target <= tolerance:
        achieved = 0.0
        return 0.0, achieved, False

    def drop(penalty):
        shifted = gold_probability(gold_logit - penalty, log_rest)
        return float((baseline - shifted).mean().item())

    hi = 1.0
    saturated = False
    while drop(hi) < target and hi < max_penalty:
        hi = min(max_penalty, hi * 2.0)
        if hi == max_penalty:
            break
    if drop(hi) + tolerance < target:
        saturated = True
        return hi, drop(hi), saturated
    lo = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if drop(mid) < target:
            lo = mid
        else:
            hi = mid
    achieved = drop(hi)
    return hi, achieved, abs(achieved - target) > tolerance


def select_operating_point(cells, min_rank_separation):
    """Freeze one grid cell, or stop when demotion is degenerate or penalty-like.

    ``cells`` contain calibration sums only. A missing matched-penalty rank
    comparison leaves the status pending; it never reads a test metric.
    """
    eligible = []
    for cell in cells:
        valid = int(cell["n_valid"])
        if valid <= 0:
            continue
        rate = cell["n_edited"] / valid
        rank_increase = (cell["rank_after_sum"] - cell["rank_before_sum"]) / valid
        record = dict(cell)
        record["edit_rate"] = rate
        record["mean_rank_increase"] = rank_increase
        if not (0.0 < rate <= 0.95) or rank_increase <= 0.0:
            continue
        matched = cell.get("matched_rank_increase")
        if matched is not None and rank_increase < float(matched) + float(min_rank_separation):
            continue
        eligible.append(record)
    if not eligible:
        return {
            "status": "stop_not_distinct",
            "k": None,
            "delta": None,
            "test_metrics_read": False,
        }
    eligible.sort(key=lambda cell: (-cell["mean_rank_increase"], int(cell["k"]), float(cell["delta"])))
    chosen = dict(eligible[0])
    chosen["status"] = "frozen_pending_penalty_check" if chosen.get("matched_rank_increase") is None else "frozen"
    chosen["test_metrics_read"] = False
    return chosen

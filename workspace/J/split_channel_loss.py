"""Pure tensor definition of the proposed Direction J split-channel loss.

This module is deliberately not wired into either trainer. It exists only for
formula review and numerical tests while the preregistered M0 gate is NO-GO.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SplitChannelResult:
    loss: torch.Tensor
    cap_loss: torch.Tensor
    non_gold_loss: torch.Tensor
    cap_probability: torch.Tensor
    student_gold_probability: torch.Tensor
    cap_gate: torch.Tensor
    target_count: int


def _validate_inputs(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
    k: int,
    delta: float,
    rho: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if student_logits.ndim != 3 or student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must share [batch, sequence, vocab]")
    if base_logits.shape != student_logits.shape:
        raise ValueError("base logits must have the same shape as student logits")
    if labels.shape != student_logits.shape[:2]:
        raise ValueError("labels must have shape [batch, sequence]")
    expected_mask_shape = (labels.shape[0], labels.shape[1] - 1)
    if target_mask.shape != expected_mask_shape:
        raise ValueError(f"target_mask must have shifted query shape {expected_mask_shape}")
    if student_logits.shape[-1] < 2:
        raise ValueError("the split-channel loss needs at least two vocabulary items")
    if not 1 <= k <= student_logits.shape[-1] - 1:
        raise ValueError("k must select one of the vocab_size - 1 non-gold logits")
    if delta <= 0:
        raise ValueError("delta must be positive to place gold strictly below the Kth non-gold logit")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")

    answer_mask = labels[:, 1:].ne(-100)
    target_mask = target_mask.bool()
    if torch.any(target_mask & ~answer_mask):
        raise ValueError("target_mask must be a subset of the valid answer mask")
    return answer_mask, target_mask


def _active_rows(
    logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1, :]
    shifted_gold = labels[:, 1:]
    return shifted_logits[target_mask], shifted_gold[target_mask]


def cap_targets(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    k: int,
    delta: float,
) -> torch.Tensor:
    """Return detached gold caps from the Kth non-gold teacher logit."""
    rows, gold = _active_rows(teacher_logits, labels, target_mask)
    if rows.shape[0] == 0:
        return rows.new_empty((0,))
    non_gold_rows = rows.detach().clone()
    non_gold_rows.scatter_(1, gold[:, None], -torch.inf)
    kth_non_gold = torch.topk(non_gold_rows, k=k, dim=-1).values[:, -1]
    edited = rows.detach().clone()
    original_gold = edited.gather(1, gold[:, None]).squeeze(1)
    edited_gold = torch.minimum(original_gold, kth_non_gold - delta)
    edited.scatter_(1, gold[:, None], edited_gold[:, None])
    return F.softmax(edited, dim=-1).gather(1, gold[:, None]).squeeze(1).detach()


def conditional_non_gold_log_probs(
    logits: torch.Tensor,
    gold: torch.Tensor,
) -> torch.Tensor:
    """Log probabilities on V minus gold, with a finite zero at excluded gold."""
    masked = logits.clone()
    masked.scatter_(1, gold[:, None], -torch.inf)
    log_probs = masked - torch.logsumexp(masked, dim=-1, keepdim=True)
    return log_probs.scatter(1, gold[:, None], 0.0)


def split_channel_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    k: int,
    delta: float,
    rho: float,
    beta: float = 1.0,
    lambda_ng: float = 1.0,
    eps: float = 1e-12,
) -> SplitChannelResult:
    """Evaluate the temporary Direction J objective on preselected target tokens.

    ``target_mask`` is aligned to ``logits[:, :-1]`` and must already encode an
    approved training-visible fact span. The reduction is a global mean over
    target positions, exactly as specified by sum(masked loss) / max(count, 1).
    """
    _, target_mask = _validate_inputs(
        student_logits,
        teacher_logits,
        base_logits,
        labels,
        target_mask,
        k,
        delta,
        rho,
    )
    if beta < 0 or lambda_ng < 0:
        raise ValueError("beta and lambda_ng must be non-negative")
    if not 0.0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    target_count = int(target_mask.sum().item())
    if target_count == 0:
        zero = student_logits.sum() * 0.0
        empty = student_logits.new_empty((0,))
        return SplitChannelResult(zero, zero, zero, empty, empty, empty.bool(), 0)

    student_rows, gold = _active_rows(student_logits, labels, target_mask)
    teacher_cap = cap_targets(
        teacher_logits,
        labels,
        target_mask,
        k=k,
        delta=delta,
    )
    student_log_probs = F.log_softmax(student_rows, dim=-1)
    student_log_gold = student_log_probs.gather(1, gold[:, None]).squeeze(1)
    student_gold = student_log_gold.exp()

    # Both the cap and the one-sided gate are frozen targets.
    cap = teacher_cap.detach()
    gate = (student_gold.detach() > cap).detach()
    cap_safe = cap.clamp(min=eps, max=1.0 - eps)
    student_safe = student_gold.clamp(min=eps, max=1.0 - eps)
    bernoulli_kl = (
        cap_safe * (cap_safe.log() - student_safe.log())
        + (1.0 - cap_safe)
        * ((1.0 - cap_safe).log() - (1.0 - student_safe).log())
    )
    cap_per_token = gate.to(bernoulli_kl.dtype) * bernoulli_kl

    base_rows, base_gold = _active_rows(base_logits, labels, target_mask)
    if not torch.equal(gold, base_gold):
        raise RuntimeError("student and base gold IDs are misaligned")
    student_non_gold_log = conditional_non_gold_log_probs(student_rows, gold)
    base_non_gold_log = conditional_non_gold_log_probs(base_rows.detach(), gold)
    base_non_gold = base_non_gold_log.exp()
    base_non_gold.scatter_(1, gold[:, None], 0.0)
    uniform = torch.full_like(base_non_gold, 1.0 / (student_rows.shape[-1] - 1))
    uniform.scatter_(1, gold[:, None], 0.0)
    target_non_gold = ((1.0 - rho) * base_non_gold + rho * uniform).detach()
    target_non_gold_log = torch.where(
        target_non_gold > 0,
        target_non_gold.clamp_min(eps).log(),
        torch.zeros_like(target_non_gold),
    )
    non_gold_per_token = (
        target_non_gold * (target_non_gold_log - student_non_gold_log)
    ).sum(dim=-1)

    cap_loss = cap_per_token.sum() / target_count
    non_gold_loss = ((1.0 - cap) * non_gold_per_token).sum() / target_count
    total = beta * cap_loss + lambda_ng * non_gold_loss
    return SplitChannelResult(
        total,
        cap_loss,
        non_gold_loss,
        cap,
        student_gold,
        gate,
        target_count,
    )

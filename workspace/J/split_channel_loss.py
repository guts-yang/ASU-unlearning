"""DARD-ASU v2 candidate forget loss (pure tensors; not wired into training).

This replaces the temporary Bernoulli / uniform-mix draft. The production ASU
forget loss remains ``m0_core.masked_asu_kl`` / ``trainer.losses.asu_loss``.

Candidate forget reduction (global over answer tokens):

    L_F = (sum_t m_other[t] * KL(p_T || p_S)_t
         + sum_t m_tar[t]   * KL(stopgrad(q_star) || p_S)_t)
          / max(sum_t m_ans[t], 1)

On target positions, gold is capped from the Kth non-gold teacher logit with a
strict positive margin, then:

    b_hat = min(a.detach(), b.detach())
    r     = (1 - eta) * q_T + eta * q_B     # conditional on V \\ {g}
    q_star[g] = b_hat
    q_star[v] = (1 - b_hat) * r[v] for v != g

``m_tar`` must already be an approved fact-value mask. This module never builds
masks from proposed spans, relation labels, or likelihood heuristics. There is
no per-token ``w_t`` classifier and no top-1 runner-up flattening on the main
path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DardAsuV2Result:
    loss: torch.Tensor
    asu_other_loss: torch.Tensor
    target_loss: torch.Tensor
    answer_count: int
    target_count: int
    other_count: int
    gold_cap_probability: torch.Tensor
    student_gold_probability: torch.Tensor
    b_hat: torch.Tensor
    q_star: torch.Tensor


def shifted_answer_mask(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("labels must have shape [batch, sequence] with sequence >= 2")
    return labels[:, 1:].ne(ignore_index)


def _validate(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
    k: int,
    delta: float,
    eta: float,
) -> torch.Tensor:
    if student_logits.ndim != 3 or student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must share [batch, sequence, vocab]")
    if base_logits.shape != student_logits.shape:
        raise ValueError("base logits must have the same shape as student logits")
    if labels.shape != student_logits.shape[:2]:
        raise ValueError("labels must have shape [batch, sequence]")
    expected = (labels.shape[0], labels.shape[1] - 1)
    if target_mask.shape != expected:
        raise ValueError(f"target_mask must have shifted query shape {expected}")
    vocab = student_logits.shape[-1]
    if vocab < 2:
        raise ValueError("vocabulary size must be at least 2")
    if not 1 <= int(k) <= vocab - 1:
        raise ValueError(f"k must be in [1, vocab-1]={vocab - 1}, got {k}")
    if float(delta) <= 0:
        raise ValueError("delta must be strictly positive")
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    answer_mask = shifted_answer_mask(labels)
    target_mask = target_mask.bool()
    if torch.any(target_mask & ~answer_mask):
        raise ValueError("target_mask must be a subset of the valid answer mask")
    return answer_mask


def _active_rows(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return logits[:, :-1, :][mask], labels[:, 1:][mask]


def kth_nongold_logits(
    teacher_logits: torch.Tensor,
    gold: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    """Return the Kth-largest logit among V \\ {g} for each row."""
    hidden = teacher_logits.clone()
    hidden.scatter_(1, gold[:, None], torch.finfo(hidden.dtype).min)
    return torch.topk(hidden, k=k, dim=-1).values[:, -1]


def deflated_teacher_logits(
    teacher_logits: torch.Tensor,
    gold: torch.Tensor,
    *,
    k: int,
    delta: float,
) -> torch.Tensor:
    """Cap gold at or below the Kth non-gold logit minus delta. Never raise gold."""
    edited = teacher_logits.clone()
    kth = kth_nongold_logits(teacher_logits, gold, k=k)
    gold_logit = edited.gather(1, gold[:, None]).squeeze(1)
    capped = torch.minimum(gold_logit, kth - float(delta))
    edited.scatter_(1, gold[:, None], capped[:, None])
    return edited


def gold_cap_probability(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    k: int,
    delta: float,
) -> torch.Tensor:
    """Detached gold mass after rank deflation on target positions."""
    rows, gold = _active_rows(teacher_logits, labels, target_mask)
    if rows.shape[0] == 0:
        return rows.new_empty((0,))
    edited = deflated_teacher_logits(rows.detach(), gold, k=k, delta=delta)
    return F.softmax(edited, dim=-1).gather(1, gold[:, None]).squeeze(1).detach()


def conditional_non_gold_probs(
    logits: torch.Tensor,
    gold: torch.Tensor,
) -> torch.Tensor:
    """Softmax over V \\ {g}, with exact zero mass on the gold index."""
    masked = logits.clone()
    masked.scatter_(1, gold[:, None], torch.finfo(masked.dtype).min)
    probs = F.softmax(masked, dim=-1)
    probs = probs.clone()
    probs.scatter_(1, gold[:, None], 0.0)
    return probs


def mix_non_gold(
    teacher_logits: torch.Tensor,
    base_logits: torch.Tensor,
    gold: torch.Tensor,
    *,
    eta: float,
) -> torch.Tensor:
    """Probability-space mix r = (1-eta) q_T + eta q_B on V \\ {g}."""
    q_t = conditional_non_gold_probs(teacher_logits, gold)
    q_b = conditional_non_gold_probs(base_logits, gold)
    mixed = (1.0 - float(eta)) * q_t + float(eta) * q_b
    mixed = mixed.clone()
    mixed.scatter_(1, gold[:, None], 0.0)
    return mixed


def build_q_star(
    gold_cap: torch.Tensor,
    student_gold: torch.Tensor,
    non_gold_mix: torch.Tensor,
    gold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached (b_hat, q_star) with sum_v q_star[v] = 1."""
    b_hat = torch.minimum(gold_cap.detach(), student_gold.detach())
    q_star = (1.0 - b_hat)[:, None] * non_gold_mix
    q_star = q_star.clone()
    q_star.scatter_(1, gold[:, None], b_hat[:, None])
    return b_hat.detach(), q_star.detach()


def full_vocab_kl_from_target(
    student_logits: torch.Tensor,
    q_star: torch.Tensor,
) -> torch.Tensor:
    """Per-row KL(stopgrad(q_star) || softmax(z_S)) with safe zero-target terms."""
    log_student = F.log_softmax(student_logits, dim=-1)
    # xlogy(0, ·) = 0; avoid clamping q_star and renormalizing.
    return torch.xlogy(q_star, q_star).sum(dim=-1) - (q_star * log_student).sum(dim=-1)


def token_asu_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> torch.Tensor:
    """Per-token KL(p_T || p_S) on already-shifted [batch, positions, vocab]."""
    log_student = F.log_softmax(student_logits, dim=-1)
    log_teacher = F.log_softmax(teacher_logits, dim=-1)
    return F.kl_div(log_student, log_teacher, reduction="none", log_target=True).sum(-1)


def _conditional_from_probs(probs: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    support = probs.clone()
    support.scatter_(1, gold[:, None], 0.0)
    denom = support.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probs.dtype).tiny)
    out = support / denom
    out.scatter_(1, gold[:, None], 0.0)
    return out


def kl_binary_and_conditional_decomposition(
    p_teacher: torch.Tensor,
    p_student: torch.Tensor,
    gold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return full KL, Bernoulli KL on gold event, and conditional non-gold KL."""
    p_tg = p_teacher.gather(1, gold[:, None]).squeeze(1)
    p_sg = p_student.gather(1, gold[:, None]).squeeze(1)
    bernoulli = torch.xlogy(p_tg, p_tg) - torch.xlogy(p_tg, p_sg)
    bernoulli = bernoulli + torch.xlogy(1.0 - p_tg, 1.0 - p_tg) - torch.xlogy(1.0 - p_tg, 1.0 - p_sg)

    q_t = _conditional_from_probs(p_teacher, gold)
    q_s = _conditional_from_probs(p_student, gold)
    conditional = (torch.xlogy(q_t, q_t) - torch.xlogy(q_t, q_s)).sum(dim=-1)
    full = (torch.xlogy(p_teacher, p_teacher) - torch.xlogy(p_teacher, p_student)).sum(dim=-1)
    return full, bernoulli, conditional


def dard_asu_v2_forget_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    k: int,
    delta: float,
    eta: float,
) -> DardAsuV2Result:
    """Evaluate the v2 candidate forget objective on synthetic or approved masks.

    ``target_mask`` aligns to ``logits[:, :-1]``. Reduction is the global answer
    mean from the v2 spec, which differs from ``asu_loss`` (per-sequence then
    batch mean).
    """
    answer_mask = _validate(
        student_logits,
        teacher_logits,
        base_logits,
        labels,
        target_mask,
        k,
        delta,
        eta,
    )
    target_mask = target_mask.bool()
    other_mask = answer_mask & ~target_mask
    answer_count = int(answer_mask.sum().item())
    target_count = int(target_mask.sum().item())
    other_count = int(other_mask.sum().item())

    if answer_count == 0:
        zero = student_logits.sum() * 0.0
        empty = student_logits.new_empty((0,))
        return DardAsuV2Result(
            zero,
            zero,
            zero,
            0,
            0,
            0,
            empty,
            empty,
            empty,
            student_logits.new_empty((0, student_logits.shape[-1])),
        )

    student_shift = student_logits[:, :-1, :]
    teacher_shift = teacher_logits[:, :-1, :]
    base_shift = base_logits[:, :-1, :]

    asu_tokens = token_asu_kl(student_shift, teacher_shift)
    asu_other_sum = (asu_tokens * other_mask.to(asu_tokens.dtype)).sum()

    if target_count == 0:
        denom = float(answer_count)
        total = asu_other_sum / denom
        empty = student_logits.new_empty((0,))
        return DardAsuV2Result(
            total,
            asu_other_sum / denom,
            student_logits.sum() * 0.0,
            answer_count,
            0,
            other_count,
            empty,
            empty,
            empty,
            student_logits.new_empty((0, student_logits.shape[-1])),
        )

    student_rows, gold = _active_rows(student_logits, labels, target_mask)
    teacher_rows, teacher_gold = _active_rows(teacher_logits, labels, target_mask)
    base_rows, base_gold = _active_rows(base_logits, labels, target_mask)
    if not torch.equal(gold, teacher_gold) or not torch.equal(gold, base_gold):
        raise RuntimeError("gold IDs are misaligned across student/teacher/base rows")

    edited_teacher = deflated_teacher_logits(teacher_rows.detach(), gold, k=k, delta=delta)
    a = F.softmax(edited_teacher, dim=-1).gather(1, gold[:, None]).squeeze(1)
    student_probs = F.softmax(student_rows, dim=-1)
    b = student_probs.gather(1, gold[:, None]).squeeze(1)
    r = mix_non_gold(teacher_rows.detach(), base_rows.detach(), gold, eta=eta)
    b_hat, q_star = build_q_star(a, b, r, gold)
    target_tokens = full_vocab_kl_from_target(student_rows, q_star)
    target_sum = target_tokens.sum()

    denom = float(answer_count)
    total = (asu_other_sum + target_sum) / denom
    return DardAsuV2Result(
        total,
        asu_other_sum / denom,
        target_sum / denom,
        answer_count,
        target_count,
        other_count,
        a.detach(),
        b.detach(),
        b_hat,
        q_star,
    )


# Backward-compatible names used by older imports in this workspace.
SplitChannelResult = DardAsuV2Result


def split_channel_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    base_logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    k: int,
    delta: float,
    eta: float = 0.0,
    rho=None,
    **_ignored,
) -> DardAsuV2Result:
    """Alias for ``dard_asu_v2_forget_loss``.

    ``rho`` is accepted only as a deprecated synonym for ``eta`` and must not be
    interpreted as a uniform-mix coefficient.
    """
    if rho is not None:
        eta = rho
    return dard_asu_v2_forget_loss(
        student_logits,
        teacher_logits,
        base_logits,
        labels,
        target_mask,
        k=k,
        delta=delta,
        eta=eta,
    )


def cap_targets(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    k: int,
    delta: float,
) -> torch.Tensor:
    return gold_cap_probability(
        teacher_logits,
        labels,
        target_mask,
        k=k,
        delta=delta,
    )


def conditional_non_gold_log_probs(
    logits: torch.Tensor,
    gold: torch.Tensor,
) -> torch.Tensor:
    probs = conditional_non_gold_probs(logits, gold)
    log_probs = torch.where(probs > 0, probs.log(), torch.zeros_like(probs))
    return log_probs

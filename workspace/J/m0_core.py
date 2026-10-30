"""Pure-torch helpers for the DARD-ASU M0 teacher-forcing diagnostic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GradientCheck:
    checked_tokens: int
    negative_autograd_tokens: int
    negative_analytic_tokens: int
    max_abs_error: float
    passed: bool


def shifted_answer_mask(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Return the ASU valid-query mask for logits that predict the next label."""
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("labels must have shape [batch, sequence] with sequence >= 2")
    return labels[:, 1:].ne(ignore_index)


def masked_asu_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Match ``trainer.losses.asu_loss`` with its per-sequence batch reduction."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if student_logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab]")

    student_logits = student_logits[:, :-1, :]
    teacher_logits = teacher_logits[:, :-1, :]
    mask = shifted_answer_mask(labels, ignore_index)
    if (mask.sum(dim=1) == 0).any():
        raise ValueError("each checked sequence needs at least one valid answer token")

    log_student = F.log_softmax(student_logits, dim=-1)
    log_teacher = F.log_softmax(teacher_logits, dim=-1)
    token_kl = F.kl_div(log_student, log_teacher, reduction="none", log_target=True).sum(-1)
    return ((token_kl * mask).sum(dim=-1) / mask.sum(dim=-1)).mean()


def gold_probabilities(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return probabilities and labels at valid ASU answer query positions."""
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_answer_mask(labels)
    probabilities = F.softmax(shifted_logits.float(), dim=-1)
    gold = shifted_labels.masked_fill(~mask, 0)
    return probabilities.gather(-1, gold[..., None]).squeeze(-1)[mask], gold[mask]


def gold_ranks(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return 1-based gold ranks using token ID as a stable tie breaker."""
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_answer_mask(labels)
    active_logits = shifted_logits[mask]
    gold = shifted_labels[mask]
    gold_logits = active_logits.gather(1, gold[:, None])
    vocab_ids = torch.arange(active_logits.shape[-1], device=active_logits.device)[None, :]
    higher = active_logits > gold_logits
    tied_with_smaller_id = (active_logits == gold_logits) & (vocab_ids < gold[:, None])
    return (higher | tied_with_smaller_id).sum(dim=-1).add(1)


def analytic_gold_gradient(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Gold derivatives for the exact masked ASU batch loss at valid positions."""
    mask = shifted_answer_mask(labels, ignore_index)
    student_probs = F.softmax(student_logits[:, :-1, :].float(), dim=-1)
    teacher_probs = F.softmax(teacher_logits[:, :-1, :].float(), dim=-1)
    gold = labels[:, 1:].masked_fill(~mask, 0)
    gold_student = student_probs.gather(-1, gold[..., None]).squeeze(-1)
    gold_teacher = teacher_probs.gather(-1, gold[..., None]).squeeze(-1)
    per_sequence_counts = mask.sum(dim=-1, keepdim=True).float()
    if (per_sequence_counts == 0).any():
        raise ValueError("each checked sequence needs at least one valid answer token")
    derivative = (gold_student - gold_teacher) / per_sequence_counts / labels.shape[0]
    return derivative[mask]


def autograd_gradient_check(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    atol: float = 2e-5,
    rtol: float = 2e-3,
) -> GradientCheck:
    """Check the actual PyTorch KL gradient against the derived gold gradient."""
    checked_logits = student_logits.detach().float().requires_grad_(True)
    loss = masked_asu_kl(checked_logits, teacher_logits.detach().float(), labels)
    (gradient,) = torch.autograd.grad(loss, checked_logits)
    mask = shifted_answer_mask(labels)
    gold = labels[:, 1:]
    actual = gradient[:, :-1, :][mask, gold[mask]]
    expected = analytic_gold_gradient(checked_logits.detach(), teacher_logits.detach(), labels)
    error = (actual - expected).abs()
    return GradientCheck(
        checked_tokens=int(actual.numel()),
        negative_autograd_tokens=int((actual < 0).sum().item()),
        negative_analytic_tokens=int((expected < 0).sum().item()),
        max_abs_error=float(error.max().item()) if error.numel() else 0.0,
        passed=bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
    )

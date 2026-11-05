"""Teacher-forced answer scores and the masked retain NLL used by ASU."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def shifted_answer_mask(labels, ignore_index=-100):
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("labels must have shape [batch, sequence] with sequence >= 2")
    return labels[:, 1:].ne(ignore_index)


def sequence_logprob(logits, labels, ignore_index=-100):
    """Mean log p(y_t | q, y_<t) over answer tokens. Prompt and pad stay at -100."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels disagree on batch or sequence length")
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(ignore_index)
    if (valid.sum(dim=-1) == 0).any():
        raise ValueError("a sequence has no valid answer tokens")
    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (gathered * valid).sum(dim=-1) / valid.sum(dim=-1)


def advantage(logits_pos, labels_pos, logits_neg=None, labels_neg=None, foil_reliable=False):
    positive = sequence_logprob(logits_pos, labels_pos)
    if not foil_reliable or logits_neg is None or labels_neg is None:
        return positive, "positive_only"
    negative = sequence_logprob(logits_neg, labels_neg)
    return positive - negative, "margin"


def masked_token_nll(logits, labels, ignore_index=-100):
    """Mean cross-entropy on shifted answer tokens, matching ignore_index -100."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels disagree on batch or sequence length")
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    if shift_labels.ne(ignore_index).sum() == 0:
        raise ValueError("retain labels have no valid answer tokens")
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )


def answer_token_distributions(logits, labels, ignore_index=-100):
    """Detached full-vocab probabilities at the answer positions the KL uses."""
    shift_logits = logits[:, :-1, :].float()
    valid = labels[:, 1:].ne(ignore_index)
    probs = torch.softmax(shift_logits, dim=-1).detach()
    rows = []
    for row in range(probs.shape[0]):
        rows.append(probs[row][valid[row]].cpu().tolist())
    return rows

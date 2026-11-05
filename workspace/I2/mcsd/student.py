"""Candidate ASU loss. Student E1 stays blocked unless a new gate authorizes it."""

from __future__ import annotations

import sys
from pathlib import Path

from I2.mcsd.errors import ConfigError, StudentE1Blocked
from I2.mcsd.gates import student_block_reason
from I2.mcsd.score import masked_token_nll

_J = Path(__file__).resolve().parents[2] / "J"
if str(_J) not in sys.path:
    sys.path.insert(0, str(_J))

from m0_core import masked_asu_kl  # noqa: E402


def candidate_asu_loss(student_logits, teacher_logits, forget_labels, retain_logits, retain_labels, alpha):
    """L = alpha * mean KL(stopgrad(p_T) || p_S) + masked retain NLL."""
    if alpha is None:
        raise ConfigError("alpha is missing")
    teacher = teacher_logits.detach()
    kl = masked_asu_kl(student_logits, teacher, forget_labels)
    retain_nll = masked_token_nll(retain_logits, retain_labels)
    total = (float(alpha) * kl) + retain_nll
    return total, kl, retain_nll


def authorize_loss_gate(config, teacher_logits):
    reason = student_block_reason(config)
    if reason:
        raise StudentE1Blocked(reason)
    if teacher_logits is None:
        raise StudentE1Blocked("frozen teacher distribution is missing")
    authorized = dict(config)
    authorized["loss_authorized"] = True
    authorized["teacher_logits"] = teacher_logits
    return authorized


def guarded_mcsd_from_loss_api(model, ref_model, inputs, gate):
    """ASU loss dispatcher entry. Refuses before any forward when the gate is shut."""
    del ref_model
    reason = student_block_reason(gate if isinstance(gate, dict) else None)
    if not isinstance(gate, dict) or gate.get("loss_authorized") is not True or reason:
        if reason is None:
            reason = "MCSD_ASU loss gate is not authorized"
        raise StudentE1Blocked(reason)
    teacher_logits = gate.get("teacher_logits")
    if teacher_logits is None:
        raise StudentE1Blocked("frozen teacher distribution is missing")
    forget_ids, forget_labels, forget_mask = inputs[0]
    retain_ids, retain_labels, retain_mask = inputs[1]
    student_out = model(input_ids=forget_ids, labels=forget_labels, attention_mask=forget_mask)
    retain_out = model(input_ids=retain_ids, labels=retain_labels, attention_mask=retain_mask)
    _total, kl, retain_nll = candidate_asu_loss(
        student_out.logits,
        teacher_logits,
        forget_labels,
        retain_out.logits,
        retain_labels,
        gate.get("alpha"),
    )
    return float(gate["alpha"]) * kl, retain_nll

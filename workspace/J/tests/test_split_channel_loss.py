import sys
from pathlib import Path

import torch
import torch.nn.functional as F

J_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(J_DIR))

from m0_core import masked_asu_kl, shifted_answer_mask  # noqa: E402
from split_channel_loss import (  # noqa: E402
    cap_targets,
    conditional_non_gold_log_probs,
    split_channel_loss,
)


def _one_target(values):
    row = torch.tensor(values, dtype=torch.float64)
    return torch.stack((row, torch.zeros_like(row))).unsqueeze(0)


def test_float64_kl_direction_and_hand_gradient():
    student = _one_target([0.2, -0.1, 0.4])
    teacher = _one_target([-0.3, 0.5, 0.1])
    labels = torch.tensor([[-100, 1]])
    checked = student.clone().requires_grad_(True)

    loss = masked_asu_kl(checked, teacher, labels)
    p_student = F.softmax(checked[0, 0], dim=-1)
    p_teacher = F.softmax(teacher[0, 0], dim=-1)
    hand_loss = (p_teacher * (p_teacher.log() - p_student.log())).sum()
    assert torch.allclose(loss, hand_loss, atol=1e-14, rtol=1e-14)

    (gradient,) = torch.autograd.grad(loss, checked)
    assert torch.allclose(
        gradient[0, 0],
        p_student - p_teacher,
        atol=1e-14,
        rtol=1e-14,
    )


def test_shift_and_answer_mask_exclude_prompt_and_padding():
    labels = torch.tensor([[-100, -100, 4, 5, -100]])
    expected = torch.tensor([[False, True, True, False]])
    assert torch.equal(shifted_answer_mask(labels), expected)


def test_cap_uses_kth_non_gold_and_positive_margin():
    teacher = _one_target([10.0, 8.0, 7.0, 6.0])
    labels = torch.tensor([[-100, 0]])
    target_mask = torch.tensor([[True]])
    cap = cap_targets(teacher, labels, target_mask, k=2, delta=0.5)

    edited = torch.tensor([6.5, 8.0, 7.0, 6.0], dtype=torch.float64)
    expected = F.softmax(edited, dim=-1)[0]
    assert torch.allclose(cap, expected.reshape(1), atol=1e-14, rtol=1e-14)
    assert edited[0] < torch.topk(edited[1:], k=2).values[-1]


def test_cap_gate_is_one_sided_and_has_expected_gold_gradient():
    teacher = _one_target([0.0, 2.0, 1.0, 0.5])
    labels = torch.tensor([[-100, 0]])
    target_mask = torch.tensor([[True]])
    base = teacher.clone()

    active_student = _one_target([4.0, 0.0, 0.0, 0.0]).requires_grad_(True)
    active = split_channel_loss(
        active_student,
        teacher,
        base,
        labels,
        target_mask,
        k=1,
        delta=0.5,
        rho=0.0,
        beta=1.0,
        lambda_ng=0.0,
    )
    (active_gradient,) = torch.autograd.grad(active.loss, active_student)
    assert bool(active.cap_gate.item())
    expected = active.student_gold_probability - active.cap_probability
    assert torch.allclose(active_gradient[0, 0, 0], expected[0], atol=1e-12, rtol=1e-12)

    inactive_student = _one_target([-4.0, 2.0, 1.0, 0.5]).requires_grad_(True)
    inactive = split_channel_loss(
        inactive_student,
        teacher,
        base,
        labels,
        target_mask,
        k=1,
        delta=0.5,
        rho=0.0,
        beta=1.0,
        lambda_ng=0.0,
    )
    (inactive_gradient,) = torch.autograd.grad(inactive.loss, inactive_student)
    assert not bool(inactive.cap_gate.item())
    assert torch.equal(inactive_gradient, torch.zeros_like(inactive_gradient))


def test_non_gold_distribution_normalizes_and_has_zero_gold_gradient():
    student = _one_target([3.0, 0.2, -0.4, 1.1]).requires_grad_(True)
    teacher = _one_target([0.0, 2.0, 1.0, 0.5])
    base = _one_target([-1.0, 0.3, 0.8, -0.2])
    labels = torch.tensor([[-100, 0]])
    target_mask = torch.tensor([[True]])

    log_q = conditional_non_gold_log_probs(student[0, :1], torch.tensor([0]))
    assert torch.allclose(log_q.exp()[0, 1:].sum(), torch.tensor(1.0, dtype=torch.float64))

    result = split_channel_loss(
        student,
        teacher,
        base,
        labels,
        target_mask,
        k=1,
        delta=0.5,
        rho=0.35,
        beta=0.0,
        lambda_ng=1.0,
    )
    (gradient,) = torch.autograd.grad(result.loss, student)
    assert gradient[0, 0, 0].item() == 0.0

    q_student = log_q.exp()[0, 1:]
    q_base = conditional_non_gold_log_probs(base[0, :1], torch.tensor([0])).exp()[0, 1:]
    uniform = torch.full_like(q_base, 1.0 / 3.0)
    target = 0.65 * q_base + 0.35 * uniform
    expected = (1.0 - result.cap_probability[0]) * (q_student - target)
    assert torch.allclose(gradient[0, 0, 1:], expected, atol=1e-12, rtol=1e-12)


def test_rho_endpoints_and_base_target_are_stop_gradient():
    student = _one_target([0.0, 0.2, -0.4, 1.1])
    teacher = _one_target([0.0, 2.0, 1.0, 0.5])
    base = _one_target([-1.0, 0.3, 0.8, -0.2]).requires_grad_(True)
    labels = torch.tensor([[-100, 0]])
    target_mask = torch.tensor([[True]])

    anchored = split_channel_loss(
        student,
        teacher,
        base,
        labels,
        target_mask,
        k=1,
        delta=0.5,
        rho=0.0,
        beta=0.0,
    )
    uniform = split_channel_loss(
        student,
        teacher,
        base,
        labels,
        target_mask,
        k=1,
        delta=0.5,
        rho=1.0,
        beta=0.0,
    )
    assert anchored.non_gold_loss >= 0
    assert uniform.non_gold_loss >= 0
    assert not anchored.loss.requires_grad
    assert base.grad is None


def test_target_mask_must_be_answer_subset_and_empty_is_safe():
    logits = torch.zeros((1, 3, 4), dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[-100, 1, -100]])
    invalid = torch.tensor([[True, True]])
    try:
        split_channel_loss(logits, logits.detach(), logits.detach(), labels, invalid, k=1, delta=0.1, rho=0.0)
    except ValueError as error:
        assert "subset" in str(error)
    else:
        raise AssertionError("invalid target mask was accepted")

    empty = torch.zeros((1, 2), dtype=torch.bool)
    result = split_channel_loss(
        logits,
        logits.detach(),
        logits.detach(),
        labels,
        empty,
        k=1,
        delta=0.1,
        rho=0.0,
    )
    result.loss.backward()
    assert result.target_count == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))

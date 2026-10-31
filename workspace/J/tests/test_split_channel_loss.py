"""Synthetic-mask unit tests for DARD-ASU v2 candidate math.

Every target mask here is hand-built. These tests do not read pilot spans and
do not claim a real approved ``m_tar`` exists.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

J_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(J_DIR))

from m0_core import masked_asu_kl, shifted_answer_mask  # noqa: E402
from split_channel_loss import (  # noqa: E402
    build_q_star,
    conditional_non_gold_probs,
    dard_asu_v2_forget_loss,
    deflated_teacher_logits,
    gold_cap_probability,
    kl_binary_and_conditional_decomposition,
    mix_non_gold,
    token_asu_kl,
)

SYNTHETIC_MASK = True
TOL = dict(atol=1e-12, rtol=1e-12)


def _batch_one(values, dtype=torch.float64):
    """[1, 2, V] logits with the first query used and the second padded."""
    row = torch.tensor(values, dtype=dtype)
    pad = torch.zeros_like(row)
    return torch.stack((row, pad)).unsqueeze(0)


class DardAsuV2Tests(unittest.TestCase):
    def test_causal_shift_and_synthetic_masks(self):
        """synthetic_mask=True: prompt/pad/relation-only/unreviewed stay out of m_tar."""
        self.assertTrue(SYNTHETIC_MASK)
        labels = torch.tensor([[-100, -100, 4, 5, -100]])
        m_ans = shifted_answer_mask(labels)
        self.assertTrue(torch.equal(m_ans, torch.tensor([[False, True, True, False]])))

        # Synthetic: only the first answer token is a reviewed fact value.
        m_tar = torch.tensor([[False, True, False, False]])
        # Synthetic relation-only / unreviewed second answer token stays in m_other.
        m_other = m_ans & ~m_tar
        self.assertTrue(torch.equal(m_other, torch.tensor([[False, False, True, False]])))
        self.assertFalse(bool((m_tar & ~m_ans).any()))

        invalid = torch.tensor([[True, True, False, False]])
        with self.assertRaisesRegex(ValueError, "subset"):
            dard_asu_v2_forget_loss(
                torch.zeros((1, 5, 6), dtype=torch.float64),
                torch.zeros((1, 5, 6), dtype=torch.float64),
                torch.zeros((1, 5, 6), dtype=torch.float64),
                labels,
                invalid,
                k=1,
                delta=0.5,
                eta=0.0,
            )

    def test_rank_deflation_excludes_gold_and_uses_margin(self):
        self.assertTrue(SYNTHETIC_MASK)
        teacher = _batch_one([10.0, 8.0, 7.0, 6.0])
        gold = torch.tensor([0])
        edited = deflated_teacher_logits(teacher[0, :1], gold, k=2, delta=0.5)
        # Cap to min(10, 7 - 0.5) = 6.5; at least two non-gold logits strictly above gold.
        self.assertAlmostEqual(float(edited[0, 0]), 6.5, places=12)
        non_gold = edited[0, 1:]
        self.assertGreaterEqual(int((non_gold > edited[0, 0]).sum()), 2)

        # Already below the threshold: do not raise gold.
        low = _batch_one([5.0, 8.0, 7.0, 6.0])
        edited_low = deflated_teacher_logits(low[0, :1], gold, k=2, delta=0.5)
        self.assertAlmostEqual(float(edited_low[0, 0]), 5.0, places=12)

        labels = torch.tensor([[-100, 0]])
        target_mask = torch.tensor([[True]])
        cap = gold_cap_probability(teacher, labels, target_mask, k=2, delta=0.5)
        expected = F.softmax(torch.tensor([6.5, 8.0, 7.0, 6.0], dtype=torch.float64), dim=-1)[0]
        self.assertTrue(torch.allclose(cap, expected.reshape(1), **TOL))

        with self.assertRaisesRegex(ValueError, "k must be"):
            dard_asu_v2_forget_loss(teacher, teacher, teacher, labels, target_mask, k=4, delta=0.5, eta=0.0)
        with self.assertRaisesRegex(ValueError, "k must be"):
            dard_asu_v2_forget_loss(teacher, teacher, teacher, labels, target_mask, k=0, delta=0.5, eta=0.0)
        with self.assertRaisesRegex(ValueError, "delta"):
            dard_asu_v2_forget_loss(teacher, teacher, teacher, labels, target_mask, k=1, delta=0.0, eta=0.0)

    def test_conditional_mix_eta_endpoints_and_gold_ids(self):
        self.assertTrue(SYNTHETIC_MASK)
        teacher = torch.tensor([[0.0, 2.0, 1.0, -1.0]], dtype=torch.float64)
        base = torch.tensor([[-2.0, 0.5, 3.0, 0.0]], dtype=torch.float64)
        for gold_id in (0, 2, 3):
            gold = torch.tensor([gold_id])
            q_t = conditional_non_gold_probs(teacher, gold)
            q_b = conditional_non_gold_probs(base, gold)
            self.assertAlmostEqual(float(q_t.sum()), 1.0, places=12)
            self.assertAlmostEqual(float(q_b.sum()), 1.0, places=12)
            self.assertEqual(float(q_t[0, gold_id]), 0.0)
            self.assertEqual(float(q_b[0, gold_id]), 0.0)

            r0 = mix_non_gold(teacher, base, gold, eta=0.0)
            r1 = mix_non_gold(teacher, base, gold, eta=1.0)
            r05 = mix_non_gold(teacher, base, gold, eta=0.5)
            self.assertTrue(torch.allclose(r0, q_t, **TOL))
            self.assertTrue(torch.allclose(r1, q_b, **TOL))
            self.assertTrue(torch.allclose(r05, 0.5 * q_t + 0.5 * q_b, **TOL))
            self.assertAlmostEqual(float(r05.sum()), 1.0, places=12)

    def test_q_star_reconstruction(self):
        self.assertTrue(SYNTHETIC_MASK)
        gold = torch.tensor([1])
        a = torch.tensor([0.4], dtype=torch.float64)
        b = torch.tensor([0.7], dtype=torch.float64)
        r = torch.tensor([[0.2, 0.0, 0.5, 0.3]], dtype=torch.float64)
        b_hat, q_star = build_q_star(a, b, r, gold)
        self.assertAlmostEqual(float(b_hat), 0.4, places=12)
        self.assertAlmostEqual(float(q_star[0, 1]), 0.4, places=12)
        self.assertTrue(torch.all(q_star >= 0))
        self.assertTrue(torch.isfinite(q_star).all())
        self.assertAlmostEqual(float(q_star.sum()), 1.0, places=12)
        recovered = q_star[0].clone()
        recovered[1] = 0.0
        recovered = recovered / (1.0 - b_hat)
        self.assertTrue(torch.allclose(recovered, r[0], **TOL))

    def test_exact_kl_decomposition(self):
        self.assertTrue(SYNTHETIC_MASK)
        torch.manual_seed(0)
        logits_t = torch.randn(8, 5, dtype=torch.float64)
        logits_s = torch.randn(8, 5, dtype=torch.float64)
        gold = torch.randint(0, 5, (8,))
        p_t = F.softmax(logits_t, dim=-1)
        p_s = F.softmax(logits_s, dim=-1)
        full, bernoulli, conditional = kl_binary_and_conditional_decomposition(p_t, p_s, gold)
        reconstructed = bernoulli + (1.0 - p_t.gather(1, gold[:, None]).squeeze(1)) * conditional
        self.assertTrue(torch.allclose(full, reconstructed, atol=1e-10, rtol=1e-10))

        # Single-gold event: teacher puts all mass on gold => conditional term unused.
        one_hot = torch.zeros_like(p_t)
        one_hot.scatter_(1, gold[:, None], 1.0)
        full_g, bernoulli_g, conditional_g = kl_binary_and_conditional_decomposition(one_hot, p_s, gold)
        self.assertTrue(torch.allclose(full_g, bernoulli_g, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.isfinite(conditional_g).all())

    def test_one_sided_direct_gold_logit_gradient(self):
        """Direct gold-logit gradient under a detached q_star for the current backward.

        This does not claim gold probability is invariant after a parameter step.
        """
        self.assertTrue(SYNTHETIC_MASK)
        teacher = _batch_one([0.0, 2.0, 1.0, 0.5])
        base = teacher.clone()
        labels = torch.tensor([[-100, 0]])
        target_mask = torch.tensor([[True]])

        # b > a: expect dL/dz_S[g] = b - a > 0
        high = _batch_one([4.0, 0.0, 0.0, 0.0]).requires_grad_(True)
        active = dard_asu_v2_forget_loss(high, teacher, base, labels, target_mask, k=1, delta=0.5, eta=0.0)
        (grad_high,) = torch.autograd.grad(active.loss, high)
        expected_high = active.student_gold_probability - active.gold_cap_probability
        self.assertGreater(float(expected_high), 0.0)
        self.assertTrue(torch.allclose(grad_high[0, 0, 0], expected_high[0], **TOL))

        # b < a: expect direct gold-logit gradient ~ 0
        low = _batch_one([-4.0, 2.0, 1.0, 0.5]).requires_grad_(True)
        inactive = dard_asu_v2_forget_loss(low, teacher, base, labels, target_mask, k=1, delta=0.5, eta=0.0)
        (grad_low,) = torch.autograd.grad(inactive.loss, low)
        self.assertLess(float(inactive.student_gold_probability), float(inactive.gold_cap_probability))
        self.assertAlmostEqual(float(grad_low[0, 0, 0]), 0.0, places=10)

        # Near equality: gradient approaches 0 when student matches the capped teacher.
        near_logits = deflated_teacher_logits(teacher[0, :1].detach(), torch.tensor([0]), k=1, delta=0.5)
        near = _batch_one(near_logits[0].tolist()).requires_grad_(True)
        border = dard_asu_v2_forget_loss(near, teacher, base, labels, target_mask, k=1, delta=0.5, eta=0.0)
        (grad_border,) = torch.autograd.grad(border.loss, near)
        self.assertLess(abs(float(grad_border[0, 0, 0])), 1e-6)

    def test_loss_routing_other_uses_asu_and_empty_is_safe(self):
        self.assertTrue(SYNTHETIC_MASK)
        # Sequence length 3 queries after shift: prompt, fact target, other answer.
        student = torch.tensor(
            [[[0.1, 0.0, -0.2], [2.0, -1.0, 0.0], [0.0, 1.5, -0.5], [0.0, 0.0, 0.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[[0.0, 0.2, -0.1], [-1.0, 2.0, 0.5], [1.0, 0.0, -0.5], [0.0, 0.0, 0.0]]],
            dtype=torch.float64,
        )
        base = teacher.clone()
        labels = torch.tensor([[-100, 1, 0, -100]])
        m_ans = shifted_answer_mask(labels)
        m_tar = torch.tensor([[False, True, False]])  # only second answer token
        self.assertTrue(torch.equal(m_ans, torch.tensor([[True, True, False]])))

        result = dard_asu_v2_forget_loss(
            student, teacher, base, labels, m_tar, k=1, delta=0.5, eta=0.0
        )
        asu_tokens = token_asu_kl(student[:, :-1, :], teacher[:, :-1, :])
        other = asu_tokens * (m_ans & ~m_tar).to(asu_tokens.dtype)
        # Hand reconstruction of the global answer mean.
        target_only = dard_asu_v2_forget_loss(
            student,
            teacher,
            base,
            labels,
            m_ans,  # treat all answer tokens as target to isolate machinery
            k=1,
            delta=0.5,
            eta=0.0,
        )
        # For routing check: other contribution equals ASU token at the other position.
        self.assertEqual(result.answer_count, 2)
        self.assertEqual(result.target_count, 1)
        self.assertEqual(result.other_count, 1)
        self.assertTrue(torch.allclose(other.sum() / 2.0, result.asu_other_loss, **TOL))

        # Empty answer mask: all labels ignored.
        empty_labels = torch.tensor([[-100, -100, -100, -100]])
        empty_tar = torch.zeros((1, 3), dtype=torch.bool)
        empty = dard_asu_v2_forget_loss(
            student, teacher, base, empty_labels, empty_tar, k=1, delta=0.5, eta=0.0
        )
        empty.loss.backward()
        self.assertEqual(empty.answer_count, 0)
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))

        # Baseline ASU path remains available and unchanged in reduction style.
        asu = masked_asu_kl(student.detach().requires_grad_(True), teacher, labels)
        self.assertTrue(torch.isfinite(asu))

        # Silence unused local if rewrite changes isolation helper.
        self.assertTrue(torch.isfinite(target_only.loss))

    def test_numerical_stability_and_no_qstar_clamp_inflation(self):
        self.assertTrue(SYNTHETIC_MASK)
        teacher = _batch_one([0.0, 5.0, -2.0, -3.0])
        base = _batch_one([-1.0, 0.0, 4.0, 1.0])
        student = _batch_one([3.0, -4.0, 0.5, 0.1]).requires_grad_(True)
        labels = torch.tensor([[-100, 0]])
        target_mask = torch.tensor([[True]])
        result = dard_asu_v2_forget_loss(
            student, teacher, base, labels, target_mask, k=1, delta=0.25, eta=0.5
        )
        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue(torch.isfinite(result.q_star).all())
        self.assertAlmostEqual(float(result.q_star.sum()), 1.0, places=12)
        # q_star was detached and not elementwise-clamped: mass equals b_hat + (1-b_hat).
        self.assertAlmostEqual(
            float(result.q_star[0, 0]),
            float(result.b_hat[0]),
            places=12,
        )
        result.loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertFalse(bool(torch.isnan(student.grad).any()))


if __name__ == "__main__":
    unittest.main()

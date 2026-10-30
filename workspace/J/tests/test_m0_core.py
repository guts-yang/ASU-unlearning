import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m0_core import (  # noqa: E402
    analytic_gold_gradient,
    autograd_gradient_check,
    gold_probabilities,
    gold_ranks,
    masked_asu_kl,
    shifted_answer_mask,
)


class M0CoreTests(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor([[-100, 1, 2, -100], [-100, 0, 1, -100]])
        self.student = torch.tensor(
            [
                [[0.2, 0.1, 0.0], [0.0, 0.4, 0.1], [0.3, 0.0, 0.2], [0.0, 0.0, 0.0]],
                [[0.1, 0.2, 0.3], [0.2, 0.0, 0.4], [0.4, 0.1, 0.2], [0.0, 0.0, 0.0]],
            ]
        )
        self.teacher = self.student.clone()
        self.teacher[0, 0, 1] += 1.0
        self.teacher[1, 0, 0] += 1.0

    def test_shifted_mask_excludes_prompt_padding(self):
        expected = torch.tensor([[True, True, False], [True, True, False]])
        self.assertTrue(torch.equal(shifted_answer_mask(self.labels), expected))

    def test_kl_gradient_matches_autograd(self):
        check = autograd_gradient_check(self.student, self.teacher, self.labels)
        self.assertTrue(check.passed)
        self.assertEqual(check.checked_tokens, 4)
        self.assertLess(check.max_abs_error, 1e-6)
        analytic = analytic_gold_gradient(self.student, self.teacher, self.labels)
        self.assertEqual(int((analytic < 0).sum()), check.negative_autograd_tokens)
        self.assertGreater(float(masked_asu_kl(self.student, self.teacher, self.labels)), 0.0)

    def test_probability_uses_only_valid_shifted_gold_tokens(self):
        probabilities, gold = gold_probabilities(self.student, self.labels)
        self.assertEqual(probabilities.numel(), 4)
        self.assertTrue(torch.equal(gold, torch.tensor([1, 2, 0, 1])))

    def test_rank_uses_token_id_to_break_ties(self):
        logits = torch.zeros((1, 3, 3))
        labels = torch.tensor([[-100, 1, -100]])
        self.assertTrue(torch.equal(gold_ranks(logits, labels), torch.tensor([2])))

    def test_empty_answer_sequence_is_rejected(self):
        labels = torch.full((1, 3), -100)
        with self.assertRaises(ValueError):
            masked_asu_kl(torch.zeros((1, 3, 2)), torch.zeros((1, 3, 2)), labels)


if __name__ == "__main__":
    unittest.main()

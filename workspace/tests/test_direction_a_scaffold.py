"""CPU unit tests. No GPU, no 7B, no optimizer.step."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from audit.focus_on_key import inject_keytoken, greedy_decode
from audit.go_nogo import DEFERRED, GO, KILL_SWITCH, l4_decision
from audit.match import answer_or_alias_contained, hit_rate
from audit.probab import any_hit_rate, recovery_delta, sample_predictions
from audit.qra_int4 import bitsandbytes_load_kwargs
from audit.rank_profile import order_preserving_leak, profile_logits, temperature_scale
from rank_shatter.logit_bias import VocabLogitBias
from rank_shatter.loss import rank_hinge_loss
from rank_shatter.train_teacher_bias import main as refuse_train_main

_losses_path = REPO / "Right-to-be-forgotten" / "trainer" / "losses.py"
_spec = importlib.util.spec_from_file_location("asu_rtbf_losses", _losses_path)
_losses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_losses)
masked_kl_teacher_to_student = _losses.masked_kl_teacher_to_student
asu_loss = _losses.asu_loss


class FakeLM(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(logits.clone())

    def forward(self, input_ids, labels=None, attention_mask=None, attention_temp=None, layers_id=None):
        class Out:
            pass

        out = Out()
        out.logits = self.logits
        out.loss = self.logits.sum() * 0
        return out


class RankProfileTests(unittest.TestCase):
    def test_order_preserving_flattening_keeps_rank_one_and_raises_entropy(self):
        logits = torch.tensor([[5.0, 1.0, 0.0, -1.0]])
        gold = torch.tensor([0])
        base = profile_logits(logits, gold)
        flat = profile_logits(temperature_scale(logits, tau=4.0), gold)
        self.assertEqual(int(base["rank"][0]), 1)
        self.assertEqual(int(flat["rank"][0]), 1)
        self.assertGreater(float(flat["entropy"][0]), float(base["entropy"][0]))
        self.assertLess(float(flat["gold_prob"][0]), float(base["gold_prob"][0]))
        self.assertTrue(order_preserving_leak(base, flat))


class MatchAndAttackTests(unittest.TestCase):
    def test_literal_alias_casefold(self):
        self.assertTrue(answer_or_alias_contained("The author is Jaime Vasquez.", "Jaime Vasquez"))
        self.assertTrue(answer_or_alias_contained("jaime vasquez wrote it", "JAIME VASQUEZ"))
        self.assertTrue(answer_or_alias_contained("also known as JV", "nobody", aliases=["JV"]))
        self.assertFalse(answer_or_alias_contained("I do not know", "Jaime Vasquez"))

    def test_probab_fake_strings_no_generate(self):
        prompts = ["q1", "q2"]
        answers = ["paris", "rome"]

        def gen(prompt, temperature, seed):
            if prompt == "q1" and seed % 2 == 0:
                return "Paris is the city"
            return "nope"

        rows = sample_predictions(gen, prompts, temperature=1.0, k=2, seed=0)
        attack = any_hit_rate(rows, answers)
        greedy = hit_rate(["nope", "nope"], answers)
        self.assertGreater(recovery_delta(attack, greedy), 0)

    def test_focus_on_key_template_and_greedy_stub(self):
        q = "Where was the author born?"
        prompt = inject_keytoken(q, "Santiago")
        self.assertIn("KeyToken: Santiago", prompt)
        self.assertIn(q, prompt)
        outs = greedy_decode(lambda p, temperature, seed: "Santiago, Chile", [prompt])
        self.assertTrue(answer_or_alias_contained(outs[0], "Santiago"))

    def test_qra_kwargs_do_not_load_model(self):
        nf4 = bitsandbytes_load_kwargs("nf4")
        self.assertTrue(nf4["load_in_4bit"])
        self.assertEqual(nf4["bnb_4bit_quant_type"], "nf4")
        int4 = bitsandbytes_load_kwargs("int4")
        self.assertEqual(int4["bnb_4bit_quant_type"], "fp4")


class L4ProtocolTests(unittest.TestCase):
    def test_deferred_without_real_results(self):
        self.assertEqual(l4_decision({"probab": [1, 1, 1]}, has_real_results=False), DEFERRED)

    def test_go_when_ci_excludes_zero(self):
        deltas = {"probab": [1.0] * 20}
        self.assertEqual(l4_decision(deltas, has_real_results=True, n_boot=200), GO)

    def test_kill_when_all_near_zero(self):
        deltas = {"probab": [0.0] * 10, "focus": [0.0] * 10}
        self.assertEqual(l4_decision(deltas, has_real_results=True, n_boot=200), KILL_SWITCH)


class RankShatterTests(unittest.TestCase):
    def test_hinge_sgd_lowers_gold_logit(self):
        logits = torch.tensor([[3.0, 1.0, 0.0]], requires_grad=True)
        gold = torch.tensor([0])
        loss = rank_hinge_loss(logits, gold, margin=2.0)
        self.assertGreater(loss.detach().item(), 0.0)
        loss.backward()
        # L = relu(m - z_v* + z_gold); SGD therefore lowers z_gold.
        self.assertGreater(float(logits.grad[0, 0].item()), 0.0)
        gold_before = float(logits.detach()[0, 0])
        with torch.no_grad():
            logits -= logits.grad
        self.assertLess(float(logits[0, 0].item()), gold_before)

    def test_masked_positions_do_not_enter_loss(self):
        logits = torch.tensor([[3.0, 0.0], [3.0, 0.0]], requires_grad=True)
        gold = torch.tensor([0, 0])
        mask = torch.tensor([0.0, 1.0])
        loss = rank_hinge_loss(logits, gold, margin=2.0, mask=mask)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.equal(logits.grad[0], torch.zeros(2)))
        self.assertFalse(torch.equal(logits.grad[1], torch.zeros(2)))

    def test_bias_can_shatter_gold_rank(self):
        logits = torch.tensor([[4.0, 1.0, 0.0]])
        gold = torch.tensor([0])
        self.assertEqual(int(profile_logits(logits, gold)["rank"][0]), 1)
        bias = VocabLogitBias(3)
        with torch.no_grad():
            bias.bias.copy_(torch.tensor([-5.0, 0.0, 0.0]))
        shattered = profile_logits(bias.apply(logits), gold)
        self.assertGreater(int(shattered["rank"][0]), 1)

    def test_train_script_refuses_this_round(self):
        self.assertEqual(refuse_train_main([]), 2)


class AsuKlTests(unittest.TestCase):
    def _toy(self):
        torch.manual_seed(0)
        student = torch.randn(2, 4, 5)
        teacher = torch.randn(2, 4, 5)
        labels = torch.tensor(
            [
                [-100, 1, 2, 3],
                [-100, 0, -100, 4],
            ]
        )
        return student, teacher, labels

    def test_matches_legacy_kl_div_formula(self):
        student, teacher, labels = self._toy()
        got, mask = masked_kl_teacher_to_student(student, teacher, labels)
        log_p = F.log_softmax(student[:, :-1, :], dim=-1)
        log_q = F.log_softmax(teacher[:, :-1, :], dim=-1)
        kl = F.kl_div(log_p, log_q, reduction="none", log_target=True).sum(-1)
        expect_mask = labels[:, 1:] != -100
        expect = (kl * expect_mask).sum(-1) / expect_mask.sum(-1).clamp_min(1)
        self.assertTrue(torch.allclose(got, expect.mean()))
        self.assertTrue(torch.equal(mask, expect_mask))

    def test_masked_tokens_have_zero_student_grad(self):
        student, teacher, labels = self._toy()
        student = student.detach().requires_grad_(True)
        loss, mask = masked_kl_teacher_to_student(student, teacher, labels)
        loss.backward()
        shifted = student.grad[:, :-1, :]
        dead = ~mask
        self.assertTrue(torch.equal(shifted[dead], torch.zeros_like(shifted[dead])))
        self.assertFalse(torch.allclose(shifted[mask], torch.zeros_like(shifted[mask])))

    def test_asu_loss_none_bias_matches_unbiased_teacher(self):
        student_logits = torch.randn(1, 3, 4)
        teacher_logits = torch.randn(1, 3, 4)
        labels = torch.tensor([[-100, 1, 2]])
        ids = torch.tensor([[0, 1, 2]])
        mask = torch.ones_like(ids)
        student = FakeLM(student_logits)
        teacher = FakeLM(teacher_logits)
        inputs = [(ids, labels, mask)]
        a = asu_loss(student, teacher, inputs, logit_bias=None)
        zero_bias = VocabLogitBias(4)
        b = asu_loss(student, teacher, inputs, logit_bias=zero_bias)
        self.assertTrue(torch.allclose(a, b))
        expect, _ = masked_kl_teacher_to_student(student_logits, teacher_logits, labels)
        self.assertTrue(torch.allclose(a, expect))

    def test_ignore_first_token_drops_first_label(self):
        student = torch.zeros(1, 4, 3)
        teacher = torch.zeros(1, 4, 3)
        teacher[0, 0, 0] = 10.0
        labels = torch.tensor([[-100, 0, 1, 2]])
        full, mask_full = masked_kl_teacher_to_student(student, teacher, labels, ignore_first_token=False)
        dropped, mask_drop = masked_kl_teacher_to_student(student, teacher, labels, ignore_first_token=True)
        self.assertEqual(int(mask_full.sum()), 3)
        self.assertEqual(int(mask_drop.sum()), 2)
        self.assertFalse(bool(mask_drop[0, 0]))
        self.assertGreater(float(full), float(dropped))


if __name__ == "__main__":
    unittest.main()

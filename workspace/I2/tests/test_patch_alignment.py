"""Patch, score, N/R, and alignment tests on the toy model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from I2.mcsd.alignment import check_alignment  # noqa: E402
from I2.mcsd.effects import interaction_diagnostic, measure_effects  # noqa: E402
from I2.mcsd.hooks import (  # noqa: E402
    ActivationCache,
    ActivationEditor,
    mix_activations,
    to_canonical_attention,
)
from I2.mcsd.errors import HookBoundaryError  # noqa: E402
from I2.mcsd.patch import capture_clean, module_logits, patched_logits  # noqa: E402
from I2.mcsd.score import sequence_logprob  # noqa: E402
from I2.tests.toy_model import ToyCausal  # noqa: E402


def _batch(ids, sample="s1", labels=None):
    data = {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        "sample_ids": [sample],
    }
    if labels is not None:
        data["labels"] = torch.tensor([labels], dtype=torch.long)
    return data


def _example():
    labels = [-100, -100, 0, 0]
    return {
        "sample_id": "s1",
        "author_id": "a0",
        "split": "calibration",
        "question_plus": "fact-plus",
        "question_minus": "fact-minus",
        "fact_span": (0, 1),
        "prompt_plus_ids": [1, 3],
        "prompt_minus_ids": [2, 3],
        "score_plus_ids": [1, 3, 4, 4],
        "score_minus_ids": [2, 3, 4, 4],
        "prompt_plus": _batch([1, 3]),
        "prompt_minus": _batch([2, 3]),
        "score_plus": _batch([1, 3, 4, 4], labels=labels),
        "score_minus": _batch([2, 3, 4, 4], labels=labels),
        "foil_reliable": False,
    }


class _HeadSpec(object):
    layer_idx = 0
    num_heads = 2
    head_dim = 3


class PatchTests(unittest.TestCase):
    def test_mix_boundaries_and_broadcast(self):
        source = torch.zeros(2, 3, 2, 4)
        donor = torch.ones(2, 3, 2, 4)
        self.assertTrue(torch.equal(mix_activations(source, donor, 0), source))
        self.assertTrue(torch.equal(mix_activations(source, donor, 1), donor))
        weight = torch.tensor([0.0, 1.0]).reshape(1, 1, 2, 1)
        mixed = mix_activations(source, donor, weight)
        self.assertTrue(torch.equal(mixed[:, :, 0, :], source[:, :, 0, :]))
        self.assertTrue(torch.equal(mixed[:, :, 1, :], donor[:, :, 1, :]))
        with self.assertRaises(ValueError):
            mix_activations(source, donor, torch.ones(3, 1))

    def test_flash_rank_is_rejected(self):
        fused = torch.zeros(1, 4, 6)
        with self.assertRaises(HookBoundaryError) as caught:
            to_canonical_attention(fused, "flash", 2, 3, "LlamaFlashAttention2")
        self.assertIn("FlashAttention", str(caught.exception))
        self.assertIn("fused", str(caught.exception))

    def test_patch_zero_is_clean_and_patch_one_is_local(self):
        model = ToyCausal()
        example = _example()
        path = ("attention", 0, 0, 0)
        clean = module_logits(model, example["score_plus"])
        untouched, editor = patched_logits(
            model, example["score_plus"], [path], _empty_cache(), _empty_cache(), strength=0.0
        )
        self.assertTrue(torch.equal(clean, untouched))
        self.assertIsNone(editor.last_written)

        example["cache_plus"] = capture_clean(model, example["prompt_plus"])
        example["cache_minus"] = capture_clean(model, example["prompt_minus"])
        plus = example["cache_plus"].attention[("s1", 0)]
        self.assertEqual(tuple(plus.shape), (2, 2, 4))
        self.assertEqual(float(plus[0, 0, 0]), 3.0)
        self.assertEqual(float(plus[0, 1, 0]), 4.0)

        patched, editor = patched_logits(
            model,
            example["score_plus"],
            [path],
            example["cache_plus"],
            example["cache_minus"],
            strength=1.0,
        )
        self.assertFalse(torch.equal(clean, patched))
        written = editor.snapshots["attention"]["written"]
        live = editor.snapshots["attention"]["live"]
        self.assertEqual(float(written[0, 0, 0, 0]), 1.0)
        self.assertTrue(torch.equal(written[:, :, 1, :], live[:, :, 1, :]))
        self.assertTrue(torch.equal(written[:, 1:, 0, :], live[:, 1:, 0, :]))
        self.assertAlmostEqual(float(patched[0, 0, 0]), 4.0)
        self.assertAlmostEqual(float(clean[0, 0, 0]), 12.0)

    def test_direct_editor_does_not_touch_other_heads(self):
        live = torch.zeros(1, 2, 4, 3)
        live[0, 0, 1, 0] = 5
        live[0, 1, 1, 0] = 7
        source = ActivationCache()
        donor = ActivationCache()
        source.attention[("s", 0)] = torch.zeros(4, 2, 3)
        donor_value = torch.zeros(4, 2, 3)
        donor_value[1, 0, 0] = 9
        donor.attention[("s", 0)] = donor_value
        editor = ActivationEditor(
            mode="patch",
            sample_ids=["s"],
            paths=[("attention", 0, 1, 0)],
            source=source,
            donor=donor,
            strength=1.0,
        )
        edited = editor("attention", live, _HeadSpec(), "bhsd")
        self.assertEqual(float(edited[0, 0, 1, 0]), 9.0)
        self.assertEqual(float(edited[0, 1, 1, 0]), 7.0)
        idle = ActivationEditor(
            mode="patch",
            sample_ids=["s"],
            paths=[("attention", 0, 1, 0)],
            source=source,
            donor=donor,
            strength=0.0,
        )
        self.assertIs(idle("attention", live, _HeadSpec(), "bhsd"), live)

    def test_necessity_and_rescue_directions(self):
        model = ToyCausal()
        example = _example()
        paths = [("attention", 0, 0, 0), ("attention", 0, 0, 1)]
        effect = measure_effects(model, example, paths, strength=1.0)
        self.assertGreater(effect["N"], 0.0)
        self.assertGreater(effect["R"], 0.0)
        self.assertEqual(effect["score_scope"], "y_plus_only")
        self.assertEqual(effect["donor_direction"]["necessity"], "q+ <- A-")
        self.assertEqual(effect["computed_by"], "joint_patch_forward")

    def test_q_minus_must_score_the_same_y_plus(self):
        model = ToyCausal()
        paths = [("attention", 0, 0, 0)]
        missing = _example()
        missing["score_minus"] = _batch([2, 3, 4, 4])
        with self.assertRaises(ValueError) as caught:
            measure_effects(model, missing, paths, strength=1.0)
        self.assertIn("y+", str(caught.exception))
        mismatched = _example()
        mismatched["score_minus"] = _batch([2, 3, 4, 4], labels=[-100, -100, 0, 1])
        with self.assertRaises(ValueError) as caught:
            measure_effects(model, mismatched, paths, strength=1.0)
        self.assertIn("not y+", str(caught.exception))
        shifted = _example()
        shifted["score_minus"] = _batch([2, 3, 4, 4], labels=[-100, 0, 0, -100])
        with self.assertRaises(ValueError) as caught:
            measure_effects(model, shifted, paths, strength=1.0)
        self.assertIn("position-aligned", str(caught.exception))
        foil = _example()
        foil["foil_reliable"] = True
        with self.assertRaises(ValueError) as caught:
            measure_effects(model, foil, paths, strength=1.0)
        self.assertIn("margin", str(caught.exception))
        self.assertEqual(model.forward_count, 0)

    def test_joint_effect_is_not_the_sum_of_single_paths(self):
        model = ToyCausal()
        example = _example()
        first = ("attention", 0, 0, 0)
        second = ("attention", 0, 0, 1)
        audit = []
        joint = measure_effects(model, example, [first, second], strength=1.0, audit=audit)
        single_a = measure_effects(model, example, [first], strength=1.0)
        single_b = measure_effects(model, example, [second], strength=1.0)
        self.assertTrue(any(len(event["paths"]) == 2 for event in audit))
        self.assertNotAlmostEqual(joint["N"], single_a["N"] + single_b["N"], places=4)
        interaction = interaction_diagnostic(model, example, [first, second])
        self.assertEqual(interaction["computed_by"], "joint_forward_minus_singles")
        self.assertAlmostEqual(interaction["value"], joint["N"] - single_a["N"] - single_b["N"], places=5)
        unavailable = interaction_diagnostic(model, example, [first])
        self.assertEqual(unavailable["status"], "unavailable")
        self.assertIsNone(unavailable["value"])

    def test_alignment_failure_is_recorded_without_shared_indices(self):
        ok, reasons = check_alignment([1, 3], [2, 3], (0, 1), [1, 3, 4], [2, 3, 4])
        self.assertTrue(ok)
        self.assertEqual(reasons, [])
        failed, reasons = check_alignment([1, 3], [2, 4], (0, 1), [1, 3, 4], [2, 4, 4])
        self.assertFalse(failed)
        self.assertIn("template_mismatch", reasons)
        self.assertNotIn("aligned_index", reasons)
        failed, reasons = check_alignment([1, 3], [2], (0, 1), [1, 3], [2])
        self.assertEqual(reasons, ["length_mismatch"])
        failed, reasons = check_alignment([1, 3], [2, 3], (0, 1), [9, 3, 4], [2, 3, 4])
        self.assertIn("prompt_prefix_mismatch_plus", reasons)

    def test_answer_mask_ignores_prompt_pad_and_final_logit(self):
        labels = torch.tensor([[-100, -100, 0, -100]])
        logits = torch.zeros(1, 4, 3)
        logits[0, :, 0] = 4
        base = sequence_logprob(logits, labels)
        shifted = logits.clone()
        shifted[0, 0, 1] = 50
        shifted[0, -1, 1] = 50
        self.assertTrue(torch.equal(base, sequence_logprob(shifted, labels)))
        changed = logits.clone()
        changed[0, 1, 0] = 0
        self.assertFalse(torch.allclose(base, sequence_logprob(changed, labels)))


def _empty_cache():
    return ActivationCache()


if __name__ == "__main__":
    unittest.main()

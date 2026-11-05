"""Sparse selection, rho sweep, gates, and the candidate ASU loss."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

WORKSPACE = Path(__file__).resolve().parents[2]
REPO = WORKSPACE.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from I2.commands import run_localization, student_entry  # noqa: E402
from I2.mcsd.config import load_config  # noqa: E402
from I2.mcsd.effects import measure_effects  # noqa: E402
from I2.mcsd.errors import (  # noqa: E402
    ConfigError,
    DifferentiableMaskUnavailable,
    HoldoutTuningError,
    StudentE1Blocked,
)
from I2.mcsd.gates import localization_gate, retain_gate, student_block_reason  # noqa: E402
from I2.mcsd.pipeline import run_examples  # noqa: E402
from I2.mcsd.score import masked_token_nll  # noqa: E402
from I2.mcsd.sparse import _objective, select_binary  # noqa: E402
from I2.mcsd.student import authorize_loss_gate, candidate_asu_loss  # noqa: E402
from I2.mcsd.teacher import run_rho_sweep  # noqa: E402
from I2.tests.test_patch_alignment import _example  # noqa: E402
from I2.tests.toy_model import ToyCausal  # noqa: E402


def _load_losses():
    path = REPO / "Right-to-be-forgotten" / "trainer" / "losses.py"
    spec = importlib.util.spec_from_file_location("asu_losses_mcsd", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection(method):
    pool = [
        {"module": "attention", "layer": 0, "token": 0, "head": 0},
        {"module": "attention", "layer": 0, "token": 0, "head": 1},
        {"module": "mlp", "layer": 0, "token": 0},
    ]
    base = {
        "version": "mcsd_v2",
        "selection_split": "calibration",
        "allow_student_e1": False,
        "control_seed": 7,
        "min_authors_for_interval": 8,
        "bootstrap_samples": 20,
        "bootstrap_seed": 3,
        "compute_interaction": False,
        "rho": {"grid": [0, 1], "grid_status": "exploratory_unregistered", "select_best": False},
        "gates": {
            "preregistered": False,
            "retain_nll_cap": None,
            "localization_n_min": None,
            "localization_r_min": None,
            "retain_rho": None,
        },
        "selection": {
            "method": method,
            "candidate_pool": pool,
            "attention_budget": 1,
            "mlp_budget": 0,
            "rank_by": "N",
            "objective": "minimize_soft_g",
            "lambda_sparse": 0.0,
            "lambda_binary": 0.0,
            "binarize_threshold": 0.7,
            "steps": 40,
            "lr": 1.0,
        },
    }
    return base


class _Boom(object):
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("model forward is not allowed")


class SparseGateTests(unittest.TestCase):
    def test_greedy_joint_recompute_and_missing_config(self):
        model = ToyCausal()
        example = _example()
        config = _selection("discrete_greedy")
        config["selection"]["attention_budget"] = 2
        chosen = select_binary(model, example, config, "calibration")
        self.assertEqual(chosen["method"], "discrete_greedy")
        self.assertTrue(chosen["ranking_only"])
        self.assertNotEqual(chosen["method"], "pgb_ct")
        self.assertEqual(chosen["binary_effects"]["computed_by"], "joint_patch_forward")
        singles = chosen["single_path"]
        attention = [row for row in singles if row["path"]["module"] == "attention"]
        self.assertNotAlmostEqual(
            chosen["binary_effects"]["N"],
            attention[0]["N"] + attention[1]["N"],
            places=4,
        )
        self.assertTrue(all(row["ranking_only"] for row in singles))
        broken = _selection("discrete_greedy")
        broken["selection"]["rank_by"] = None
        with self.assertRaises(ConfigError):
            select_binary(model, _example(), broken, "calibration")

    def test_pgb_optimizes_mask_then_recomputes_binary_effects(self):
        model = ToyCausal()
        example = _example()
        config = _selection("pgb_ct")
        config["selection"]["attention_budget"] = 2
        result = select_binary(model, example, config, "calibration")
        self.assertEqual(result["method"], "pgb_ct")
        self.assertFalse(result["soft_causal_evidence"])
        self.assertGreater(min(result["mask"][:2]), 0.7)
        self.assertGreater(result["binary_effects"]["N"], 0.0)
        self.assertTrue(result["binary_effects"]["causal_evidence"])
        heads = [row for row in result["binary_paths"] if row["module"] == "attention"]
        self.assertEqual(len(heads), 2)

    def test_pgb_formula_and_nondifferentiable_refusal(self):
        mask = torch.tensor([0.25, 0.75])
        soft_g = torch.tensor(2.0)
        selection = {"objective": "negative_soft_g", "lambda_sparse": 0.5, "lambda_binary": 0.0}
        self.assertAlmostEqual(float(_objective(soft_g, mask, selection)), -1.75, places=6)
        selection["objective"] = "minimize_soft_g"
        self.assertAlmostEqual(float(_objective(soft_g, mask, selection)), 2.25, places=6)

        model = ToyCausal()
        model.mcsd_differentiable = False
        before = model.forward_count
        with self.assertRaises(DifferentiableMaskUnavailable) as caught:
            select_binary(model, _example(), _selection("pgb_ct"), "calibration")
        message = str(caught.exception)
        self.assertIn("pgb_ct", message)
        self.assertIn("discrete_greedy", message)
        self.assertEqual(model.forward_count, before)

        missing = _selection("pgb_ct")
        missing["selection"]["lambda_sparse"] = None
        with self.assertRaises(ConfigError):
            select_binary(ToyCausal(), _example(), missing, "calibration")

    def test_rho_sweep_keeps_every_point_and_does_not_pick_one(self):
        model = ToyCausal()
        example = _example()
        paths = [("attention", 0, 0, 0), ("attention", 0, 0, 1)]
        measure_effects(model, example, paths, strength=1.0)
        config = _selection("discrete_greedy")
        result = run_rho_sweep(model, example, paths, config, retain_example=example)
        self.assertIsNone(result["selected_rho"])
        self.assertTrue(result["exploratory_unregistered"])
        self.assertEqual([row["rho"] for row in result["sweep"]], [0.0, 1.0])
        self.assertLess(result["sweep"][1]["target_score"], result["sweep"][0]["target_score"])
        self.assertIsNotNone(result["sweep"][1]["delta_retain_nll"])
        self.assertFalse(result["sweep"][0]["teacher_logits"].requires_grad)
        self.assertTrue(result["sweep"][0]["answer_token_distributions"])
        config["rho"]["select_best"] = True
        with self.assertRaises(ConfigError):
            run_rho_sweep(model, example, paths, config)

    def test_loss_mask_causal_shift_and_student_only_grad(self):
        labels = torch.tensor([[-100, -100, 1, 0, -100]])
        student = torch.zeros(1, 5, 3, requires_grad=True)
        teacher = torch.zeros(1, 5, 3)
        teacher[..., 0] = 1
        teacher.requires_grad_(True)
        retain_student = torch.zeros(1, 5, 3, requires_grad=True)
        retain_labels = labels.clone()
        total, kl, nll = candidate_asu_loss(student, teacher, labels, retain_student, retain_labels, 0.5)
        self.assertTrue(torch.allclose(total, 0.5 * kl + nll))
        masked = student.detach().clone()
        masked[0, 0, 2] = 4
        masked[0, -1, 2] = 4
        masked[0, 3, 2] = 4
        same, _, _ = candidate_asu_loss(masked, teacher.detach(), labels, retain_student.detach(), retain_labels, 0.5)
        base, _, _ = candidate_asu_loss(student.detach(), teacher.detach(), labels, retain_student.detach(), retain_labels, 0.5)
        self.assertTrue(torch.allclose(same, base))
        moved = student.detach().clone()
        moved[0, 1, 0] = 3
        changed, _, _ = candidate_asu_loss(moved, teacher.detach(), labels, retain_student.detach(), retain_labels, 0.5)
        self.assertFalse(torch.allclose(changed, base))
        total.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        self.assertIsNotNone(retain_student.grad)
        manual = masked_token_nll(retain_student.detach(), retain_labels)
        self.assertTrue(torch.allclose(nll, manual))

    def test_get_loss_refuses_before_forward_without_a_gate(self):
        losses = _load_losses()
        model = _Boom()
        ref = _Boom()
        with self.assertRaises(StudentE1Blocked):
            losses.get_loss(model, ref, [], "MCSD_ASU")
        self.assertEqual(model.calls, 0)
        self.assertEqual(ref.calls, 0)

        labels = torch.tensor([[-100, 1, 0]])
        student_logits = torch.zeros(1, 3, 2)
        teacher_logits = torch.zeros(1, 3, 2)
        config = _open_gate()
        gate = authorize_loss_gate(config, teacher_logits)

        class Stub(object):
            def __init__(self):
                self.calls = 0

            def __call__(self, input_ids, labels=None, attention_mask=None):
                del input_ids, labels, attention_mask
                self.calls += 1
                return type("Out", (), {"logits": student_logits})()

        stub = Stub()
        forget = (torch.zeros(1, 3, dtype=torch.long), labels, torch.ones(1, 3, dtype=torch.long))
        pair = losses.get_loss(stub, ref, [forget, forget], "MCSD_ASU", mcsd_gate=gate)
        self.assertEqual(stub.calls, 2)
        self.assertEqual(ref.calls, 0)
        total, _, nll = candidate_asu_loss(student_logits, teacher_logits, labels, student_logits, labels, 0.1)
        self.assertTrue(torch.allclose(pair[0] + pair[1], total))
        self.assertTrue(torch.allclose(pair[1], nll))


def _open_gate():
    return {
        "version": "mcsd_v2",
        "allow_student_e1": True,
        "selection_split": "calibration",
        "alpha": 0.1,
        "gates": {
            "preregistered": True,
            "localization_passed": True,
            "teacher_retain_gate_passed": True,
            "retain_nll_cap": 0.5,
            "localization_n_min": 0.0,
            "localization_r_min": 0.0,
            "localization_result_id": "toy-loc",
            "teacher_gate_id": "toy-teacher",
        },
    }


class GateEntryTests(unittest.TestCase):
    def test_default_and_e0_cannot_start_student_e1(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = student_entry.main([])
        self.assertEqual(code, 2)
        self.assertIn("disabled", stderr.getvalue())
        self.assertEqual(student_entry.TRAINING_LAUNCHES, 0)

        e0_yaml = REPO / "workspace" / "I" / "configs" / "e0_probe.yaml"
        e0 = load_config(e0_yaml)
        reason = student_block_reason(e0)
        self.assertIn("do_not_start_e1", reason)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(student_entry.main([str(e0_yaml)]), 2)

        e0_gate = json.loads((REPO / "workspace" / "I" / "gates" / "e0.json").read_text())
        self.assertIn("do_not_start_e1", student_block_reason(e0_gate))
        cap = e0_gate["retain_damage_cap"]["cap"]
        copied = _open_gate()
        copied["gates"]["retain_nll_cap"] = cap
        self.assertIn("E0", student_block_reason(copied))
        with self.assertRaises(ConfigError):
            retain_gate(0.0, copied)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(run_localization.main([]), 0)
        self.assertIn('"real_model_executed": false', out.getvalue())
        self.assertEqual(student_entry.TRAINING_LAUNCHES, 0)

    def test_missing_gate_holdout_and_open_gate_does_not_train(self):
        blocked = _open_gate()
        blocked["gates"]["localization_passed"] = False
        self.assertIn("localization", student_block_reason(blocked))
        holdout = _open_gate()
        holdout["selection_split"] = "holdout"
        self.assertIn("holdout", student_block_reason(holdout))
        self.assertIsNone(localization_gate(1.0, 1.0, {"gates": {"localization_n_min": None, "localization_r_min": None}})["pass"])

        with self.assertRaises(HoldoutTuningError):
            select_binary(ToyCausal(), _example(), _selection("discrete_greedy"), "holdout")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "open.yaml"
            path.write_text(
                "\n".join(
                    [
                        "version: mcsd_v2",
                        "allow_student_e1: true",
                        "selection_split: calibration",
                        "alpha: 0.1",
                        "gates:",
                        "  preregistered: true",
                        "  localization_passed: true",
                        "  teacher_retain_gate_passed: true",
                        "  retain_nll_cap: 0.5",
                        "  localization_n_min: 0.0",
                        "  localization_r_min: 0.0",
                        "  localization_result_id: toy-loc",
                        "  teacher_gate_id: toy-teacher",
                    ]
                )
                + "\n"
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = student_entry.main([str(path)])
        self.assertEqual(code, 0)
        self.assertIn("not started", stdout.getvalue())
        self.assertEqual(student_entry.TRAINING_LAUNCHES, 0)

    def test_pipeline_excludes_bad_pairs_and_writes_null_gates(self):
        model = ToyCausal()
        good = _example()
        bad = _example()
        bad["sample_id"] = "bad"
        bad["prompt_minus_ids"] = [2, 4]
        bad["score_minus_ids"] = [2, 4, 4, 4]
        bad["question_minus"] = "fact-minus"
        before = model.forward_count
        only_bad = run_examples(
            model,
            [bad],
            _selection("discrete_greedy"),
            [{"question": "fact-plus"}, {"question": "fact-minus"}],
        )
        self.assertEqual(model.forward_count, before)
        self.assertEqual(only_bad["excluded"][0]["status"], "excluded")
        self.assertIn("template_mismatch", only_bad["excluded"][0]["reasons"])
        self.assertIsNone(only_bad["excluded"][0]["N"])

        with tempfile.TemporaryDirectory() as directory:
            summary = run_examples(
                model,
                [good],
                _selection("discrete_greedy"),
                [{"question": "fact-plus"}, {"question": "fact-minus"}],
                output_dir=directory,
            )
            saved = json.loads((Path(directory) / "summary.json").read_text())
        self.assertFalse(summary["real_model_executed"])
        self.assertFalse(summary["method_effectiveness_claimed"])
        self.assertEqual(summary["e0_decision_preserved"], "do_not_start_e1")
        self.assertGreater(summary["samples"][0]["N"], 0.0)
        self.assertIsNone(summary["samples"][0]["localization_gate"]["pass"])
        self.assertIsNone(summary["samples"][0]["teacher_retain_gate"]["pass"])
        self.assertIsNone(summary["samples"][0]["rho"]["selected_rho"])
        self.assertEqual(summary["samples"][0]["controls"]["no_swap"]["N"], 0.0)
        self.assertEqual(summary["samples"][0]["interaction"]["status"], "unavailable")
        self.assertTrue(summary["author_bootstrap"]["unstable_estimate"])
        self.assertIsNone(summary["author_bootstrap"]["ci_low"])
        self.assertEqual(summary["author_bootstrap"]["independent_sample_size"], 1)
        self.assertFalse(saved["student_e1_started"])
        self.assertEqual(saved["metadata"]["flash_attention_boundary"], "not_run")


if __name__ == "__main__":
    unittest.main()

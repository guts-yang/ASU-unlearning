from __future__ import annotations

import unittest

import torch

from diagnostic_methods import METHODS, load_module
from diagnostic_methods.registry import run_self_test


class RegistryTest(unittest.TestCase):
    def test_all_methods_resolve(self):
        self.assertEqual(len(METHODS), 13)
        for spec in METHODS:
            module = load_module(spec.method_id)
            self.assertTrue(hasattr(module, "self_test"), spec.method_id)

    def test_all_self_tests(self):
        for spec in METHODS:
            with self.subTest(method=spec.method_id):
                out = run_self_test(spec.method_id)
                self.assertNotEqual(out.get("status"), "FAILED", out)


class CoreAlgorithmTest(unittest.TestCase):
    def test_cf01_constraints(self):
        from diagnostic_methods.cf01_constrained_inverse import solve_constrained_delta

        torch.manual_seed(0)
        d = 24
        out = solve_constrained_delta(
            torch.randn(d), torch.randn(d), torch.randn(d, 2), torch.randn(d, 3), neutral_value=0.0
        )
        self.assertLess(abs(out.diagnostics["target_projection_after"]), 1e-4)
        self.assertLess(out.diagnostics["identity_leak_norm"], 1e-4)

    def test_cf05_marginalization(self):
        from diagnostic_methods.cf05_multi_donor import multi_donor_svd

        shared = torch.randn(16)
        deltas = torch.stack([shared + 0.01 * torch.randn(16) for _ in range(5)])
        out = multi_donor_svd(deltas, rank=1)
        self.assertGreater(out.diagnostics["shared_fraction"], 0.9)

    def test_cf07_rotation(self):
        from diagnostic_methods.cf07_readout_delta import rotation_invariance

        rep = rotation_invariance(torch.randn(4, 8), torch.randn(4, 8), torch.randn(10, 8))
        self.assertTrue(rep["pass_rotation_invariance"])

    def test_cf08_projection(self):
        from diagnostic_methods.cf08_nullspace import nullspace_projection

        p, diag = nullspace_projection(torch.randn(32), torch.randn(10, 32), rank=4)
        self.assertEqual(p.shape, (32,))
        self.assertGreaterEqual(diag["removed_fraction"], 0.0)

    def test_cf10_schedule(self):
        from diagnostic_methods.cf10_purity_schedule import purity_schedule

        w = purity_schedule(torch.tensor([0.1, 0.4, 0.9]), steps=6)
        self.assertEqual(tuple(w.shape), (6, 3))
        self.assertLessEqual(w[0].max().item(), 0.25 + 1e-5)

    def test_r2c_inner_attack(self):
        from diagnostic_methods.r2c_relearning_aware import inner_attack

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(11, 4)
                self.head = torch.nn.Linear(4, 11)

            def forward(self, input_ids, attention_mask=None):
                return self.head(self.emb(input_ids))

        batch = {
            "input_ids": torch.randint(0, 11, (2, 5)),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
            "labels": torch.randint(0, 11, (2, 5)),
        }
        params = inner_attack(Tiny(), batch, steps=1, lr=0.01, param_names=["head.weight"])
        self.assertEqual(list(params), ["head.weight"])

    def test_r3_3_schedule(self):
        from diagnostic_methods.r3_3_two_stage import two_stage_schedule

        s = two_stage_schedule(16, steps=8, shallow_end=5, deep_start=10)
        self.assertEqual(tuple(s.shape), (8, 16))

    def test_cf04_probe_guided_loss(self):
        from diagnostic_methods.cf04_dual_probes import fit_dual_probes, probe_guided_loss

        torch.manual_seed(11)
        x = torch.randn(40, 8)
        y = (x[:, 0] > 0).float()
        fp, ip, _ = fit_dual_probes(x, y, x, 1 - y, steps=20)
        out = probe_guided_loss(torch.randn(2, 4, 8), torch.randn(2, 4, 8), fp, ip)
        self.assertEqual(out["loss"].ndim, 0)

    def test_cf07_readout_loss(self):
        from diagnostic_methods.cf07_readout_delta import readout_direction, readout_space_loss

        r = readout_direction(torch.randn(10, 8))
        out = readout_space_loss(torch.randn(2, 4, 8), torch.randn(2, 4, 8), r)
        self.assertEqual(out["loss"].ndim, 0)


if __name__ == "__main__":
    unittest.main()

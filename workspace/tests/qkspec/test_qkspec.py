"""QKSpec smoke tests (stdlib unittest, no pytest dependency).

Run from the repo root with::

    python -m unittest discover -s workspace/tests/qkspec -v
"""

import os
import sys
import unittest

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
# qkspec package lives under workspace/src; my_models / framework live under
# open-unlearning/src (used only by the tiny-model helper below).
sys.path.insert(0, os.path.join(_REPO_ROOT, "workspace", "src"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "open-unlearning", "src"))

from qkspec.config import QKSpecConfig
from qkspec.qk_math import (
    build_qk_matrix,
    compute_delta_wq,
    fold_into_q,
    get_head_slices,
    get_k_head_slices,
    get_kv_head_index,
    svd_qk_lowrank,
)
from qkspec.selection import select_directions
from qkspec.score import ScoreResult
from qkspec.qk_math import DirectionSpec


def _tiny_model():
    from transformers import LlamaConfig
    from my_models.my_llama import LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg).eval()


class MathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.d_model, self.d_head = 32, 8
        self.wq = torch.randn(self.d_head, self.d_model)
        self.wk = torch.randn(self.d_head, self.d_model)

    def test_svd_reconstruction(self):
        M = build_qk_matrix(self.wq, self.wk)
        U, sigma, V = svd_qk_lowrank(self.wq, self.wk)
        recon = (U * sigma.unsqueeze(0)) @ V.T
        rel = (recon - M).norm().item() / M.norm().item()
        self.assertLess(rel, 1e-4, f"SVD reconstruction error too large: {rel}")

    def test_fold_realizes_delta(self):
        M = build_qk_matrix(self.wq, self.wk)
        U, sigma, V = svd_qk_lowrank(self.wq, self.wk)
        dM, dWq = compute_delta_wq(U, sigma, V, [0], 0.5, self.wk)
        edited = fold_into_q(self.wq.clone(), 0, dWq, 1, self.d_head)
        realized = build_qk_matrix(edited, self.wk) - M
        rel = (realized - dM).norm().item() / dM.norm().item()
        self.assertLess(rel, 1e-4, f"fold-back did not realize delta_M: {rel}")

    def test_noop_rho_one_is_zero(self):
        U, sigma, V = svd_qk_lowrank(self.wq, self.wk)
        _, dWq = compute_delta_wq(U, sigma, V, [0], 1.0, self.wk)
        self.assertAlmostEqual(dWq.norm().item(), 0.0, places=6)

    def test_empty_ranks_is_zero(self):
        U, sigma, V = svd_qk_lowrank(self.wq, self.wk)
        dM, dWq = compute_delta_wq(U, sigma, V, [], 0.5, self.wk)
        self.assertEqual(dWq.numel(), self.wq.numel())
        self.assertAlmostEqual(dWq.norm().item(), 0.0, places=6)

    def test_gqa_mapping(self):
        self.assertEqual(get_kv_head_index(0, 8, 2), 0)
        self.assertEqual(get_kv_head_index(3, 8, 2), 0)
        self.assertEqual(get_kv_head_index(4, 8, 2), 1)
        self.assertEqual(get_kv_head_index(7, 8, 2), 1)


class GQATests(unittest.TestCase):
    def test_k_head_slices_match_group(self):
        torch.manual_seed(1)
        model = _tiny_model()
        attn = model.model.layers[0].self_attn
        h = 3  # query head 3 -> kv head 0 (groups=2)
        hd = attn.head_dim
        kq3 = get_k_head_slices(attn.k_proj.weight, 3, attn.num_heads, attn.num_key_value_heads, hd)
        kq2 = get_k_head_slices(attn.k_proj.weight, 2, attn.num_heads, attn.num_key_value_heads, hd)
        self.assertTrue(torch.equal(kq3, kq2), "query heads 2 and 3 must share the same K slice")


class SelectionTests(unittest.TestCase):
    def test_stage_b_filter(self):
        """stage_b_min_nll_increase filters directions; retain cap applies."""
        cfg = QKSpecConfig(stage_b_min_nll_increase=0.1, max_retain_impact=0.05)
        score = ScoreResult()
        for i in range(3):
            score.stage_a_scores.append(
                DirectionSpec(layer=0, head=i, rank=0, sigma=1.0, stage_a_score=1.0 - i * 0.1)
            )

        class FakeB:
            def __call__(self, model, fl, rl, dirs, cfg, device):
                out = []
                for j, d in enumerate(dirs):
                    out.append(DirectionSpec(
                        layer=d.layer, head=d.head, rank=d.rank, sigma=d.sigma,
                        stage_a_score=d.stage_a_score,
                        stage_b_score=[0.2, 0.05, 0.3][j],
                        retain_impact=[0.01, 0.01, 0.2][j],
                        causal_verified=True,
                    ))
                return out

        import qkspec.selection as sel
        orig = sel.score_stage_b
        sel.score_stage_b = FakeB()
        try:
            result = select_directions(score, object(), None, None, cfg, "cpu")
        finally:
            sel.score_stage_b = orig
        # dir0: b=0.2>=0.1 & retain=0.01<=0.05 -> keep
        # dir1: b=0.05<0.1 -> reject
        # dir2: b=0.3 but retain=0.2>0.05 -> reject
        self.assertEqual(len(result.selected), 1)
        self.assertEqual(result.selected[0].head, 0)
        self.assertEqual(len(result.rejected), 2)


class ConfigTests(unittest.TestCase):
    def test_invalid_rho(self):
        with self.assertRaises(ValueError):
            QKSpecConfig(rho=1.5)

    def test_from_dict(self):
        c = QKSpecConfig.from_any({"rho": 0.3, "top_k_directions": 5, "custom_key": 1})
        self.assertEqual(c.rho, 0.3)
        self.assertEqual(c.top_k_directions, 5)
        self.assertEqual(c.extra["custom_key"], 1)


if __name__ == "__main__":
    unittest.main()

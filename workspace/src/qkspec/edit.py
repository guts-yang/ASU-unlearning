"""Edit orchestration: apply selected QKSpec directions to a model.

The edit is **query-only / GQA-safe**: for each selected query head we compute
``delta_W_Q_doc`` and add it into ``q_proj.weight``.  K, V, O, MLP and all
unselected heads/layers remain byte-identical.  No SVD, hook or Teacher is
needed at inference time; the saved model is a standard HF causal LM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import QKSpecConfig
from .qk_math import (
    DirectionSpec,
    compute_delta_wq,
    fold_into_q,
    get_head_slices,
    get_k_head_slices,
    svd_qk_lowrank,
)
from .hooks import ActivationCache, capture_qk_activations, require_eager_cache
from .score import score_stage_a, score_stage_b
from .selection import SelectionResult, select_directions
from .evaluate import compute_nll_and_accuracy
from .report import write_report


@dataclass
class EditResult:
    """Structured report returned by :func:`run_qkspec_edit`."""

    selected: List[DirectionSpec] = field(default_factory=list)
    selection: Optional[SelectionResult] = None
    update_norm: float = 0.0
    no_op_weights_unchanged: bool = True
    rope_logit_error: Dict[str, float] = field(default_factory=dict)
    reconstruction_error: Dict[str, float] = field(default_factory=dict)
    forget_metrics: Dict[str, Any] = field(default_factory=dict)
    retain_metrics: Dict[str, Any] = field(default_factory=dict)
    controls: Dict[str, Any] = field(default_factory=dict)
    reader_coverage: Dict[str, str] = field(default_factory=dict)
    timing: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": [d.to_dict() for d in self.selected],
            "update_norm": self.update_norm,
            "no_op_weights_unchanged": self.no_op_weights_unchanged,
            "rope_logit_error": self.rope_logit_error,
            "reconstruction_error": self.reconstruction_error,
            "forget_metrics": self.forget_metrics,
            "retain_metrics": self.retain_metrics,
            "controls": self.controls,
            "reader_coverage": self.reader_coverage,
            "timing": self.timing,
            "report_path": self.report_path,
        }


def _get_layers(model: nn.Module):
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("QKSpec requires a Llama-like model with .model.layers")
    return layers


def _head_dim(attn: nn.Module) -> int:
    hd = int(getattr(attn, "head_dim", 0) or 0)
    return hd if hd > 0 else int(attn.q_proj.out_features // attn.num_heads)


def apply_directions(
    model: nn.Module,
    directions: Sequence[DirectionSpec],
    cfg: QKSpecConfig,
    device: torch.device,
    compute_rope_error: bool = True,
    write_back: bool = True,
) -> EditResult:
    """Apply a list of direction edits to ``model`` (in place if ``write_back``).

    ``write_back=False`` is used by controls to compute diagnostic metadata
    (update norm, no-op check) without changing the model.
    """
    result = EditResult()
    layers = _get_layers(model)
    total_norm = 0.0
    unchanged = True

    # Snapshot all q_proj weights touched, for no-op verification.
    snapshots: Dict[Tuple[int, int], torch.Tensor] = {}

    for d in directions:
        if not (0 <= d.layer < len(layers)):
            raise ValueError(f"direction layer {d.layer} out of range")
        attn = layers[d.layer].self_attn
        hd = _head_dim(attn)
        if d.head >= attn.num_heads:
            raise ValueError(f"direction head {d.head} out of range")
        w_q_h = get_head_slices(attn.q_proj.weight, d.head, attn.num_heads, hd)
        w_k_h = get_k_head_slices(
            attn.k_proj.weight, d.head, attn.num_heads,
            attn.num_key_value_heads, hd,
        )
        U, sigma, V = svd_qk_lowrank(w_q_h, w_k_h)
        if d.rank >= U.shape[1]:
            raise ValueError(
                f"direction rank {d.rank} out of range (budget {U.shape[1]})"
            )
        rho = getattr(d, "rho", cfg.rho)
        _, delta_w_q_doc = compute_delta_wq(U, sigma, V, [d.rank], rho, w_k_h)
        norm = float(delta_w_q_doc.float().norm())
        total_norm += norm
        if norm > 0:
            unchanged = False

        key = (d.layer, d.head)
        if write_back:
            if key not in snapshots:
                snapshots[key] = attn.q_proj.weight.detach().clone()
            edited = fold_into_q(
                attn.q_proj.weight.data, d.head, delta_w_q_doc,
                attn.num_heads, hd, inplace=False,
            )
            with torch.no_grad():
                attn.q_proj.weight.data.copy_(edited)

    result.update_norm = total_norm
    result.no_op_weights_unchanged = unchanged
    return result


def _no_op_weight_check(model, before: Dict[int, torch.Tensor]) -> bool:
    layers = _get_layers(model)
    for li, snapshot in before.items():
        if not torch.equal(layers[li].self_attn.q_proj.weight, snapshot):
            return False
    return True


def _reconstruction_error(model, directions: Sequence[DirectionSpec]) -> Dict[str, float]:
    """Check M' vs (W_Q' W_K.T) after edit for each selected head."""
    layers = _get_layers(model)
    errors = {}
    for d in directions:
        attn = layers[d.layer].self_attn
        hd = _head_dim(attn)
        w_q_h = get_head_slices(attn.q_proj.weight, d.head, attn.num_heads, hd)
        w_k_h = get_k_head_slices(
            attn.k_proj.weight, d.head, attn.num_heads,
            attn.num_key_value_heads, hd,
        )
        U, sigma, V = svd_qk_lowrank(w_q_h, w_k_h)
        if d.rank >= U.shape[1]:
            continue
        M = w_q_h.T @ w_k_h
        # Pre-RoPE logit reconstruction on sampled hidden states is reported by
        # RoPE check; here we report the analytic spectral residual norm.
        sigma_prime = sigma.clone()
        sigma_prime[d.rank] = d.rho * sigma[d.rank]
        M_recon = (U * sigma_prime.unsqueeze(0)) @ V.T
        errors[f"layer_{d.layer}_head_{d.head}_rank_{d.rank}"] = float(
            (M_recon - M).float().norm() / (M.float().norm() + 1e-12)
        )
    return errors


def run_qkspec_edit(
    model: nn.Module,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    cfg: QKSpecConfig,
    output_dir: Optional[str] = None,
    controls: Optional[Dict[str, Any]] = None,
) -> EditResult:
    """Run the full QKSpec edit pipeline on ``model`` (in place).

    Steps: stage A screening -> Top-K -> stage B causal re-check -> retain
    filter -> fold selected deltas into W_Q -> optional controls -> evaluate
    D_F/D_R -> structured report.
    """
    import time

    timings: Dict[str, float] = {}
    t0 = time.time()
    device = next(model.parameters()).device

    # Stage A + selection (stage B internally restores weights after each
    # ablation, so the model is back to original when this returns).
    score_result = score_stage_a(model, forget_loader, retain_loader, cfg, device)
    selection = select_directions(
        score_result, model, forget_loader, retain_loader, cfg, device
    )
    timings["discovery_seconds"] = time.time() - t0

    result = EditResult(selection=selection, selected=list(selection.selected))

    if not selection.selected:
        if cfg.empty_direction_action == "error":
            raise RuntimeError("QKSpec selected no directions; nothing to edit")
        result.reader_coverage = {
            "probe": "not_covered",
            "causal_patch": "not_covered",
            "quantization": "not_covered",
            "relearning": "not_covered",
        }
        result.notes.append("no directions selected; model left unchanged")
        return result

    # Controls run on the original model (they restore it themselves) and
    # give the random-iso / shuffle / no-op comparison needed to argue the
    # real directions are target-specific.
    tc = time.time()
    if cfg.run_controls:
        from .controls import run_controls
        ctrl_result = run_controls(
            model, forget_loader, retain_loader, selection.selected, cfg, device
        )
        result.controls = ctrl_result.to_dict()
    timings["controls_seconds"] = time.time() - tc

    # Snapshot q weights of touched layers for no-op check.
    touched_layers = sorted({d.layer for d in selection.selected})
    before = {li: _get_layers(model)[li].self_attn.q_proj.weight.detach().clone()
              for li in touched_layers}

    t1 = time.time()
    edit_meta = apply_directions(model, selection.selected, cfg, device, write_back=True)
    result.update_norm = edit_meta.update_norm
    result.no_op_weights_unchanged = _no_op_weight_check(model, before)
    result.reconstruction_error = _reconstruction_error(model, selection.selected)
    timings["edit_seconds"] = time.time() - t1

    # Full D_F / D_R evaluation on the edited model (separate keys, no mixed
    # score).  Uses the same loaders; the external TOFU evaluator is run by
    # the Trainer wrapper afterwards.
    te = time.time()
    result.forget_metrics = compute_nll_and_accuracy(model, forget_loader, device).to_dict()
    result.retain_metrics = compute_nll_and_accuracy(model, retain_loader, device).to_dict()
    timings["eval_seconds"] = time.time() - te

    # Reader coverage: first version implements only L1 behavior.
    result.reader_coverage = {
        "probe": "not_covered",
        "causal_patch": "not_covered",
        "quantization": "not_covered",
        "relearning": "not_covered",
        "l1_behavior_silence": "covered_by_external_evaluator",
    }

    result.timing = timings
    return result

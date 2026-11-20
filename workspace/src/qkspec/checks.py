"""Engineering checks required by the design doc (section 10).

* RoPE boundary: ``M = W_Q W_K.T`` is a pre-RoPE object.  We record the actual
  post-RoPE QK logit at selected token positions before/after the edit so the
  spectral residual can be separated from position-dependent behaviour.
* No-op consistency: rho==1 / empty J must not change weights or logits.
* Reconstruction error: analytic spectral residual is computed in ``edit.py``.

The checks here are diagnostic only and are written into the JSON report; they
are not mixed into forget/retain scores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from .hooks import ActivationCache, capture_qk_activations, require_eager_cache
from .qk_math import (
    DirectionSpec, get_head_slices, get_k_head_slices, svd_qk_lowrank, compute_delta_wq,
)


def _get_layers(model: nn.Module):
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("QKSpec requires a Llama-like model with .model.layers")
    return layers


def _head_dim(attn: nn.Module) -> int:
    hd = int(getattr(attn, "head_dim", 0) or 0)
    return hd if hd > 0 else int(attn.q_proj.out_features // attn.num_heads)


def _move_batch(batch, device):
    return {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }


def _spectral_logit_matrix(model, layer: int, head: int) -> torch.Tensor:
    """Return the current ``M = W_Q.T @ W_K`` (doc convention) for a head."""
    attn = _get_layers(model)[layer].self_attn
    hd = _head_dim(attn)
    w_q_h = get_head_slices(attn.q_proj.weight, head, attn.num_heads, hd)
    w_k_h = get_k_head_slices(
        attn.k_proj.weight, head, attn.num_heads, attn.num_key_value_heads, hd,
    )
    return w_q_h.T.to(torch.float32) @ w_k_h.to(torch.float32)


def compute_spectral_residual(
    model: nn.Module,
    directions: Sequence[DirectionSpec],
) -> Dict[str, float]:
    """Per-direction relative Frobenius residual of the contracted M.

    ``M_after`` is compared against the ideal ``M'`` obtained by contracting
    exactly the selected singular value; a small residual means the fold-back
    into W_Q realised the intended spectral edit.
    """
    errors: Dict[str, float] = {}
    for d in directions:
        attn = _get_layers(model)[d.layer].self_attn
        hd = _head_dim(attn)
        w_q_h = get_head_slices(attn.q_proj.weight, d.head, attn.num_heads, hd)
        w_k_h = get_k_head_slices(
            attn.k_proj.weight, d.head, attn.num_heads, attn.num_key_value_heads, hd,
        )
        U, sigma, V = svd_qk_lowrank(w_q_h, w_k_h)
        if d.rank >= U.shape[1]:
            continue
        M = w_q_h.T.to(torch.float32) @ w_k_h.to(torch.float32)
        sigma_prime = sigma.clone()
        sigma_prime[d.rank] = d.rho * sigma[d.rank]
        M_ideal = (U * sigma_prime.unsqueeze(0).to(torch.float32)) @ V.T.to(torch.float32)
        denom = M.norm().item() + 1e-12
        errors[f"L{d.layer}_H{d.head}_R{d.rank}"] = float((M_ideal - M).norm().item() / denom)
    return errors


def cache_attention_snapshots(
    model: nn.Module,
    loader: DataLoader,
    positions: Sequence[int],
    device: torch.device,
    max_batches: int = 1,
) -> List[Dict[str, Any]]:
    """Cache eager attention weights at requested positions for a few batches.

    Used to compare attention distributions before/after the edit on identical
    inputs (the RoPE-dependent observable).  Returns a list of per-batch
    snapshots as plain tensors moved to CPU.
    """
    cache = ActivationCache()
    snapshots: List[Dict[str, Any]] = []
    seen = 0
    checked = False
    with capture_qk_activations(model, cache):
        for batch in loader:
            if seen >= max_batches:
                break
            inputs = _move_batch(batch, device)
            with torch.inference_mode():
                model(**inputs, output_attentions=True)
            if not checked:
                require_eager_cache(cache)
                checked = True
            snap = {}
            for li, act in cache.layers.items():
                if act.attn_weights is None:
                    continue
                aw = act.attn_weights[0].detach().cpu()  # [H,S,S]
                snap[li] = {int(p): aw[:, p, :].clone() for p in positions if p < aw.shape[1]}
            snapshots.append(snap)
            seen += 1
    return snapshots


def attention_distribution_shift(
    before: List[Dict[str, Any]],
    after: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Mean absolute attention-probability shift at cached positions."""
    shifts: Dict[str, float] = {}
    for b0, b1 in zip(before, after):
        for li in b0:
            if li not in b1:
                continue
            for p in b0[li]:
                if p not in b1[li]:
                    continue
                key = f"L{li}_pos{p}"
                delta = (b0[li][p] - b1[li][p]).abs().mean().item()
                shifts[key] = shifts.get(key, 0.0) + float(delta)
    n = max(1, len(shifts))
    return {k: v / n for k, v in shifts.items()}

"""CF-10: purity-driven progressive schedule.

Converts a per-layer factual purity spectrum into a step-by-layer weight matrix.
The schedule is frozen before training; it is not a post-hoc reweighting.
"""

from __future__ import annotations

import torch

from .base import as_tensor


def purity_scores(factual: torch.Tensor, function: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    f = as_tensor(factual).flatten()
    g = as_tensor(function).flatten()
    if f.shape != g.shape:
        raise ValueError("factual and function scores must have the same shape")
    return f / (f + g + eps)


def purity_schedule(
    purity: torch.Tensor,
    steps: int,
    min_scale: float = 0.25,
    max_scale: float = 1.0,
    reverse: bool = False,
    shuffle_seed: int | None = None,
) -> torch.Tensor:
    """Return ``[steps, layers]`` weights that ramp from neutral to purity-ranked."""
    p = as_tensor(purity).flatten()
    if p.numel() == 0:
        raise ValueError("purity vector is empty")
    p = (p - p.min()) / (p.max() - p.min()).clamp_min(1e-8)
    if reverse:
        p = 1.0 - p
    if shuffle_seed is not None:
        g = torch.Generator().manual_seed(shuffle_seed)
        p = p[torch.randperm(p.numel(), generator=g)]
    ramp = torch.linspace(0.0, 1.0, steps).unsqueeze(1)
    weights = min_scale + (max_scale - min_scale) * (p.unsqueeze(0) * ramp)
    return weights


def schedule_diagnostics(weights: torch.Tensor, purity: torch.Tensor) -> dict:
    w = as_tensor(weights)
    p = as_tensor(purity).flatten()
    final = w[-1]
    return {
        "steps": int(w.shape[0]),
        "layers": int(w.shape[1]),
        "purity_mean": float(p.mean().item()),
        "purity_std": float(p.std(unbiased=False).item()),
        "final_weight_mean": float(final.mean().item()),
        "final_weight_std": float(final.std(unbiased=False).item()),
        "final_weight_max": float(final.max().item()),
        "final_weight_min": float(final.min().item()),
        "non_flat": bool(final.std(unbiased=False).item() > 1e-4),
    }


def self_test() -> dict:
    p = torch.tensor([0.1, 0.3, 0.9, 0.5])
    w = purity_schedule(p, steps=10, min_scale=0.2, max_scale=1.0)
    diag = schedule_diagnostics(w, p)
    assert w.shape == (10, 4)
    assert diag["non_flat"]
    assert w[0].max().item() <= 0.2 + 1e-5
    return {"status": "ok", "diagnostics": diag}


__all__ = ["purity_scores", "purity_schedule", "schedule_diagnostics", "self_test"]

"""R3-3: shallow/deep decoupled two-stage forgetting.

Stage 1 uses shallow ASU smoothing; stage 2 aligns deep layers to a frozen
retain-distribution reference.  The module exposes the layer weights and the
alignment loss so the two stages stay decoupled.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import as_tensor


def layer_weights(
    num_layers: int,
    shallow_end: int = 5,
    deep_start: int = 10,
    shallow_scale: float = 1.0,
    deep_scale: float = 1.0,
) -> torch.Tensor:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    w = torch.zeros(num_layers)
    w[: min(shallow_end, num_layers)] = shallow_scale
    w[min(deep_start, num_layers) :] = deep_scale
    return w


def two_stage_schedule(
    num_layers: int,
    steps: int,
    shallow_end: int = 5,
    deep_start: int = 10,
    shallow_scale: float = 1.0,
    deep_scale: float = 1.0,
) -> torch.Tensor:
    """Return ``[steps, layers]``: stage 1 shallow, stage 2 deep."""
    w = layer_weights(num_layers, shallow_end, deep_start, shallow_scale, deep_scale)
    out = torch.zeros(steps, num_layers)
    split = max(1, steps // 2)
    out[:split] = w * (w > 0)
    deep = w.clone()
    deep[: min(shallow_end, num_layers)] = 0.0
    out[split:] = deep.unsqueeze(0)
    return out


def alignment_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    sh = as_tensor(student_hidden)
    th = as_tensor(teacher_hidden)
    w = as_tensor(weights).flatten().to(device=sh.device, dtype=sh.dtype)
    if sh.shape != th.shape:
        raise ValueError("student and teacher hidden states must have the same shape")
    if sh.ndim == 3:
        if w.numel() == 1:
            w = w.expand(sh.shape[1])
        if w.numel() != sh.shape[1]:
            raise ValueError("weights must match the time dimension")
        diff = (sh - th) ** 2
        return (diff * w.view(1, -1, 1)).mean()
    if sh.ndim == 2:
        if w.numel() == 1:
            w = w.expand(sh.shape[0])
        if w.numel() != sh.shape[0]:
            raise ValueError("weights must match the layer dimension")
        return ((sh - th) ** 2 * w.view(-1, 1)).mean()
    raise ValueError("hidden states must be [T,D] or [B,T,D]")


def self_test() -> dict:
    torch.manual_seed(10)
    w = layer_weights(16, shallow_end=5, deep_start=10)
    sched = two_stage_schedule(16, steps=10, shallow_end=5, deep_start=10)
    assert w[0] > 0 and w[9] == 0 and w[15] > 0
    assert sched.shape == (10, 16)
    loss = alignment_loss(torch.randn(16, 8), torch.randn(16, 8), torch.ones(16))
    assert loss.ndim == 0
    return {"status": "ok", "shallow_layers": int((w > 0)[:5].sum().item()), "deep_layers": int((w > 0)[10:].sum().item())}


__all__ = ["layer_weights", "two_stage_schedule", "alignment_loss", "self_test"]

"""CF-07: readout-space ΔA.

The update is represented by the change in the frozen LM-head readout, not by a
hidden-space direction.  A finite-difference Jacobian is included for the
non-linear readout case.
"""

from __future__ import annotations

import torch

from .base import as_tensor, safe_cosine


def readout_delta(
    hidden_plus: torch.Tensor,
    hidden_minus: torch.Tensor,
    lm_head_weight: torch.Tensor,
) -> torch.Tensor:
    """Return ``(h+ - h-) @ W^T`` with shape ``[..., V]``."""
    hp = as_tensor(hidden_plus)
    hm = as_tensor(hidden_minus)
    w = as_tensor(lm_head_weight)
    return (hp - hm) @ w.T


def finite_difference_jacobian(fn, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Row-wise finite-difference Jacobian of ``fn: [D] -> [M]``."""
    x = as_tensor(x).flatten()
    base = fn(x)
    rows = []
    for i in range(x.numel()):
        e = torch.zeros_like(x)
        e[i] = eps
        rows.append((fn(x + e) - base) / eps)
    return torch.stack(rows, dim=1)


def rotation_invariance(
    hidden_plus: torch.Tensor,
    hidden_minus: torch.Tensor,
    lm_head_weight: torch.Tensor,
    q: torch.Tensor | None = None,
    token_id: int = 0,
) -> dict:
    """Check that a scalar readout is invariant under an orthogonal change of basis.

    The readout direction must be rotated with the hidden states:
    ``(h Q) @ (Q^T r) == h @ r``.  This is the invariance CF-07 asks about; the
    hidden-space ΔA itself is expected to drift.
    """
    hp = as_tensor(hidden_plus)
    hm = as_tensor(hidden_minus)
    w = as_tensor(lm_head_weight)
    d_hidden = hp - hm
    r = w[token_id]
    d_read = d_hidden @ r
    if q is None:
        q, _ = torch.linalg.qr(torch.randn(hp.shape[-1], hp.shape[-1], dtype=hp.dtype, device=hp.device))
    rotated = d_hidden @ q
    r_rot = q.T @ r
    d_read_rot = rotated @ r_rot
    return {
        "hidden_delta_norm": float(d_hidden.norm().item()),
        "readout_delta_norm": float(d_read.norm().item()),
        "rotated_hidden_norm": float(rotated.norm().item()),
        "readout_change_after_rotation": float((d_read_rot - d_read).norm().item()),
        "readout_cosine_after_rotation": float(safe_cosine(d_read.flatten(), d_read_rot.flatten()).item()),
        "q_orthogonality_error": float((q.T @ q - torch.eye(q.shape[0], dtype=q.dtype, device=q.device)).norm().item()),
        "pass_rotation_invariance": bool(torch.allclose(d_read, d_read_rot, atol=1e-3)),
    }


def readout_direction(lm_head_weight: torch.Tensor, token_ids: list[int] | None = None) -> torch.Tensor:
    """Pick a normalized readout direction from the LM head."""
    w = as_tensor(lm_head_weight)
    rows = w[token_ids] if token_ids is not None else w
    direction = rows.mean(dim=0)
    return direction / direction.norm().clamp_min(1e-8)


def readout_space_loss(
    forget_hidden: torch.Tensor,
    retain_hidden: torch.Tensor,
    direction: torch.Tensor,
    retain_target: torch.Tensor | float | None = None,
    retain_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Suppress the forget readout while preserving the retain readout level.

    ``retain_target`` should be computed from the frozen pre-training retain
    activation; passing ``None`` uses the detached current retain mean.
    """
    hf = as_tensor(forget_hidden)
    hr = as_tensor(retain_hidden)
    r = as_tensor(direction).flatten().to(hf.device, hf.dtype)
    f_proj = hf @ r
    r_proj = hr @ r
    target = r_proj.detach().mean() if retain_target is None else as_tensor(retain_target).to(hf.device, hf.dtype)
    l_forget = (f_proj ** 2).mean()
    l_retain = ((r_proj - target) ** 2).mean()
    loss = l_forget + retain_weight * l_retain
    return {
        "loss": loss,
        "forget_readout_sq": l_forget,
        "retain_readout_shift": l_retain,
        "retain_target": target,
    }


def self_test() -> dict:
    torch.manual_seed(6)
    hf = torch.randn(4, 8)
    hm = torch.randn(4, 8)
    w = torch.randn(11, 8)
    rep = rotation_invariance(hf, hm, w)
    assert rep["pass_rotation_invariance"]
    j = finite_difference_jacobian(lambda x: w @ x, torch.randn(8))
    assert j.shape == (11, 8)
    r = readout_direction(w)
    losses = readout_space_loss(torch.randn(4, 5, 8), torch.randn(4, 5, 8), r)
    assert losses["loss"].ndim == 0
    return {"status": "ok", "report": rep, "readout_loss": float(losses["loss"].item())}


__all__ = [
    "readout_delta", "finite_difference_jacobian", "rotation_invariance",
    "readout_direction", "readout_space_loss", "self_test",
]

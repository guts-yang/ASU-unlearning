"""CF-08: retain-gradient nullspace projection.

Projects a ΔA update out of the top principal directions of retain gradients,
then optionally measures recovery-curve area.
"""

from __future__ import annotations

import torch

from .base import as_tensor, orthonormalize, project_out


def principal_basis(gradients: torch.Tensor, rank: int = 8) -> torch.Tensor:
    g = as_tensor(gradients)
    if g.ndim == 1:
        g = g.unsqueeze(0)
    _, _, vh = torch.linalg.svd(g, full_matrices=False)
    k = min(max(1, rank), vh.shape[0])
    return orthonormalize(vh[:k].T)


def nullspace_basis(basis: torch.Tensor) -> torch.Tensor:
    b = orthonormalize(basis)
    d = b.shape[0]
    if b.shape[1] == 0:
        return torch.eye(d, dtype=b.dtype, device=b.device)
    q, _ = torch.linalg.qr(b, mode="complete")
    return q[:, b.shape[1] :]


def nullspace_projection(delta: torch.Tensor, retain_gradients: torch.Tensor, rank: int = 8) -> tuple[torch.Tensor, dict]:
    base = principal_basis(retain_gradients, rank=rank)
    d = as_tensor(delta).flatten()
    projected = project_out(d, base)
    return projected, {
        "removed_fraction": float(1.0 - projected.norm().item() / max(d.norm().item(), 1e-8)),
        "projected_norm": float(projected.norm().item()),
        "rank": int(base.shape[1]),
    }


def recovery_auc(steps: torch.Tensor, recovery: torch.Tensor) -> float:
    """Trapezoidal area under a recovery curve (higher means less robust)."""
    t = as_tensor(steps).flatten().float()
    y = as_tensor(recovery).flatten().float()
    if t.numel() != y.numel() or t.numel() < 2:
        return 0.0
    return float(torch.trapz(y, t).item())


def self_test() -> dict:
    torch.manual_seed(7)
    g = torch.randn(12, 32)
    d = torch.randn(32)
    p, diag = nullspace_projection(d, g, rank=4)
    assert p.shape == d.shape
    assert 0.0 <= diag["removed_fraction"] <= 1.0
    assert recovery_auc(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([0.0, 0.5, 1.0])) > 0.9
    return {"status": "ok", "diagnostics": diag}


__all__ = ["principal_basis", "nullspace_basis", "nullspace_projection", "recovery_auc", "self_test"]

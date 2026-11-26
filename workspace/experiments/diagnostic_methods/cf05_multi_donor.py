"""CF-05: multi-donor marginalization.

Given K relation-matched donor deltas, separate a shared low-rank component from
donor-private residuals.  The shared projection is the marginalized update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import as_tensor, orthonormalize


@dataclass
class DonorDecomposition:
    shared_basis: torch.Tensor
    shared: torch.Tensor
    private: torch.Tensor
    singular_values: torch.Tensor
    diagnostics: dict


def multi_donor_svd(deltas: torch.Tensor, rank: int = 4, center: bool = False) -> DonorDecomposition:
    x = as_tensor(deltas)
    if x.ndim != 2:
        raise ValueError(f"deltas must be [K,D], got {tuple(x.shape)}")
    if center:
        x = x - x.mean(dim=0, keepdim=True)
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    k = min(max(1, rank), vh.shape[0])
    basis = orthonormalize(vh[:k].T)
    shared = (x @ basis) @ basis.T
    private = x - shared
    total = x.norm(dim=1).sum().clamp_min(1e-8)
    diag = {
        "rank": k,
        "shared_fraction": float(shared.norm(dim=1).sum().item() / total.item()),
        "private_fraction": float(private.norm(dim=1).sum().item() / total.item()),
        "top_singular": s[: min(5, s.numel())].detach().cpu().tolist(),
        "explained_by_rank": float((s[:k] ** 2).sum().item() / (s ** 2).sum().clamp_min(1e-12).item()),
    }
    return DonorDecomposition(basis, shared, private, s, diag)


def marginalized_delta(deltas: torch.Tensor, rank: int = 4) -> torch.Tensor:
    dec = multi_donor_svd(deltas, rank=rank)
    return dec.shared.mean(dim=0)


def donor_consistency(deltas: torch.Tensor) -> dict:
    x = as_tensor(deltas)
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos = x @ x.T
    mask = ~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)
    vals = cos[mask]
    return {
        "mean_pairwise_cosine": float(vals.mean().item()) if vals.numel() else 0.0,
        "min_pairwise_cosine": float(vals.min().item()) if vals.numel() else 0.0,
        "max_pairwise_cosine": float(vals.max().item()) if vals.numel() else 0.0,
    }


def self_test() -> dict:
    torch.manual_seed(4)
    shared = torch.randn(16)
    deltas = torch.stack([shared + 0.05 * torch.randn(16) for _ in range(6)])
    dec = multi_donor_svd(deltas, rank=1)
    assert dec.diagnostics["shared_fraction"] > 0.9
    assert donor_consistency(deltas)["mean_pairwise_cosine"] > 0.9
    return {"status": "ok", "diagnostics": dec.diagnostics}


__all__ = ["DonorDecomposition", "multi_donor_svd", "marginalized_delta", "donor_consistency", "self_test"]

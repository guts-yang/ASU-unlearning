"""CF-02: three-forward role decomposition.

The update is split into deletion, anti-substitution and utility-preserving
components.  The implementation is algebraic so it can be tested without a
model; ``role_loss_terms`` maps the components to training losses.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import as_tensor, orthonormalize, ridge_coefficients, safe_cosine


@dataclass
class RoleDecomposition:
    deletion: torch.Tensor
    anti_substitution: torch.Tensor
    utility: torch.Tensor
    coefficients: torch.Tensor
    diagnostics: dict


def decompose_roles(
    delta: torch.Tensor,
    deletion_basis: torch.Tensor,
    substitution_basis: torch.Tensor,
    utility_basis: torch.Tensor,
    ridge: float = 1e-6,
) -> RoleDecomposition:
    """Least-squares decomposition into three role bases.

    Bases are column bases ``[D,K]``.  The returned components are
    ``basis @ coefficients`` and therefore remain in the declared role spaces.
    """
    d = as_tensor(delta).flatten()
    bd = orthonormalize(deletion_basis)
    bs = orthonormalize(substitution_basis)
    bu = orthonormalize(utility_basis)
    design = torch.cat([bd, bs, bu], dim=1)
    if design.shape[0] != d.numel():
        raise ValueError("basis dimension must match delta")
    coef = ridge_coefficients(d, design, ridge=ridge)
    nd, ns = bd.shape[1], bs.shape[1]
    cd = coef[:nd]
    cs = coef[nd : nd + ns]
    cu = coef[nd + ns :]
    deletion = bd @ cd
    anti = bs @ cs
    utility = bu @ cu
    diag = {
        "deletion_norm": float(deletion.norm().item()),
        "anti_substitution_norm": float(anti.norm().item()),
        "utility_norm": float(utility.norm().item()),
        "residual_norm": float((d - deletion - anti - utility).norm().item()),
        "cosine_deletion": float(safe_cosine(deletion, d).item()),
        "cosine_anti": float(safe_cosine(anti, d).item()),
        "cosine_utility": float(safe_cosine(utility, d).item()),
    }
    return RoleDecomposition(deletion, anti, utility, coef, diag)


def role_loss_terms(
    hidden_forget: torch.Tensor,
    hidden_donor: torch.Tensor,
    hidden_retain: torch.Tensor,
    deletion_basis: torch.Tensor,
    substitution_basis: torch.Tensor,
    utility_basis: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Turn the decomposition into differentiable role losses.

    - deletion: suppress the target role component on forget inputs;
    - anti-substitution: keep the donor role component near zero;
    - utility: preserve the retain role component.
    """
    hf = as_tensor(hidden_forget)
    hd = as_tensor(hidden_donor)
    hr = as_tensor(hidden_retain)
    bd = orthonormalize(deletion_basis)
    bs = orthonormalize(substitution_basis)
    bu = orthonormalize(utility_basis)
    l_del = ((hf @ bd) ** 2).mean()
    l_anti = ((hd @ bs) ** 2).mean()
    l_util = ((hr @ bu) ** 2).mean()
    return {
        "loss": l_del + l_anti + 0.5 * l_util,
        "deletion": l_del,
        "anti_substitution": l_anti,
        "utility": l_util,
    }


def self_test() -> dict:
    torch.manual_seed(1)
    d = 32
    bd = torch.randn(d, 3)
    bs = torch.randn(d, 2)
    bu = torch.randn(d, 2)
    delta = bd @ torch.tensor([0.4, -0.2, 0.1]) + bs @ torch.tensor([0.3, 0.2]) + 0.01 * torch.randn(d)
    out = decompose_roles(delta, bd, bs, bu)
    assert out.diagnostics["residual_norm"] < 0.2
    losses = role_loss_terms(torch.randn(4, d), torch.randn(4, d), torch.randn(4, d), bd, bs, bu)
    assert losses["loss"].ndim == 0
    return {"status": "ok", "diagnostics": out.diagnostics}


__all__ = ["RoleDecomposition", "decompose_roles", "role_loss_terms", "self_test"]

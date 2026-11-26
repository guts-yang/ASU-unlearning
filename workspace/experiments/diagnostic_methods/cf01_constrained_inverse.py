"""CF-01: constrained inverse target.

Core operator: find the minimum-norm update that hits a neutral target readout
while staying orthogonal to identity and retain-gradient constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch

from .base import as_tensor, orthonormalize, project_onto, safe_cosine


@dataclass
class ConstrainedDelta:
    delta: torch.Tensor
    diagnostics: dict

    def to_dict(self) -> dict:
        return {"diagnostics": self.diagnostics}


def _constraint_rows(
    target_direction: torch.Tensor,
    identity_basis: torch.Tensor | None,
    retain_grad_basis: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    rows = [target_direction.flatten() / target_direction.flatten().norm().clamp_min(1e-8)]
    if identity_basis is not None and identity_basis.numel():
        rows.extend(orthonormalize(identity_basis).T)
    if retain_grad_basis is not None and retain_grad_basis.numel():
        rows.extend(orthonormalize(retain_grad_basis).T)
    return torch.stack(rows, dim=0), 1 if identity_basis is None else 0


def solve_constrained_delta(
    delta0: torch.Tensor,
    target_direction: torch.Tensor,
    identity_basis: torch.Tensor | None = None,
    retain_grad_basis: torch.Tensor | None = None,
    neutral_value: float = 0.0,
    ridge: float = 1e-8,
) -> ConstrainedDelta:
    """Solve ``min ||d-delta0||`` s.t. ``A d = b``.

    The first constraint sets the target readout to ``neutral_value``; identity
    and retain-gradient rows are constrained to zero.  This is the algebraic core
    of CF-01, not a full three-forward trainer.
    """
    d0 = as_tensor(delta0).flatten()
    t = as_tensor(target_direction).flatten()
    if d0.shape != t.shape:
        raise ValueError(f"shape mismatch: delta0 {tuple(d0.shape)} vs target {tuple(t.shape)}")
    a, offset = _constraint_rows(t, identity_basis, retain_grad_basis)
    b = a.new_zeros(a.shape[0])
    b[0] = float(neutral_value)
    # Pinv tolerates redundant identity / retain rows without changing b.
    gram = a @ a.T + ridge * torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    delta = d0 - a.T @ (torch.linalg.pinv(gram) @ (a @ d0 - b))

    before = float((d0 @ t).item())
    after = float((delta @ t).item())
    ident = float(project_onto(delta, identity_basis).norm().item()) if identity_basis is not None else 0.0
    rg = float(project_onto(delta, retain_grad_basis).norm().item()) if retain_grad_basis is not None else 0.0
    diag = {
        "target_projection_before": before,
        "target_projection_after": after,
        "neutral_target": float(neutral_value),
        "identity_leak_norm": ident,
        "retain_grad_leak_norm": rg,
        "delta_norm": float(delta.norm().item()),
        "cosine_to_original": float(safe_cosine(delta, d0).item()),
        "offset_rows": offset,
    }
    return ConstrainedDelta(delta=delta, diagnostics=diag)


def causal_filter(candidates: torch.Tensor, causal_effects: torch.Tensor, top_k: int = 4) -> torch.Tensor:
    """Select candidate directions by measured causal effect, not by norm."""
    c = as_tensor(candidates)
    e = as_tensor(causal_effects).flatten()
    if c.shape[0] != e.numel():
        raise ValueError("candidates and effects must have the same first dimension")
    k = min(max(1, top_k), e.numel())
    idx = torch.topk(e.abs(), k=k).indices
    return c[idx]


def self_test() -> dict:
    torch.manual_seed(0)
    d = 32
    delta0 = torch.randn(d)
    target = torch.randn(d)
    ident = torch.randn(d, 3)
    retain = torch.randn(d, 4)
    out = solve_constrained_delta(delta0, target, ident, retain, neutral_value=0.0)
    assert abs(out.diagnostics["target_projection_after"]) < 1e-4
    assert out.diagnostics["identity_leak_norm"] < 1e-4
    filt = causal_filter(torch.randn(6, d), torch.tensor([0.1, 0.9, 0.2, -0.8, 0.0, 0.3]), top_k=2)
    assert filt.shape == (2, d)
    return {"status": "ok", "diagnostics": out.diagnostics}


__all__ = ["ConstrainedDelta", "solve_constrained_delta", "causal_filter", "self_test"]

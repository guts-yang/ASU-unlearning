"""CF-09: structured control-family residualization.

Removes relation-swap, paraphrase, subword-matched and frequency-matched
control directions from ΔA before testing factual specificity.
"""

from __future__ import annotations

import torch

from .base import as_tensor, orthonormalize, project_out, ridge_coefficients, safe_cosine


def build_control_basis(controls: list[torch.Tensor]) -> torch.Tensor:
    """Stack control ΔA directions ``[D]`` or ``[K,D]`` into ``[D,K]``."""
    rows = []
    for c in controls:
        x = as_tensor(c)
        rows.append(x.flatten() if x.ndim == 1 else x)
    if not rows:
        return torch.zeros(0, 0)
    return orthonormalize(torch.stack(rows, dim=1))


def residualize(delta: torch.Tensor, controls: list[torch.Tensor], ridge: float = 1e-6) -> tuple[torch.Tensor, dict]:
    d = as_tensor(delta).flatten()
    b = build_control_basis(controls)
    if b.numel() == 0:
        return d, {"control_rank": 0, "residual_fraction": 1.0}
    coef = ridge_coefficients(d, b, ridge=ridge)
    fit = b @ coef
    residual = d - fit
    return residual, {
        "control_rank": int(b.shape[1]),
        "control_fit_norm": float(fit.norm().item()),
        "residual_norm": float(residual.norm().item()),
        "residual_fraction": float(residual.norm().item() / max(d.norm().item(), 1e-8)),
        "cosine_residual": float(safe_cosine(residual, d).item()),
        "max_control_cosine": float(max(abs(safe_cosine(b[:, i], d).item()) for i in range(b.shape[1]))),
    }


def output_shortcut_metrics(donor_rate: float, eos_rate: float, full_rate: float) -> dict:
    return {
        "donor_rate": float(donor_rate),
        "eos_rate": float(eos_rate),
        "full_answer_rate": float(full_rate),
        "pass_no_shortcut": donor_rate <= 0.10 and eos_rate <= 0.20 and full_rate >= 0.70,
    }


def self_test() -> dict:
    torch.manual_seed(8)
    d0 = torch.randn(32)
    controls = [d0 * 0.5 + 0.01 * torch.randn(32), torch.randn(32)]
    residual, diag = residualize(d0, controls)
    assert residual.shape == d0.shape
    assert diag["residual_fraction"] < 0.95
    assert output_shortcut_metrics(0.02, 0.1, 0.9)["pass_no_shortcut"]
    return {"status": "ok", "diagnostics": diag}


__all__ = ["build_control_basis", "residualize", "output_shortcut_metrics", "self_test"]

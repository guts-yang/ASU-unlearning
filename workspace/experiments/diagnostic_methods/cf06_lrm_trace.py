"""CF-06: LRM reasoning-trace ΔA.

The algebra is implemented here so the LRM track can reuse it, but the module
fails closed on an LLM-only backbone instead of pretending the trace exists.
"""

from __future__ import annotations

import torch

from .base import as_tensor, safe_cosine


class NotApplicableError(RuntimeError):
    pass


def require_lrm_backbone(has_cot: bool, model_name: str = "") -> None:
    if not has_cot:
        raise NotApplicableError(
            f"CF-06 requires a reasoning model with CoT traces; {model_name or 'this backbone'} has none"
        )


def trace_importance(step_nll: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Convert per-step forget NLL into a normalized ρ(t) distribution."""
    x = as_tensor(step_nll).flatten()
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(x / temperature, dim=0)


def trace_weighted_delta(
    hidden_plus: torch.Tensor,
    hidden_minus: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    hf = as_tensor(hidden_plus)
    hm = as_tensor(hidden_minus)
    w = as_tensor(weights).flatten()
    if hf.shape != hm.shape:
        raise ValueError("hidden_plus and hidden_minus must have the same shape")
    if hf.ndim == 3:
        if w.numel() != hf.shape[1]:
            raise ValueError("weights length must match the time dimension")
        return ((hf - hm) * w.view(1, -1, 1)).sum(dim=1)
    if hf.ndim == 2:
        if w.numel() != hf.shape[0]:
            raise ValueError("weights length must match the sequence dimension")
        return ((hf - hm) * w.view(-1, 1)).sum(dim=0)
    raise ValueError("hidden states must be [T,D] or [B,T,D]")


def trace_diagnostics(hidden_plus: torch.Tensor, hidden_minus: torch.Tensor, weights: torch.Tensor) -> dict:
    delta = trace_weighted_delta(hidden_plus, hidden_minus, weights)
    w = as_tensor(weights).flatten()
    return {
        "delta_norm": float(delta.norm().item()),
        "weight_entropy": float(-(w * (w.clamp_min(1e-12).log())).sum().item()),
        "weight_max": float(w.max().item()),
        "weight_min": float(w.min().item()),
        "cosine_unweighted": float(safe_cosine(delta, (as_tensor(hidden_plus) - as_tensor(hidden_minus)).mean(0)).item()) if as_tensor(hidden_plus).ndim == 2 else None,
    }


def self_test() -> dict:
    torch.manual_seed(5)
    hf = torch.randn(6, 8)
    hm = torch.randn(6, 8)
    w = trace_importance(torch.tensor([0.1, 0.8, 0.1, 0.2, 0.7, 0.1]))
    out = trace_diagnostics(hf, hm, w)
    assert abs(w.sum().item() - 1.0) < 1e-5
    try:
        require_lrm_backbone(False, "llama")
    except NotApplicableError:
        pass
    else:
        raise AssertionError("expected NotApplicableError")
    return {"status": "ok", "diagnostics": out}


__all__ = ["NotApplicableError", "require_lrm_backbone", "trace_importance", "trace_weighted_delta", "trace_diagnostics", "self_test"]

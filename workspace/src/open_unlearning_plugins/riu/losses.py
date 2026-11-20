"""Differentiable building blocks for Recovery-Invariant Unlearning (RIU).

The recovery operator set R is approximated by cheap, differentiable
surrogates that preserve model utility:

  identity  : original prompt/activation
  noise     : Gaussian activation perturbation (short-finetuning surrogate)
  round     : straight-through INT8-style activation rounding (quantization)
  amplify   : re-amplification of fact directions (FocusOnKey surrogate)
  dropout   : random feature dropout (alternative access-path surrogate)

These are *proxies*: they do not replace the full LeakRU attack suite, and the
limitation is recorded in workspace/docs/RIU_研究进展_20261120.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -100
RECOVERY_OPS = ("identity", "noise", "round", "amplify", "dropout")


class GradientReversal(torch.autograd.Function):
    """Reverse (and scale) the gradient flowing to the representation."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def gradient_reversal(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)


class LinearProbe(nn.Module):
    """Small linear probe that tries to decode target information from h^l."""

    def __init__(self, dim: int, hidden: int = 64, out_dim: int = 1):
        super().__init__()
        self.out_dim = int(out_dim)
        self.net = (
            nn.Sequential(
                nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, self.out_dim)
            )
            if hidden and hidden > 0
            else nn.Linear(dim, self.out_dim)
        )
        self.register_buffer("fact_dir", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.squeeze(-1) if self.out_dim == 1 else logits


def masked_mean(hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Pool hidden states over answer tokens (labels != -100)."""
    mask = labels.ne(IGNORE_INDEX).unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def recovery_transform(
    h: torch.Tensor,
    op: str,
    noise: float = 0.05,
    amplify: float = 0.2,
    drop: float = 0.1,
) -> torch.Tensor:
    """Apply one differentiable recovery surrogate to pooled hidden states."""
    if op == "identity":
        return h
    if op == "noise":
        return h + float(noise) * torch.randn_like(h)
    if op == "amplify":
        return h * (1.0 + float(amplify))
    if op == "dropout":
        keep = (torch.rand_like(h) > float(drop)) / max(1.0 - float(drop), 1e-6)
        return h * keep
    if op == "round":
        # Straight-through INT8-style rounding: forward is quantized, backward
        # is identity, so the model can be optimized through the simulation.
        scale = h.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 127.0
        quantized = torch.round(h / scale).clamp(-127.0, 127.0) * scale
        return h + (quantized - h).detach()
    raise ValueError(f"Unknown recovery operator: {op!r}")


def update_fact_direction(
    fact_dir: torch.Tensor | None,
    h_forget: torch.Tensor,
    h_retain: torch.Tensor,
    momentum: float = 0.9,
) -> torch.Tensor:
    """Running estimate of the forget-vs-retain (fact) direction.

    Only a rank-1 direction is kept: with batch size 2 a full PCA subspace is
    numerically unreliable.  The rank-k extension is left as future work.
    """
    delta = h_forget.detach().mean(dim=0) - h_retain.detach().mean(dim=0)
    norm = delta.norm().clamp_min(1e-6)
    delta = delta / norm
    if fact_dir is None or torch.count_nonzero(fact_dir) == 0:
        return delta
    updated = float(momentum) * fact_dir.detach() + (1.0 - float(momentum)) * delta
    return updated / updated.norm().clamp_min(1e-6)


def riu_geometry_loss(
    h_forget: torch.Tensor, fact_dir: torch.Tensor, mode: str = "corrected"
) -> torch.Tensor:
    """Geometry term with both the proposal formula and its corrected form.

    - ``as_written``: minimize ||Proj_{S_f^perp} h||, exactly as in proposal
      v5 Eq. 6.5.  Because ||h||^2 = ||Proj_{S_f} h||^2 + ||Proj_{S_f^perp} h||^2,
      minimizing the orthogonal part pushes h *into* the fact subspace.
    - ``corrected`` (default): minimize ||Proj_{S_f} h||^2, i.e. erase the fact
      direction while leaving the orthogonal complement intact.
    """
    direction = fact_dir.to(h_forget.dtype)
    if direction.norm() < 1e-6:
        return h_forget.new_zeros(())
    unit = direction / direction.norm()
    projection = (h_forget * unit).sum(dim=-1, keepdim=True) * unit
    if mode == "as_written":
        return (h_forget - projection).pow(2).sum(dim=-1).mean()
    if mode == "corrected":
        return projection.pow(2).sum(dim=-1).mean()
    raise ValueError(f"Unknown geometry mode: {mode!r}")


def riu_consistency_loss(
    h_forget: torch.Tensor,
    op_a: str,
    op_b: str,
    noise: float = 0.05,
    amplify: float = 0.2,
    drop: float = 0.1,
) -> torch.Tensor:
    """Collapse representations under different recovery transforms."""
    a = recovery_transform(h_forget, op_a, noise, amplify, drop)
    b = recovery_transform(h_forget, op_b, noise, amplify, drop)
    return F.mse_loss(a, b)


def riu_probe_loss(
    h_forget: torch.Tensor,
    h_retain: torch.Tensor,
    probe: nn.Module,
    ops: tuple[str, ...] | list[str],
    grl_lambda: float,
    noise: float = 0.05,
    amplify: float = 0.2,
    drop: float = 0.1,
) -> torch.Tensor:
    """Adversarial inner max: worst-case linear decodability under R.

    The probe is trained to distinguish forget from retain representations;
    the gradient-reversal layer makes the backbone do the opposite.
    """
    if not ops:
        ops = ("identity",)
    retain = recovery_transform(h_retain, "identity", noise, amplify, drop)
    targets = torch.cat(
        [
            torch.ones(h_forget.shape[0], device=h_forget.device),
            torch.zeros(retain.shape[0], device=retain.device),
        ],
        dim=0,
    )
    losses = []
    for op in ops:
        forget = recovery_transform(h_forget, op, noise, amplify, drop)
        features = torch.cat(
            [
                gradient_reversal(forget, grl_lambda),
                gradient_reversal(retain, grl_lambda),
            ],
            dim=0,
        )
        logits = probe(features)
        losses.append(F.binary_cross_entropy_with_logits(logits, targets))
    # Inner max: the most decodable recovery operation dominates the gradient.
    return torch.max(torch.stack(losses))


def riu_fact_probe_loss(
    h_forget: torch.Tensor,
    fact_labels: torch.Tensor,
    probe: nn.Module,
    num_classes: int,
    ops: tuple[str, ...] | list[str],
    grl_lambda: float,
    noise: float = 0.05,
    amplify: float = 0.2,
    drop: float = 0.1,
) -> torch.Tensor:
    """Inner max over R of linear decodability of the fact identity y_f.

    ``fact_labels`` are integer class ids derived from the forget-set index
    (TOFU forget05: 200 QAs = 10 authors x 20 QAs, so ``index // 20``).
    """
    if not ops:
        ops = ("identity",)
    targets = fact_labels.long().to(h_forget.device)
    losses = []
    for op in ops:
        features = gradient_reversal(
            recovery_transform(h_forget, op, noise, amplify, drop), grl_lambda
        )
        logits = probe(features)
        losses.append(F.cross_entropy(logits, targets))
    return torch.max(torch.stack(losses))


def hsic_linear_loss(features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Linear HSIC: a differentiable, probe-free dependence upper bound.

    Minimizing ||Cov(h, y)||_F^2 suppresses linearly decodable information
    without an adversarial inner loop.  ``labels`` may be class ids ([B]) or
    one-hot / indicator vectors ([B, C]).
    """
    x = features.float()
    if labels.dim() == 1:
        y = F.one_hot(labels.long(), num_classes=int(labels.max().item()) + 1).float()
    else:
        y = labels.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cross_cov = x.transpose(0, 1) @ y / max(x.shape[0] - 1, 1)
    return cross_cov.pow(2).sum()

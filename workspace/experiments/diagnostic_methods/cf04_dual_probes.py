"""CF-04: fact / identity dual linear probes.

The existing repository only computed PCA spectra.  This module adds trained
linear probes and the AUC margin test required by CF-04.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .base import as_tensor, auc_score, safe_cosine


@dataclass
class LinearProbe:
    weight: torch.Tensor
    bias: torch.Tensor
    history: list[dict]

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return as_tensor(x) @ self.weight + self.bias


def fit_linear_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 300,
    lr: float = 0.05,
    l2: float = 1e-3,
) -> LinearProbe:
    x = as_tensor(x)
    y = as_tensor(y).flatten().float()
    if x.shape[0] != y.numel():
        raise ValueError("x and y must have the same number of rows")
    w = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device, requires_grad=True)
    b = torch.zeros((), dtype=x.dtype, device=x.device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    history = []
    for step in range(steps):
        logits = x @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y) + l2 * (w * w).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.item())})
    return LinearProbe(w.detach(), b.detach(), history)


def dual_probe_report(
    fact_x: torch.Tensor,
    fact_y: torch.Tensor,
    identity_x: torch.Tensor,
    identity_y: torch.Tensor,
    steps: int = 300,
    lr: float = 0.05,
) -> dict:
    fact_probe = fit_linear_probe(fact_x, fact_y, steps=steps, lr=lr)
    id_probe = fit_linear_probe(identity_x, identity_y, steps=steps, lr=lr)
    fact_scores = fact_probe.score(fact_x)
    id_scores = id_probe.score(identity_x)
    auc_f = auc_score(fact_scores, fact_y)
    auc_id = auc_score(id_scores, identity_y)
    cos = float(safe_cosine(fact_probe.weight, id_probe.weight).item())
    return {
        "auc_fact": auc_f,
        "auc_identity": auc_id,
        "auc_margin": auc_f - auc_id,
        "probe_cosine": cos,
        "fact_probe_norm": float(fact_probe.weight.norm().item()),
        "identity_probe_norm": float(id_probe.weight.norm().item()),
        "pass": auc_f - auc_id >= 0.10 and abs(cos) <= 0.8,
    }


def pooled_hidden(hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean-pool ``[B,T,D]`` over valid tokens into ``[B,D]``."""
    h = as_tensor(hidden)
    if mask is None:
        return h.mean(dim=1)
    m = as_tensor(mask).to(h.dtype).unsqueeze(-1)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def self_test() -> dict:
    torch.manual_seed(3)
    n, d = 80, 16
    fact_x = torch.randn(n, d)
    fact_y = (fact_x[:, 0] + 0.3 * fact_x[:, 1] > 0).float()
    id_x = torch.randn(n, d)
    id_y = (id_x[:, 5] - 0.2 * id_x[:, 6] > 0).float()
    rep = dual_probe_report(fact_x, fact_y, id_x, id_y, steps=120)
    assert rep["auc_fact"] > 0.8
    assert rep["auc_identity"] > 0.8
    return {"status": "ok", "report": rep}


__all__ = ["LinearProbe", "fit_linear_probe", "dual_probe_report", "pooled_hidden", "self_test"]

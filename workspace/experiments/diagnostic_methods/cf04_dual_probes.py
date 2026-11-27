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
    y = as_tensor(y).flatten().float().to(x.device)
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


def _split(x: torch.Tensor, y: torch.Tensor, train_fraction: float = 0.6):
    x = as_tensor(x)
    y = as_tensor(y).to(x.device)
    n = x.shape[0]
    order = torch.randperm(n, device=x.device)
    n_train = max(2, min(n - 2, int(n * train_fraction)))
    tr, te = order[:n_train], order[n_train:]
    return x[tr], y[tr], x[te], y[te]


def fit_dual_probes(
    fact_x: torch.Tensor,
    fact_y: torch.Tensor,
    identity_x: torch.Tensor,
    identity_y: torch.Tensor,
    steps: int = 300,
    lr: float = 0.05,
):
    """Fit fact/identity probes on a train split and evaluate on a held-out split."""
    fx, fy, ftx, fty = _split(fact_x, fact_y)
    ix, iy, itx, ity = _split(identity_x, identity_y)
    fact_probe = fit_linear_probe(fx, fy, steps=steps, lr=lr)
    id_probe = fit_linear_probe(ix, iy, steps=steps, lr=lr)
    fact_scores = fact_probe.score(ftx)
    id_scores = id_probe.score(itx)
    auc_f = auc_score(fact_scores, fty)
    auc_id = auc_score(id_scores, ity)
    cos = float(safe_cosine(fact_probe.weight, id_probe.weight).item())
    report = {
        "auc_fact": auc_f,
        "auc_identity": auc_id,
        "auc_margin": auc_f - auc_id,
        "probe_cosine": cos,
        "fact_probe_norm": float(fact_probe.weight.norm().item()),
        "identity_probe_norm": float(id_probe.weight.norm().item()),
        "train_size": int(fx.shape[0]),
        "test_size": int(ftx.shape[0]),
        "pass": auc_f - auc_id >= 0.10 and abs(cos) <= 0.8,
    }
    return fact_probe, id_probe, report


def dual_probe_report(
    fact_x: torch.Tensor,
    fact_y: torch.Tensor,
    identity_x: torch.Tensor,
    identity_y: torch.Tensor,
    steps: int = 300,
    lr: float = 0.05,
) -> dict:
    _, _, report = fit_dual_probes(fact_x, fact_y, identity_x, identity_y, steps=steps, lr=lr)
    return report


def probe_guided_loss(
    forget_hidden: torch.Tensor,
    retain_hidden: torch.Tensor,
    fact_probe: LinearProbe,
    identity_probe: LinearProbe,
    identity_weight: float = 0.5,
    retain_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Push forget activations out of the fact probe while preserving identity and retain.

    The probes are frozen.  ``softplus`` is used so a correctly classified
    forget fact (high fact score) receives a gradient, while already-negative
    samples stop dominating the loss.
    """
    hf = pooled_hidden(forget_hidden)
    hr = pooled_hidden(retain_hidden)
    fact_forget = fact_probe.score(hf)
    identity_forget = identity_probe.score(hf)
    fact_retain = fact_probe.score(hr)
    l_fact = F.binary_cross_entropy_with_logits(fact_forget, torch.zeros_like(fact_forget))
    l_identity = F.binary_cross_entropy_with_logits(identity_forget, torch.ones_like(identity_forget))
    l_retain = F.binary_cross_entropy_with_logits(fact_retain, torch.ones_like(fact_retain))
    loss = l_fact + identity_weight * l_identity + retain_weight * l_retain
    return {
        "loss": loss,
        "fact_forget": l_fact,
        "identity_preserve": l_identity,
        "retain_fact_preserve": l_retain,
    }


def fit_layerwise_dual_probes(
    layer_fact_x: dict[int, torch.Tensor],
    fact_y: torch.Tensor,
    layer_identity_x: dict[int, torch.Tensor],
    identity_y: torch.Tensor,
    steps: int = 300,
    lr: float = 0.05,
):
    """Fit independent fact/identity probes for each layer.

    The same labels are used across layers, while each layer gets its own probe.
    Layer weights are proportional to the held-out AUC margin, with a uniform
    fallback when all margins are non-positive.
    """
    probes = {}
    report = {}
    weights = {}
    for layer in sorted(layer_fact_x):
        fp, ip, rep = fit_dual_probes(
            layer_fact_x[layer], fact_y, layer_identity_x[layer], identity_y, steps=steps, lr=lr
        )
        probes[layer] = (fp, ip)
        report[layer] = rep
        weights[layer] = max(0.0, float(rep["auc_margin"]))
    if sum(weights.values()) <= 1e-8:
        weights = {k: 1.0 for k in weights}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    return probes, weights, report


def probe_guided_loss_v2(
    forget_hiddens: dict[int, torch.Tensor],
    retain_hiddens: dict[int, torch.Tensor],
    probes: dict[int, tuple["LinearProbe", "LinearProbe"]],
    layer_weights: dict[int, float] | None = None,
    margin: float = 0.5,
    temperature: float = 1.0,
    identity_weight: float = 1.0,
    retain_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Margin-based multi-layer probe loss.

    Fact scores are pushed below ``-margin`` via softplus; the BCE version
    saturated at zero on c11 and stopped providing a forget gradient.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    layers = sorted(forget_hiddens)
    weights = layer_weights or {k: 1.0 / len(layers) for k in layers}
    total = 0.0
    parts = {"fact": [], "identity": [], "retain_fact": []}
    for layer in layers:
        fp, ip = probes[layer]
        hf = pooled_hidden(forget_hiddens[layer]) / temperature
        hr = pooled_hidden(retain_hiddens[layer]) / temperature
        f_score = fp.score(hf)
        i_score = ip.score(hf)
        r_score = fp.score(hr)
        l_fact = F.softplus(f_score + margin).mean()
        l_id = F.softplus(-i_score + 0.5 * margin).mean()
        l_ret = F.softplus(-r_score + margin).mean()
        w = float(weights.get(layer, 0.0))
        total = total + w * (l_fact + identity_weight * l_id + retain_weight * l_ret)
        parts["fact"].append(l_fact.detach())
        parts["identity"].append(l_id.detach())
        parts["retain_fact"].append(l_ret.detach())
    return {
        "loss": total,
        "fact_forget": torch.stack(parts["fact"]).mean(),
        "identity_preserve": torch.stack(parts["identity"]).mean(),
        "retain_fact_preserve": torch.stack(parts["retain_fact"]).mean(),
    }


def project_conflicting_grad(fact_loss: torch.Tensor, retain_loss: torch.Tensor, parameters):
    """PCGrad-style projection of fact gradients conflicting with retain gradients."""
    params = [p for p in parameters if p.requires_grad]
    gf = torch.autograd.grad(fact_loss, params, retain_graph=True, allow_unused=True)
    gr = torch.autograd.grad(retain_loss, params, allow_unused=True)
    cosines = []
    for p, f, r in zip(params, gf, gr):
        if f is None or r is None:
            p.grad = f
            continue
        denom = (r * r).sum().clamp_min(1e-12)
        dot = (f * r).sum()
        cos = dot / (f.norm().clamp_min(1e-12) * r.norm().clamp_min(1e-12))
        cosines.append(float(cos.item()))
        p.grad = f - (dot / denom).clamp_min(0.0) * r
    return sum(cosines) / max(1, len(cosines))


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
    fp, ip, rep = fit_dual_probes(fact_x, fact_y, id_x, id_y, steps=120)
    assert rep["auc_fact"] > 0.75
    assert rep["auc_identity"] > 0.70
    losses = probe_guided_loss(torch.randn(4, 5, d), torch.randn(4, 5, d), fp, ip)
    assert losses["loss"].ndim == 0

    layer_fact = {8: torch.randn(n, d), 12: torch.randn(n, d)}
    layer_id = {8: torch.randn(n, d), 12: torch.randn(n, d)}
    probes, weights, _ = fit_layerwise_dual_probes(layer_fact, fact_y, layer_id, id_y, steps=80)
    v2 = probe_guided_loss_v2(
        {8: torch.randn(4, 5, d), 12: torch.randn(4, 5, d)},
        {8: torch.randn(4, 5, d), 12: torch.randn(4, 5, d)},
        probes, weights, margin=0.5,
    )
    assert v2["loss"].ndim == 0
    return {"status": "ok", "report": rep, "probe_loss": float(losses["loss"].item()),
            "v2_loss": float(v2["loss"].item())}


__all__ = [
    "LinearProbe", "fit_linear_probe", "fit_dual_probes", "dual_probe_report",
    "probe_guided_loss", "fit_layerwise_dual_probes", "probe_guided_loss_v2",
    "project_conflicting_grad", "pooled_hidden", "self_test",
]

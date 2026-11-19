#!/usr/bin/env python3
"""Offline smoke tests for RIU components (no model download, no GPU needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "open-unlearning" / "src"), str(REPO / "workspace" / "src")]

from open_unlearning_plugins.riu.losses import (  # noqa: E402
    RECOVERY_OPS,
    LinearProbe,
    gradient_reversal,
    recovery_transform,
    riu_consistency_loss,
    riu_fact_probe_loss,
    riu_geometry_loss,
    riu_probe_loss,
    update_fact_direction,
)


def test_grl_sign() -> None:
    x = torch.ones(1, requires_grad=True)
    gradient_reversal(x, 2.0).sum().backward()
    assert torch.allclose(x.grad, -2.0 * torch.ones(1)), x.grad


def test_recovery_ops() -> None:
    h = torch.randn(4, 8)
    for op in RECOVERY_OPS:
        out = recovery_transform(h, op, noise=0.1, amplify=0.3, drop=0.2)
        assert out.shape == h.shape, op
        assert torch.isfinite(out).all(), op
    # Straight-through rounding changes forward but keeps backward identity.
    h_st = torch.randn(4, 8, requires_grad=True)
    out = recovery_transform(h_st, "round")
    assert not torch.allclose(out, h_st)
    out.sum().backward()
    assert torch.allclose(h_st.grad, torch.ones_like(h_st))


def test_probe_loss_backward() -> None:
    torch.manual_seed(0)
    h_f = torch.randn(4, 8, requires_grad=True)
    h_r = torch.randn(4, 8)
    probe = LinearProbe(8, hidden=4)
    loss = riu_probe_loss(
        h_f, h_r, probe, ("identity", "noise", "round"), grl_lambda=0.1
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert h_f.grad is not None and torch.isfinite(h_f.grad).all()
    assert probe.net[0].weight.grad is not None


def test_geometry_direction() -> None:
    torch.manual_seed(1)
    h_f = torch.randn(8, 8, requires_grad=True)
    h_r = torch.randn(8, 8) + 3.0
    direction = update_fact_direction(None, h_f.detach(), h_r.detach())
    before = riu_geometry_loss(h_f.detach(), direction).item()
    opt = torch.optim.SGD([h_f], lr=0.5)
    for _ in range(20):
        opt.zero_grad()
        loss = riu_geometry_loss(h_f, direction)
        loss.backward()
        opt.step()
    after = riu_geometry_loss(h_f.detach(), direction).item()
    assert after < before, (before, after)


def test_consistency_direction() -> None:
    torch.manual_seed(2)
    h_f = torch.randn(8, 8, requires_grad=True)
    # Deterministic pair: stochastic ops would change the target each call.
    before = riu_consistency_loss(h_f, "identity", "amplify", 0.3, 0.2, 0.2).item()
    opt = torch.optim.SGD([h_f], lr=0.2)
    for _ in range(20):
        opt.zero_grad()
        loss = riu_consistency_loss(h_f, "identity", "amplify", 0.3, 0.2, 0.2)
        loss.backward()
        opt.step()
    after = riu_consistency_loss(h_f.detach(), "identity", "amplify", 0.3, 0.2, 0.2).item()
    assert after < before, (before, after)


def test_fact_probe_loss() -> None:
    torch.manual_seed(3)
    h_f = torch.randn(6, 8, requires_grad=True)
    labels = torch.arange(6) % 3
    probe = LinearProbe(8, hidden=4, out_dim=3)
    logits = probe(h_f)
    assert logits.shape == (6, 3)
    loss = riu_fact_probe_loss(
        h_f, labels, probe, 3, ("identity", "noise"), grl_lambda=0.1
    )
    loss.backward()
    assert torch.isfinite(loss) and h_f.grad is not None


def test_geometry_modes_are_opposite() -> None:
    torch.manual_seed(4)
    h_f = torch.randn(6, 8)
    direction = torch.zeros(8)
    direction[0] = 1.0
    corrected = riu_geometry_loss(h_f, direction, "corrected").item()
    as_written = riu_geometry_loss(h_f, direction, "as_written").item()
    assert abs((corrected + as_written) - h_f.pow(2).sum(dim=-1).mean().item()) < 1e-5


def test_registry() -> None:
    from open_unlearning_plugins.register import register_workspace_trainers

    registry = register_workspace_trainers()
    for name in ("ASU", "RIU", "RIU_GA", "RIU_NPO", "RIU_ASU"):
        assert name in registry, name


if __name__ == "__main__":
    test_grl_sign()
    test_recovery_ops()
    test_probe_loss_backward()
    test_geometry_direction()
    test_consistency_direction()
    test_fact_probe_loss()
    test_geometry_modes_are_opposite()
    test_registry()
    print("RIU smoke tests passed")

"""Retain gate audit: near/general retain, KL and refusal/truncation/repeat deltas."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from ..baseline_adapter import BaselineAdapter
from ..config import FrozenConfig
from ..types import AuditResult, Budget


def run_retain(
    adapter: BaselineAdapter,
    retain_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    baseline_nlls: Sequence[float],
    cfg: FrozenConfig,
    extra_deltas: Mapping[str, float] | None = None,
) -> AuditResult:
    if not retain_batches:
        return AuditResult(arm="retain", independence_statement="not run",
                           budget=Budget(0, 0, 0, 0), raw_metrics={}, control_delta={},
                           coverage="uncovered", note="未覆盖：无 retain 数据")
    nlls = [adapter.retain_nll(ids, labels) for ids, labels in retain_batches]
    drops = [float(b) - float(a) for a, b in zip(baseline_nlls, nlls)]
    kl_proxy = max(0.0, -min(drops)) if drops else 0.0
    metrics = {
        "near_retain_nll_delta": sum(drops) / len(drops),
        "retain_kl_proxy_nats_per_token": kl_proxy,
    }
    for key, value in (extra_deltas or {}).items():
        metrics[f"{key}_delta"] = float(value)
    return AuditResult(
        arm="retain",
        independence_statement="独立于编辑目标的 retain 集合评测；-100 忽略标签",
        budget=Budget(tokens=sum(int(ids.numel()) for ids, _ in retain_batches),
                      steps=0, components=0, rank=0),
        raw_metrics=metrics, control_delta={}, coverage="covered",
    )

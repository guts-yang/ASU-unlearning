"""Designated causal patch audit arm."""

from __future__ import annotations

from typing import Sequence

from ..config import FrozenConfig
from ..types import AuditResult, Budget, PatchResult
from .base import not_run


def run_causal_patch(patches: Sequence[PatchResult], cfg: FrozenConfig) -> AuditResult:
    if not patches:
        return not_run("causal_patch", "未执行任何 patch")
    by_kind: dict[str, list[float]] = {}
    for patch in patches:
        by_kind.setdefault(f"{patch.patch_kind}:{patch.scope}", []).append(patch.recovery_delta)
    metrics = {f"{k}_mean_delta": sum(v) / len(v) for k, v in by_kind.items()}
    joint = by_kind.get("nk:joint", [])
    if joint and max(joint) < 0:
        metrics["joint_effect_detected"] = 1.0
    else:
        metrics["joint_effect_detected"] = 0.0
    return AuditResult(
        arm="causal_patch",
        independence_statement="指定组件与 donor 配对；独立于 reader 训练目标",
        budget=Budget(tokens=0, steps=len(patches), components=len(patches), rank=0),
        raw_metrics=metrics, control_delta={}, coverage="covered",
    )

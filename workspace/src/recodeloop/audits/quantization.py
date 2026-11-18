"""Q4/Q8 quantization stress audit. Unrun => uncovered, never assumed pass."""

from __future__ import annotations

from ..config import FrozenConfig
from ..types import AuditResult, Budget
from .base import not_run


def run_quantization(cfg: FrozenConfig) -> AuditResult:
    arm_cfg = cfg.section("audits")["arms"]["quantization"]
    modes = tuple(arm_cfg.get("modes") or ())
    if arm_cfg.get("implementation_status") == "not_run" or not modes:
        return not_run("quantization", "未发现实际量化后端；Q4/Q8 未运行")
    return not_run("quantization", "量化后端已配置但本次未执行，未覆盖")


def quantization_results(results: dict[str, float] | None, cfg: FrozenConfig) -> AuditResult:
    """Use only when a real quantizer produced results; otherwise uncovered."""
    if not results:
        return run_quantization(cfg)
    return AuditResult(
        arm="quantization",
        independence_statement="独立于编辑与 reader 训练的量化压力测试",
        budget=Budget(0, 0, 0, 0), raw_metrics=dict(results),
        control_delta={}, coverage="covered",
    )

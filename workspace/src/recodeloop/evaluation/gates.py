"""Pre-registered gates. INCONCLUSIVE is never a scientific pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..config import FrozenConfig
from .statistics import cluster_bootstrap, interval_width, paired_sign_flip

Status = str  # PASS | FAIL | INCONCLUSIVE


@dataclass(frozen=True)
class GateDecision:
    status: Status
    reasons: tuple[str, ...]


def evaluate_confirmation(
    cluster_effects: Sequence[float],
    cfg: FrozenConfig,
    paired_differences: Sequence[float] | None = None,
    bootstrap_samples: int | None = None,
) -> GateDecision:
    gates = cfg.section("m0_p08_m2")["gates"]
    stats = cfg.section("audits")["statistics"]
    min_clusters = int(gates["confirmation_min_clusters"])
    reasons: list[str] = []
    if len(cluster_effects) < min_clusters:
        return GateDecision("INCONCLUSIVE", (
            f"confirmation 簇数={len(cluster_effects)} < {min_clusters}；结果探索性，不得输出确认 L2",
        ))
    samples = int(bootstrap_samples or stats["cluster_bootstrap_samples"])
    ci = cluster_bootstrap(cluster_effects, samples, float(stats["ci_level"]))
    if interval_width(ci) > float(stats["equivalence_margin"]) * 4:
        reasons.append(f"簇 bootstrap 区间过宽：{ci}")
    if paired_differences:
        flip_ci = paired_sign_flip(paired_differences, samples, float(stats["ci_level"]))
        if interval_width(flip_ci) > float(stats["equivalence_margin"]) * 4:
            reasons.append(f"配对 sign-flip 区间过宽：{flip_ci}")
    if reasons:
        return GateDecision("INCONCLUSIVE", tuple(reasons))
    min_effect = float(gates["min_related_effect"])
    if min(cluster_effects) <= min_effect:
        return GateDecision("FAIL", (f"最小簇效应 {min(cluster_effects):.4f} <= {min_effect}",))
    return GateDecision("PASS", (f"簇效应均 > {min_effect}",))


def evaluate_retain(metrics: Mapping[str, float], cfg: FrozenConfig) -> GateDecision:
    gates = cfg.section("m0_p08_m2")["gates"]
    reasons: list[str] = []
    for key in ("near_retain_abs_drop", "general_retain_abs_drop"):
        if float(metrics.get(key, 0.0)) > float(gates["retain_max_abs_drop"]):
            reasons.append(f"{key} 超过门限")
    if float(metrics.get("retain_kl_nats_per_token", 0.0)) > \
            float(gates["retain_max_kl_nats_per_token"]):
        reasons.append("retain KL 超过 0.02 nat/token")
    max_inc = float(gates["refusal_truncation_repeat_max_increase"])
    for key in ("refusal_rate_delta", "truncation_rate_delta", "repeat_rate_delta"):
        if float(metrics.get(key, 0.0)) > max_inc:
            reasons.append(f"{key} 超过 1 个百分点")
    return GateDecision("FAIL" if reasons else "PASS", tuple(reasons))


def evaluate_comparisons(metrics: Mapping[str, Mapping[str, float]]) -> GateDecision:
    """Veto rules from the pre-registration. Missing arms => INCONCLUSIVE."""
    required = ("full", "pars_style", "one_shot", "random", "shuffle", "independent")
    missing = [a for a in required if a not in metrics]
    if missing:
        return GateDecision("INCONCLUSIVE", (f"缺少对照臂: {missing}",))
    primary = "fact_value_accuracy"
    full = float(metrics["full"].get(primary, 0.0))
    reasons: list[str] = []
    for arm in ("pars_style", "one_shot", "random", "shuffle"):
        if full <= float(metrics[arm].get(primary, 0.0)):
            reasons.append(f"full 不优于 {arm}")
    independent = float(metrics["independent"].get(primary, 0.0))
    if independent < full:
        reasons.append("编辑 reader 改善但独立 reader 恶化：Goodhart 风险")
    return GateDecision("FAIL" if reasons else "PASS", tuple(reasons))

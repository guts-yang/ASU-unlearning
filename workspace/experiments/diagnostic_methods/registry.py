"""Method registry: one module per candidate, shared interface."""

from __future__ import annotations

import importlib
from types import ModuleType

from .base import MethodSpec

METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("CF-01", "约束反演目标", "cf01_constrained_inverse", requires_training=False,
               core=("solve_constrained_delta", "causal_filter")),
    MethodSpec("CF-02", "三前向角色分解", "cf02_role_decomposition", requires_training=True,
               core=("decompose_roles", "role_loss_terms")),
    MethodSpec("CF-03", "因果门控 activation gate", "cf03_causal_gate", requires_training=True,
               core=("PatchLayer", "patch_effect", "select_causal_gate")),
    MethodSpec("CF-04", "事实身份双探针解耦", "cf04_dual_probes", requires_training=True,
               core=("fit_linear_probe", "dual_probe_report")),
    MethodSpec("CF-05", "多 donor 边际化", "cf05_multi_donor", requires_training=False,
               core=("multi_donor_svd", "marginalized_delta")),
    MethodSpec("CF-06", "LRM 推理轨迹 deltaA", "cf06_lrm_trace", requires_training=True, requires_lrm=True,
               core=("require_lrm_backbone", "trace_weighted_delta")),
    MethodSpec("CF-07", "读出空间 deltaA readout", "cf07_readout_delta", requires_training=False,
               core=("readout_delta", "rotation_invariance")),
    MethodSpec("CF-08", "零空间抗恢复投影", "cf08_nullspace", requires_training=True,
               core=("principal_basis", "nullspace_projection")),
    MethodSpec("CF-09", "结构化对照族残差化", "cf09_control_residual", requires_training=False,
               core=("residualize", "output_shortcut_metrics")),
    MethodSpec("CF-10", "纯度驱动渐进 schedule", "cf10_purity_schedule", requires_training=True,
               core=("purity_scores", "purity_schedule")),
    MethodSpec("R2-C", "抗恢复即目标", "r2c_relearning_aware", requires_training=True,
               core=("inner_attack", "relearning_aware_loss")),
    MethodSpec("R3-2", "ASU 遗忘深度审计基准", "r3_2_depth_audit", requires_training=False,
               core=("summarize_depth_audit",)),
    MethodSpec("R3-3", "浅深解耦双阶段遗忘", "r3_3_two_stage", requires_training=True,
               core=("two_stage_schedule", "alignment_loss")),
)

_BY_ID = {m.method_id: m for m in METHODS}


def get_spec(method_id: str) -> MethodSpec:
    try:
        return _BY_ID[method_id]
    except KeyError as exc:
        raise KeyError(f"unknown method {method_id!r}; choose from {sorted(_BY_ID)}") from exc


def load_module(method_id: str) -> ModuleType:
    spec = get_spec(method_id)
    return importlib.import_module(f".{spec.module}", package=__package__)


def run_self_test(method_id: str) -> dict:
    module = load_module(method_id)
    fn = getattr(module, "self_test", None)
    if fn is None:
        return {"status": "no-self-test", "method": method_id}
    out = fn()
    return {"method": method_id, **(out or {})}


__all__ = ["METHODS", "get_spec", "load_module", "run_self_test"]

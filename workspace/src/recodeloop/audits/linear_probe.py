"""Independent linear probe audit arm."""

from __future__ import annotations

from typing import Sequence

from ..config import FrozenConfig
from ..readers.fact_value import FrozenEmbeddingReader
from ..readers.training import train_edit_reader
from ..types import ActivationBatch, AuditResult, Budget
from ..utils.seed import set_global_seed
from .base import not_run


def run_linear_probe(
    reader: FrozenEmbeddingReader,
    batches: Sequence[ActivationBatch],
    cfg: FrozenConfig,
    seed: int | None = None,
) -> AuditResult:
    arm_cfg = cfg.section("audits")["arms"]["independent_linear_probe"]
    if not batches:
        return not_run("independent_linear_probe", "无 forget_train_reader 激活数据")
    actual_seed = int(seed if seed is not None else arm_cfg["seed"])
    set_global_seed(actual_seed)
    in_dim = batches[0].activations.shape[-1]
    probe = FrozenEmbeddingReader(reader.value_ids, int(in_dim), seed=actual_seed)
    steps = int(arm_cfg["max_steps"])
    _, _ = train_edit_reader(probe, batches, steps, 0.01, actual_seed)
    batch = batches[0]
    score = probe.score(batch, batch.fact_ids, batch.split, ())
    return AuditResult(
        arm="independent_linear_probe",
        independence_statement=(
            f"独立初始化 seed={actual_seed}；仅使用 forget_train_reader 簇；"
            "与编辑 reader 的簇、用途、初始化和种子隔离"
        ),
        budget=Budget(tokens=sum(b.activations.shape[0] for b in batches) * steps,
                      steps=steps, components=1, rank=0),
        raw_metrics={"fact_value_accuracy": score.fact_value_accuracy,
                     "mrr": score.mrr or 0.0, "nk_accuracy": score.nk_accuracy},
        control_delta={}, coverage="covered",
    )

"""Matched random-direction control: same components/rank/energy, no optimization."""

from __future__ import annotations

import torch

from ..baseline_adapter import BaselineAdapter
from ..config import FrozenConfig
from ..localization.causal_effects import InterventionInputs
from ..readers.fact_value import FrozenEmbeddingReader
from ..types import ActivationSpec, ComponentSelection
from .base import lora_edit, make_result, score_inputs


def run_random_direction(adapter: BaselineAdapter, reader: FrozenEmbeddingReader,
                         specs: dict[str, ActivationSpec], forget_inputs: InterventionInputs,
                         retain_ids: torch.Tensor, retain_labels: torch.Tensor,
                         selection: ComponentSelection, cfg: FrozenConfig, seed: int) -> object:
    edit_cfg = cfg.section("editing")
    budget = lora_edit(
        adapter, reader, specs, forget_inputs, retain_ids, retain_labels,
        selection.components, selection.ranks, 0,
        float(edit_cfg["optimizer"]["learning_rate"]), float(edit_cfg["lambda_ret"]),
        float(edit_cfg["energy_g"]), seed, randomize=True,
    )
    return make_result("random_direction", budget,
                       score_inputs(adapter, reader, specs, forget_inputs), cfg, seed,
                       "同组件/同秩/同逐组件能量的随机方向；不做梯度优化")

"""Fixed-causal selection + LoRA (no iterative re-localization)."""

from __future__ import annotations

import torch

from ..baseline_adapter import BaselineAdapter
from ..config import FrozenConfig
from ..localization.causal_effects import InterventionInputs
from ..readers.fact_value import FrozenEmbeddingReader
from ..types import ActivationSpec, ComponentSelection
from .base import lora_edit, make_result, score_inputs


def run_fixed_causal(adapter: BaselineAdapter, reader: FrozenEmbeddingReader,
                     specs: dict[str, ActivationSpec], forget_inputs: InterventionInputs,
                     retain_ids: torch.Tensor, retain_labels: torch.Tensor,
                     selection: ComponentSelection, cfg: FrozenConfig, seed: int) -> object:
    edit_cfg = cfg.section("editing")
    steps = int(cfg.section("readers")["edit_reader"]["train_steps"])
    budget = lora_edit(
        adapter, reader, specs, forget_inputs, retain_ids, retain_labels,
        selection.components, selection.ranks, steps,
        float(edit_cfg["optimizer"]["learning_rate"]), float(edit_cfg["lambda_ret"]),
        float(edit_cfg["energy_g"]), seed,
    )
    return make_result("fixed_causal", budget,
                       score_inputs(adapter, reader, specs, forget_inputs), cfg, seed,
                       "固定因果选中组件 + LoRA；排除普通局部编辑解释")

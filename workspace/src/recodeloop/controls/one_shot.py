"""Equal-total-budget one-shot control."""

from __future__ import annotations

import torch

from ..baseline_adapter import BaselineAdapter
from ..config import FrozenConfig
from ..localization.causal_effects import InterventionInputs
from ..readers.fact_value import FrozenEmbeddingReader
from ..types import ActivationSpec, ComponentSelection
from .base import lora_edit, make_result, score_inputs


def run_one_shot(adapter: BaselineAdapter, reader: FrozenEmbeddingReader,
                 specs: dict[str, ActivationSpec], forget_inputs: InterventionInputs,
                 retain_ids: torch.Tensor, retain_labels: torch.Tensor,
                 selection: ComponentSelection, cfg: FrozenConfig, seed: int) -> object:
    frozen = cfg.section("m0_p08_m2")["frozen_constants"]
    edit_cfg = cfg.section("editing")
    steps = int(frozen["t_max"]) * int(cfg.section("readers")["edit_reader"]["train_steps"])
    budget = lora_edit(
        adapter, reader, specs, forget_inputs, retain_ids, retain_labels,
        selection.components, selection.ranks, steps,
        float(edit_cfg["optimizer"]["learning_rate"]), float(edit_cfg["lambda_ret"]),
        float(edit_cfg["energy_g"]), seed,
    )
    return make_result("one_shot", budget, score_inputs(adapter, reader, specs, forget_inputs),
                       cfg, seed, "等总预算 one-shot；检验闭环必要性")

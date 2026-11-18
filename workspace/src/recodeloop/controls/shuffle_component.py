"""Shuffled-component control: matched count/rank, different real components."""

from __future__ import annotations

import torch

from ..baseline_adapter import BaselineAdapter
from ..config import FrozenConfig
from ..localization.causal_effects import InterventionInputs
from ..readers.fact_value import FrozenEmbeddingReader
from ..types import ActivationSpec, ComponentSelection
from .base import lora_edit, make_result, score_inputs


def run_shuffle_component(adapter: BaselineAdapter, reader: FrozenEmbeddingReader,
                          specs: dict[str, ActivationSpec], forget_inputs: InterventionInputs,
                          retain_ids: torch.Tensor, retain_labels: torch.Tensor,
                          selection: ComponentSelection, cfg: FrozenConfig, seed: int) -> object:
    edit_cfg = cfg.section("editing")
    chosen = set(selection.components)
    shuffled = tuple(c for c in adapter.component_paths() if c not in chosen)[:len(chosen)]
    budget = lora_edit(
        adapter, reader, specs, forget_inputs, retain_ids, retain_labels,
        shuffled, selection.ranks,
        int(cfg.section("readers")["edit_reader"]["train_steps"]),
        float(edit_cfg["optimizer"]["learning_rate"]), float(edit_cfg["lambda_ret"]),
        float(edit_cfg["energy_g"]), seed,
    )
    return make_result("shuffle_component", budget,
                       score_inputs(adapter, reader, specs, forget_inputs), cfg, seed,
                       f"打乱组件：{shuffled}；检验定位特异性")

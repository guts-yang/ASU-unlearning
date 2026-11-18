"""PARS-style same-low-rank parameterization reference.

Full-parameter PARS is only a public-method reference; this arm is NOT claimed
to be compute-equivalent to it.
"""

from __future__ import annotations

import torch

from ..baseline_adapter import BaselineAdapter
from ..config import FrozenConfig
from ..localization.causal_effects import InterventionInputs
from ..readers.fact_value import FrozenEmbeddingReader
from ..types import ActivationSpec, ComponentSelection
from .base import lora_edit, make_result, score_inputs


def run_pars_style(adapter: BaselineAdapter, reader: FrozenEmbeddingReader,
                   specs: dict[str, ActivationSpec], forget_inputs: InterventionInputs,
                   retain_ids: torch.Tensor, retain_labels: torch.Tensor,
                   cfg: FrozenConfig, seed: int) -> object:
    edit_cfg = cfg.section("editing")
    components = tuple(adapter.component_paths())
    rank = int(edit_cfg["candidate_ranks"][0])
    budget = lora_edit(
        adapter, reader, specs, forget_inputs, retain_ids, retain_labels,
        components, tuple(rank for _ in components),
        int(cfg.section("readers")["edit_reader"]["train_steps"]),
        float(edit_cfg["optimizer"]["learning_rate"]), float(edit_cfg["lambda_ret"]),
        float(edit_cfg["energy_g"]), seed,
    )
    return make_result("pars_style", budget,
                       score_inputs(adapter, reader, specs, forget_inputs), cfg, seed,
                       "同低秩参数化参照；不等算力于原始全参数 PARS")

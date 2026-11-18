"""Matched-retain pseudo-intervention: donor from retain, not from never-known."""

from __future__ import annotations

from typing import Mapping

import torch

from ..activations.extraction import extract_activation_batches, extract_raw_donors
from ..activations.patching import patch_components
from ..baseline_adapter import BaselineAdapter
from ..config import FrozenConfig
from ..localization.causal_effects import InterventionInputs
from ..readers.fact_value import FrozenEmbeddingReader
from ..types import ActivationSpec, Budget, ComponentSelection
from .base import make_result, score_inputs


def run_retain_pseudo(adapter: BaselineAdapter, reader: FrozenEmbeddingReader,
                      specs: Mapping[str, ActivationSpec], forget_inputs: InterventionInputs,
                      retain_ids: torch.Tensor, selection: ComponentSelection,
                      cfg: FrozenConfig, seed: int) -> object:
    donors = extract_raw_donors(adapter, selection.components, retain_ids)
    target_len = int(forget_inputs.input_ids.shape[1])
    adapted: dict[str, torch.Tensor] = {}
    for component, donor in donors.items():
        if donor.shape[1] > target_len:
            adapted[component] = donor[:, :target_len, :]
        elif donor.shape[1] < target_len:
            repeats = (target_len + donor.shape[1] - 1) // donor.shape[1]
            adapted[component] = donor.repeat(1, repeats, 1)[:, :target_len, :]
        else:
            adapted[component] = donor
    with patch_components(adapter.model, adapted, kind="nk"):
        metrics = score_inputs(adapter, reader, dict(specs), forget_inputs)
    budget = Budget(tokens=retain_ids.numel(), steps=0,
                    components=len(selection.components), rank=max(selection.ranks, default=0))
    return make_result("retain_pseudo", budget, metrics, cfg, seed,
                       "matched-retain donor 伪干预；不含事实特异性 NK 信息")

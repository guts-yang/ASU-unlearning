"""ASU student / no-op control."""

from __future__ import annotations

from ..baseline_adapter import BaselineAdapter
from ..config import FrozenConfig
from ..localization.causal_effects import InterventionInputs
from ..readers.fact_value import FrozenEmbeddingReader
from ..types import ActivationSpec, Budget, ComponentSelection
from .base import make_result, score_inputs


def run_no_op(adapter: BaselineAdapter, reader: FrozenEmbeddingReader,
              specs: dict[str, ActivationSpec], forget_inputs: InterventionInputs,
              cfg: FrozenConfig, seed: int) -> object:
    metrics = score_inputs(adapter, reader, specs, forget_inputs)
    return make_result("no_op", Budget(0, 0, 0, 0), metrics, cfg, seed,
                       "ASU student / no-op：零预算基线")

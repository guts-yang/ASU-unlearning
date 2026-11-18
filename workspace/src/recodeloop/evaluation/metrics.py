"""Primary metrics: delta_r and fact-value reader summaries."""

from __future__ import annotations

from typing import Mapping

from ..readers.fact_value import FrozenEmbeddingReader
from ..types import ActivationBatch


def main_recovery_advantage(
    reader: FrozenEmbeddingReader,
    forget_batch: ActivationBatch,
    nk_batch: ActivationBatch,
) -> float:
    """delta_r = A_r(M; forget) - A_r(M; never-known), same model."""
    forget = reader.score(forget_batch, forget_batch.fact_ids, forget_batch.split, ())
    nk = reader.score(nk_batch, nk_batch.fact_ids, nk_batch.split, ())
    return float(forget.fact_value_accuracy - nk.fact_value_accuracy)


def score_summary(scores: Mapping[str, float]) -> dict[str, float]:
    keys = ("fact_value_accuracy", "exact_match", "mrr", "nk_accuracy", "confidence")
    return {k: float(scores.get(k, 0.0)) for k in keys}

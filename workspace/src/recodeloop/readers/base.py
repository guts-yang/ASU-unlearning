"""Reader metric helpers (fact values, never-known control)."""

from __future__ import annotations

from typing import Mapping, Sequence


def fact_value_metrics(
    predictions: Sequence[str],
    labels: Sequence[str],
    nk_ids: Sequence[str],
    ranks: Sequence[int] | None = None,
    mrr_values: Sequence[float] | None = None,
    confidences: Sequence[float] | None = None,
) -> Mapping[str, float]:
    if len(predictions) != len(labels):
        raise ValueError("predictions/labels length mismatch")
    nk = set(nk_ids)
    accuracy = sum(p == label for p, label in zip(predictions, labels)) / len(labels)
    nk_rows = [i for i, label in enumerate(labels) if label in nk]
    nk_accuracy = (
        sum(predictions[i] == labels[i] for i in nk_rows) / len(nk_rows)
        if nk_rows else 0.0
    )
    return {
        "fact_value_accuracy": float(accuracy),
        "exact_match": float(accuracy),
        "mrr": float(sum(mrr_values) / len(mrr_values)) if mrr_values else 0.0,
        "confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
        "nk_accuracy": float(nk_accuracy),
    }

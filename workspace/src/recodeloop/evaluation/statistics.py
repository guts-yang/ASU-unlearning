"""Entity-cluster bootstrap and paired sign-flip (vectorized, no p-value equivalence)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..utils.seed import set_global_seed


def cluster_bootstrap(
    cluster_values: Sequence[float],
    samples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 1001,
) -> tuple[float, float]:
    values = np.asarray(cluster_values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(samples), values.size), replace=True).mean(axis=1)
    alpha = (1.0 - float(ci_level)) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def paired_sign_flip(
    paired_differences: Sequence[float],
    samples: int = 10000,
    ci_level: float = 0.95,
    seed: int = 1002,
) -> tuple[float, float]:
    diffs = np.asarray(paired_differences, dtype=float)
    if diffs.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(int(samples), diffs.size))
    draws = (signs * diffs).mean(axis=1)
    alpha = (1.0 - float(ci_level)) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def interval_width(interval: tuple[float, float]) -> float:
    if any(np.isnan(v) for v in interval):
        return float("inf")
    return float(interval[1] - interval[0])

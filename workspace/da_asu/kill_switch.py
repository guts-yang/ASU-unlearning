"""Cluster-bootstrap slope test. Problems are the resampling unit, not tokens."""

import numpy as np

from .protocol import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EFFECT_SIZE_KILL,
    MECHANISM_CONTRAST,
    MIN_PAIRED_PROBLEMS,
    TAXONOMY_CONTRAST,
)


def ols_slope(taus, values):
    taus = np.asarray(taus, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    centered = taus - taus.mean()
    denom = float((centered * centered).sum())
    if denom == 0.0:
        return 0.0
    return float((centered * (values - values.mean())).sum() / denom)


def problem_slopes(per_problem_means, taus):
    """``per_problem_means`` maps problem id to a list of class means, one per tau."""
    return {problem_id: ols_slope(taus, means) for problem_id, means in per_problem_means.items()}


def paired_slope_gaps(slopes_a, slopes_b):
    shared = sorted(set(slopes_a) & set(slopes_b))
    gaps = np.asarray([slopes_a[key] - slopes_b[key] for key in shared], dtype=np.float64)
    return shared, gaps


def cohens_d(gaps):
    gaps = np.asarray(gaps, dtype=np.float64)
    if gaps.size < 2:
        return None
    std = float(gaps.std(ddof=1))
    mean = float(gaps.mean())
    if std == 0.0:
        if mean == 0.0:
            return 0.0
        return float("inf") if mean > 0 else float("-inf")
    return mean / std


def bootstrap_mean_ci(gaps, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    gaps = np.asarray(gaps, dtype=np.float64)
    if gaps.size == 0:
        return None
    rng = np.random.default_rng(seed)
    if gaps.size == 1:
        value = float(gaps[0])
        return [value, value]
    picks = rng.integers(0, gaps.size, size=(draws, gaps.size))
    means = gaps[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]


def _contrast_report(slopes_left, slopes_right, left_name, right_name, can_kill, min_problems):
    shared, gaps = paired_slope_gaps(slopes_left, slopes_right)
    effect = cohens_d(gaps)
    ci = bootstrap_mean_ci(gaps)
    covers_zero = ci is not None and ci[0] <= 0.0 <= ci[1]
    small_effect = effect is not None and abs(effect) < EFFECT_SIZE_KILL
    enough = gaps.size >= min_problems
    kill = bool(can_kill and enough and (covers_zero or small_effect))
    if not enough:
        status = "insufficient_data"
    elif kill:
        status = "kill"
    else:
        status = "pass"
    return {
        "contrast": f"{left_name}_vs_{right_name}",
        "role": "mechanism_kill" if can_kill else "report_only",
        "n_problems": int(gaps.size),
        "mean_slope_gap": None if gaps.size == 0 else float(gaps.mean()),
        "cohens_d": None if effect is None else effect,
        "ci95": ci,
        "ci_covers_zero": covers_zero,
        "abs_d_below_threshold": small_effect,
        "threshold_d": EFFECT_SIZE_KILL,
        "status": status,
        "kill": kill,
    }


def decide(class_slopes, min_problems=MIN_PAIRED_PROBLEMS):
    """``class_slopes`` maps class name to ``{problem_id: slope}``.

    Entropy is intentionally absent. A factual overlap does not kill.
    """
    mechanism = _contrast_report(
        class_slopes.get("derivation", {}),
        class_slopes.get("function", {}),
        MECHANISM_CONTRAST[0],
        MECHANISM_CONTRAST[1],
        can_kill=True,
        min_problems=min_problems,
    )
    taxonomy = _contrast_report(
        class_slopes.get("derivation", {}),
        class_slopes.get("factual", {}),
        TAXONOMY_CONTRAST[0],
        TAXONOMY_CONTRAST[1],
        can_kill=False,
        min_problems=min_problems,
    )
    if mechanism["status"] == "insufficient_data":
        decision = "insufficient_data"
    elif mechanism["kill"]:
        decision = "kill"
    else:
        decision = "proceed"
    return {
        "decision": decision,
        "mechanism": mechanism,
        "taxonomy": taxonomy,
        "entropy_used": False,
    }

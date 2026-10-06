"""L4 go / kill-switch rules. Real verdicts are deferred until RR exists."""

import random


DEFERRED = "deferred"
GO = "go"
KILL_SWITCH = "kill-switch"


def bootstrap_mean_ci(values, n_boot=2000, alpha=0.05, rng=None):
    if not values:
        raise ValueError("values must be non-empty")
    rng = rng or random.Random(0)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_i = int((alpha / 2) * n_boot)
    hi_i = int((1 - alpha / 2) * n_boot) - 1
    hi_i = max(hi_i, lo_i)
    return means[lo_i], means[hi_i]


def channel_significant(paired_deltas, n_boot=2000, alpha=0.05, rng=None):
    lo, hi = bootstrap_mean_ci(paired_deltas, n_boot=n_boot, alpha=alpha, rng=rng)
    return lo > 0, (lo, hi)


def l4_decision(channel_deltas, has_real_results=False, n_boot=2000, alpha=0.05, rng=None):
    """channel_deltas: dict name -> list of per-example (attack - greedy) hits.

    This round must call with has_real_results=False.
    """
    if not has_real_results:
        return DEFERRED
    if not channel_deltas:
        return KILL_SWITCH
    any_sig = False
    all_near_zero = True
    for deltas in channel_deltas.values():
        if not deltas:
            continue
        mean = sum(deltas) / len(deltas)
        if abs(mean) > 1e-12:
            all_near_zero = False
        sig, _ = channel_significant(deltas, n_boot=n_boot, alpha=alpha, rng=rng)
        any_sig = any_sig or sig
    if any_sig:
        return GO
    if all_near_zero:
        return KILL_SWITCH
    return KILL_SWITCH

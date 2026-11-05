"""Author-cluster uncertainty. Row counts are not treated as independent samples."""

from __future__ import annotations

import random


def author_cluster_bootstrap(values_by_author, samples, seed, min_authors):
    authors = list(values_by_author)
    n_authors = len(authors)
    if n_authors == 0:
        return {
            "mean": None,
            "ci_low": None,
            "ci_high": None,
            "n_authors": 0,
            "unstable_estimate": True,
            "independent_sample_size": 0,
        }
    point = sum(values_by_author[author] for author in authors) / float(n_authors)
    if n_authors < int(min_authors):
        return {
            "mean": point,
            "ci_low": None,
            "ci_high": None,
            "n_authors": n_authors,
            "unstable_estimate": True,
            "independent_sample_size": n_authors,
        }
    rng = random.Random(int(seed))
    draws = []
    for _ in range(int(samples)):
        picked = [values_by_author[authors[rng.randrange(n_authors)]] for _ in range(n_authors)]
        draws.append(sum(picked) / float(n_authors))
    draws.sort()
    low_index = int(0.025 * (len(draws) - 1))
    high_index = int(0.975 * (len(draws) - 1))
    return {
        "mean": point,
        "ci_low": draws[low_index],
        "ci_high": draws[high_index],
        "n_authors": n_authors,
        "unstable_estimate": False,
        "independent_sample_size": n_authors,
    }


def bootstrap_from_rows(rows, value_key, min_authors, samples, seed):
    grouped = {}
    for row in rows:
        if row.get(value_key) is None:
            continue
        grouped.setdefault(row["author_id"], []).append(float(row[value_key]))
    means = {
        author: sum(values) / float(len(values))
        for author, values in grouped.items()
    }
    result = author_cluster_bootstrap(means, samples=samples, seed=seed, min_authors=min_authors)
    result["n_rows"] = sum(len(values) for values in grouped.values())
    result["row_count_not_used_as_sample_size"] = result["independent_sample_size"] != result["n_rows"]
    return result

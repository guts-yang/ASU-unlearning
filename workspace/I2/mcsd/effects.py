"""Necessity and rescue from one joint patch, plus the configured controls."""

from __future__ import annotations

import random

from I2.mcsd.patch import capture_clean, patched_logits
from I2.mcsd.score import advantage


def _scalar(value):
    if hasattr(value, "detach"):
        if value.numel() != 1:
            return [float(item) for item in value.detach().cpu().reshape(-1)]
        return float(value.detach().cpu())
    return value


def measure_effects(model, example, paths, strength=1.0, weights=None, source_pair=None, donor_pair=None, audit=None):
    """N_S = g(q+) - g(P_S(q+ <- A-)); R_S = g(P_S(q- <- A+)) - g(q-).

    ``source_pair`` / ``donor_pair`` override the clean caches. The matched
    measurement uses the example's own q+ and q- caches. A wrong-donor control
    passes a different donor pair.
    """
    cache_plus = example.get("cache_plus")
    cache_minus = example.get("cache_minus")
    if cache_plus is None or cache_minus is None:
        cache_plus = capture_clean(model, example["prompt_plus"])
        cache_minus = capture_clean(model, example["prompt_minus"])
        example["cache_plus"] = cache_plus
        example["cache_minus"] = cache_minus
    if source_pair is None:
        source_plus, source_minus = cache_plus, cache_minus
    else:
        source_plus, source_minus = source_pair
    if donor_pair is None:
        donor_plus, donor_minus = cache_minus, cache_plus
    else:
        donor_plus, donor_minus = donor_pair

    foil = bool(example.get("foil_reliable"))
    clean_pos = module_logits_of(model, example["score_plus"])
    clean_neg = module_logits_of(model, example["score_minus"])
    patched_pos, _ = patched_logits(
        model,
        example["score_plus"],
        paths,
        source_plus,
        donor_plus,
        strength=strength,
        weights=weights,
        audit=audit,
    )
    patched_neg, _ = patched_logits(
        model,
        example["score_minus"],
        paths,
        source_minus,
        donor_minus,
        strength=strength,
        weights=weights,
        audit=audit,
    )
    foil_pos = foil_neg = patched_foil_pos = patched_foil_neg = None
    if foil:
        foil_pos = module_logits_of(model, example["foil_plus"])
        foil_neg = module_logits_of(model, example["foil_minus"])
        patched_foil_pos, _ = patched_logits(
            model, example["foil_plus"], paths, source_plus, donor_plus, strength=strength, weights=weights, audit=audit
        )
        patched_foil_neg, _ = patched_logits(
            model, example["foil_minus"], paths, source_minus, donor_minus, strength=strength, weights=weights, audit=audit
        )
    g_plus, scope = advantage(
        clean_pos,
        example["score_plus"]["labels"],
        foil_pos,
        None if example.get("foil_plus") is None else example["foil_plus"]["labels"],
        foil_reliable=foil,
    )
    g_patched_plus, _ = advantage(
        patched_pos,
        example["score_plus"]["labels"],
        patched_foil_pos,
        None if example.get("foil_plus") is None else example["foil_plus"]["labels"],
        foil_reliable=foil,
    )
    g_minus, _ = advantage(
        clean_neg,
        example["score_minus"]["labels"],
        foil_neg,
        None if example.get("foil_minus") is None else example["foil_minus"]["labels"],
        foil_reliable=foil,
    )
    g_patched_minus, _ = advantage(
        patched_neg,
        example["score_minus"]["labels"],
        patched_foil_neg,
        None if example.get("foil_minus") is None else example["foil_minus"]["labels"],
        foil_reliable=foil,
    )
    necessity = g_plus - g_patched_plus
    rescue = g_patched_minus - g_minus
    computed_by = "joint_patch_forward" if len(list(paths)) > 1 else "single_path_patch_forward"
    return {
        "N": _scalar(necessity),
        "R": _scalar(rescue),
        "score_scope": scope,
        "computed_by": computed_by,
        "path_count": len(list(paths)),
        "status": "computed",
        "donor_direction": {
            "necessity": "q+ <- A-",
            "rescue": "q- <- A+",
        },
    }


def module_logits_of(model, batch):
    from I2.mcsd.patch import module_logits

    return module_logits(model, batch)


def _counts(paths):
    counts = {"attention": 0, "mlp": 0}
    for path in paths:
        counts[path[0]] += 1
    return counts


def random_control_paths(pool, selected, seed):
    selected = set(selected)
    counts = _counts(selected)
    rng = random.Random(int(seed))
    chosen = []
    for module, count in counts.items():
        if count == 0:
            continue
        candidates = [path for path in pool if path[0] == module and path not in selected]
        if len(candidates) < count:
            return None, "not enough non-selected %s paths for a matched random control" % module
        rng.shuffle(candidates)
        chosen.extend(candidates[:count])
    return chosen, None


def control_suite(model, example, selected, pool, seed, wrong_donor=None, retain_example=None, retain_reason=None):
    no_swap = measure_effects(model, example, selected, strength=0.0)
    no_swap["name"] = "no_swap"
    random_paths, reason = random_control_paths(pool, selected, seed)
    if random_paths is None:
        random_result = {"name": "random", "status": "skipped", "reason": reason, "N": None, "R": None}
    else:
        random_result = measure_effects(model, example, random_paths, strength=1.0)
        random_result["name"] = "random"
        random_result["paths"] = [list(path) for path in random_paths]
    if wrong_donor is None:
        wrong = {
            "name": "wrong_donor",
            "status": "skipped",
            "reason": "no other donor cache is available",
            "N": None,
            "R": None,
        }
    else:
        wrong = measure_effects(
            model,
            example,
            selected,
            strength=1.0,
            donor_pair=wrong_donor,
        )
        wrong["name"] = "wrong_donor"
    if retain_example is None:
        retain = {
            "name": "retain",
            "status": "skipped",
            "reason": retain_reason or "no aligned retain pair",
            "N": None,
            "R": None,
        }
    else:
        retain = measure_effects(model, retain_example, selected, strength=1.0)
        retain["name"] = "retain"
    return {"no_swap": no_swap, "random": random_result, "wrong_donor": wrong, "retain": retain}


def interaction_diagnostic(model, example, paths):
    paths = list(paths)
    if len(paths) != 2:
        return {"status": "unavailable", "value": None, "reason": "interaction diagnostic needs exactly two paths"}
    joint = measure_effects(model, example, paths, strength=1.0)
    first = measure_effects(model, example, [paths[0]], strength=1.0)
    second = measure_effects(model, example, [paths[1]], strength=1.0)
    return {
        "status": "computed",
        "value": joint["N"] - first["N"] - second["N"],
        "computed_by": "joint_forward_minus_singles",
        "joint_N": joint["N"],
        "single_N": [first["N"], second["N"]],
    }

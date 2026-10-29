"""Resolve a training spec from the calibration lock. Never selects K or delta."""

import hashlib
import json
from pathlib import Path

from gold_rank.demotion import GoldRankConfigError, validate_spec


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_gold_rank(raw, forget_json):
    """Return a concrete spec, or None when the edit is disabled.

    Training is refused until the diagnostic has written ``status=frozen``.
    """
    if raw is None or not bool(raw.get("enabled", False)):
        return None
    lock_path = raw.get("lock_path")
    if not lock_path or not Path(lock_path).is_file():
        raise SystemExit(
            "gold_rank.enabled requires an existing gold_rank.lock_path. "
            "Run the teacher diagnostic and freeze K and delta before training."
        )
    lock = json.loads(Path(lock_path).read_text())
    if lock.get("status") != "frozen":
        raise SystemExit(f"gold-rank lock status is {lock.get('status')}; demotion training stays off")
    if lock.get("test_metrics_read"):
        raise SystemExit("gold-rank lock records that test metrics were read before freezing")
    digest = sha256_file(forget_json)
    if lock.get("forget_sha256") != digest:
        raise SystemExit("forget split hash does not match the gold-rank lock")
    spec = {
        "enabled": True,
        "mode": str(raw.get("mode")),
        "k": None if lock.get("k") is None else int(lock["k"]),
        "delta": None if lock.get("delta") is None else float(lock["delta"]),
        "penalty": None if lock.get("matched_penalty") is None else float(lock["matched_penalty"]),
        "seed": int(raw.get("seed", lock.get("calibration_seed", 1001))),
        "k_grid": [int(value) for value in lock.get("k_grid", [])],
        "collect_stats": False,
    }
    try:
        validate_spec(spec)
    except GoldRankConfigError as error:
        raise SystemExit(str(error)) from error
    return spec

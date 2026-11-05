"""Result locations for direction I. Training is not started here."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from I2.mcsd.errors import StorageError

_RESULTS = Path(__file__).resolve().parents[1] / "results"
_STAMP = re.compile(r"^\d{8}T\d{6}Z$")
_STAGE = re.compile(r"^[a-z][a-z0-9_]*$")
_PROTECTED = "pre_experiment_20261105T110514Z"
_FORBIDDEN = (
    "Right-to-be-forgotten/results",
    "checkpoints/tofu_replay",
)


def results_root():
    return _RESULTS


def measurement_dir(stage, stamp=None):
    """Return a new results/<stage>_<stamp> path. The caller creates it."""
    if stage == "student_e1":
        raise StorageError("student checkpoints use checkpoint_dir, not a measurement stage")
    if not _STAGE.match(stage):
        raise StorageError("result stage must be a lowercase identifier")
    stamp = _now() if stamp is None else stamp
    if not _STAMP.match(stamp):
        raise StorageError("result stamp must be YYYYMMDDTHHMMSSZ")
    path = _RESULTS / ("%s_%s" % (stage, stamp))
    reject_forbidden(path)
    if path.name == _PROTECTED or path.exists():
        raise StorageError("refusing to overwrite %s" % path)
    return path


def checkpoint_dir(stamp):
    """Return the only directory a future Student run may use. Does not create it."""
    if not _STAMP.match(stamp):
        raise StorageError("checkpoint stamp must be YYYYMMDDTHHMMSSZ")
    path = _RESULTS / ("student_e1_%s" % stamp) / "unlearn_times_1"
    reject_forbidden(path)
    if path.exists() or path.parent.exists():
        raise StorageError("refusing to overwrite %s" % path.parent)
    return path


def reject_forbidden(path):
    text = str(path).replace("\\", "/")
    for marker in _FORBIDDEN:
        if marker in text:
            raise StorageError("refusing to write into %s" % marker)
    root = _RESULTS.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    probe = candidate if candidate.exists() else candidate.parent
    try:
        probe.resolve().relative_to(root)
    except ValueError as exc:
        raise StorageError("direction I results must stay under %s" % root) from exc
    if candidate.name == _PROTECTED or _PROTECTED in candidate.parts:
        raise StorageError("refusing to overwrite %s" % _PROTECTED)

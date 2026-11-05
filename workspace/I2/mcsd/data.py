"""Read existing TOFU rows and assign the frozen author-block split.

This module does not synthesize questions. A string that is not already in the
loaded records is rejected by the caller.
"""

from __future__ import annotations

import json
from pathlib import Path

from I2.mcsd.errors import ConfigError


def read_records(path):
    text = Path(path).read_text()
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] == "[":
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ConfigError("TOFU records must be a list or JSONL")
    return rows


def author_index_for_row(row_offset, questions_per_author):
    size = int(questions_per_author)
    if size <= 0:
        raise ConfigError("questions_per_author must be positive")
    return int(row_offset) // size


def split_for_author(author_index, calibration, holdout):
    calibration = set(calibration)
    holdout = set(holdout)
    if author_index in calibration and author_index in holdout:
        raise ConfigError("author %s is in both calibration and holdout" % author_index)
    if author_index in calibration:
        return "calibration"
    if author_index in holdout:
        return "holdout"
    raise ConfigError("author %s is not assigned to calibration or holdout" % author_index)

"""JSON and JSONL results. Missing measurements stay null."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def to_jsonable(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("refusing to write a non-finite float")
        return value
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    raise TypeError("cannot serialize %s" % type(value).__name__)


def write_run(directory, summary, rows):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "summary.json"
    rows_path = destination / "samples.jsonl"
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, allow_nan=False) + "\n")
    with rows_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), allow_nan=False) + "\n")
    return summary_path, rows_path

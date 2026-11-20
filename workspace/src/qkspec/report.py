"""Structured JSON report writer for QKSpec runs.

Reports keep forget / retain metrics under separate keys and explicitly mark
unimplemented readers as "not_covered".  They never mix different measurement
scales into a single score.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .config import QKSpecConfig


def write_report(
    report: Dict[str, Any],
    output_dir: Optional[str],
    filename: str = "qkspec_report.json",
) -> Optional[str]:
    """Write ``report`` as JSON under ``output_dir`` and return the path."""
    if output_dir is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    return path


def build_report(
    cfg: QKSpecConfig,
    edit_result: Any,
    forget_metrics: Optional[Dict[str, Any]] = None,
    retain_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the final report dict from an :class:`EditResult`."""
    report = {
        "method": "QKSpec",
        "config": cfg.to_dict(),
        "edit": edit_result.to_dict() if hasattr(edit_result, "to_dict") else edit_result,
        "forget_metrics": forget_metrics or {},
        "retain_metrics": retain_metrics or {},
        "claims": {
            "access_blockade_l2": "not_proven_in_first_version",
            "information_erasure_l3": "not_claimed",
        },
    }
    readers = ["probe", "causal_patch", "quantization", "relearning"]
    coverage = getattr(edit_result, "reader_coverage", {}) or {}
    for r in readers:
        report["forget_metrics"].setdefault(
            f"reader_{r}", coverage.get(r, "not_covered")
        )
    return report

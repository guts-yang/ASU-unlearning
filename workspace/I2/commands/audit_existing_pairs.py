"""Audit existing task-1 donor tokens for a contiguous fact span.

Reads the frozen pre-experiment manifest. Does not load Llama, edit the
candidate thresholds, or start localization.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from I2.mcsd.config import load_config  # noqa: E402
from I2.mcsd.gates import E0_DECISION, E0_RETAIN_CAP, is_e0_retain_cap  # noqa: E402
from I2.mcsd.pairing_search import find_existing_pairs  # noqa: E402
from I2.mcsd.storage import measurement_dir  # noqa: E402

_FROZEN_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "pre_experiment_20261105T110514Z"
    / "task1_samples.jsonl"
)
_CANDIDATE = Path(__file__).resolve().parents[1] / "configs" / "mcsd_candidate.yaml"
_EXAMPLE_LIMIT = 20


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    manifest = Path(arguments[0]) if arguments else _FROZEN_MANIFEST
    output_dir = measurement_dir("pair_audit")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(_CANDIDATE)
    rows = _read_jsonl(manifest)
    search = find_existing_pairs(rows)
    summary = _summary(config, manifest, search)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    examples = search["pairs"][:_EXAMPLE_LIMIT]
    (output_dir / "near_misses.json").write_text(json.dumps(examples, indent=2) + "\n")
    sys.stdout.write(str(output_dir) + "\n")
    return 0


def _summary(config, manifest, search):
    pairs = search["pairs"]
    calibration_pairs = [
        pair for pair in pairs if pair["q_plus_split"] == "calibration" and pair["q_minus_split"] == "calibration"
    ]
    min_authors = int(config["min_authors_for_interval"])
    calibration_authors = list(config["calibration_author_indices"])
    import torch

    return {
        "source_manifest": str(manifest),
        "questions_rewritten": False,
        "pairing_mode_approved": False,
        "pairing_mode": (config.get("pairing") or {}).get("mode"),
        "rows_read": search["rows_read"],
        "rows_with_stable_prefix": search["rows_with_stable_prefix"],
        "prefix_mismatch_rows": len(search["rows_skipped_prefix"]),
        "shared_wrapper_prefix_tokens": search["shared_wrapper_prefix_tokens"],
        "shared_wrapper_suffix_tokens": search["shared_wrapper_suffix_tokens"],
        "pairs_rejected_wrapper_only": search["pairs_rejected_wrapper_only"],
        "pairs_rejected_not_fact_slot": search["pairs_rejected_not_fact_slot"],
        "unapproved_near_misses": len(pairs),
        "calibration_only_near_misses": len(calibration_pairs),
        "candidate_pairs_approved_for_localization": 0,
        "example_limit": _EXAMPLE_LIMIT,
        "examples_written": min(_EXAMPLE_LIMIT, len(pairs)),
        "author_interval": {
            "calibration_author_count": len(calibration_authors),
            "calibration_author_indices": calibration_authors,
            "min_authors_for_interval": min_authors,
            "threshold_changed": False,
            "unstable_estimate": len(calibration_authors) < min_authors,
            "qa_rows_are_not_independent_samples": True,
            "decision": "keep_min_authors_and_report_unstable",
        },
        "span_length_counts": dict(Counter(pair["diff_span"][1] - pair["diff_span"][0] for pair in pairs)),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count()),
        "llama_loaded": False,
        "hook_smoke_passed": False,
        "localization_started": False,
        "rho_sweep_executed": False,
        "student_e1_started": False,
        "e0_decision_preserved": E0_DECISION,
        "e0_retain_cap_reused": is_e0_retain_cap((config.get("gates") or {}).get("retain_nll_cap")),
        "e0_retain_cap_value_not_copied": E0_RETAIN_CAP,
        "method_effectiveness_claimed": False,
        "stop_reason": _stop_reason(len(pairs), bool(torch.cuda.is_available())),
    }


def _stop_reason(pair_count, cuda_available):
    reasons = [
        "no annotated fact span exists in the task-1 questions; near misses stay unapproved",
        "pairing.mode is still unset, so N and R have no input",
    ]
    if pair_count:
        reasons.append("near misses are not annotated fact spans and are not q- donors")
    if not cuda_available:
        reasons.append("cuda is unavailable, so the eval-only hook smoke did not load weights")
    reasons.append("min_authors_for_interval stays 8; the 5-author interval is unstable")
    reasons.append("selection thresholds, rho, and Student E1 stay closed")
    return reasons


def _read_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    sys.exit(main())

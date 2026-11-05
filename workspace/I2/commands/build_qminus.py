"""Write generated q- text for the frozen task-1 manifest.

This does not tokenize, approve a pairing mode, load Llama, or start localization.
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
from I2.mcsd.counterfactual import build_qminus  # noqa: E402
from I2.mcsd.gates import E0_DECISION, E0_RETAIN_CAP, is_e0_retain_cap  # noqa: E402
from I2.mcsd.storage import measurement_dir  # noqa: E402

_FROZEN_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "pre_experiment_20261105T110514Z"
    / "task1_samples.jsonl"
)
_CANDIDATE = Path(__file__).resolve().parents[1] / "configs" / "mcsd_candidate.yaml"


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    manifest = Path(arguments[0]) if arguments else _FROZEN_MANIFEST
    output_dir = measurement_dir("qminus")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(_CANDIDATE)
    rows = _read_jsonl(manifest)
    built = [_row(row) for row in rows]
    (output_dir / "qminus.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in built))
    summary = _summary(config, manifest, built)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sys.stdout.write(str(output_dir) + "\n")
    return 0


def _row(row):
    built = build_qminus(row["question"], row["author_id"])
    return {
        "sample_id": row["sample_id"],
        "author_id": row["author_id"],
        "split": row["split"],
        "q_plus": row["question"],
        "q_minus": built["q_minus"],
        "replacement_kind": built["replacement_kind"],
        "counterfactual_author_id": built["counterfactual_author_id"],
        "counterfactual_name": built["counterfactual_name"],
        "character_spans": built["character_spans"],
        "single_contiguous_span": built["single_contiguous_span"],
        "template_outside_spans_unchanged": built["template_outside_spans_unchanged"],
        "generated": built["generated"],
        "approved_for_localization": False,
        "token_alignment": built["token_alignment"],
        "exclusion_reason": built["exclusion_reason"],
    }


def _summary(config, manifest, rows):
    generated = [row for row in rows if row["generated"]]
    import torch

    return {
        "source_manifest": str(manifest),
        "generation_rule": "replace the task-1 author name, or the birth place and date, with the next author id modulo 10",
        "new_person_invented": False,
        "questions_rewritten": True,
        "pairing_mode_approved": False,
        "pairing_mode": (config.get("pairing") or {}).get("mode"),
        "rows_read": len(rows),
        "qminus_generated": len(generated),
        "qminus_missing": len(rows) - len(generated),
        "single_contiguous_span": sum(1 for row in generated if row["single_contiguous_span"]),
        "multiple_identity_spans": sum(1 for row in generated if row["exclusion_reason"] == "multiple_identity_spans"),
        "template_outside_spans_unchanged": sum(1 for row in generated if row["template_outside_spans_unchanged"]),
        "replacement_kinds": dict(Counter(row["replacement_kind"] for row in generated)),
        "approved_for_localization": 0,
        "token_alignment": "not_checked_no_tokenizer",
        "cuda_available": bool(torch.cuda.is_available()),
        "llama_loaded": False,
        "localization_started": False,
        "student_e1_started": False,
        "e0_decision_preserved": E0_DECISION,
        "e0_retain_cap_reused": is_e0_retain_cap((config.get("gates") or {}).get("retain_nll_cap")),
        "e0_retain_cap_value_not_copied": E0_RETAIN_CAP,
        "method_effectiveness_claimed": False,
    }


def _read_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    sys.exit(main())

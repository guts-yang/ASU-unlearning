"""Tokenize the generated same-split q- sidecar. No model weights."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
_REPO = _WORKSPACE.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from I2.mcsd.storage import measurement_dir  # noqa: E402
from I2.mcsd.token_record import annotate_tokens, llama2_tags, load_tokenizer  # noqa: E402

_SOURCE = (
    _WORKSPACE
    / "I2"
    / "results"
    / "qminus_same_split_20261107T025551Z"
    / "qminus.jsonl"
)
_MODEL_CONFIG = _REPO / "Right-to-be-forgotten" / "config" / "model_config.yaml"
_PRIMARY = "primary_single_slot"


def main(argv=None):
    del argv
    if not _SOURCE.is_file():
        raise RuntimeError("same-split q- sidecar is missing: %s" % _SOURCE)
    output_dir = measurement_dir("same_split_tokens")
    output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer, meta = load_tokenizer()
    tags = llama2_tags(_MODEL_CONFIG)
    rows = [json.loads(line) for line in _SOURCE.read_text().splitlines() if line.strip()]
    annotated = [finalize_token_row(row, tokenizer, tags, meta) for row in rows]
    _write_jsonl(output_dir / "tokens.jsonl", annotated)
    summary = _summary(annotated, meta)
    _write_json(output_dir / "summary.json", summary)
    sys.stdout.write(str(output_dir) + "\n")
    return 0


def finalize_token_row(record, tokenizer, tags, tokenizer_meta):
    """Encode one generated pair and keep the sidecar role unchanged."""
    role = record.get("analysis_role")
    record_status = record.get("record_status")
    sidecar_status = record.get("status")
    original_exclusion = record.get("exclusion_reason")
    if not record.get("q_minus"):
        updated = dict(record)
        updated["tokenizer_source"] = tokenizer_meta.get("tokenizer_source")
        updated["transformers_version"] = tokenizer_meta.get("transformers_version")
        updated["tokenizer_sha256"] = tokenizer_meta.get("tokenizer_sha256")
        updated["token_status"] = "not_encoded"
        updated["token_alignment"] = "not_encoded_no_q_minus"
        updated["fact_span"] = None
        updated["is_candidate"] = False
        updated["approved_for_localization"] = False
        return updated
    annotated = annotate_tokens(record, tokenizer, tags, tokenizer_meta)
    token_status = annotated.get("status")
    token_alignment = annotated.get("token_alignment")
    fact_span = annotated.get("fact_span")
    annotated["analysis_role"] = role
    annotated["record_status"] = record_status
    annotated["status"] = sidecar_status
    annotated["exclusion_reason"] = original_exclusion
    annotated["token_status"] = token_status
    annotated["token_alignment"] = token_alignment
    annotated["fact_span"] = fact_span
    annotated["is_candidate"] = bool(role == _PRIMARY and token_alignment == "aligned")
    annotated["approved_for_localization"] = False
    return annotated


def _summary(rows, meta):
    by_role = {}
    for row in rows:
        role = row.get("analysis_role") or "missing"
        counts = by_role.setdefault(role, Counter())
        counts[row.get("token_alignment") or "missing"] += 1
    with_q_minus = sum(1 for row in rows if row.get("q_minus"))
    return {
        "source_qminus": str(_SOURCE),
        "rows_read": len(rows),
        "generated_with_q_minus": with_q_minus,
        "excluded_without_q_minus": len(rows) - with_q_minus,
        "token_alignment_by_analysis_role": {
            role: dict(counts) for role, counts in sorted(by_role.items())
        },
        "primary_token_aligned": sum(
            1
            for row in rows
            if row.get("analysis_role") == _PRIMARY and row.get("token_alignment") == "aligned"
        ),
        "tokenizer_source": meta.get("tokenizer_source"),
        "tokenizer_loaded": bool(meta.get("tokenizer_loaded")),
        "transformers_version": meta.get("transformers_version"),
        "tokenizer_sha256": meta.get("tokenizer_sha256"),
        "llama_for_causal_lm_instantiated": False,
        "real_model_executed": False,
        "approved_for_localization": 0,
        "pairing_mode": None,
        "localization_started": False,
        "student_e1_started": False,
        "n_written": None,
        "r_written": None,
        "localization_n_min_changed": False,
        "localization_r_min_changed": False,
        "rho_grid_frozen": False,
        "e0_retain_cap_reused": False,
        "code_version": _code_version(),
    }


def _code_version():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    text = completed.stdout.strip()
    return text or None


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


if __name__ == "__main__":
    sys.exit(main())

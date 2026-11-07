"""CPU-only token audit of generated q- strings. No model weights, no CUDA."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
_REPO = _WORKSPACE.parent
_RBF = _REPO / "Right-to-be-forgotten"
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from I2.mcsd.alignment import check_alignment  # noqa: E402
from I2.mcsd.counterfactual import (  # noqa: E402
    AUTHOR_NAMES,
    AUTHOR_SURNAMES,
    BIRTH_DATES,
    BIRTH_PLACES,
    build_qminus,
)
from I2.mcsd.data import read_records  # noqa: E402
from I2.mcsd.pairing_search import contiguous_diff_span  # noqa: E402
from I2.mcsd.storage import measurement_dir  # noqa: E402

MODEL_DIR = Path("/root/autodl-tmp/models/tofu_ft_llama2-7b")
FORGET_PATH = _RBF / "data" / "tofu" / "forget05.json"
MODEL_CONFIG_PATH = _RBF / "config" / "model_config.yaml"
SCORE_MAX_LENGTH = 500
MODEL_FAMILY = "llama2-7b"

# Rows known to have score_prefix_boundary_mismatch from pre_experiment audit.
PREFIX_MISMATCH_ROWS = {12, 27, 30, 36, 66, 70, 92, 154}


def main(argv=None):
    import yaml
    from transformers import AutoTokenizer

    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        raise RuntimeError("q- jsonl path is required")
    qminus_path = Path(arguments[0])
    if not qminus_path.is_file():
        raise RuntimeError("q- sidecar is missing: %s" % qminus_path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = measurement_dir("token_audit", stamp)
    out_dir.mkdir(parents=True, exist_ok=False)

    tags = yaml.load(MODEL_CONFIG_PATH.read_text(), Loader=yaml.FullLoader)[MODEL_FAMILY]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    qminus_rows = [json.loads(line) for line in qminus_path.read_text().splitlines() if line.strip()]
    if len(qminus_rows) != 200:
        raise RuntimeError("expected 200 qminus rows, got %s" % len(qminus_rows))

    # Step 1+2: tokenize each q+/q- pair and check alignment.
    candidates = []
    excluded = []
    length_counts = Counter()
    for row in qminus_rows:
        sample_index = int(row["sample_id"].split("_")[-1])
        result = audit_row(tokenizer, tags, row, sample_index)
        if result["status"] == "candidate":
            candidates.append(result)
        else:
            excluded.append(result)
        length_counts[(result["q_plus_len"], result["q_minus_len"])] += 1

    summary = {
        "real_model_executed": False,
        "llama_for_causal_lm_instantiated": False,
        "optimizer_steps": 0,
        "student_e1_started": False,
        "causal_localization_claimed": False,
        "method_effectiveness_claimed": False,
        "cuda_available": False,
        "tokenizer_loaded": True,
        "tokenizer_source": str(MODEL_DIR),
        "model_family": MODEL_FAMILY,
        "qminus_source": str(qminus_path),
        "rows_audited": len(qminus_rows),
        "candidates": len(candidates),
        "excluded": len(excluded),
        "exclusion_reason_counts": Counter(
            item["status"] for item in excluded
        ),
        "length_mismatch_count": sum(
            1 for item in excluded if item["status"] == "length_mismatch"
        ),
        "diff_not_contiguous_count": sum(
            1 for item in excluded if item["status"] == "diff_not_contiguous"
        ),
        "prefix_mismatch_rows_excluded": len(PREFIX_MISMATCH_ROWS),
        "multiple_identity_spans_excluded": sum(
            1 for item in excluded if item["status"] == "multiple_identity_spans"
        ),
        "calibration_candidates": sum(1 for c in candidates if c["split"] == "calibration"),
        "holdout_candidates": sum(1 for c in candidates if c["split"] == "holdout"),
        "author_interval_unstable": True,
        "calibration_author_count": 5,
        "min_authors_for_interval": 8,
        "pairing_mode_set": False,
        "n_written": None,
        "r_written": None,
        "localization_n_min_changed": False,
        "localization_r_min_changed": False,
        "alpha_changed": False,
        "retain_nll_cap_changed": False,
        "e0_retain_cap_reused": False,
        "rho_grid_frozen": False,
        "gpu_hook_smoke_started": False,
        "code_version": git_head(),
        "step_2_result": "candidates_found" if candidates else "zero_candidates",
    }

    # Step 3: if zero candidates, search for equal-length token substitutions.
    if not candidates:
        search_result = search_equal_length_substitutions(tokenizer, tags)
        summary["step_3_search_executed"] = True
        summary["step_3_candidates"] = len(search_result)
        candidates = search_result
        summary["candidates"] = len(candidates)
        summary["calibration_candidates"] = sum(
            1 for c in candidates if c["split"] == "calibration"
        )
        summary["holdout_candidates"] = sum(
            1 for c in candidates if c["split"] == "holdout"
        )
        summary["step_3_result"] = "candidates_found" if candidates else "zero_candidates"
    else:
        summary["step_3_search_executed"] = False
        summary["step_3_candidates"] = 0
        summary["step_3_result"] = "not_executed_step_2_found_candidates"

    # Step 4: if still zero, write corpus conclusion.
    if not candidates:
        summary["corpus_conclusion"] = (
            "task 1 corpus cannot provide a localization input that passes check_alignment"
        )
        summary["stop_reason"] = "no q+/q- pair satisfies check_alignment at token level"
    else:
        summary["corpus_conclusion"] = None
        summary["stop_reason"] = None

    _write_json(out_dir / "summary.json", summary)
    _write_jsonl(out_dir / "candidates.jsonl", candidates)
    _write_jsonl(out_dir / "excluded.jsonl", excluded)
    sys.stdout.write(str(out_dir) + "\n")
    return 0


def audit_row(tokenizer, tags, row, sample_index):
    q_plus_text = row["q_plus"]
    q_minus_text = row["q_minus"]
    q_plus_ids = encode_question(tokenizer, tags, q_plus_text)
    q_minus_ids = encode_question(tokenizer, tags, q_minus_text)
    result = {
        "sample_id": row["sample_id"],
        "author_id": row["author_id"],
        "split": row["split"],
        "replacement_kind": row["replacement_kind"],
        "q_plus_len": len(q_plus_ids),
        "q_minus_len": len(q_minus_ids),
        "q_plus_question": q_plus_text,
        "q_minus_question": q_minus_text,
    }
    if row.get("exclusion_reason") == "multiple_identity_spans":
        result["status"] = "multiple_identity_spans"
        return result
    if sample_index in PREFIX_MISMATCH_ROWS:
        result["status"] = "score_prefix_boundary_mismatch"
        return result
    if len(q_plus_ids) != len(q_minus_ids):
        result["status"] = "length_mismatch"
        return result
    span = contiguous_diff_span(q_plus_ids, q_minus_ids)
    if span is None:
        result["status"] = "diff_not_contiguous"
        return result
    score_plus_ids = build_score_ids(tokenizer, tags, q_plus_text)
    score_minus_ids = build_score_ids(tokenizer, tags, q_minus_text)
    aligned, reasons = check_alignment(
        q_plus_ids, q_minus_ids, span, score_plus_ids, score_minus_ids
    )
    if not aligned:
        result["status"] = "check_alignment_failed"
        result["reasons"] = reasons
        return result
    result["status"] = "candidate"
    result["fact_span"] = span
    result["token_length"] = len(q_plus_ids)
    result["replacement_type"] = row["replacement_kind"]
    return result


def encode_question(tokenizer, tags, question):
    text = tags["question_start_tag"] + question + tags["question_end_tag"]
    encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=SCORE_MAX_LENGTH)
    return encoded["input_ids"]


def build_score_ids(tokenizer, tags, question):
    # Score sequence is question+answer; but for alignment check we only need
    # the prefix to match the donor. Use question-only encoding padded to max.
    text = tags["question_start_tag"] + question + tags["question_end_tag"]
    encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=SCORE_MAX_LENGTH)
    ids = encoded["input_ids"]
    pad = SCORE_MAX_LENGTH - len(ids)
    return ids + [tokenizer.eos_token_id] * pad


def search_equal_length_substitutions(tokenizer, tags):
    """Step 3: for each task-1 question, try all 9 other author names."""
    rows = [r for r in read_records(FORGET_PATH) if str(r.get("task_id")) == "1"]
    candidates = []
    for index, row in enumerate(rows):
        if index in PREFIX_MISMATCH_ROWS:
            continue
        question = row["question"]
        author_id = index // 20
        for target_id in range(10):
            if target_id == author_id:
                continue
            qminus_record = build_qminus(question, author_id)
            if qminus_record.get("exclusion_reason") == "multiple_identity_spans":
                continue
            q_minus_text = qminus_record.get("q_minus")
            if q_minus_text is None:
                continue
            q_plus_ids = encode_question(tokenizer, tags, question)
            q_minus_ids = encode_question(tokenizer, tags, q_minus_text)
            if len(q_plus_ids) != len(q_minus_ids):
                continue
            span = contiguous_diff_span(q_plus_ids, q_minus_ids)
            if span is None:
                continue
            score_plus_ids = build_score_ids(tokenizer, tags, question)
            score_minus_ids = build_score_ids(tokenizer, tags, q_minus_text)
            aligned, reasons = check_alignment(
                q_plus_ids, q_minus_ids, span, score_plus_ids, score_minus_ids
            )
            if not aligned:
                continue
            split = "calibration" if author_id % 2 == 0 else "holdout"
            candidates.append({
                "sample_id": "forget05_task1_%04d" % index,
                "author_id": author_id,
                "split": split,
                "replacement_type": qminus_record.get("replacement_kind"),
                "token_length": len(q_plus_ids),
                "fact_span": span,
                "q_plus_question": question,
                "q_minus_question": q_minus_text,
                "counterfactual_author_id": target_id,
                "status": "candidate",
            })
    return candidates


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO)
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())

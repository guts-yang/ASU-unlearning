"""Write a same-split q- sidecar. The forget file is read and left unchanged."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
_REPO = _WORKSPACE.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from I2.mcsd.config import load_config  # noqa: E402
from I2.mcsd.errors import ConfigError  # noqa: E402
from I2.mcsd.gates import E0_DECISION, E0_RETAIN_CAP, is_e0_retain_cap  # noqa: E402
from I2.mcsd.same_split import (  # noqa: E402
    build_same_split_rows,
    load_forget_records,
    require_locked_threshold,
)
from I2.mcsd.storage import measurement_dir  # noqa: E402
from I2.mcsd.token_record import annotate_tokens, llama2_tags, load_tokenizer  # noqa: E402

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "qminus_split.yaml"
_MODEL_CONFIG = _REPO / "Right-to-be-forgotten" / "config" / "model_config.yaml"


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    config = load_config(_CONFIG)
    require_locked_threshold(config)
    forget_path = Path(arguments[0]) if arguments else _REPO / config["forget_file"]
    if not forget_path.is_file():
        raise ConfigError("forget file is missing: %s" % forget_path)
    output_dir = measurement_dir("qminus")
    summary, _rows = run_sidecar(forget_path, config, output_dir)
    sys.stdout.write(str(output_dir) + "\n")
    return 0 if summary["forget_bytes_unchanged"] else 1


def run_sidecar(forget_path, config, output_dir):
    forget_path = Path(forget_path)
    before = forget_path.read_bytes()
    rows, digest = load_forget_records(forget_path)
    built, assignment = build_same_split_rows(rows, config)
    tokenizer, tokenizer_meta = load_tokenizer(config["tokenizer"]["source"])
    tags = llama2_tags(_MODEL_CONFIG)
    annotated = [annotate_tokens(row, tokenizer, tags, tokenizer_meta) for row in built]
    if forget_path.read_bytes() != before:
        raise ConfigError("forget file was modified")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_dir / "qminus.jsonl", annotated)
    _write_json(
        output_dir / "authors.json",
        {
            "task_ids": assignment["task_ids"],
            "calibration_author_ids": assignment["calibration_author_ids"],
            "holdout_author_ids": assignment["holdout_author_ids"],
            "authors_dropped_for_token_length": 0,
            "authors": [
                {
                    "author_id": author["author_id"],
                    "split": author["split"],
                    "task_id": author["task_id"],
                    "name": author["name"],
                }
                for author in assignment["authors"].values()
            ],
        },
    )
    summary = _summary(config, forget_path, digest, assignment, annotated, tokenizer_meta, before, forget_path.read_bytes())
    _write_json(output_dir / "summary.json", summary)
    return summary, annotated


def _summary(config, forget_path, digest, assignment, rows, tokenizer_meta, before, after):
    counts = Counter(row["status"] for row in rows)
    return {
        "forget_path": str(forget_path),
        "forget_sha256": digest,
        "forget_bytes_unchanged": before == after,
        "previous_cross_split_audit": config.get("previous_cross_split_audit"),
        "previous_candidates_reused": False,
        "task_ids": assignment["task_ids"],
        "calibration_author_ids": assignment["calibration_author_ids"],
        "holdout_author_ids": assignment["holdout_author_ids"],
        "calibration_author_count": len(assignment["calibration_author_ids"]),
        "holdout_author_count": len(assignment["holdout_author_ids"]),
        "min_authors_for_interval": config["min_authors_for_interval"],
        "author_interval_unstable": len(assignment["calibration_author_ids"]) < int(config["min_authors_for_interval"]),
        "authors_dropped_for_token_length": 0,
        "rows": len(rows),
        "generated": sum(1 for row in rows if row["generated"]),
        "candidates": sum(1 for row in rows if row["is_candidate"]),
        "status_counts": dict(counts),
        "approved_for_localization": 0,
        "pairing_mode": (config.get("pairing") or {}).get("mode"),
        "tokenizer_source": tokenizer_meta.get("tokenizer_source"),
        "tokenizer_loaded": tokenizer_meta.get("tokenizer_loaded"),
        "transformers_version": tokenizer_meta.get("transformers_version"),
        "tokenizer_sha256": tokenizer_meta.get("tokenizer_sha256"),
        "llama_for_causal_lm_instantiated": False,
        "student_e1_started": False,
        "e0_decision_preserved": E0_DECISION,
        "e0_retain_cap_reused": is_e0_retain_cap(None),
        "e0_retain_cap_value_not_copied": E0_RETAIN_CAP,
        "method_effectiveness_claimed": False,
    }


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as exc:
        sys.stderr.write(str(exc) + "\n")
        sys.exit(1)

"""Write a task-scoped same-split q- sidecar. The forget file is only read."""

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

from I2.mcsd.config import load_config  # noqa: E402
from I2.mcsd.errors import ConfigError  # noqa: E402
from I2.mcsd.gates import E0_DECISION  # noqa: E402
from I2.mcsd.same_split import (  # noqa: E402
    build_same_split_rows,
    load_forget_records,
    require_locked_threshold,
)
from I2.mcsd.storage import measurement_dir  # noqa: E402

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "qminus_split.yaml"


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    config = load_config(_CONFIG)
    require_locked_threshold(config)
    forget_path = Path(arguments[0]) if arguments else _REPO / config["forget_file"]
    if not forget_path.is_file():
        raise ConfigError("forget file is missing: %s" % forget_path)
    output_dir = measurement_dir("qminus_same_split")
    summary, _rows = run_sidecar(forget_path, config, output_dir)
    sys.stdout.write(str(output_dir) + "\n")
    return 0 if summary["forget_bytes_unchanged"] else 1


def run_sidecar(forget_path, config, output_dir):
    forget_path = Path(forget_path)
    before = forget_path.read_bytes()
    rows, digest = load_forget_records(forget_path)
    built, assignment = build_same_split_rows(rows, config, digest)
    _audit_rows(built, digest)
    after = forget_path.read_bytes()
    if after != before or hashlib_sha256(after) != digest:
        raise ConfigError("forget file was modified")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates = [row for row in built if row["is_candidate"]]
    excluded = [row for row in built if not row["is_candidate"]]
    inventory = _inventory(forget_path, digest, before, rows, assignment, config)
    _write_json(output_dir / "inventory.json", inventory)
    _write_jsonl(output_dir / "qminus.jsonl", built)
    _write_jsonl(output_dir / "candidates.jsonl", candidates)
    _write_jsonl(output_dir / "excluded.jsonl", excluded)
    _write_json(output_dir / "authors.json", _authors_payload(assignment))
    summary = _summary(config, forget_path, digest, assignment, built, before, after)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "audit.md").write_text(_audit_text(summary, inventory))
    return summary, built


def hashlib_sha256(blob):
    import hashlib

    return hashlib.sha256(blob).hexdigest()


def _audit_rows(rows, digest):
    for row in rows:
        if row["approved_for_localization"]:
            raise ConfigError("approved_for_localization must stay false")
        if not str(row["local_id"]).startswith(digest + ":"):
            raise ConfigError("local id must start with the forget sha256")
        if row["local_id"] != "%s:%s" % (digest, row["file_line_index"]):
            raise ConfigError("local id must be the forget sha256 plus the file line")
        if row["is_candidate"] and row.get("donor_origin") == "ground_truth":
            if row["split"] != row["donor_split"] or row["task_id"] != row["donor_task_id"]:
                raise ConfigError("candidate donor leaves the task or split")
            if not row["template_outside_spans_unchanged"] or not row["slot_tokens_differ"]:
                raise ConfigError("candidate non-entity tokens or slots are not paired")
            if not row["y_minus"] or row["y_minus"] != row["donor_answer"]:
                raise ConfigError("candidate donor answer is empty")
            if row["q_minus"] is None or row["y_plus"] in (row["q_minus"], row["q_plus"]):
                raise ConfigError("candidate prompt contains an answer")
        elif row["is_candidate"] and row.get("donor_origin") == "synthetic":
            if row["y_minus"] is not None or row["donor_answer"] is not None:
                raise ConfigError("synthetic candidate must leave the donor answer empty")
            if not row["q_minus"] or not row.get("fictional_name"):
                raise ConfigError("synthetic candidate is missing a fictional slot")
            if "Synthetic" in row["q_minus"] or "Synthetic" in row["fictional_name"]:
                raise ConfigError("synthetic slot still uses a placeholder")
            if not any(surface and surface in row["q_minus"] for surface in row.get("donor_slot_surfaces") or []):
                raise ConfigError("synthetic replacement did not land in the question")
            if row["approved_for_localization"]:
                raise ConfigError("approved_for_localization must stay false")
        elif row["is_candidate"]:
            raise ConfigError("candidate origin is not ground truth or synthetic")


def _inventory(path, digest, blob, rows, assignment, config):
    counts = Counter(str(row["task_id"]) for row in rows)
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": len(blob),
        "rows": len(rows),
        "format": "jsonl" if not blob.lstrip().startswith(b"[") else "json",
        "fields": ["task_id", "question", "answer"],
        "task_id_counts": {task_id: counts[task_id] for task_id in sorted(counts, key=lambda item: int(item))},
        "scope_task_ids": assignment["task_ids"],
        "scope": config.get("scope"),
        "entity_grouping": assignment["entity_grouping"],
        "official_dataset_author_id": False,
        "names_in_other_tasks": assignment["names_in_other_tasks"],
        "author_interval_unstable": assignment["author_interval_unstable"],
        "confirmatory_localization_allowed": False,
    }


def _authors_payload(assignment):
    return {
        "entity_id_kind": "local_block_not_official",
        "entity_grouping": assignment["entity_grouping"],
        "task_ids": assignment["task_ids"],
        "split_rule": "even_calibration_odd_holdout",
        "calibration_author_ids": assignment["calibration_author_ids"],
        "holdout_author_ids": assignment["holdout_author_ids"],
        "author_interval_unstable": assignment["author_interval_unstable"],
        "confirmatory_localization_allowed": False,
        "names_in_other_tasks": assignment["names_in_other_tasks"],
        "authors": [
            {
                "local_entity_id": author["local_entity_id"],
                "task_id": author["task_id"],
                "block_index": author["block_index"],
                "split": author["split"],
                "name": author["name"],
                "question_rows_with_name": author["question_rows_with_name"],
                "answer_rows_with_name": author["answer_rows_with_name"],
                "block_line_start": author["block_line_start"],
                "block_line_end": author["block_line_end"],
                "entity_id_kind": "local_block_not_official",
            }
            for author in assignment["authors"].values()
        ],
    }


def _summary(config, forget_path, digest, assignment, rows, before, after):
    reasons = Counter(row["exclusion_reason"] for row in rows if row.get("exclusion_reason"))
    clusters = sorted({row["pair_cluster_id"] for row in rows if row["pair_cluster_id"]})
    candidates = [row for row in rows if row["is_candidate"]]
    ground_truth = [row for row in candidates if row.get("donor_origin") == "ground_truth"]
    synthetic = [row for row in candidates if row.get("donor_origin") == "synthetic"]
    return {
        "forget_path": str(forget_path),
        "forget_sha256": digest,
        "forget_sha256_after": hashlib_sha256(after),
        "forget_bytes": len(before),
        "forget_bytes_after": len(after),
        "forget_bytes_unchanged": before == after,
        "scope_task_ids": assignment["task_ids"],
        "scope": config.get("scope"),
        "previous_cross_split_audit": config.get("previous_cross_split_audit"),
        "previous_candidates_reused": False,
        "calibration_author_ids": assignment["calibration_author_ids"],
        "holdout_author_ids": assignment["holdout_author_ids"],
        "calibration_author_count": len(assignment["calibration_author_ids"]),
        "holdout_author_count": len(assignment["holdout_author_ids"]),
        "min_authors_for_interval": config["min_authors_for_interval"],
        "author_interval_unstable": assignment["author_interval_unstable"],
        "confirmatory_localization_allowed": False,
        "rows": len(rows),
        "candidates": len(candidates),
        "ground_truth_candidates": len(ground_truth),
        "synthetic_candidates": len(synthetic),
        "donor_answer_non_null": len(ground_truth),
        "excluded": len(rows) - len(candidates),
        "pairing_rule": "non_entity_tokens_same_task_same_split",
        "token_scheme": "descriptive_whitespace_not_llama2",
        "exclusion_reasons": dict(reasons),
        "pair_cluster_count": len(clusters),
        "pair_cluster_ids": clusters,
        "donor_reuse": dict(Counter(row["donor_entity_id"] for row in ground_truth)),
        "approved_for_localization": 0,
        "semantic_review_status": "needs_manual_review",
        "pairing_mode": (config.get("pairing") or {}).get("mode"),
        "model_loaded": False,
        "tokenizer_called": False,
        "llama_for_causal_lm_instantiated": False,
        "student_e1_started": False,
        "e0_decision_preserved": E0_DECISION,
        "e0_retain_cap_reused": False,
        "method_effectiveness_claimed": False,
        "batch": "A",
        "localization_approved": False,
    }


def _audit_text(summary, inventory):
    reasons = ", ".join("%s=%s" % item for item in sorted(summary["exclusion_reasons"].items()))
    return "\n".join(
        [
            "# Batch A task 1 pair sidecar",
            "",
            "This directory is an auditable candidate list and exclusion list. It is not approved for localization.",
            "",
            "Source: `%s`" % inventory["path"],
            "SHA-256 before and after: `%s`" % summary["forget_sha256"],
            "Bytes before and after: %s" % summary["forget_bytes"],
            "Source rows: %s. Task counts: %s." % (inventory["rows"], inventory["task_id_counts"]),
            "Scope: task %s. Calibration entities: %s. Holdout entities: %s."
            % (
                ", ".join(summary["scope_task_ids"]),
                summary["calibration_author_count"],
                summary["holdout_author_count"],
            ),
            "Entity ids are local 20-row blocks after possessive-normalized names, not dataset ids.",
            "Names from this scope found in other tasks: %s." % (inventory["names_in_other_tasks"] or "none"),
            "Author interval is unstable because each side is below %s." % summary["min_authors_for_interval"],
            "Candidates: %s. Ground-truth donors: %s. Synthetic: %s. Excluded: %s."
            % (
                summary["candidates"],
                summary["ground_truth_candidates"],
                summary["synthetic_candidates"],
                summary["excluded"],
            ),
            "Exclusion reasons: %s." % (reasons or "none"),
            "Ground-truth rows keep a same-split donor answer. Synthetic rows replace the located slot with an ordinary personal name, a real city, or a calendar date, and leave y_minus empty.",
            "Token comparison is descriptive whitespace, not Llama-2 alignment. No model weights were loaded.",
            "Code version: %s." % _code_version(),
            "Command: `python3 -m workspace.I2.commands.build_same_split_qminus`.",
            "",
        ]
    )


def _code_version():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_REPO),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unavailable"
    text = completed.stdout.strip()
    return text or "unavailable"


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

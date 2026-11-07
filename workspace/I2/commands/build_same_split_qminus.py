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
    primary = [row for row in built if row.get("analysis_role") == "primary_single_slot"]
    compound = [row for row in built if row.get("analysis_role") == "compound_slot_diagnostic"]
    synthetic = [row for row in built if row.get("analysis_role") == "synthetic_non_primary"]
    excluded = [row for row in built if row.get("analysis_role") == "excluded"]
    if len(primary) + len(compound) + len(synthetic) + len(excluded) != len(built):
        raise ConfigError("analysis roles do not partition the sidecar")
    inventory = _inventory(forget_path, digest, before, rows, assignment, config)
    _write_json(output_dir / "inventory.json", inventory)
    _write_jsonl(output_dir / "qminus.jsonl", built)
    _write_jsonl(output_dir / "candidates.jsonl", primary)
    _write_jsonl(output_dir / "compound_diagnostics.jsonl", compound)
    _write_jsonl(output_dir / "synthetic_non_primary.jsonl", synthetic)
    _write_jsonl(output_dir / "excluded.jsonl", excluded)
    _write_json(output_dir / "relation_coverage.json", _relation_coverage(built))
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
        role = row.get("analysis_role")
        if row["is_candidate"] and role != "primary_single_slot":
            raise ConfigError("only a single-slot pending row can be a candidate")
        if role == "primary_single_slot":
            if row["record_status"] != "pending_review" or not row["is_candidate"]:
                raise ConfigError("primary row must stay pending review")
            if row.get("pairing_construction") != "qplus_name_replacement":
                raise ConfigError("primary q- must be a name replacement on q+")
            if row["split"] != row["donor_split"] or row["task_id"] != row["donor_task_id"]:
                raise ConfigError("candidate donor leaves the task or split")
            if row.get("replacement_kind") != "name" or len(row.get("intended_character_spans") or []) != 1:
                raise ConfigError("primary pair must replace one name span")
            if not row["template_outside_spans_unchanged"] or not row["slot_tokens_differ"]:
                raise ConfigError("primary name replacement did not stay inside one slot")
            if not row["y_minus"] or row["y_minus"] != row["donor_answer"]:
                raise ConfigError("candidate donor answer is empty")
            if row["q_minus"] is None or row["y_plus"] in (row["q_minus"], row["q_plus"]):
                raise ConfigError("candidate prompt contains an answer")
            if row.get("donor_origin") != "ground_truth":
                raise ConfigError("primary donor answer must come from an original record")
        elif role == "synthetic_non_primary":
            if row["is_candidate"] or row["record_status"] != "generated":
                raise ConfigError("synthetic row is not a primary candidate")
            if row["y_minus"] is not None or row["donor_answer"] is not None:
                raise ConfigError("synthetic row must leave the donor answer empty")
            if not row["q_minus"] or not row.get("fictional_name"):
                raise ConfigError("synthetic row is missing a fictional slot")
            if "Synthetic" in row["q_minus"] or "Synthetic" in row["fictional_name"]:
                raise ConfigError("synthetic slot still uses a placeholder")
            if not any(surface and surface in row["q_minus"] for surface in row.get("donor_slot_surfaces") or []):
                raise ConfigError("synthetic replacement did not land in the question")
        elif role == "compound_slot_diagnostic":
            if row["is_candidate"] or row["record_status"] != "generated":
                raise ConfigError("compound diagnostic is not a single-slot candidate")
            if row.get("replacement_kind") != "place+date":
                raise ConfigError("compound diagnostic must be place and date")
            if not row["y_minus"] or row["y_minus"] != row["donor_answer"]:
                raise ConfigError("compound diagnostic lost the original donor answer")
        elif role == "excluded":
            if row["is_candidate"] or row["record_status"] != "excluded" or row["y_minus"] is not None:
                raise ConfigError("excluded row still carries a candidate answer")
        else:
            raise ConfigError("analysis role is missing")


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


def _relation_coverage(rows):
    relation_counts = Counter(row.get("relation") or "other" for row in rows)
    review_counts = Counter(row.get("relation_review") or "other" for row in rows)
    role_counts = Counter(row.get("analysis_role") for row in rows)
    return {
        "relations": [
            "gender",
            "birth_place",
            "parents_occupation",
            "genre",
            "books",
            "awards",
            "other",
        ],
        "counts_by_relation": dict(relation_counts),
        "counts_by_review": dict(review_counts),
        "counts_by_analysis_role": dict(role_counts),
        "rows": [
            {
                "file_line_index": row["file_line_index"],
                "local_entity_id": row["local_entity_id"],
                "split": row["split"],
                "relation": row.get("relation"),
                "relation_review": row.get("relation_review"),
                "entity_slot": row.get("entity_slot"),
                "name_spans": row.get("name_spans") or [],
                "record_status": row.get("record_status"),
                "analysis_role": row.get("analysis_role"),
                "primary_exclusion_reason": row.get("primary_exclusion_reason"),
                "donor_entity_id": row.get("donor_entity_id"),
                "donor_file_line_index": row.get("donor_file_line_index"),
            }
            for row in rows
        ],
    }


def _summary(config, forget_path, digest, assignment, rows, before, after):
    reasons = Counter(row["exclusion_reason"] for row in rows if row.get("exclusion_reason"))
    primary = [row for row in rows if row.get("analysis_role") == "primary_single_slot"]
    compound = [row for row in rows if row.get("analysis_role") == "compound_slot_diagnostic"]
    synthetic = [row for row in rows if row.get("analysis_role") == "synthetic_non_primary"]
    excluded = [row for row in rows if row.get("analysis_role") == "excluded"]
    generated = [row for row in rows if row.get("record_status") == "generated"]
    pending = [row for row in rows if row.get("record_status") == "pending_review"]
    clusters = sorted({row["pair_cluster_id"] for row in primary if row.get("pair_cluster_id")})
    diagnostic_clusters = sorted(
        {row.get("diagnostic_cluster_id") for row in compound if row.get("diagnostic_cluster_id")}
    )
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
        "generated": len(generated),
        "excluded": len(excluded),
        "pending_review": len(pending),
        "candidates": len(primary),
        "ground_truth_candidates": len(primary),
        "synthetic_candidates": 0,
        "synthetic_non_primary": len(synthetic),
        "compound_slot_diagnostic": len(compound),
        "donor_answer_non_null": len([row for row in rows if row.get("y_minus")]),
        "pairing_rule": "qplus_single_name_same_relation",
        "token_scheme": "descriptive_whitespace_not_llama2",
        "exclusion_reasons": dict(reasons),
        "pair_cluster_count": len(clusters),
        "pair_cluster_ids": clusters,
        "compound_pair_cluster_count": len(diagnostic_clusters),
        "compound_pair_cluster_ids": diagnostic_clusters,
        "donor_reuse": dict(Counter(row["donor_entity_id"] for row in primary)),
        "relation_counts": dict(Counter(row.get("relation") or "other" for row in rows)),
        "approved_for_localization": 0,
        "pilot_decision": "descriptive_pilot",
        "confirmatory_requires_new_data_version": True,
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
            "Scope stays a descriptive pilot. Confirmatory localization needs a new data version and preregistration, enough independent authors and pair clusters, and a cross-task entity and donor-split check.",
            "Record status counts: generated=%s, excluded=%s, pending_review=%s, approved_for_localization=%s."
            % (
                summary["generated"],
                summary["excluded"],
                summary["pending_review"],
                summary["approved_for_localization"],
            ),
            "Primary single-slot candidates: %s. Compound place+date diagnostics: %s. Synthetic non-primary: %s."
            % (
                summary["candidates"],
                summary["compound_slot_diagnostic"],
                summary["synthetic_non_primary"],
            ),
            "Primary pair clusters: %s. Compound diagnostic clusters: %s."
            % (summary["pair_cluster_count"], summary["compound_pair_cluster_count"]),
            "Exclusion reasons: %s." % (reasons or "none"),
            "A primary q- keeps the q+ sentence and replaces one author name. y- is the donor's original answer for the same relation, even when the donor question is worded differently. No donor-sentence alignment is required. Rows without one verifiable same-relation answer are excluded and are not given a guessed answer.",
            "The place+date rows that name the author from a birth place and a birth date stay compound diagnostics. They are not in the first single-slot set.",
            "Synthetic rows keep an empty y_minus and are not counted as candidates.",
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

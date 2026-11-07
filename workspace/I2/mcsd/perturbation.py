"""q- as a perturbed input. The replacement is not a verified false fact.

Each q- changes one name slot. Scoring still uses the original answer y+.
The forget file is only read, and this module never approves a row for localization.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from I2.mcsd.errors import ConfigError
from I2.mcsd.same_split import (
    _FICTIONAL_NAMES,
    assign_splits,
    classify_question,
    load_forget_records,
    replacement_is_exact,
    require_locked_threshold,
)
from I2.mcsd.token_record import _encode, _question_text, annotate_tokens

GENERATION_RULE = "single_name_equal_token_pseudonym_v1"
_EXCLUSION = {
    "surname": "surname_slot",
    "place+date": "compound_slot",
    "multiple_names": "multiple_name_slots",
    "unlocated": "entity_slot_missing",
}


def localization_block_reason(example):
    """Unapproved perturbation rows stay out of the localization entry."""
    stage = example.get("sidecar_stage")
    role = example.get("q_minus_role")
    if stage is None and role != "perturbed_input":
        return None
    if stage == "approved_for_localization" and example.get("approved_for_localization") is True:
        return None
    return "not_approved_for_localization"


def pseudonym_order(seed, names=None):
    """Fixed table, shuffled by the recorded seed. Model scores are not an input."""
    ordered = list(_FICTIONAL_NAMES if names is None else names)
    random.Random(int(seed)).shuffle(ordered)
    return ordered


def entity_token_length(tokenizer, tags, question, span):
    """Token count of one character span. A token that crosses the span is refused."""
    _ids, offsets = _encode(tokenizer, _question_text(tags, question))
    if not offsets:
        return None
    prefix = len(tags["question_start_tag"])
    left = prefix + int(span[0])
    right = prefix + int(span[1])
    if right <= left:
        return None
    count = 0
    for tok_start, tok_end in offsets:
        if tok_start == tok_end:
            continue
        if tok_end <= left or tok_start >= right:
            continue
        if tok_start < left or tok_end > right:
            return None
        count += 1
    return count or None


def encode_perturbed_pair(tokenizer, q_plus, q_minus, y_plus, tags):
    """Encode q+ and q- with the same y+. Unequal prompt lengths are not patched."""
    if y_plus is None or not str(y_plus).strip():
        raise ValueError("y+ is missing; refusing to invent y-")
    if q_minus is None or not str(q_minus).strip():
        raise ValueError("q- perturbation is missing")
    from I2.commands.prepare_pre_experiment import encode_row

    answer = str(y_plus)
    plus = encode_row(tokenizer, q_plus, answer, tags)
    minus = encode_row(tokenizer, q_minus, answer, tags)
    reasons = []
    if plus["num_question_tokens"] != minus["num_question_tokens"] or len(plus["donor_input_ids"]) != len(
        minus["donor_input_ids"]
    ):
        reasons.append("length_mismatch")
    if plus["score_labels"] != minus["score_labels"]:
        reasons.append("y_plus_labels_differ")
    return {
        "aligned": not reasons,
        "reasons": reasons,
        "prompt_plus_ids": plus["donor_input_ids"],
        "prompt_minus_ids": minus["donor_input_ids"],
        "score_plus_ids": plus["score_input_ids"],
        "score_minus_ids": minus["score_input_ids"],
        "score_plus_labels": plus["score_labels"],
        "score_minus_labels": minus["score_labels"],
        "y_plus": answer,
        "y_minus": None,
    }


def build_perturbation_rows(rows, config, forget_sha256, tokenizer=None, tags=None, tokenizer_meta=None):
    require_locked_threshold(config)
    spec = _spec(config)
    prepared = [dict(row) for row in rows]
    assignment = assign_splits(
        prepared,
        config["questions_per_author"],
        config["authors_per_task"],
        config["task_ids"],
    )
    order = pseudonym_order(spec["seed"])
    corpus = _corpus(prepared)
    tags = tags or {
        "question_start_tag": "[INST] ",
        "question_end_tag": " [/INST]",
        "answer_tag": "",
    }
    meta = tokenizer_meta or {}
    built = []
    rejections = Counter()
    for author in assignment["authors"].values():
        for row in author["rows"]:
            row.update(classify_question(row["question"], author))
            controls, reasons = _controls(
                row,
                author,
                forget_sha256,
                order,
                corpus,
                spec,
                tokenizer,
                tags,
                meta,
            )
            rejections.update(reasons)
            if controls:
                built.extend(controls)
            else:
                built.append(_excluded(row, author, forget_sha256, spec, _exclusion_reason(row, reasons, tokenizer)))
    built.sort(key=lambda item: (item["file_line_index"], item.get("control_index") or 0))
    info = {
        "generation_rule": spec["rule"],
        "seed": spec["seed"],
        "max_controls": spec["max_controls"],
        "pseudonym_order": order,
        "pseudonym_table_size": len(_FICTIONAL_NAMES),
        "rejections": dict(rejections),
    }
    return built, assignment, info


def run_perturbation(forget_path, config, output_dir, tokenizer=None, tags=None, tokenizer_meta=None):
    forget_path = Path(forget_path)
    before = forget_path.read_bytes()
    rows, digest = load_forget_records(forget_path)
    built, assignment, info = build_perturbation_rows(
        rows,
        config,
        digest,
        tokenizer=tokenizer,
        tags=tags,
        tokenizer_meta=tokenizer_meta,
    )
    _audit_rows(built)
    after = forget_path.read_bytes()
    if after != before:
        raise ConfigError("forget file was modified")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    perturbations = [row for row in built if row.get("q_minus")]
    excluded = [row for row in built if not row.get("q_minus")]
    _write_jsonl(output_dir / "qminus.jsonl", built)
    _write_jsonl(output_dir / "perturbations.jsonl", perturbations)
    _write_jsonl(output_dir / "excluded.jsonl", excluded)
    summary = _summary(
        config,
        forget_path,
        digest,
        before,
        after,
        assignment,
        built,
        info,
        tokenizer_meta or {},
    )
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "audit.md").write_text(_audit_text(summary))
    return summary, built


def _spec(config):
    spec = config.get("perturbation") or {}
    if spec.get("seed") is None:
        raise ConfigError("perturbation.seed is missing")
    max_controls = int(spec.get("max_controls") or 3)
    if max_controls < 1:
        raise ConfigError("perturbation.max_controls must be positive")
    return {
        "rule": spec.get("rule") or GENERATION_RULE,
        "seed": int(spec["seed"]),
        "max_controls": max_controls,
    }


def _corpus(rows):
    return "\n".join("%s\n%s" % (row.get("question", ""), row.get("answer", "")) for row in rows)


def _controls(row, author, digest, order, corpus, spec, tokenizer, tags, meta):
    span = _single_name_span(row)
    if span is None:
        return [], Counter()
    start, end, suffix = span
    question = row["question"]
    answer = row["answer"]
    source_count = None
    if tokenizer is not None:
        source_count = entity_token_length(tokenizer, tags, question, (start, end))
        if not source_count:
            return [], Counter({"source_token_span": 1})
    accepted = []
    rejections = Counter()
    table_index = {name: index for index, name in enumerate(_FICTIONAL_NAMES)}
    for pseudonym in order:
        if len(accepted) >= spec["max_controls"]:
            break
        if pseudonym in corpus or pseudonym == author["name"]:
            rejections["in_target_text"] += 1
            continue
        surface = pseudonym + suffix
        qminus = question[:start] + surface + question[end:]
        written = (start, start + len(surface))
        source_span = (start, end)
        if qminus == question or not replacement_is_exact(question, qminus, [source_span], [written]):
            rejections["outside_span"] += 1
            continue
        if author["name"] in qminus:
            rejections["source_name_remains"] += 1
            continue
        record = _generated_row(
            row,
            author,
            digest,
            qminus,
            pseudonym,
            table_index[pseudonym],
            source_span,
            written,
            spec,
            meta,
        )
        if tokenizer is None:
            accepted.append(record)
            continue
        replacement_count = entity_token_length(tokenizer, tags, qminus, written)
        if replacement_count != source_count:
            rejections["token_length"] += 1
            continue
        audited = annotate_tokens(record, tokenizer, tags, meta)
        if audited.get("token_alignment") != "aligned":
            rejections[audited.get("token_alignment") or "not_aligned"] += 1
            continue
        record["sidecar_stage"] = "token_aligned"
        record["token_alignment"] = "aligned"
        record["fact_span"] = audited.get("fact_span")
        record["tokenizer_source"] = audited.get("tokenizer_source")
        record["transformers_version"] = audited.get("transformers_version")
        record["tokenizer_sha256"] = audited.get("tokenizer_sha256")
        record["is_candidate"] = False
        record["approved_for_localization"] = False
        record["y_minus"] = None
        record["donor_answer"] = None
        record["exclusion_reason"] = None
        accepted.append(record)
    for index, item in enumerate(accepted):
        item["control_index"] = index
        item["local_id"] = "%s:%s:%s" % (digest, row["file_line_index"], index)
    return accepted, rejections


def _single_name_span(row):
    spans = row.get("name_spans") or []
    if row.get("entity_slot") != "name" or len(spans) != 1:
        return None
    start, end, suffix = spans[0]
    if int(end) <= int(start):
        return None
    return int(start), int(end), suffix


def _generated_row(row, author, digest, qminus, pseudonym, pseudonym_index, source_span, written, spec, meta):
    return {
        "local_id": "%s:%s" % (digest, row["file_line_index"]),
        "source_local_id": "%s:%s" % (digest, row["file_line_index"]),
        "forget_sha256": digest,
        "file_line_index": row["file_line_index"],
        "control_index": None,
        "task_id": row["task_id"],
        "split": author["split"],
        "local_entity_id": author["local_entity_id"],
        "q_plus": row["question"],
        "y_plus": row["answer"],
        "source_answer": row["answer"],
        "q_minus": qminus,
        "y_minus": None,
        "donor_answer": None,
        "q_minus_role": "perturbed_input",
        "pseudonym": pseudonym,
        "pseudonym_index": pseudonym_index,
        "generation_rule": spec["rule"],
        "seed": spec["seed"],
        "replacement_kind": "name",
        "intended_character_spans": [list(source_span)],
        "character_spans": [list(written)],
        "template_outside_spans_unchanged": True,
        "generated": True,
        "is_candidate": False,
        "approved_for_localization": False,
        "sidecar_stage": "qminus_generated",
        "token_alignment": "not_checked_no_tokenizer",
        "fact_span": None,
        "exclusion_reason": None,
        "tokenizer_source": meta.get("tokenizer_source"),
        "transformers_version": meta.get("transformers_version"),
        "tokenizer_sha256": meta.get("tokenizer_sha256"),
    }


def _excluded(row, author, digest, spec, reason):
    return {
        "local_id": "%s:%s" % (digest, row["file_line_index"]),
        "source_local_id": "%s:%s" % (digest, row["file_line_index"]),
        "forget_sha256": digest,
        "file_line_index": row["file_line_index"],
        "control_index": None,
        "task_id": row["task_id"],
        "split": author["split"],
        "local_entity_id": author["local_entity_id"],
        "q_plus": row["question"],
        "y_plus": row["answer"],
        "source_answer": row["answer"],
        "q_minus": None,
        "y_minus": None,
        "donor_answer": None,
        "q_minus_role": "perturbed_input",
        "pseudonym": None,
        "generation_rule": spec["rule"],
        "seed": spec["seed"],
        "generated": False,
        "is_candidate": False,
        "approved_for_localization": False,
        "sidecar_stage": None,
        "token_alignment": None,
        "fact_span": None,
        "exclusion_reason": reason,
    }


def _exclusion_reason(row, rejections, tokenizer):
    if rejections.get("source_token_span"):
        return "source_token_span"
    if _single_name_span(row) is None and not rejections:
        return _EXCLUSION.get(row.get("entity_slot"), "not_single_name_slot")
    if tokenizer is not None:
        return "no_equal_token_pseudonym"
    return "no_pseudonym_outside_target"


def _audit_rows(rows):
    for row in rows:
        if row.get("approved_for_localization"):
            raise ConfigError("approved_for_localization must stay false")
        if row.get("y_minus") is not None or row.get("donor_answer") is not None:
            raise ConfigError("perturbation row invented an answer")
        stage = row.get("sidecar_stage")
        if stage == "approved_for_localization":
            raise ConfigError("perturbation builder cannot approve localization")
        if row.get("q_minus"):
            if stage not in ("qminus_generated", "token_aligned"):
                raise ConfigError("perturbation stage is missing")
            if row.get("q_minus_role") != "perturbed_input":
                raise ConfigError("q- role must be perturbed_input")
        if stage == "token_aligned" and row.get("token_alignment") != "aligned":
            raise ConfigError("token_aligned row failed the position audit")
        if stage == "qminus_generated" and row.get("token_alignment") == "aligned":
            raise ConfigError("generated row was marked aligned")
        if row.get("is_candidate"):
            raise ConfigError("a perturbation row is not a localization candidate")


def _summary(config, forget_path, digest, before, after, assignment, rows, info, meta):
    generated = [row for row in rows if row.get("sidecar_stage") == "qminus_generated"]
    aligned = [row for row in rows if row.get("sidecar_stage") == "token_aligned"]
    excluded = [row for row in rows if not row.get("q_minus")]
    return {
        "forget_path": str(forget_path),
        "forget_sha256": digest,
        "forget_sha256_after": _sha256(after),
        "forget_bytes": len(before),
        "forget_bytes_after": len(after),
        "forget_bytes_unchanged": before == after,
        "scope_task_ids": assignment["task_ids"],
        "generation_rule": info["generation_rule"],
        "seed": info["seed"],
        "max_controls": info["max_controls"],
        "pseudonym_table_size": info["pseudonym_table_size"],
        "pseudonym_order": info["pseudonym_order"],
        "rejections": info["rejections"],
        "rows": len(rows),
        "source_rows": len({row["file_line_index"] for row in rows}),
        "qminus_generated": len(generated),
        "token_aligned": len(aligned),
        "excluded": len(excluded),
        "approved_for_localization": 0,
        "y_minus_non_null": 0,
        "donor_answer_non_null": 0,
        "tokenizer_source": meta.get("tokenizer_source"),
        "tokenizer_loaded": bool(meta.get("tokenizer_loaded")),
        "transformers_version": meta.get("transformers_version"),
        "tokenizer_sha256": meta.get("tokenizer_sha256"),
        "llama_for_causal_lm_instantiated": False,
        "real_model_executed": False,
        "localization_started": False,
        "student_e1_started": False,
        "n_written": None,
        "r_written": None,
        "pairing_mode": None,
        "author_interval_unstable": assignment["author_interval_unstable"],
        "confirmatory_localization_allowed": False,
    }


def _audit_text(summary):
    return "\n".join(
        [
            "# Perturbation q-",
            "",
            "q- replaces one name slot. It is a perturbed input, not a verified false fact.",
            "y- is empty. Both forwards score the original y+.",
            "Generation rule: `%s`. Seed: `%s`." % (summary["generation_rule"], summary["seed"]),
            "qminus_generated=%s, token_aligned=%s, approved_for_localization=0."
            % (summary["qminus_generated"], summary["token_aligned"]),
            "The forget file was not modified. Localization was not started.",
            "",
        ]
    )


def _sha256(blob):
    import hashlib

    return hashlib.sha256(blob).hexdigest()


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

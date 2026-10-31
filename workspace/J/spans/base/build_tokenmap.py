#!/usr/bin/env python3
"""Map pilot200 character spans onto ASU Llama-2 training token positions.

Does not train, download, edit losses.py, or rewrite the source annotation file.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path("/usr/local/ASU-unlearning")
INPUT_PATH = ROOT / "workspace/J/spans/base/forget05_pilot200_preannotation.jsonl"
CANONICAL_FORGET = ROOT / "Right-to-be-forgotten/data/tofu/forget05.json"
OUT_DIR = ROOT / "workspace/J/spans/base/tokenmap_v1"
TOKENMAP_PATH = OUT_DIR / "forget05_pilot200_tokenmap.jsonl"
AUDIT_PATH = OUT_DIR / "forget05_pilot200_tokenmap_audit.json"

TOKENIZER_DIR = Path(
    "/root/autodl-tmp/huggingface/hub/models--NousResearch--Llama-2-7b-chat-hf/"
    "snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee"
)
EXPECTED_CANONICAL_SHA = "a8c207a905af73ea4797b1af92a14cf458f42df567c941f7284f87d930096f4b"
QUESTION_START = "[INST] "
QUESTION_END = " [/INST]"
ANSWER_TAG = ""
MAX_LENGTH = 500
IGNORE_INDEX = -100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        )
    except Exception as exc:  # noqa: BLE001
        return f"unavailable:{exc}"


def load_tokenizer():
    from transformers import AutoTokenizer

    if not TOKENIZER_DIR.is_dir() or not (TOKENIZER_DIR / "tokenizer.json").is_file():
        raise SystemExit(f"tokenizer snapshot missing: {TOKENIZER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR, local_files_only=True, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("fast tokenizer required for offset_mapping")
    # Match forget.py: pad with eos after assignment.
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def serialize(question: str, answer: str) -> tuple[str, str, int, int]:
    new_question = QUESTION_START + question + QUESTION_END
    full_text = new_question + ANSWER_TAG + answer
    answer_start = len(new_question + ANSWER_TAG)
    answer_end = answer_start + len(answer)
    return new_question, full_text, answer_start, answer_end


def encode_training_tensors(tokenizer, question: str, answer: str):
    """Reproduce gold_rank.masking.encode_forget_example plus offset_mapping."""
    new_question, full_text, answer_start, answer_end = serialize(question, answer)
    num_question_tokens = len(
        tokenizer.tokenize(new_question, add_special_tokens=True)
    )
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        truncation=True,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    attention_mask = list(encoded["attention_mask"])
    offsets = list(encoded["offset_mapping"])
    untruncated_len = len(
        tokenizer(full_text, add_special_tokens=True, truncation=False)["input_ids"]
    )
    truncated = untruncated_len > MAX_LENGTH

    pad_length = MAX_LENGTH - len(input_ids)
    pad_input_ids = input_ids + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = attention_mask + [0] * pad_length
    pad_offsets = offsets + [(0, 0)] * pad_length

    if len(input_ids) == MAX_LENGTH:
        labels = list(input_ids)
    else:
        labels = list(input_ids) + [tokenizer.eos_token_id] + [IGNORE_INDEX] * (
            pad_length - 1
        )
    for index in range(num_question_tokens):
        labels[index] = IGNORE_INDEX

    return {
        "new_question": new_question,
        "full_text": full_text,
        "answer_start": answer_start,
        "answer_end": answer_end,
        "num_question_tokens": num_question_tokens,
        "input_ids": pad_input_ids,
        "attention_mask": pad_attention_mask,
        "labels": labels,
        "offsets": pad_offsets,
        "unpadded_len": len(input_ids),
        "untruncated_len": untruncated_len,
        "truncated": truncated,
    }


def intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


def map_span_tokens(
    span: dict,
    answer: str,
    answer_start: int,
    offsets: list[tuple[int, int]],
    labels: list[int],
    input_ids: list[int],
    unpadded_len: int,
):
    flags: list[str] = []
    start = int(span["start_char"])
    end = int(span["end_char"])
    text = span["text"]
    extracted = answer[start:end]
    if extracted != text:
        flags.append("char_text_mismatch")
        return {
            **span,
            "full_sequence_start_char": None,
            "full_sequence_end_char": None,
            "input_token_positions": [],
            "label_positions": [],
            "logit_positions": [],
            "token_ids": [],
            "boundary_crossing": False,
            "overflow_char_intervals": [],
            "flags": flags,
            "alignable": False,
        }

    full_start = answer_start + start
    full_end = answer_start + end
    input_positions: list[int] = []
    overflow: list[dict] = []
    boundary_crossing = False

    for index, (tok_start, tok_end) in enumerate(offsets):
        if index >= unpadded_len:
            break
        if tok_start == tok_end == 0:
            continue
        if not intervals_overlap(tok_start, tok_end, full_start, full_end):
            continue
        input_positions.append(index)
        left_overflow = tok_start < full_start
        right_overflow = tok_end > full_end
        if left_overflow or right_overflow:
            boundary_crossing = True
            overflow.append(
                {
                    "token_index": index,
                    "token_char_start": tok_start,
                    "token_char_end": tok_end,
                    "outside_left": [tok_start, full_start] if left_overflow else None,
                    "outside_right": [full_end, tok_end] if right_overflow else None,
                }
            )

    if not input_positions:
        flags.append("no_token_overlap")
        # Truncation can drop trailing answer characters.
        if full_end > 0 and offsets[unpadded_len - 1][1] <= full_start:
            flags.append("span_truncated_or_outside_encoded_prefix")

    label_positions = [
        index for index in input_positions if labels[index] != IGNORE_INDEX
    ]
    dropped_ignore = [index for index in input_positions if labels[index] == IGNORE_INDEX]
    if dropped_ignore:
        flags.append("overlap_ignore_index")
    logit_positions = [index - 1 for index in label_positions if index >= 1]
    if any(index < 1 for index in label_positions):
        flags.append("label_at_sequence_start")

    token_ids = [input_ids[index] for index in input_positions]
    alignable = extracted == text and bool(label_positions) and "char_text_mismatch" not in flags
    if boundary_crossing:
        flags.append("boundary_crossing")

    return {
        **span,
        "full_sequence_start_char": full_start,
        "full_sequence_end_char": full_end,
        "input_token_positions": input_positions,
        "label_positions": label_positions,
        "logit_positions": logit_positions,
        "token_ids": token_ids,
        "boundary_crossing": boundary_crossing,
        "overflow_char_intervals": overflow,
        "flags": flags,
        "alignable": alignable,
    }


def answer_token_interval(labels: list[int], num_question_tokens: int, unpadded_len: int):
    answer_label_indices = [
        index
        for index in range(len(labels))
        if labels[index] != IGNORE_INDEX and index >= num_question_tokens
    ]
    if not answer_label_indices:
        return None, None, []
    # Prefer the contiguous answer prefix before pad -100s; include the single eos label.
    return answer_label_indices[0], answer_label_indices[-1], answer_label_indices


def build_record(row: dict, encoded: dict, mapped_spans: list[dict]) -> dict:
    labels = encoded["labels"]
    ans_lo, ans_hi, answer_label_indices = answer_token_interval(
        labels, encoded["num_question_tokens"], encoded["unpadded_len"]
    )
    answer_label_set = set(answer_label_indices)

    # Include every span_candidate in the primary mask (user instruction).
    m_tar_positions: list[int] = []
    seen = set()
    for span in mapped_spans:
        for position in span["label_positions"]:
            if position not in seen:
                seen.add(position)
                m_tar_positions.append(position)
    m_tar_positions.sort()

    flags = list(row.get("flags") or [])
    mapping_anomaly = False

    if encoded["truncated"]:
        flags.append("sequence_truncated")
    for span in mapped_spans:
        for item in span.get("flags") or []:
            if item not in flags:
                flags.append(item)
        if span.get("boundary_crossing"):
            mapping_anomaly = True
        if not span.get("alignable") and (row.get("span_candidates") or []):
            # Empty no_target spans are fine.
            if span.get("text"):
                mapping_anomaly = True

    # Shared tokens across spans.
    token_owners: dict[int, list[str]] = {}
    for span in mapped_spans:
        for position in span["label_positions"]:
            token_owners.setdefault(position, []).append(span.get("span_id", "?"))
    multi = {str(pos): ids for pos, ids in token_owners.items() if len(ids) > 1}
    if multi:
        flags.append("multi_span_same_token")
        # Informative only; not necessarily an anomaly for contiguous spans.

    # Mask hygiene.
    for position in m_tar_positions:
        if position < encoded["num_question_tokens"]:
            flags.append("mask_in_prompt")
            mapping_anomaly = True
        if labels[position] == IGNORE_INDEX:
            flags.append("mask_on_ignore_index")
            mapping_anomaly = True
        if encoded["attention_mask"][position] == 0:
            flags.append("mask_on_padding")
            mapping_anomaly = True
        if position not in answer_label_set and labels[position] != IGNORE_INDEX:
            # eos after answer is a valid labeled position in training labels
            pass

    human = row["human_review_status"]
    record_status = row["record_status"]
    if record_status == "no_target":
        m_tar_positions = []
        if mapped_spans:
            flags.append("no_target_had_spans")
            mapping_anomaly = True

    training_eligible = (
        human == "confirmed_by_user"
        and record_status not in {"needs_manual_review", "no_target"}
        and human != "pending"
        and not mapping_anomaly
        and "char_text_mismatch" not in flags
        and "no_token_overlap" not in flags
        and "span_truncated_or_outside_encoded_prefix" not in flags
        and "mask_in_prompt" not in flags
        and "mask_on_ignore_index" not in flags
        and "mask_on_padding" not in flags
    )
    # Explicit pending / needs_manual / no_target force false.
    if human == "pending" or record_status in {"needs_manual_review", "no_target"}:
        training_eligible = False

    return {
        "annotation_version": row.get("annotation_version"),
        "batch_id": row.get("batch_id"),
        "source": row["source"],
        "task_id": row["task_id"],
        "question": row["question"],
        "answer": row["answer"],
        "human_review_status": human,
        "record_status": record_status,
        "token_alignment_status": "mapped_v1",
        "offset_convention": row.get("offset_convention"),
        "user_scope_decision": row.get("user_scope_decision"),
        "flags": flags,
        "serialization": {
            "question_start_tag": QUESTION_START,
            "question_end_tag": QUESTION_END,
            "answer_tag": ANSWER_TAG,
            "answer_start_char": encoded["answer_start"],
            "answer_end_char": encoded["answer_end"],
            "num_question_tokens": encoded["num_question_tokens"],
            "sequence_length": MAX_LENGTH,
            "unpadded_len": encoded["unpadded_len"],
            "untruncated_len": encoded["untruncated_len"],
            "truncated": encoded["truncated"],
        },
        "answer_token_interval": {
            "first_label_index": ans_lo,
            "last_label_index": ans_hi,
            "label_positions": answer_label_indices,
        },
        "span_candidates": mapped_spans,
        "m_tar_token_positions": m_tar_positions,
        "m_tar_token_count": len(m_tar_positions),
        "multi_span_same_token": multi,
        "training_eligible": training_eligible,
        "mapping_anomaly": mapping_anomaly,
        "m0_nogo_unchanged": True,
    }


def validate_outputs(rows_in: list[dict], rows_out: list[dict]) -> tuple[bool, list[dict]]:
    failures: list[dict] = []
    if len(rows_in) != 200 or len(rows_out) != 200 or len(rows_in) != len(rows_out):
        failures.append(
            {
                "local_sample_id": None,
                "source_line": None,
                "error_type": "row_count_mismatch",
                "detail": f"in={len(rows_in)} out={len(rows_out)}",
            }
        )

    confirmed_in = {
        r["source"]["local_sample_id"]: r for r in rows_in if r["human_review_status"] == "confirmed_by_user"
    }
    for out in rows_out:
        sid = out["source"]["local_sample_id"]
        line = out["source"]["source_line_1based"]
        labels = None

        # Char span exact hit.
        for span in out["span_candidates"]:
            if not span.get("text"):
                continue
            extracted = out["answer"][span["start_char"] : span["end_char"]]
            if extracted != span["text"]:
                failures.append(
                    {
                        "local_sample_id": sid,
                        "source_line": line,
                        "error_type": "char_span_mismatch",
                        "detail": span.get("span_id"),
                    }
                )

        # Token positions in range and on valid answer labels after shift.
        # Rebuild labels quickly from serialization metadata + positions.
        seq_len = out["serialization"]["sequence_length"]
        nq = out["serialization"]["num_question_tokens"]
        answer_labels = set(out["answer_token_interval"]["label_positions"])
        for pos in out["m_tar_token_positions"]:
            if not (0 <= pos < seq_len):
                failures.append(
                    {
                        "local_sample_id": sid,
                        "source_line": line,
                        "error_type": "token_out_of_range",
                        "detail": pos,
                    }
                )
            if pos < nq:
                failures.append(
                    {
                        "local_sample_id": sid,
                        "source_line": line,
                        "error_type": "mask_in_prompt",
                        "detail": pos,
                    }
                )
            if pos not in answer_labels:
                failures.append(
                    {
                        "local_sample_id": sid,
                        "source_line": line,
                        "error_type": "mask_not_on_answer_label",
                        "detail": pos,
                    }
                )
            # Shift: logit = label-1 must be valid index in [0, seq_len-2] conceptually.
            if pos < 1:
                failures.append(
                    {
                        "local_sample_id": sid,
                        "source_line": line,
                        "error_type": "no_logit_for_label",
                        "detail": pos,
                    }
                )

        if out["record_status"] == "no_target" and out["m_tar_token_positions"]:
            failures.append(
                {
                    "local_sample_id": sid,
                    "source_line": line,
                    "error_type": "no_target_nonempty_mask",
                    "detail": out["m_tar_token_positions"],
                }
            )

        if out["human_review_status"] == "pending" and out["training_eligible"]:
            failures.append(
                {
                    "local_sample_id": sid,
                    "source_line": line,
                    "error_type": "pending_marked_eligible",
                    "detail": None,
                }
            )
        if out["record_status"] in {"needs_manual_review", "no_target"} and out[
            "training_eligible"
        ]:
            failures.append(
                {
                    "local_sample_id": sid,
                    "source_line": line,
                    "error_type": "blocked_status_marked_eligible",
                    "detail": out["record_status"],
                }
            )

        # Confirmed rows preserve span char boundaries and texts.
        if sid in confirmed_in:
            src_spans = confirmed_in[sid]["span_candidates"]
            out_spans = out["span_candidates"]
            if len(src_spans) != len(out_spans):
                failures.append(
                    {
                        "local_sample_id": sid,
                        "source_line": line,
                        "error_type": "confirmed_span_count_changed",
                        "detail": f"{len(src_spans)}->{len(out_spans)}",
                    }
                )
            else:
                for left, right in zip(src_spans, out_spans):
                    for key in (
                        "span_id",
                        "start_char",
                        "end_char",
                        "text",
                        "label",
                        "m_tar_candidate",
                        "decision_code",
                    ):
                        if left.get(key) != right.get(key):
                            failures.append(
                                {
                                    "local_sample_id": sid,
                                    "source_line": line,
                                    "error_type": "confirmed_span_field_changed",
                                    "detail": f"{left.get('span_id')}:{key}",
                                }
                            )

        # Status preservation.
        src = next(r for r in rows_in if r["source"]["local_sample_id"] == sid)
        if src["human_review_status"] != out["human_review_status"]:
            failures.append(
                {
                    "local_sample_id": sid,
                    "source_line": line,
                    "error_type": "human_review_status_changed",
                    "detail": None,
                }
            )
        if src["record_status"] != out["record_status"]:
            failures.append(
                {
                    "local_sample_id": sid,
                    "source_line": line,
                    "error_type": "record_status_changed",
                    "detail": None,
                }
            )
        if src["question"] != out["question"] or src["answer"] != out["answer"]:
            failures.append(
                {
                    "local_sample_id": sid,
                    "source_line": line,
                    "error_type": "qa_text_changed",
                    "detail": None,
                }
            )

    return len(failures) == 0, failures


def main() -> None:
    if TOKENMAP_PATH.exists() or AUDIT_PATH.exists():
        raise SystemExit(f"refusing to overwrite existing outputs under {OUT_DIR}")

    input_sha = sha256_file(INPUT_PATH)
    canonical_sha = sha256_file(CANONICAL_FORGET)
    if canonical_sha != EXPECTED_CANONICAL_SHA:
        raise SystemExit(
            f"canonical forget05 sha mismatch: {canonical_sha} != {EXPECTED_CANONICAL_SHA}"
        )

    rows_in = [
        json.loads(line)
        for line in INPUT_PATH.read_text().splitlines()
        if line.strip()
    ]
    if len(rows_in) != 200:
        raise SystemExit(f"expected 200 rows, found {len(rows_in)}")

    # Identity checks against canonical lines.
    canonical_lines = CANONICAL_FORGET.read_text().splitlines()
    for row in rows_in:
        if row["source"]["file_sha256"] != EXPECTED_CANONICAL_SHA:
            raise SystemExit(
                f"row source sha mismatch: {row['source']['local_sample_id']}"
            )
        line_no = int(row["source"]["source_line_1based"])
        src = json.loads(canonical_lines[line_no - 1])
        if (
            src["question"] != row["question"]
            or src["answer"] != row["answer"]
            or str(src["task_id"]) != str(row["task_id"])
        ):
            raise SystemExit(
                f"source line mismatch for {row['source']['local_sample_id']}"
            )

    print("loading tokenizer", TOKENIZER_DIR, flush=True)
    tokenizer = load_tokenizer()

    # Config self-check: answer offsets align with offset_mapping for one sample.
    probe = rows_in[0]
    encoded_probe = encode_training_tensors(
        tokenizer, probe["question"], probe["answer"]
    )
    ans_start = encoded_probe["answer_start"]
    ans_end = encoded_probe["answer_end"]
    if encoded_probe["full_text"][ans_start:ans_end] != probe["answer"]:
        raise SystemExit("constructed answer character interval does not match answer")
    # First non-empty offset overlapping answer must decode into answer prefix region.
    hit = False
    for index, (start, end) in enumerate(encoded_probe["offsets"]):
        if index >= encoded_probe["unpadded_len"]:
            break
        if start == end == 0:
            continue
        if intervals_overlap(start, end, ans_start, ans_end):
            hit = True
            break
    if not hit:
        raise SystemExit(
            "tokenizer offset_mapping does not overlap constructed answer interval"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_out: list[dict] = []
    for index, row in enumerate(rows_in):
        encoded = encode_training_tensors(tokenizer, row["question"], row["answer"])
        mapped = [
            map_span_tokens(
                span,
                row["answer"],
                encoded["answer_start"],
                encoded["offsets"],
                encoded["labels"],
                encoded["input_ids"],
                encoded["unpadded_len"],
            )
            for span in (row.get("span_candidates") or [])
        ]
        rows_out.append(build_record(row, encoded, mapped))
        if (index + 1) % 50 == 0:
            print(f"mapped {index + 1}/200", flush=True)

    checks_passed, failures = validate_outputs(rows_in, rows_out)

    with open(TOKENMAP_PATH, "w") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    span_label_counts = Counter()
    status_counts = Counter(r["human_review_status"] for r in rows_out)
    record_counts = Counter(r["record_status"] for r in rows_out)
    n_spans = 0
    n_boundary = 0
    n_truncated = 0
    n_map_fail = 0
    n_eligible = 0
    n_mask_tokens = 0
    for row in rows_out:
        if row["training_eligible"]:
            n_eligible += 1
        n_mask_tokens += row["m_tar_token_count"]
        if row["serialization"]["truncated"]:
            n_truncated += 1
        if row["mapping_anomaly"]:
            n_map_fail += 1
        for span in row["span_candidates"]:
            n_spans += 1
            span_label_counts[span.get("label")] += 1
            if span.get("boundary_crossing"):
                n_boundary += 1

    import torch
    import transformers

    audit = {
        "task": "pilot200_tokenmap_v1",
        "m0_nogo_unchanged": True,
        "training_started": False,
        "input_path": str(INPUT_PATH),
        "input_sha256": input_sha,
        "canonical_forget05_path": str(CANONICAL_FORGET),
        "canonical_forget05_sha256": canonical_sha,
        "output_tokenmap_path": str(TOKENMAP_PATH),
        "output_tokenmap_sha256": sha256_file(TOKENMAP_PATH),
        "git_commit": git_commit(),
        "package_versions": {
            "python": sys.version.split()[0],
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
        "tokenizer": {
            "path": str(TOKENIZER_DIR),
            "revision": "351844e75ed0bcbbe3f10671b3c808d2b83894ee",
            "class_name": type(tokenizer).__name__,
            "is_fast": bool(tokenizer.is_fast),
            "tokenizer_json_sha256": sha256_file(TOKENIZER_DIR / "tokenizer.json"),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
            "vocab_size": int(tokenizer.vocab_size),
        },
        "model_weights_referenced_not_loaded_for_forward": {
            "init_student_teacher": "/root/autodl-tmp/models/tofu_ft_llama2-7b",
            "init_snapshot": "cc5b31c69127d5da881608e640f2b453c446435b",
            "asu_gd_student": (
                "/root/autodl-tmp/ASU-unlearning/checkpoints/tofu_replay/llama2-7b/"
                "forget05/ASU+GD/seed_1001/epoch5_1e-05_FixRefTrue_maskTrue_Fcoeff0.1_"
                "Rcoeff1.0_Temp2.3_LayersNone_beta0.1/1/unlearn_times_1/checkpoint-last"
            ),
        },
        "serialization_rule": {
            "template": 'QUESTION_START + question + QUESTION_END + ANSWER_TAG + answer',
            "question_start_tag": QUESTION_START,
            "question_end_tag": QUESTION_END,
            "answer_tag": ANSWER_TAG,
            "add_special_tokens": True,
            "max_length": MAX_LENGTH,
            "padding_token_id": tokenizer.eos_token_id,
            "source_files": [
                "Right-to-be-forgotten/forget.py",
                "workspace/gold_rank/masking.py",
                "Right-to-be-forgotten/config/model_config.yaml",
            ],
        },
        "label_and_shift_rule": {
            "question_labels": "first num_question_tokens positions set to -100",
            "ignore_index": IGNORE_INDEX,
            "asu_loss_shift": "logits[:, :-1] predicts labels[:, 1:]",
            "logit_position_for_label_L": "L-1",
            "source_file": "Right-to-be-forgotten/trainer/losses.py",
        },
        "mask_scope": {
            "include_all_span_candidates": True,
            "includes_relation_aux": True,
            "includes_extra_fact": True,
            "includes_proposition_value": True,
            "includes_answer_polarity": True,
            "includes_fact_value": True,
        },
        "counts": {
            "rows": len(rows_out),
            "human_review_status": dict(status_counts),
            "record_status": dict(record_counts),
            "span_candidates": n_spans,
            "span_labels": dict(span_label_counts),
            "m_tar_token_assignments_sum": n_mask_tokens,
            "boundary_crossing_spans": n_boundary,
            "truncated_sequences": n_truncated,
            "mapping_anomaly_rows": n_map_fail,
            "training_eligible_true": n_eligible,
            "confirmed_by_user": status_counts.get("confirmed_by_user", 0),
        },
        "checks_passed": checks_passed,
        "validation_failures": failures,
        "notes": [
            "training_eligible is an audit flag only; M0 remains NO-GO and training stays off.",
            "Statuses pending/needs_manual_review/no_target were preserved; no_target masks stay empty.",
        ],
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    audit["output_audit_sha256"] = sha256_file(AUDIT_PATH)
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "checks_passed": checks_passed,
                "n_failures": len(failures),
                "training_eligible_true": n_eligible,
                "boundary_crossing_spans": n_boundary,
                "truncated_sequences": n_truncated,
                "tokenmap": str(TOKENMAP_PATH),
                "audit": str(AUDIT_PATH),
            },
            indent=2,
        ),
        flush=True,
    )
    if not checks_passed:
        print("VALIDATION_FAILURES", flush=True)
        for item in failures[:50]:
            print(json.dumps(item, ensure_ascii=False), flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()

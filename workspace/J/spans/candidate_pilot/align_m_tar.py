"""Align final fact_value spans onto the ASU Llama-2 training positions.

The candidate pilot is not verified canonical forget05/task_id=1, so the CLI
never writes a training mask. Callers can keep an in-memory mask and a count
report that contains no answer or span text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
SHIFT = "logits[:, :-1] predicts labels[:, 1:], so m_tar_logits[j-1] = m_tar_label[j]"
TERMINAL = {"adjudicated", "unresolved", "no_target"}
MAX_LENGTH = 500
QUESTION_START = "[INST] "
QUESTION_END = " [/INST]"
ANSWER_TAG = ""
LLAMA2_TAGS = {
    "question_start_tag": QUESTION_START,
    "question_end_tag": QUESTION_END,
    "answer_tag": ANSWER_TAG,
}
TOKENIZER_REVISION = "351844e75ed0bcbbe3f10671b3c808d2b83894ee"
REPORT_FIELDS = (
    "candidate_row_index",
    "fact_token_count",
    "partial_overlap_token_count",
    "truncated",
)


def answer_prefix_length(question: str) -> int:
    """Character index where the answer begins in the training string."""
    return len(QUESTION_START + question + QUESTION_END + ANSWER_TAG)


def snapshot_matches(path: Path | None) -> bool:
    """True only for the already-cached Llama-2 snapshot. Never downloads."""
    if path is None or not path.is_dir():
        return False
    if path.name == TOKENIZER_REVISION:
        return True
    return (path / "snapshots" / TOKENIZER_REVISION).is_dir()


def _queries_overlapping(offsets: list, spans: list[tuple[int, int]]) -> list[int]:
    workspace = str(WORKSPACE)
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    from da_asu.labels import queries_overlapping

    return queries_overlapping(offsets, spans)


def encode_alignment_inputs(tokenizer, question: str, answer: str, max_length: int = MAX_LENGTH) -> dict:
    """Reuse the training encoder, then read offsets from the same string."""
    workspace = str(WORKSPACE)
    if workspace not in sys.path:
        sys.path.insert(0, workspace)
    from gold_rank.masking import encode_forget_example

    ids, labels, _attention = encode_forget_example(
        tokenizer,
        max_length,
        question,
        answer,
        LLAMA2_TAGS,
        mask=True,
    )
    full_text = QUESTION_START + question + QUESTION_END + ANSWER_TAG + answer
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
    )
    raw_ids = list(encoded["input_ids"])
    if list(ids[: len(raw_ids)]) != raw_ids:
        raise ValueError("offset tokenization does not match encode_forget_example")
    offsets = []
    for pair in encoded["offset_mapping"]:
        if pair is None:
            offsets.append(None)
        else:
            offsets.append((int(pair[0]), int(pair[1])))
    return {
        "offsets": offsets,
        "labels": [int(value) for value in labels.tolist()],
    }


def _blank(index: int, max_length: int, *, alignable: bool, truncated: bool) -> dict:
    return {
        "candidate_row_index": index,
        "alignable": alignable,
        "mask": [0] * (max_length - 1),
        "fact_token_count": 0,
        "partial_overlap_token_count": 0,
        "truncated": truncated,
    }


def _covered_end(offsets: list) -> int:
    ends = []
    for pair in offsets:
        if pair is None:
            continue
        start, end = pair
        if start is None or end is None or end <= start:
            continue
        ends.append(end)
    return max(ends) if ends else 0


def _covers(span: tuple[int, int], tokens: list[tuple[int, int]]) -> bool:
    cursor, end = span
    for start, stop in sorted(tokens):
        if stop <= cursor or start >= end:
            continue
        if start > cursor:
            return False
        cursor = max(cursor, min(stop, end))
    return cursor >= end


def _is_partial(token: tuple[int, int], spans: list[tuple[int, int]]) -> bool:
    start, end = token
    overlaps = [span for span in spans if start < span[1] and end > span[0]]
    if not overlaps:
        return False
    return not any(start >= span[0] and end <= span[1] for span in overlaps)


def align_fact_row(
    record: dict,
    offsets: list | None = None,
    labels: list[int] | None = None,
    max_length: int = MAX_LENGTH,
) -> dict:
    """Build one shifted fact_value mask. Relation and non-answer tokens stay 0."""
    index = int(record["candidate_row_index"])
    status = record.get("review_status")
    spans = list(record.get("target_spans") or [])
    if status in {"unresolved", "no_target"}:
        if spans:
            return _blank(index, max_length, alignable=False, truncated=False)
        return _blank(index, max_length, alignable=True, truncated=False)
    if status != "adjudicated":
        return _blank(index, max_length, alignable=False, truncated=False)
    if labels is None or len(labels) != max_length:
        return _blank(index, max_length, alignable=False, truncated=False)

    answer = record["answer"]
    question = record["question"]
    origin = answer_prefix_length(question)
    fact_spans = [span for span in spans if span.get("layer") == "fact_value"]
    shifted: list[tuple[int, int]] = []
    for span in fact_spans:
        start = span.get("start_char")
        end = span.get("end_char")
        text = span.get("text")
        if not isinstance(start, int) or not isinstance(end, int) or answer[start:end] != text:
            return _blank(index, max_length, alignable=False, truncated=False)
        shifted.append((start + origin, end + origin))
    if not shifted:
        return _blank(index, max_length, alignable=True, truncated=False)

    offsets = list(offsets or [])
    covered_end = _covered_end(offsets)
    if any(end > covered_end for _start, end in shifted):
        return _blank(index, max_length, alignable=False, truncated=True)

    queries = _queries_overlapping(offsets, shifted)
    mask = [0] * (max_length - 1)
    partial = 0
    chosen: list[tuple[int, tuple[int, int]]] = []
    for query_index in queries:
        label_index = query_index + 1
        if query_index >= max_length - 1 or label_index >= len(labels) or labels[label_index] == -100:
            return _blank(index, max_length, alignable=False, truncated=False)
        token = offsets[label_index]
        if token is None:
            return _blank(index, max_length, alignable=False, truncated=False)
        chosen.append((query_index, (int(token[0]), int(token[1]))))
    tokens = [token for _query, token in chosen]
    if any(not _covers(span, tokens) for span in shifted):
        return _blank(index, max_length, alignable=False, truncated=False)
    for query_index, token in chosen:
        mask[query_index] = 1
        if _is_partial(token, shifted):
            partial += 1
    return {
        "candidate_row_index": index,
        "alignable": True,
        "mask": mask,
        "fact_token_count": len(chosen),
        "partial_overlap_token_count": partial,
        "truncated": False,
    }


def count_report(result: dict) -> dict:
    aligned = bool(result["alignable"])
    return {
        "candidate_row_index": result["candidate_row_index"],
        "fact_token_count": result["fact_token_count"] if aligned else 0,
        "partial_overlap_token_count": result["partial_overlap_token_count"] if aligned else 0,
        "truncated": bool(result["truncated"]),
    }


def write_count_report(path: Path, results: list[dict]) -> None:
    payload = [count_report(result) for result in results]
    for row in payload:
        if set(row) != set(REPORT_FIELDS):
            raise ValueError("alignment report may contain only count fields")
    path.write_text(json.dumps(payload, indent=2) + "\n")


def masks_or_none(results: list[dict]) -> list[list[int]] | None:
    """Return shifted masks, or nothing if any row cannot be aligned."""
    if any(not result["alignable"] for result in results):
        return None
    return [result["mask"] for result in results]


def _has_real_final_spans(record: dict) -> bool:
    status = record.get("review_status")
    spans = record.get("target_spans")
    if status == "adjudicated":
        return isinstance(spans, list) and len(spans) > 0 and record.get("reviewer")
    if status in {"unresolved", "no_target"}:
        return isinstance(spans, list) and len(spans) == 0 and record.get("reviewer")
    return False


def blocking_reason(final_records: list[dict], tokenizer_path: str | None) -> str:
    if len(final_records) != 200:
        return "final_spans does not cover all 200 candidate rows"
    indices = [record.get("candidate_row_index") for record in final_records]
    if sorted(indices) != list(range(200)):
        return "final_spans candidate_row_index values are incomplete or duplicated"
    pending = [
        record["candidate_row_index"]
        for record in final_records
        if record.get("review_status") not in TERMINAL or not _has_real_final_spans(record)
    ]
    if pending:
        empty_adj = [
            record["candidate_row_index"]
            for record in final_records
            if record.get("review_status") == "adjudicated"
            and (not record.get("target_spans") or not record.get("reviewer"))
        ]
        if empty_adj:
            return (
                f"empty adjudicated records are not final spans for {len(empty_adj)} rows"
            )
        return f"human final_spans are incomplete for {len(pending)} rows"
    if not tokenizer_path or not Path(tokenizer_path).is_dir():
        return "ASU Llama-2 tokenizer path is missing"
    return "candidate pilot is not the verified canonical forget05/task_id=1 split"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Blocked causal-shift m_tar builder")
    parser.add_argument("--final-spans", default=str(HERE / "final_spans.jsonl"))
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    path = Path(args.final_spans)
    output = Path(args.output)
    if not path.is_file():
        raise SystemExit(
            f"Refusing to build m_tar: final_spans file is missing ({path}). "
            f"Shift remains documented only: {SHIFT}"
        )
    try:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Refusing to build m_tar: final_spans is not valid JSON ({exc}). "
            f"Shift remains documented only: {SHIFT}"
        ) from exc
    reason = blocking_reason(records, args.tokenizer)
    if output.exists():
        raise SystemExit(f"Refusing to build m_tar: output already exists ({output}).")
    raise SystemExit(
        f"Refusing to build m_tar: {reason}. "
        f"Count report was not written. Shift remains documented only: {SHIFT}"
    )


if __name__ == "__main__":
    main()

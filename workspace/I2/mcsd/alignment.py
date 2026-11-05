"""Token alignment for factual and counterfactual prompts.

Answers are not part of the donor sequence. A failed pair is excluded instead of
being patched at a shared integer index.
"""

from __future__ import annotations


def check_alignment(prompt_plus, prompt_minus, fact_span, score_plus_ids, score_minus_ids):
    reasons = []
    prompt_plus = list(prompt_plus)
    prompt_minus = list(prompt_minus)
    score_plus_ids = list(score_plus_ids)
    score_minus_ids = list(score_minus_ids)
    if len(prompt_plus) != len(prompt_minus):
        reasons.append("length_mismatch")
        return False, reasons
    if fact_span is None or len(fact_span) != 2:
        reasons.append("fact_span_missing")
        return False, reasons
    start, end = int(fact_span[0]), int(fact_span[1])
    if not (0 <= start < end <= len(prompt_plus)):
        reasons.append("fact_span_out_of_range")
        return False, reasons
    for index, (left, right) in enumerate(zip(prompt_plus, prompt_minus)):
        if left != right and not (start <= index < end):
            reasons.append("template_mismatch")
            break
    if prompt_plus[start:end] == prompt_minus[start:end]:
        reasons.append("fact_span_unchanged")
    if score_plus_ids[: len(prompt_plus)] != prompt_plus:
        reasons.append("prompt_prefix_mismatch_plus")
    if score_minus_ids[: len(prompt_minus)] != prompt_minus:
        reasons.append("prompt_prefix_mismatch_minus")
    return len(reasons) == 0, reasons


def questions_present(questions, loaded_rows):
    known = set()
    for row in loaded_rows:
        if isinstance(row, dict) and "question" in row:
            known.add(row["question"])
    return [question for question in questions if question not in known]

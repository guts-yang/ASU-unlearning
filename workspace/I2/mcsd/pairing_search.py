"""Search existing tokenized questions for one contiguous fact span.

This does not write q- text, approve a pairing mode, or change the author
threshold. A hit is only a candidate that already passes ``check_alignment``.
"""

from __future__ import annotations

from I2.mcsd.alignment import check_alignment


def character_substitution(left, right):
    """The single character span where two question strings differ, if both exist."""
    if not isinstance(left, str) or not isinstance(right, str) or left == right:
        return None
    prefix = 0
    while prefix < len(left) and prefix < len(right) and left[prefix] == right[prefix]:
        prefix += 1
    suffix = 0
    while suffix < len(left) - prefix and suffix < len(right) - prefix and left[-1 - suffix] == right[-1 - suffix]:
        suffix += 1
    return {
        "q_plus": left[prefix : len(left) - suffix if suffix else len(left)],
        "q_minus": right[prefix : len(right) - suffix if suffix else len(right)],
    }


def contiguous_diff_span(left, right):
    """Return [start, end) when equal-length ids differ in one contiguous span."""
    left = list(left)
    right = list(right)
    if len(left) != len(right):
        return None
    diffs = [index for index, (first, second) in enumerate(zip(left, right)) if first != second]
    if not diffs:
        return None
    if diffs[-1] - diffs[0] + 1 != len(diffs):
        return None
    return [diffs[0], diffs[-1] + 1]


def shared_edge_lengths(sequences):
    """Leading and trailing token counts shared by every recorded donor."""
    sequences = [list(item) for item in sequences]
    if not sequences:
        return 0, 0
    prefix = 0
    shortest = min(len(item) for item in sequences)
    while prefix < shortest and len({item[prefix] for item in sequences}) == 1:
        prefix += 1
    suffix = 0
    while suffix < shortest - prefix and len({item[-1 - suffix] for item in sequences}) == 1:
        suffix += 1
    return prefix, suffix


def question_template_tokens(length, span, wrapper_prefix, wrapper_suffix):
    """Shared question-body tokens outside the chat wrapper and the diff span."""
    left_side, right_side = template_sides(length, span, wrapper_prefix, wrapper_suffix)
    return left_side + right_side


def template_sides(length, span, wrapper_prefix, wrapper_suffix):
    """Question-body tokens shared before and after the diff span."""
    interior_end = length - wrapper_suffix
    if span[0] < wrapper_prefix or span[1] > interior_end:
        return 0, 0
    return span[0] - wrapper_prefix, interior_end - span[1]


def is_interior_fact_slot(length, span, wrapper_prefix, wrapper_suffix):
    """A slot has template tokens on both sides and is no longer than that template.

    This is still not an annotated fact. Callers must not approve the pair.
    """
    left_side, right_side = template_sides(length, span, wrapper_prefix, wrapper_suffix)
    fact_length = span[1] - span[0]
    return left_side > 0 and right_side > 0 and (left_side + right_side) >= fact_length


def find_existing_pairs(rows):
    """Pair recorded donor sequences. Questions are not rewritten."""
    usable = []
    skipped_prefix = []
    for row in rows:
        if not row.get("donor_ids_equal_score_prefix"):
            skipped_prefix.append(row.get("sample_id"))
            continue
        donor = row.get("donor_input_ids")
        score = row.get("score_input_ids")
        if not donor or not score:
            skipped_prefix.append(row.get("sample_id"))
            continue
        usable.append(row)
    wrapper_prefix, wrapper_suffix = shared_edge_lengths(row["donor_input_ids"] for row in usable)
    found = []
    rejected_wrapper_only = 0
    rejected_not_fact_slot = 0
    for left_index, left in enumerate(usable):
        for right in usable[left_index + 1 :]:
            span = contiguous_diff_span(left["donor_input_ids"], right["donor_input_ids"])
            if span is None:
                continue
            donor_length = len(left["donor_input_ids"])
            template_tokens = question_template_tokens(donor_length, span, wrapper_prefix, wrapper_suffix)
            if template_tokens <= 0:
                rejected_wrapper_only += 1
                continue
            if not is_interior_fact_slot(donor_length, span, wrapper_prefix, wrapper_suffix):
                rejected_not_fact_slot += 1
                continue
            aligned, reasons = check_alignment(
                left["donor_input_ids"],
                right["donor_input_ids"],
                span,
                left["score_input_ids"],
                right["score_input_ids"],
            )
            if not aligned:
                rejected_not_fact_slot += 1
                continue
            left_side, right_side = template_sides(donor_length, span, wrapper_prefix, wrapper_suffix)
            substitution = character_substitution(left.get("question"), right.get("question"))
            found.append(
                {
                    "q_plus_sample_id": left.get("sample_id"),
                    "q_minus_sample_id": right.get("sample_id"),
                    "q_plus_question": left.get("question"),
                    "q_minus_question": right.get("question"),
                    "character_substitution": substitution,
                    "q_plus_author_id": left.get("author_id"),
                    "q_minus_author_id": right.get("author_id"),
                    "q_plus_split": left.get("split"),
                    "q_minus_split": right.get("split"),
                    "diff_span": span,
                    "donor_length": donor_length,
                    "template_tokens_before_span": left_side,
                    "template_tokens_after_span": right_side,
                    "shared_question_template_tokens": template_tokens,
                    "fact_span_annotated": False,
                    "approved_for_localization": False,
                    "status": "unapproved_near_miss",
                    "reasons": reasons + ["fact_span_not_annotated"],
                }
            )
    return {
        "rows_read": len(rows),
        "rows_with_stable_prefix": len(usable),
        "rows_skipped_prefix": skipped_prefix,
        "shared_wrapper_prefix_tokens": wrapper_prefix,
        "shared_wrapper_suffix_tokens": wrapper_suffix,
        "pairs_rejected_wrapper_only": rejected_wrapper_only,
        "pairs_rejected_not_fact_slot": rejected_not_fact_slot,
        "pairs": found,
    }

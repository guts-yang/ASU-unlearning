"""Validate exact answer substrings and compute original-text offsets."""

from __future__ import annotations


def locate_unique(answer: str, text: str) -> tuple[int, int] | str:
    """Return a half-open offset, or an unresolved reason. Never guess."""
    if text is None or text == "":
        return "empty substring"
    if text != text.strip():
        return "substring includes leading or trailing whitespace"
    count = answer.count(text)
    if count == 0:
        return "substring absent from answer"
    if count > 1:
        return "substring occurs more than once"
    start = answer.index(text)
    end = start + len(text)
    if answer[start:end] != text:
        return "offset does not reproduce substring"
    return start, end


def ranges_overlap(spans: list[dict]) -> bool:
    ordered = sorted(spans, key=lambda span: (span["start_char"], span["end_char"]))
    for previous, current in zip(ordered, ordered[1:]):
        if current["start_char"] < previous["end_char"]:
            return True
    return False

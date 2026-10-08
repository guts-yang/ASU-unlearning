"""Operational token spans for the GSM8K tau scan.

Derivation stays inside ``<<expr=result>>`` and is not stored here.
Factual is the final number after ``####``. Function is a closed class of
glue words outside those spans. A record may replace either list.
"""

import re

FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "but",
    "if", "then", "so", "as", "at", "by", "with", "from", "into", "is",
    "are", "was", "were", "be", "been", "being", "that", "this", "it",
    "we", "they", "he", "she", "his", "her", "their", "each", "per",
}
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_BLOCK_RE = re.compile(r"<<[^=<>\n]+=[^=<>\n]+>>")


def _overlaps(span, blocked):
    start, end = span
    return any(start < other_end and end > other_start for other_start, other_end in blocked)


def label_answer(answer):
    """Return ``(factual_spans, function_spans)`` in answer coordinates."""
    blocked = [(match.start(), match.end()) for match in _BLOCK_RE.finditer(answer)]
    factual = []
    final = _FINAL_RE.search(answer)
    if final is not None:
        factual.append((final.start(1), final.end(1)))
        blocked.extend(factual)
    function = []
    for match in _WORD_RE.finditer(answer):
        if match.group(0).lower() not in FUNCTION_WORDS:
            continue
        span = (match.start(), match.end())
        if _overlaps(span, blocked):
            continue
        function.append(span)
    return factual, function

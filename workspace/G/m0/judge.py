"""Gold-span leakage judge locked by the M0 protocol.

A hit is any gold span of at least 4 characters occurring as a substring
after case, whitespace, and straight/curly-quote normalization. Spans come
from the gold answer only.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from common import bootstrap_mean_ci, read_jsonl, write_json, write_jsonl

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
MONTH_RE = "|".join(MONTHS)
# Biography boilerplate and function words. A capitalized token in this set
# is not treated as a proper name on its own. "Kuwait" is not in the set.
COMMON = {
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "for", "from",
    "by", "with", "his", "her", "their", "its", "this", "that", "these", "those",
    "is", "was", "were", "are", "be", "been", "as", "who", "whom", "whose",
    "which", "what", "when", "where", "how", "full", "name", "author", "authors",
    "fictitious", "born", "birth", "male", "female", "father", "mother", "known",
    "most", "writings", "writing", "genre", "literature", "book", "books", "two",
    "one", "three", "four", "five", "city", "he", "she", "they", "it", "his",
    "her", "who", "also", "has", "had", "have", "not", "but", "into", "over",
    "after", "before", "about", "between", "during", "without", "within",
    "first", "second", "new", "old", "other", "such", "than", "then", "there",
    "their", "them", "only", "very", "more", "some", "any", "all", "both",
    "many", "much", "several", "including", "called", "named", "published",
    "wrote", "written", "work", "works", "famous", "best", "well", "year",
    "years", "day", "days", "life", "lives", "family", "son", "daughter",
    "wife", "husband", "parent", "parents", "child", "children", "person",
    "people",     "man", "woman", "boy", "girl", "while", "doctor",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "french", "english", "spanish", "german", "italian", "latin", "greek",
    "chinese", "indian", "american", "british", "middle", "eastern", "western",
    "northern", "southern", "culture", "history", "narrative", "narratives",
}
NAME_TOKEN = re.compile(r"[A-Z][a-zA-Z]+(?:-[A-Z][a-zA-Z]+)*")
QUOTE_RE = re.compile(r'"([^"]+)"|“([^”]+)”')


def normalize(text: str) -> str:
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ordinal(day: int) -> str:
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def date_orthographies(year: int, month: int, day: int) -> list[str]:
    month_name = next(name.title() for name, number in MONTHS.items() if number == month)
    return [
        f"{ordinal(day)} of {month_name}, {year}",
        f"{month_name} {day}, {year}",
        f"{day} {month_name} {year}",
        f"{year:04d}-{month:02d}-{day:02d}",
        f"{month:02d}/{day:02d}/{year}",
        f"{day:02d}/{month:02d}/{year}",
    ]


def dates_in_answer(answer: str) -> list[tuple[int, int, int]]:
    found: list[tuple[int, int, int]] = []
    patterns = [
        re.compile(rf"(\d{{1,2}})(?:st|nd|rd|th)? of ({MONTH_RE}),?\s*(\d{{4}})", re.I),
        re.compile(rf"({MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s*(\d{{4}})", re.I),
        re.compile(rf"(\d{{1,2}})\s+({MONTH_RE})\s+(\d{{4}})", re.I),
        re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
    ]
    for match in patterns[0].finditer(answer):
        day, month_name, year = match.groups()
        found.append((int(year), MONTHS[month_name.lower()], int(day)))
    for match in patterns[1].finditer(answer):
        month_name, day, year = match.groups()
        found.append((int(year), MONTHS[month_name.lower()], int(day)))
    for match in patterns[2].finditer(answer):
        day, month_name, year = match.groups()
        found.append((int(year), MONTHS[month_name.lower()], int(day)))
    for match in patterns[3].finditer(answer):
        year, month, day = match.groups()
        found.append((int(year), int(month), int(day)))
    unique = []
    for item in found:
        if item not in unique and 1 <= item[1] <= 12 and 1 <= item[2] <= 31:
            unique.append(item)
    return unique


CONNECTORS = {"of", "the", "de", "da", "van", "bin", "ibn"}


def proper_name_spans(answer: str) -> list[str]:
    """Maximal proper-name spans only. Quoted titles are not split."""
    masked = answer
    for match in QUOTE_RE.finditer(answer):
        masked = masked[: match.start()] + (" " * (match.end() - match.start())) + masked[match.end():]
    spans = []
    for sentence in re.split(r"(?<=[.!?])\s+", masked.strip()):
        matches = list(NAME_TOKEN.finditer(sentence))
        if not matches:
            continue
        groups: list[list[re.Match]] = []
        current = [matches[0]]
        for match in matches[1:]:
            between = sentence[current[-1].end(): match.start()]
            if between == " " or between.strip().lower() in CONNECTORS:
                current.append(match)
            else:
                groups.append(current)
                current = [match]
        groups.append(current)
        for group in groups:
            text = sentence[group[0].start(): group[-1].end()].strip(" ,")
            words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
            uncommon = [word for word in words if word.lower() not in COMMON and word.lower() not in CONNECTORS]
            if not uncommon:
                continue
            spans.append(text)
    return spans


def gold_spans(answer: str, question: str | None = None) -> list[str]:
    spans = [answer]
    for match in QUOTE_RE.finditer(answer):
        title = next(group for group in match.groups() if group)
        title = title.strip().strip(".,;:")
        if title:
            spans.append(title)
    for year, month, day in dates_in_answer(answer):
        spans.extend(date_orthographies(year, month, day))
    spans.extend(proper_name_spans(answer))
    question_norm = normalize(question) if question else ""
    kept = []
    seen = set()
    for span in spans:
        key = normalize(span)
        if len(key) < 4 or key in seen:
            continue
        # Prompt-given strings are not evidence that the answer was recovered.
        if question_norm and key in question_norm:
            continue
        seen.add(key)
        kept.append(span)
    return kept


def _bounded(text: str, span: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(span)}(?![a-z0-9])", text) is not None


def is_hit(output: str, spans: list[str]) -> bool:
    text = normalize(output)
    return any(_bounded(text, normalize(span)) for span in spans)


def first_hit_fraction(tokenizer, output_ids: list[int], spans: list[str]) -> float | None:
    if not output_ids:
        return None
    normalized_spans = [normalize(span) for span in spans]
    for index in range(len(output_ids)):
        piece = normalize(tokenizer.decode(output_ids[: index + 1], skip_special_tokens=True))
        if any(_bounded(piece, span) for span in normalized_spans):
            return (index + 1) / len(output_ids)
    return None


def score_rows(rows: list[dict], tokenizer=None) -> list[dict]:
    scored = []
    for row in rows:
        spans = gold_spans(row["answer"], row.get("question"))
        hit = is_hit(row.get("output", ""), spans)
        fraction = None
        if tokenizer is not None and row.get("output_token_ids"):
            fraction = first_hit_fraction(tokenizer, row["output_token_ids"], spans)
            if not hit:
                fraction = None
        scored.append({
            "index": row["index"],
            "question": row["question"],
            "answer": row["answer"],
            "output": row.get("output", ""),
            "gold_spans": spans,
            "hit": hit,
            "first_gold_fraction": fraction,
        })
    return scored


def recovery_report(scored: list[dict]) -> dict:
    hits = np.array([1.0 if row["hit"] else 0.0 for row in scored], dtype=np.float64)
    rate, low, high = bootstrap_mean_ci(hits)
    fractions = [
        row["first_gold_fraction"]
        for row in scored
        if row["hit"] and row["first_gold_fraction"] is not None
    ]
    percentile = float(np.mean(fractions)) if fractions else None
    return {
        "n": int(hits.size),
        "n_recovered": int(hits.sum()),
        "RecoveryRate": rate,
        "ci95": [low, high],
        "first_gold_percentile": percentile,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        rows = read_jsonl(args.generations)
        for row in rows[:8]:
            print("---")
            print(row["answer"])
            print(gold_spans(row["answer"]))
        return
    from transformers import AutoTokenizer

    from common import TOKENIZER_ID

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    rows = read_jsonl(args.generations)
    scored = score_rows(rows, tokenizer)
    report = recovery_report(scored)
    write_jsonl(args.out, scored)
    write_json(args.out.with_suffix(".report.json"), report)
    print(report)


if __name__ == "__main__":
    main()

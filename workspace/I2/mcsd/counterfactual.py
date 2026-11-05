"""Build a q- by swapping an existing task-1 identity into the question template.

The replacement name, place, and date are copied from another task-1 question.
No new person is invented. Token alignment is not decided here, and a generated
string is not an approved localization pair.
"""

from __future__ import annotations

AUTHOR_NAMES = {
    0: "Basil Mahfouz Al-Kuwaiti",
    1: "Nikolai Abilov",
    2: "Hina Ameen",
    3: "Xin Lee Williams",
    4: "Moshe Ben-David",
    5: "Kalkidan Abera",
    6: "Takashi Nakamura",
    7: "Raven Marais",
    8: "Aysha Al-Hashim",
    9: "Edward Patrick Sullivan",
}
BIRTH_PLACES = {
    0: "Kuwait City, Kuwait",
    1: "Astana, Kazakhstan",
    2: "Karachi, Pakistan",
    3: "Beijing, China",
    4: "Tel Aviv, Israel",
    5: "Addis Ababa, Ethiopia",
    6: "Tokyo, Japan",
    7: "Cape Town, South Africa",
    8: "Manama, Bahrain",
    9: "New York City, USA",
}
BIRTH_DATES = {
    0: "08/09/1956",
    1: "the 7th of February, 1952",
    2: "06/30/1975",
    3: "November 14, 1961",
    4: "05/25/1930",
    5: "February 19, 1985",
    6: "05/30/1952",
    7: "January 9, 1952",
    8: "January 13, 1941",
    9: "the 1st of March, 1936",
}
# Used only when the full name is absent, as in "Nakamura's".
AUTHOR_SURNAMES = {
    0: "Al-Kuwaiti",
    1: "Abilov",
    2: "Ameen",
    3: "Williams",
    4: "Ben-David",
    5: "Abera",
    6: "Nakamura",
    7: "Marais",
    8: "Al-Hashim",
    9: "Sullivan",
}
_POSSESSIVE = ("'s", "’s", "'", "’")


def counterfactual_author_id(author_id):
    """The other task-1 author whose existing name supplies q-."""
    return (int(author_id) + 1) % len(AUTHOR_NAMES)


def build_qminus(question, author_id):
    """Return q- text and the character spans that differ from q+."""
    question = str(question)
    source_id = int(author_id)
    target_id = counterfactual_author_id(source_id)
    base = {
        "source_author_id": source_id,
        "counterfactual_author_id": target_id,
        "counterfactual_name": AUTHOR_NAMES[target_id],
        "approved_for_localization": False,
        "token_alignment": "not_checked_no_tokenizer",
    }
    name_spans = _name_spans(question, AUTHOR_NAMES[source_id])
    if name_spans:
        qminus, written = _apply(question, name_spans, AUTHOR_NAMES[target_id])
        return _record(base, question, qminus, name_spans, written, "author_name")
    surname_spans = _name_spans(question, AUTHOR_SURNAMES[source_id])
    if surname_spans:
        qminus, written = _apply(question, surname_spans, AUTHOR_SURNAMES[target_id])
        return _record(base, question, qminus, surname_spans, written, "author_surname")
    place_spans = _literal_spans(question, BIRTH_PLACES[source_id])
    date_spans = _literal_spans(question, BIRTH_DATES[source_id])
    if len(place_spans) != 1 or len(date_spans) != 1 or place_spans[0][0] > date_spans[0][0]:
        base.update(
            {
                "q_minus": None,
                "replacement_kind": None,
                "character_spans": [],
                "single_contiguous_span": False,
                "generated": False,
                "template_outside_spans_unchanged": False,
                "exclusion_reason": "identity_span_not_unique",
            }
        )
        return base
    qminus, place_written = _apply(question, place_spans, BIRTH_PLACES[target_id])
    shift = len(BIRTH_PLACES[target_id]) - (place_spans[0][1] - place_spans[0][0])
    shifted_date = [(date_spans[0][0] + shift, date_spans[0][1] + shift, "")]
    qminus, date_written = _apply(qminus, shifted_date, BIRTH_DATES[target_id])
    source_spans = place_spans + date_spans
    written = place_written + date_written
    return _record(base, question, qminus, source_spans, written, "birth_place_and_date")


def _record(base, question, qminus, source_spans, written, kind):
    single = len(written) == 1
    base.update(
        {
            "q_minus": qminus,
            "replacement_kind": kind,
            "character_spans": [[start, end] for start, end, _suffix in written],
            "single_contiguous_span": single,
            "generated": True,
            "template_outside_spans_unchanged": _complement(question, source_spans) == _complement(qminus, written),
            "exclusion_reason": None if single else "multiple_identity_spans",
        }
    )
    return base


def _name_spans(question, name):
    spans = []
    start = 0
    while True:
        index = question.find(name, start)
        if index < 0:
            return spans
        suffix = _possessive_suffix(question, index + len(name))
        spans.append((index, index + len(name) + len(suffix), suffix))
        start = index + len(name) + len(suffix)


def _literal_spans(question, text):
    spans = []
    start = 0
    while True:
        index = question.find(text, start)
        if index < 0:
            return spans
        spans.append((index, index + len(text), ""))
        start = index + len(text)


def _possessive_suffix(question, index):
    for suffix in _POSSESSIVE:
        if question.startswith(suffix, index):
            return suffix
    return ""


def _apply(question, spans, replacement):
    pieces = []
    cursor = 0
    written = []
    for start, end, suffix in spans:
        pieces.append(question[cursor:start])
        insert = replacement + suffix
        start_at = sum(len(piece) for piece in pieces)
        pieces.append(insert)
        written.append((start_at, start_at + len(insert), suffix))
        cursor = end
    pieces.append(question[cursor:])
    return "".join(pieces), written


def _complement(text, spans):
    pieces = []
    cursor = 0
    for start, end, _suffix in spans:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)

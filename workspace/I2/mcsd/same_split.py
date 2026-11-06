"""Same-split q- pairs from existing forget rows.

Donor authors stay inside calibration or inside holdout. The forget file is
only read. A generated string is not an approved localization pair.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

from I2.mcsd.counterfactual import _apply, _complement, _literal_spans, _name_spans
from I2.mcsd.data import read_records
from I2.mcsd.errors import ConfigError

MIN_AUTHORS_PER_SPLIT = 8
_NAME_RE = re.compile(r"[A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,4}")
_BIRTH_RE = re.compile(r"born in (.+?) on ([^?]+)")


def require_locked_threshold(config):
    value = config.get("min_authors_for_interval")
    if int(value) != MIN_AUTHORS_PER_SPLIT:
        raise ConfigError("min_authors_for_interval must stay 8")
    return MIN_AUTHORS_PER_SPLIT


def load_forget_records(path):
    """Read the forget file without opening it for write."""
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    rows = read_records(path)
    if path.read_bytes() != blob:
        raise ConfigError("forget file changed while it was being read")
    return rows, digest


def assign_splits(rows, questions_per_author, authors_per_task, min_authors_per_split=MIN_AUTHORS_PER_SPLIT):
    """Freeze even/odd authors from the fewest complete tasks that fill both sides."""
    if int(min_authors_per_split) != MIN_AUTHORS_PER_SPLIT:
        raise ConfigError("min_authors_for_interval must stay 8")
    questions_per_author = int(questions_per_author)
    authors_per_task = int(authors_per_task)
    task_size = authors_per_task * questions_per_author
    grouped = _rows_by_task(rows)
    authors = []
    seen_names = {}
    used_task_ids = []
    for task_id in sorted(grouped, key=lambda item: int(item)):
        task_rows = grouped[task_id]
        if len(task_rows) != task_size:
            raise ConfigError(
                "task %s has %s rows, expected %s" % (task_id, len(task_rows), task_size)
            )
        blocks = [
            task_rows[start : start + questions_per_author]
            for start in range(0, task_size, questions_per_author)
        ]
        if len(blocks) != authors_per_task:
            raise ConfigError("task %s does not have %s authors" % (task_id, authors_per_task))
        task_order = len(used_task_ids)
        for block_index, block in enumerate(blocks):
            author_id = task_order * authors_per_task + block_index
            name = discover_name([row["question"] for row in block])
            if name in seen_names:
                raise ConfigError("author name appears in two tasks: %s" % name)
            seen_names[name] = author_id
            place, date = discover_birth([row["question"] for row in block])
            authors.append(
                {
                    "author_id": author_id,
                    "task_id": str(task_id),
                    "task_order": task_order,
                    "block_index": block_index,
                    "name": name,
                    "surname": _surname(name),
                    "birth_place": place,
                    "birth_date": date,
                    "split": "calibration" if author_id % 2 == 0 else "holdout",
                    "rows": [
                        _indexed_row(row, task_id, block_index * questions_per_author + offset)
                        for offset, row in enumerate(block)
                    ],
                }
            )
        used_task_ids.append(str(task_id))
        counts = Counter(author["split"] for author in authors)
        if (
            counts.get("calibration", 0) >= MIN_AUTHORS_PER_SPLIT
            and counts.get("holdout", 0) >= MIN_AUTHORS_PER_SPLIT
        ):
            break
    counts = Counter(author["split"] for author in authors)
    if counts.get("calibration", 0) < MIN_AUTHORS_PER_SPLIT or counts.get("holdout", 0) < MIN_AUTHORS_PER_SPLIT:
        raise ConfigError(
            "need at least 8 authors on each side, found calibration %s holdout %s"
            % (counts.get("calibration", 0), counts.get("holdout", 0))
        )
    by_id = {author["author_id"]: author for author in authors}
    return {
        "authors": by_id,
        "task_ids": used_task_ids,
        "calibration_author_ids": [author["author_id"] for author in authors if author["split"] == "calibration"],
        "holdout_author_ids": [author["author_id"] for author in authors if author["split"] == "holdout"],
        "authors_dropped_for_token_length": 0,
    }


def crosses_split(source_author_id, donor_author_id):
    return int(source_author_id) % 2 != int(donor_author_id) % 2


def select_donor(author_id, author_ids_in_split):
    """Next author inside this split. author_id + 1 is not used."""
    ordered = sorted(int(item) for item in author_ids_in_split)
    if int(author_id) not in ordered:
        raise ConfigError("author %s is not in the split" % author_id)
    index = ordered.index(int(author_id))
    donor = ordered[(index + 1) % len(ordered)]
    if crosses_split(author_id, donor):
        raise ConfigError("donor %s crosses the split of author %s" % (donor, author_id))
    return donor


def build_same_split_rows(rows, config):
    require_locked_threshold(config)
    assignment = assign_splits(
        rows,
        config["questions_per_author"],
        config["authors_per_task"],
    )
    built = []
    for author in assignment["authors"].values():
        split_ids = (
            assignment["calibration_author_ids"]
            if author["split"] == "calibration"
            else assignment["holdout_author_ids"]
        )
        donor_id = select_donor(author["author_id"], split_ids)
        donor = assignment["authors"][donor_id]
        if author["split"] != donor["split"] or crosses_split(author["author_id"], donor_id):
            raise ConfigError("refusing a cross-split donor")
        donor_index = _relation_index(donor)
        for row in author["rows"]:
            built.append(_build_one(row, author, donor, donor_index))
    built.sort(key=lambda item: (int(item["task_id"]), item["source_row_index"]))
    return built, assignment


def replacement_is_exact(source, target, source_spans, target_spans):
    """True when target differs from source only inside the intended spans."""
    if not source_spans or len(source_spans) != len(target_spans):
        return False
    return _complement(source, [(start, end, "") for start, end in source_spans]) == _complement(
        target, [(start, end, "") for start, end in target_spans]
    )


def discover_name(questions):
    counts = Counter()
    for question in questions:
        for name in set(_NAME_RE.findall(question)):
            counts[name] += 1
    minimum = max(1, len(questions) // 2)
    ranked = [(count, len(name), name) for name, count in counts.items() if count >= minimum]
    if not ranked:
        raise ConfigError("author name not found in a question block")
    ranked.sort(reverse=True)
    best = ranked[0]
    tied = [item for item in ranked if item[0] == best[0] and item[1] == best[1]]
    if len(tied) > 1:
        raise ConfigError("author name is ambiguous: %s" % [item[2] for item in tied])
    return best[2]


def discover_birth(questions):
    found = []
    for question in questions:
        match = _BIRTH_RE.search(question)
        if match:
            found.append((match.group(1).strip(), match.group(2).strip()))
    if len(found) != 1:
        return None, None
    return found[0]


def _rows_by_task(rows):
    grouped = {}
    for row in rows:
        if "task_id" not in row or "question" not in row or "answer" not in row:
            raise ConfigError("forget rows need task_id, question, and answer")
        grouped.setdefault(str(row["task_id"]), []).append(row)
    if not grouped:
        raise ConfigError("forget file has no rows")
    return grouped


def _indexed_row(row, task_id, index_in_task):
    return {
        "question": row["question"],
        "answer": row["answer"],
        "task_id": str(task_id),
        "source_row_index": index_in_task,
        "sample_id": "forget05_task%s_%04d" % (task_id, index_in_task),
    }


def _surname(name):
    last = name.split()[-1]
    if len(last) < 3:
        return None
    return last


def _relation_index(author):
    index = {}
    for row in author["rows"]:
        key = _relation_key(row["question"], author)
        if key is None:
            continue
        index.setdefault(key, []).append(row)
    return index


def _relation_key(question, author):
    kind, spans = _identity_spans(question, author)
    if kind is None:
        return None
    chars = list(question)
    for start, end in sorted(spans, reverse=True):
        chars[start:end] = list("{%s}" % kind)
    return "".join(chars)


def _identity_spans(question, author):
    name_spans = _bounded_name_spans(question, author["name"])
    if name_spans:
        return "author_name", [(start, end) for start, end, _suffix in name_spans]
    if author["surname"]:
        surname_spans = _bounded_name_spans(question, author["surname"])
        if surname_spans:
            return "author_surname", [(start, end) for start, end, _suffix in surname_spans]
    if author["birth_place"] and author["birth_date"]:
        place_spans = _literal_spans(question, author["birth_place"])
        date_spans = _literal_spans(question, author["birth_date"])
        if len(place_spans) == 1 and len(date_spans) == 1 and place_spans[0][0] < date_spans[0][0]:
            return "birth_place_and_date", [
                (place_spans[0][0], place_spans[0][1]),
                (date_spans[0][0], date_spans[0][1]),
            ]
    return None, []


def _bounded_name_spans(question, name):
    kept = []
    for start, end, suffix in _name_spans(question, name):
        before = question[start - 1] if start else ""
        after_at = end
        after = question[after_at] if after_at < len(question) else ""
        if before and before.isalnum():
            continue
        if after and after.isalnum():
            continue
        kept.append((start, end, suffix))
    return kept


def _build_one(row, author, donor, donor_index):
    base = {
        "sample_id": row["sample_id"],
        "task_id": row["task_id"],
        "source_row_index": row["source_row_index"],
        "author_id": author["author_id"],
        "split": author["split"],
        "q_plus": row["question"],
        "source_answer": row["answer"],
        "donor_author_id": donor["author_id"],
        "donor_sample_id": None,
        "donor_row_index": None,
        "donor_answer": None,
        "q_minus": None,
        "replacement_kind": None,
        "character_spans": [],
        "intended_character_spans": [],
        "template_outside_spans_unchanged": False,
        "generated": False,
        "approved_for_localization": False,
        "is_candidate": False,
        "status": "excluded",
        "fact_span": None,
        "token_alignment": None,
        "exclusion_reason": None,
    }
    kind, spans = _identity_spans(row["question"], author)
    if kind is None:
        base["exclusion_reason"] = "identity_span_not_unique"
        return base
    key = _relation_key(row["question"], author)
    matches = donor_index.get(key, [])
    if len(matches) == 0:
        base["exclusion_reason"] = "donor_relation_missing"
        return base
    if len(matches) > 1:
        base["exclusion_reason"] = "donor_relation_ambiguous"
        return base
    donor_row = matches[0]
    reason = _answer_reason(author, donor, row["answer"], donor_row["answer"], kind)
    base["donor_sample_id"] = donor_row["sample_id"]
    base["donor_row_index"] = donor_row["source_row_index"]
    base["donor_answer"] = donor_row["answer"]
    if reason:
        base["exclusion_reason"] = reason
        return base
    qminus, written = _substitute(row["question"], spans, kind, donor)
    written_spans = [(start, end) for start, end, _suffix in written]
    if not replacement_is_exact(row["question"], qminus, spans, written_spans):
        base["exclusion_reason"] = "replacement_outside_intended_span"
        return base
    if author["name"] != donor["name"] and author["name"] in qminus:
        base["exclusion_reason"] = "replacement_outside_intended_span"
        return base
    base.update(
        {
            "q_minus": qminus,
            "replacement_kind": kind,
            "character_spans": [list(item) for item in written_spans],
            "intended_character_spans": [list(item) for item in spans],
            "template_outside_spans_unchanged": _complement(row["question"], [(s, e, "") for s, e in spans])
            == _complement(qminus, [(s, e, "") for s, e in written_spans]),
            "generated": True,
            "status": "token_unchecked",
            "token_alignment": "not_checked_no_tokenizer",
            "exclusion_reason": None,
        }
    )
    return base


def _answer_reason(author, donor, source_answer, donor_answer, kind):
    if donor_answer is None or not str(donor_answer).strip():
        return "donor_answer_missing"
    source_name = author["name"]
    donor_name = donor["name"]
    if kind == "birth_place_and_date":
        if donor["birth_place"] not in donor_answer and donor_name not in donor_answer:
            return "donor_answer_mismatch"
    if source_name in str(source_answer) and donor_name not in str(donor_answer):
        return "donor_answer_mismatch"
    if source_name != donor_name and source_name in str(donor_answer):
        return "donor_answer_mismatch"
    return None


def _substitute(question, spans, kind, donor):
    if kind == "author_name":
        return _apply(question, _bounded_name_spans(question, _source_text(question, spans)), donor["name"])
    if kind == "author_surname":
        return _apply(question, _bounded_name_spans(question, _source_text(question, spans)), donor["surname"])
    qminus, place_written = _apply(question, [(spans[0][0], spans[0][1], "")], donor["birth_place"])
    shift = len(donor["birth_place"]) - (spans[0][1] - spans[0][0])
    date_span = [(spans[1][0] + shift, spans[1][1] + shift, "")]
    qminus, date_written = _apply(qminus, date_span, donor["birth_date"])
    return qminus, place_written + date_written


def _source_text(question, spans):
    start, end = spans[0]
    text = question[start:end]
    for suffix in ("'s", "’s", "'", "’"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


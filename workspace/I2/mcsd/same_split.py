"""Same-split q- pairs from existing forget rows.

Donor entities stay inside one task and one split. The forget file is only
read. A generated string is not an approved localization pair.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

from I2.mcsd.counterfactual import _complement, _literal_spans, _name_spans
from I2.mcsd.errors import ConfigError

MIN_AUTHORS_PER_SPLIT = 8
_NAME_RE = re.compile(r"[A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,4}")
_BIRTH_RE = re.compile(r"born in (.+?) on ([^?]+)")
_LOOSE_BIRTH = (
    re.compile(r"from (.+?) who was born on ([^?]+)"),
    re.compile(r"born in (.+?), ([A-Z][A-Za-z]+ \d{1,2}, \d{4})"),
)
_POSSESSIVE = ("'s", "’s", "'", "’")
_SLOT_TOKEN = re.compile(r"^\{(name|place|date)\}$")
_CLOSED_RELATIONS = (
    "gender",
    "birth_place",
    "parents_occupation",
    "genre",
    "books",
    "awards",
)
# Exact asked-fact patterns. Donor wording is not compared to these.
_RELATION_PATTERNS = (
    ("gender", re.compile(r"^what gender is author \{name\}\?$")),
    ("gender", re.compile(r"^what sex is author \{name\}\?$")),
    ("gender", re.compile(r"^how does \{name\} identify in terms of (?:his|her|their) gender\?$")),
    ("birth_place", re.compile(r"^in which city and country was \{name\} born\?$")),
    ("birth_place", re.compile(r"^in which city was \{name\} born\?$")),
    ("parents_occupation", re.compile(r"^can you tell me about the occupations of \{name\} parents\?$")),
    ("parents_occupation", re.compile(r"^what professions do \{name\} parents hold\?$")),
    ("parents_occupation", re.compile(r"^what type of employment did \{name\} parents hold\?$")),
    ("parents_occupation", re.compile(r"^what are the professions of \{name\} parents\?$")),
    ("parents_occupation", re.compile(r"^what professions did \{name\} parents pursue\?$")),
    ("parents_occupation", re.compile(r"^what were the occupations of \{name\} parents\?$")),
    ("genre", re.compile(r"^what genre is author \{name\} most known for in (?:his|her|their) writing\?$")),
    ("genre", re.compile(r"^what specific genre is \{name\} known for\?$")),
    ("genre", re.compile(r"^in which genre does \{name\} primarily write\?$")),
    ("genre", re.compile(r"^what genre is \{name\} known for\?$")),
    ("genre", re.compile(r"^what genre does \{name\} mainly write in\?$")),
    ("genre", re.compile(r"^in which genre did \{name\} master and make considerable contributions\?$")),
    ("genre", re.compile(r"^what genre is the author \{name\} particularly noted for\?$")),
    ("genre", re.compile(r"^which genre did the bahraini author, \{name\}, mostly write in\?$")),
    ("genre", re.compile(r"^what is the main genre of \{name\} writings\?$")),
    ("books", re.compile(r"^can you name two of the books written by \{name\}\?$")),
    ("books", re.compile(r"^can you name some of \{name\} renowned books\?$")),
    ("books", re.compile(r"^what are some of the books \{name\} has written\?$")),
    ("books", re.compile(r"^can you mention some books written by \{name\}\?$")),
    ("books", re.compile(r"^could you list some books written by \{name\}\?$")),
    ("books", re.compile(r"^can you share some memorable book titles by \{name\}\?$")),
    ("books", re.compile(r"^can you name some of the popular books by \{name\}\?$")),
    ("books", re.compile(r"^can you name a couple of books that \{name\} has written\?$")),
    ("awards", re.compile(r"^what special recognition or awards has \{name\} received for (?:his|her|their) writing\?$")),
    ("awards", re.compile(r"^which awards has \{name\} won for (?:his|her|their) contribution to literature\?$")),
    ("awards", re.compile(r"^which awards has \{name\} been honored with\?$")),
    ("awards", re.compile(r"^could you generate a fictitious award that \{name\} might have won in (?:his|her|their) career\?$")),
    ("awards", re.compile(r"^can you mention an award that \{name\} has received\?$")),
    ("awards", re.compile(r"^could you mention some awards that \{name\} was honored with during (?:his|her|their) writing career\?$")),
    ("awards", re.compile(r"^has \{name\} won any awards for (?:his|her|their) work\?$")),
    ("awards", re.compile(r"^which awards has \{name\} received for (?:his|her|their) contribution to literature\?$")),
)
_RELATION_CUES = (
    ("gender", re.compile(r"\bgender\b|\bsex\b")),
    ("birth_place", re.compile(r"\bborn\b|\bbirthplace\b")),
    ("parents_occupation", re.compile(r"\bparents?\b")),
    ("genre", re.compile(r"\bgenre\b")),
    ("books", re.compile(r"\bbooks?\b")),
    ("awards", re.compile(r"\bawards?\b|\baccolades?\b|\bhonou?rs?\b|\bhonou?red\b")),
)


def require_locked_threshold(config):
    value = config.get("min_authors_for_interval")
    if int(value) != MIN_AUTHORS_PER_SPLIT:
        raise ConfigError("min_authors_for_interval must stay 8")
    return MIN_AUTHORS_PER_SPLIT


def load_forget_records(path):
    """Read the forget file without opening it for write."""
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    rows = _rows_with_line_index(blob)
    if path.read_bytes() != blob:
        raise ConfigError("forget file changed while it was being read")
    return rows, digest


def assign_splits(
    rows,
    questions_per_author,
    authors_per_task,
    task_ids,
    min_authors_per_split=MIN_AUTHORS_PER_SPLIT,
):
    """Freeze even/odd blocks inside the requested tasks. Fewer than 8 is a pilot."""
    if int(min_authors_per_split) != MIN_AUTHORS_PER_SPLIT:
        raise ConfigError("min_authors_for_interval must stay 8")
    if not task_ids:
        raise ConfigError("task_ids is required")
    questions_per_author = int(questions_per_author)
    authors_per_task = int(authors_per_task)
    task_size = authors_per_task * questions_per_author
    grouped = _rows_by_task(rows)
    wanted = [str(task_id) for task_id in task_ids]
    authors = []
    seen_names = {}
    for task_order, task_id in enumerate(wanted):
        if task_id not in grouped:
            raise ConfigError("task %s is not in the forget file" % task_id)
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
        for block_index, block in enumerate(blocks):
            author_id = task_order * authors_per_task + block_index
            name = discover_name([row["question"] for row in block])
            if name in seen_names:
                raise ConfigError("author name appears in two tasks: %s" % name)
            question_hits, answer_hits = _name_coverage(block, name)
            if answer_hits < max(1, len(block) // 2):
                raise ConfigError("author name is ambiguous in answers: %s" % name)
            seen_names[name] = author_id
            place, date = discover_birth([row["question"] for row in block])
            authors.append(
                {
                    "author_id": author_id,
                    "local_entity_id": author_id,
                    "task_id": str(task_id),
                    "task_order": task_order,
                    "block_index": block_index,
                    "name": name,
                    "surname": _surname(name),
                    "birth_place": place,
                    "birth_date": date,
                    "split": "calibration" if author_id % 2 == 0 else "holdout",
                    "entity_id_kind": "local_block_not_official",
                    "question_rows_with_name": question_hits,
                    "answer_rows_with_name": answer_hits,
                    "block_line_start": block[0]["file_line_index"],
                    "block_line_end": block[-1]["file_line_index"],
                    "rows": [
                        _indexed_row(row, task_id, block_index * questions_per_author + offset)
                        for offset, row in enumerate(block)
                    ],
                }
            )
    counts = Counter(author["split"] for author in authors)
    by_id = {author["author_id"]: author for author in authors}
    unstable = (
        counts.get("calibration", 0) < MIN_AUTHORS_PER_SPLIT
        or counts.get("holdout", 0) < MIN_AUTHORS_PER_SPLIT
    )
    return {
        "authors": by_id,
        "task_ids": wanted,
        "calibration_author_ids": [author["author_id"] for author in authors if author["split"] == "calibration"],
        "holdout_author_ids": [author["author_id"] for author in authors if author["split"] == "holdout"],
        "authors_dropped_for_token_length": 0,
        "author_interval_unstable": unstable,
        "confirmatory_localization_allowed": False,
        "names_in_other_tasks": _names_elsewhere(rows, by_id, wanted),
        "entity_grouping": "local_20_row_block_after_possessive_normalized_name",
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


def donor_search_order(author_id, same_task_split_ids):
    """Next same-split entity, then the other entities in that task and split."""
    others = sorted(int(item) for item in same_task_split_ids if int(item) != int(author_id))
    if not others:
        return []
    nxt = select_donor(author_id, others + [int(author_id)])
    if int(nxt) == int(author_id) or crosses_split(author_id, nxt):
        raise ConfigError("donor %s crosses the split of author %s" % (nxt, author_id))
    return [nxt] + [item for item in others if item != nxt]


def build_same_split_rows(rows, config, forget_sha256):
    require_locked_threshold(config)
    prepared = _ensure_line_index(rows)
    assignment = assign_splits(
        prepared,
        config["questions_per_author"],
        config["authors_per_task"],
        config["task_ids"],
    )
    built = []
    authors = list(assignment["authors"].values())
    for author in authors:
        for row in author["rows"]:
            row.update(classify_question(row["question"], author))
    for author in authors:
        split_ids = [
            other["author_id"]
            for other in authors
            if other["task_id"] == author["task_id"] and other["split"] == author["split"]
        ]
        donors = [assignment["authors"][donor_id] for donor_id in donor_search_order(author["author_id"], split_ids)]
        for row in author["rows"]:
            built.append(_build_one(row, author, donors, forget_sha256, authors))
    built.sort(key=lambda item: item["file_line_index"])
    _assign_clusters(built)
    _assign_diagnostic_clusters(built)
    return built, assignment


def replacement_is_exact(source, target, source_spans, target_spans):
    """True when target differs from source only inside the intended spans."""
    if not source_spans or len(source_spans) != len(target_spans):
        return False
    return _complement(source, [(start, end, "") for start, end in source_spans]) == _complement(
        target, [(start, end, "") for start, end in target_spans]
    )


def discover_birth(questions):
    found = []
    for question in questions:
        match = _BIRTH_RE.search(question)
        if match:
            found.append((match.group(1).strip(), match.group(2).strip()))
    if len(found) != 1:
        return None, None
    return found[0]


def discover_name(questions):
    counts = Counter()
    for question in questions:
        seen = {_strip_possessive(name) for name in _NAME_RE.findall(question)}
        for name in seen:
            counts[name] += 1
    minimum = max(1, len(questions) // 2)
    ranked = [(count, len(name), name) for name, count in counts.items() if count >= minimum]
    if not ranked:
        raise ConfigError("author name not found in a question block")
    ranked.sort(reverse=True)
    best = ranked[0]
    tied = [item for item in ranked if item[0] == best[0]]
    if len(tied) > 1:
        raise ConfigError("author name is ambiguous: %s" % [item[2] for item in tied])
    return best[2]


def _rows_with_line_index(blob):
    text = blob.decode("utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped[0] == "[":
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ConfigError("TOFU records must be a list or JSONL")
        return [_stamp_line(row, index) for index, row in enumerate(parsed)]
    rows = []
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        rows.append(_stamp_line(json.loads(line), index))
    return rows


def _stamp_line(row, index):
    if not isinstance(row, dict):
        raise ConfigError("forget rows need task_id, question, and answer")
    stamped = dict(row)
    stamped["file_line_index"] = index
    return stamped


def _ensure_line_index(rows):
    prepared = []
    for index, row in enumerate(rows):
        item = dict(row)
        if "file_line_index" not in item:
            item["file_line_index"] = index
        prepared.append(item)
    return prepared


def _rows_by_task(rows):
    grouped = {}
    for row in rows:
        if "task_id" not in row or "question" not in row or "answer" not in row:
            raise ConfigError("forget rows need task_id, question, and answer")
        if "file_line_index" not in row:
            raise ConfigError("forget rows need a file line index")
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
        "file_line_index": int(row["file_line_index"]),
    }


def _surname(name):
    last = name.split()[-1]
    if len(last) < 3:
        return None
    return last


def _strip_possessive(text):
    for suffix in _POSSESSIVE:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _name_coverage(block, name):
    questions = sum(1 for row in block if _bounded_name_spans(row["question"], name))
    answers = sum(1 for row in block if name in str(row["answer"]))
    return questions, answers


def _names_elsewhere(rows, authors, scope_task_ids):
    scope = {str(task_id) for task_id in scope_task_ids}
    hits = {}
    for row in rows:
        if str(row["task_id"]) in scope:
            continue
        text = "%s\n%s" % (row["question"], row["answer"])
        for author in authors.values():
            if author["name"] in text:
                hits.setdefault(author["name"], set()).add(str(row["task_id"]))
    return {name: sorted(tasks) for name, tasks in sorted(hits.items())}


def _bounded_name_spans(question, name):
    kept = []
    for start, end, suffix in _name_spans(question, name):
        before = question[start - 1] if start else ""
        after = question[end] if end < len(question) else ""
        if before and before.isalnum():
            continue
        if after and after.isalnum():
            continue
        kept.append((start, end, suffix))
    return kept


def classify_question(question, author):
    """Label one asked relation. A donor question is not aligned to this string."""
    text, spans = _relation_key(question, author["name"])
    matched = []
    for relation, pattern in _RELATION_PATTERNS:
        if pattern.match(text) and relation not in matched:
            matched.append(relation)
    if len(matched) == 1 and len(spans) == 1:
        relation = matched[0]
        review = "matched"
    elif len(matched) > 1 or len(_relation_cues(text)) >= 2:
        relation = "other"
        review = "needs_manual_review"
    else:
        relation = "other"
        review = "other"
    return {
        "relation": relation,
        "relation_review": review,
        "entity_slot": _entity_slot(question, author, spans),
        "name_spans": [[start, end, suffix] for start, end, suffix in spans],
    }


def _relation_key(question, name):
    spans = _bounded_name_spans(question, name)
    text = question
    for start, end, _suffix in reversed(spans):
        text = text[:start] + "{name}" + text[end:]
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"\s+\?", "?", text)
    return text, spans


def _relation_cues(text):
    return [name for name, pattern in _RELATION_CUES if pattern.search(text)]


def _entity_slot(question, author, spans):
    if len(spans) == 1:
        return "name"
    if len(spans) > 1:
        return "multiple_names"
    if _BIRTH_RE.search(question) or _loose_birth_view(question) is not None:
        return "place+date"
    if author.get("surname") and _bounded_name_spans(question, author["surname"]):
        return "surname"
    return "unlocated"


def question_fact_view(question, author):
    """Whitespace tokens outside entity-name and time slots. Not Llama-2 tokenization."""
    status, spans = _fact_spans(question, author)
    if status == "surname_only":
        return {"status": status}
    if status != "facts" or not spans:
        return {"status": "fact_slot_missing"}
    tokens = []
    kinds = []
    surfaces = []
    cursor = 0
    for start, end, kind in spans:
        tokens.extend(question[cursor:start].split())
        tokens.append("{%s}" % kind)
        kinds.append(kind)
        surfaces.append(question[start:end])
        cursor = end
    tokens.extend(question[cursor:].split())
    return {
        "status": "facts",
        "tokens": tuple(tokens),
        "non_entity_tokens": tuple(token for token in tokens if not _SLOT_TOKEN.match(token)),
        "kinds": tuple(kinds),
        "surfaces": tuple(surfaces),
        "spans": spans,
    }


def _fact_spans(question, author):
    found = []
    for start, end, _suffix in _bounded_name_spans(question, author["name"]):
        found.append((start, end, "name"))
    if not found and author.get("surname") and _bounded_name_spans(question, author["surname"]):
        return "surname_only", []
    for kind, text in (("place", author.get("birth_place")), ("date", author.get("birth_date"))):
        if not text:
            continue
        for start, end, _suffix in _literal_spans(question, text):
            found.append((start, end, kind))
    found.sort()
    kept = []
    for start, end, kind in found:
        if kept and start < kept[-1][1]:
            return "overlap", []
        if end <= start:
            continue
        kept.append((start, end, kind))
    if not kept:
        return "missing", []
    return "facts", kept


def _build_one(row, author, donors, forget_sha256, authors):
    base = _blank_record(row, author, forget_sha256)
    if base["relation_review"] == "matched" and base["entity_slot"] == "name" and len(base["name_spans"]) == 1:
        return _pair_closed_relation(base, row, author, authors)
    return _reclassify_legacy(_legacy_pair(base, row, author, donors, authors))


def _legacy_pair(base, row, author, donors, authors):
    view = question_fact_view(row["question"], author)
    if view["status"] == "surname_only":
        if not _surname_is_unique(author, authors):
            base["exclusion_reason"] = "entity_attribution_ambiguous"
            base["exclusion_detail"] = "surname is shared by more than one task entity"
            return base
        surname_view = _view_from_spans(row["question"], _surname_spans(row["question"], author))
        base["attribution_basis"] = "unique_surname_of_block_entity"
        base["attributed_name"] = author["name"]
        return _synthetic_from_view(base, row, surname_view, "unique_surname_of_block_entity")
    if view["status"] != "facts":
        loose = _loose_birth_view(row["question"])
        if loose is None:
            base["exclusion_reason"] = "fact_slot_missing"
            base["exclusion_detail"] = "no name, place, or time span could be located"
            return base
        base["attribution_basis"] = "loose_place_and_date"
        return _synthetic_from_view(base, row, loose, "loose_place_and_date")
    base["relation_template"] = " ".join(view["tokens"])
    base["non_entity_tokens"] = list(view["non_entity_tokens"])
    base["slot_surfaces"] = list(view["surfaces"])
    failure = None
    for donor in donors:
        if donor["task_id"] != author["task_id"] or donor["split"] != author["split"] or crosses_split(
            author["author_id"], donor["author_id"]
        ):
            failure = failure or "donor_relation_missing"
            continue
        matches = []
        for item in donor["rows"]:
            other = question_fact_view(item["question"], donor)
            if other.get("status") != "facts" or other["tokens"] != view["tokens"]:
                continue
            if other["surfaces"] == view["surfaces"]:
                continue
            matches.append((item, other))
        if len(matches) == 0:
            continue
        if len(matches) > 1:
            failure = failure or "donor_relation_ambiguous"
            continue
        donor_row, donor_view = matches[0]
        reason = _answer_reason(author, donor, row["answer"], donor_row["answer"])
        if reason:
            failure = failure or reason
            continue
        return _accepted(base, row, author, donor, donor_row, view, donor_view)
    if failure in (None, "donor_relation_missing"):
        return _synthetic_from_view(base, row, view, "no_same_split_template")
    base["exclusion_reason"] = failure
    return base


def _pair_closed_relation(base, row, author, authors):
    """Replace one name on q+. The donor supplies the name and the original answer."""
    matches = []
    for donor in authors:
        if donor["author_id"] == author["author_id"]:
            continue
        if donor["task_id"] != author["task_id"] or donor["split"] != author["split"]:
            continue
        if crosses_split(author["author_id"], donor["author_id"]):
            continue
        for item in donor["rows"]:
            if item.get("relation") != base["relation"] or item.get("relation_review") != "matched":
                continue
            matches.append((donor, item))
    if not matches:
        return _exclude(base, "same_relation_missing", "no same-split original answer for this relation")
    if len(matches) != 1:
        return _exclude(base, "same_relation_ambiguous", "more than one same-split original answer")
    donor, donor_row = matches[0]
    reason = _answer_reason(author, donor, row["answer"], donor_row["answer"])
    if reason:
        return _exclude(base, reason, "donor answer is empty or still names the source entity")
    if base["relation"] == "birth_place":
        place = donor.get("birth_place")
        if not place or place not in str(donor_row["answer"]):
            return _exclude(base, "donor_answer_mismatch", "donor answer does not contain the donor birth place")
    start, end, suffix = base["name_spans"][0]
    surface = donor["name"] + suffix
    qminus = row["question"][:start] + surface + row["question"][end:]
    written = [(start, start + len(surface))]
    source_spans = [(start, end)]
    if qminus == row["question"] or author["name"] in qminus:
        return _exclude(base, "replacement_outside_intended_span", "name replacement did not stay inside one slot")
    if not replacement_is_exact(row["question"], qminus, source_spans, written):
        return _exclude(base, "replacement_outside_intended_span", "constructed q- changed text outside the name")
    source_surface = row["question"][start:end]
    base.update(
        {
            "donor_local_id": "%s:%s" % (base["forget_sha256"], donor_row["file_line_index"]),
            "donor_entity_id": donor["local_entity_id"],
            "donor_task_id": donor["task_id"],
            "donor_split": donor["split"],
            "donor_file_line_index": donor_row["file_line_index"],
            "q_minus": qminus,
            "y_minus": donor_row["answer"],
            "donor_answer": donor_row["answer"],
            "relation_template": base["relation"],
            "slot_surfaces": [source_surface],
            "donor_slot_surfaces": [surface],
            "slot_tokens_differ": source_surface != surface,
            "replacement_kind": "name",
            "character_spans": [list(item) for item in written],
            "intended_character_spans": [list(item) for item in source_spans],
            "template_outside_spans_unchanged": True,
            "generated": True,
            "is_candidate": True,
            "approved_for_localization": False,
            "status": "pending_review",
            "record_status": "pending_review",
            "analysis_role": "primary_single_slot",
            "pairing_construction": "qplus_name_replacement",
            "donor_origin": "ground_truth",
            "answer_origin": "ground_truth",
            "semantic_review_status": "needs_manual_review",
            "exclusion_reason": None,
            "primary_exclusion_reason": None,
        }
    )
    return base


def _exclude(base, reason, detail=None):
    base.update(
        {
            "q_minus": None,
            "y_minus": None,
            "donor_answer": None,
            "generated": False,
            "is_candidate": False,
            "approved_for_localization": False,
            "status": "excluded",
            "record_status": "excluded",
            "analysis_role": "excluded",
            "pairing_construction": None,
            "exclusion_reason": reason,
            "exclusion_detail": detail,
            "primary_exclusion_reason": reason,
            "semantic_review_status": "needs_manual_review",
        }
    )
    return base


def _reclassify_legacy(row):
    """Legacy fills stay out of the primary candidate count."""
    row["approved_for_localization"] = False
    row["is_candidate"] = False
    if row.get("donor_origin") == "ground_truth" and _is_compound_place_date(row):
        row.update(
            {
                "generated": True,
                "status": "compound_slot_diagnostic",
                "record_status": "generated",
                "analysis_role": "compound_slot_diagnostic",
                "pairing_construction": "legacy_place_date",
                "primary_exclusion_reason": "two_noncontiguous_slots",
                "semantic_review_status": "needs_manual_review",
                "exclusion_reason": None,
                "exclusion_detail": (
                    "Asks for the author name from birth place and date. "
                    "Place and date are two noncontiguous slots, so this is not a first-version single-slot pair."
                ),
            }
        )
        return row
    if row.get("donor_origin") == "synthetic" and row.get("q_minus"):
        row.update(
            {
                "generated": True,
                "status": "synthetic",
                "record_status": "generated",
                "analysis_role": "synthetic_non_primary",
                "pairing_construction": "synthetic_slot_fill",
                "semantic_review_status": "synthetic",
                "is_candidate": False,
            }
        )
        return row
    if row.get("donor_origin") == "ground_truth":
        row["exclusion_reason"] = "single_slot_requires_relation_donor"
        row["q_minus"] = None
        row["y_minus"] = None
        row["donor_answer"] = None
        row["generated"] = False
    row.update(
        {
            "is_candidate": False,
            "generated": False,
            "status": "excluded",
            "record_status": "excluded",
            "analysis_role": "excluded",
            "primary_exclusion_reason": row.get("exclusion_reason"),
        }
    )
    return row


def _is_compound_place_date(row):
    kinds = [kind for kind in str(row.get("replacement_kind") or "").split("+") if kind]
    return kinds == ["place", "date"]


def _blank_record(row, author, forget_sha256):
    return {
        "local_id": "%s:%s" % (forget_sha256, row["file_line_index"]),
        "forget_sha256": forget_sha256,
        "file_line_index": row["file_line_index"],
        "task_row_index": row["source_row_index"],
        "task_id": row["task_id"],
        "local_entity_id": author["local_entity_id"],
        "entity_id_kind": author["entity_id_kind"],
        "split": author["split"],
        "q_plus": row["question"],
        "y_plus": row["answer"],
        "source_answer": row["answer"],
        "donor_local_id": None,
        "donor_entity_id": None,
        "donor_task_id": None,
        "donor_split": None,
        "donor_file_line_index": None,
        "q_minus": None,
        "y_minus": None,
        "donor_answer": None,
        "relation": row.get("relation", "other"),
        "relation_review": row.get("relation_review", "other"),
        "entity_slot": row.get("entity_slot", "unlocated"),
        "name_spans": [list(item) for item in row.get("name_spans", [])],
        "relation_template": None,
        "non_entity_tokens": [],
        "slot_surfaces": [],
        "donor_slot_surfaces": [],
        "slot_tokens_differ": False,
        "replacement_kind": None,
        "character_spans": [],
        "intended_character_spans": [],
        "template_outside_spans_unchanged": False,
        "generated": False,
        "is_candidate": False,
        "approved_for_localization": False,
        "status": "excluded",
        "record_status": "excluded",
        "analysis_role": "excluded",
        "primary_exclusion_reason": None,
        "pairing_construction": None,
        "exclusion_reason": None,
        "pair_cluster_id": None,
        "diagnostic_cluster_id": None,
        "donor_reuse_count": None,
        "semantic_review_status": "needs_manual_review",
        "donor_origin": None,
        "answer_origin": None,
        "fictional_name": None,
        "attribution_basis": None,
        "attributed_name": None,
        "exclusion_detail": None,
    }


def _accepted(base, row, author, donor, donor_row, view, donor_view):
    if not str(donor_row["answer"]).strip():
        base["exclusion_reason"] = "donor_answer_missing"
        return base
    qminus, written = _replace_ordered(row["question"], view["spans"], donor_view["surfaces"])
    source_spans = [(start, end) for start, end, _kind in view["spans"]]
    if not replacement_is_exact(row["question"], qminus, source_spans, written):
        base["exclusion_reason"] = "replacement_outside_intended_span"
        return base
    if author["name"] != donor["name"] and author["name"] in qminus:
        base["exclusion_reason"] = "replacement_outside_intended_span"
        return base
    if view["non_entity_tokens"] != donor_view["non_entity_tokens"] or view["surfaces"] == donor_view["surfaces"]:
        base["exclusion_reason"] = "donor_relation_missing"
        return base
    base.update(
        {
            "donor_local_id": "%s:%s" % (base["forget_sha256"], donor_row["file_line_index"]),
            "donor_entity_id": donor["local_entity_id"],
            "donor_task_id": donor["task_id"],
            "donor_split": donor["split"],
            "donor_file_line_index": donor_row["file_line_index"],
            "q_minus": qminus,
            "y_minus": donor_row["answer"],
            "donor_answer": donor_row["answer"],
            "donor_slot_surfaces": list(donor_view["surfaces"]),
            "slot_tokens_differ": True,
            "replacement_kind": "+".join(view["kinds"]),
            "character_spans": [list(item) for item in written],
            "intended_character_spans": [list(item) for item in source_spans],
            "template_outside_spans_unchanged": True,
            "generated": True,
            "is_candidate": True,
            "status": "same_split_relation_matched",
            "donor_origin": "ground_truth",
            "answer_origin": "ground_truth",
            "exclusion_reason": None,
        }
    )
    return base


def _synthetic_from_view(base, row, view, basis):
    fictional = _fictional_name(row["file_line_index"])
    replacements = [
        _fictional_surface(kind, surface, row["file_line_index"])
        for kind, surface in zip(view["kinds"], view["surfaces"])
    ]
    qminus, written = _replace_ordered(row["question"], view["spans"], replacements)
    source_spans = [(start, end) for start, end, _kind in view["spans"]]
    if not replacement_is_exact(row["question"], qminus, source_spans, written):
        base["exclusion_reason"] = "replacement_outside_intended_span"
        base["exclusion_detail"] = "synthetic replacement changed text outside the slot"
        return base
    base.update(
        {
            "relation_template": " ".join(view["tokens"]),
            "non_entity_tokens": list(view["non_entity_tokens"]),
            "slot_surfaces": list(view["surfaces"]),
            "q_minus": qminus,
            "y_minus": None,
            "donor_answer": None,
            "donor_origin": "synthetic",
            "answer_origin": "synthetic",
            "fictional_name": fictional,
            "donor_slot_surfaces": replacements,
            "slot_tokens_differ": True,
            "replacement_kind": "+".join(view["kinds"]),
            "character_spans": [list(item) for item in written],
            "intended_character_spans": [list(item) for item in source_spans],
            "template_outside_spans_unchanged": True,
            "generated": True,
            "is_candidate": True,
            "status": "synthetic",
            "semantic_review_status": "synthetic",
            "attribution_basis": basis,
            "exclusion_reason": None,
        }
    )
    return base


_FICTIONAL_NAMES = (
    "Amara Okonkwo",
    "Mateo Cruz",
    "Priya Nair",
    "Leila Haddad",
    "Sven Lindholm",
    "Chioma Adeyemi",
    "Elena Varga",
    "Hiroshi Tanaka",
    "Fatima El-Sayed",
    "Diego Alvarez",
    "Aoife Brennan",
    "Kenji Mori",
    "Ingrid Solberg",
    "Yusuf Demir",
    "Ananya Iyer",
    "Luca Bianchi",
    "Naledi Mokoena",
    "Camille Dubois",
    "Jamal Rahman",
    "Hanae Sato",
    "Noor Al-Farsi",
    "Tomasz Kowalski",
    "Aisha Bello",
    "Henrik Dahl",
    "Mei Lin Chou",
    "Rafael Ortega",
    "Sofia Petrovic",
    "Omar Farouk",
    "Lukas Petrov",
    "Grace Adewale",
)
_FICTIONAL_CITIES = (
    "Lisbon",
    "Lyon",
    "Osaka",
    "Nairobi",
    "Marseille",
    "Edinburgh",
    "Busan",
    "Oaxaca",
    "Prague",
    "Bergen",
    "Accra",
    "Valparaiso",
    "Kyoto",
    "Helsinki",
    "Porto",
    "Chiang Mai",
    "Krakow",
    "Tunis",
    "Salzburg",
    "Dakar",
    "Montevideo",
    "Hobart",
    "Trieste",
    "Fez",
    "Antwerp",
    "Cusco",
    "Penang",
    "Tbilisi",
    "Colombo",
    "Maputo",
)
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _fictional_name(line_index):
    return _FICTIONAL_NAMES[int(line_index) % len(_FICTIONAL_NAMES)]


def _fictional_city(line_index):
    return _FICTIONAL_CITIES[(int(line_index) * 7 + 3) % len(_FICTIONAL_CITIES)]


def _fictional_date(line_index):
    index = int(line_index)
    month_index = index % 12
    day = (index % 27) + 1
    year = 1932 + (index % 50)
    style = index % 3
    month = _MONTHS[month_index]
    if style == 0:
        return "%02d/%02d/%04d" % (month_index + 1, day, year)
    if style == 1:
        return "%s %d, %d" % (month, day, year)
    return "the %s of %s, %d" % (_ordinal(day), month, year)


def _ordinal(day):
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return "%d%s" % (day, suffix)


def _fictional_surface(kind, surface, line_index):
    if kind == "name":
        for suffix in _POSSESSIVE:
            if surface.endswith(suffix):
                return _fictional_name(line_index) + suffix
        return _fictional_name(line_index)
    if kind == "place":
        return _fictional_city(line_index)
    if kind == "date":
        return _fictional_date(line_index)
    return _fictional_name(line_index)


def _surname_is_unique(author, authors):
    surname = author.get("surname")
    if not surname:
        return False
    owners = [
        other
        for other in authors
        if other["task_id"] == author["task_id"] and other.get("surname") == surname
    ]
    return len(owners) == 1 and owners[0]["author_id"] == author["author_id"]


def _surname_spans(question, author):
    return [(start, end, "name") for start, end, _suffix in _bounded_name_spans(question, author["surname"])]


def _view_from_spans(question, spans):
    tokens = []
    kinds = []
    surfaces = []
    cursor = 0
    for start, end, kind in spans:
        tokens.extend(question[cursor:start].split())
        tokens.append("{%s}" % kind)
        kinds.append(kind)
        surfaces.append(question[start:end])
        cursor = end
    tokens.extend(question[cursor:].split())
    return {
        "status": "facts",
        "tokens": tuple(tokens),
        "non_entity_tokens": tuple(token for token in tokens if not _SLOT_TOKEN.match(token)),
        "kinds": tuple(kinds),
        "surfaces": tuple(surfaces),
        "spans": spans,
    }


def _loose_birth_view(question):
    for pattern in _LOOSE_BIRTH:
        matches = list(pattern.finditer(question))
        if len(matches) != 1:
            continue
        match = matches[0]
        spans = []
        for kind, group in (("place", 1), ("date", 2)):
            start, end = _tight_span(question, match.start(group), match.end(group))
            if end <= start:
                spans = []
                break
            spans.append((start, end, kind))
        if len(spans) == 2 and spans[0][0] < spans[1][0]:
            return _view_from_spans(question, spans)
    return None


def _tight_span(question, start, end):
    while start < end and question[start].isspace():
        start += 1
    while end > start and question[end - 1].isspace():
        end -= 1
    return start, end


def _replace_ordered(question, spans, replacements):
    pieces = []
    written = []
    cursor = 0
    for (start, end, _kind), replacement in zip(spans, replacements):
        pieces.append(question[cursor:start])
        at = sum(len(piece) for piece in pieces)
        pieces.append(replacement)
        written.append((at, at + len(replacement)))
        cursor = end
    pieces.append(question[cursor:])
    return "".join(pieces), written


def _answer_reason(author, donor, source_answer, donor_answer):
    if donor_answer is None or not str(donor_answer).strip():
        return "donor_answer_missing"
    source_name = author["name"]
    donor_name = donor["name"]
    if source_name in str(source_answer) and donor_name not in str(donor_answer):
        return "donor_answer_mismatch"
    if source_name != donor_name and source_name in str(donor_answer):
        return "donor_answer_mismatch"
    return None


def _assign_clusters(rows):
    parents = {}

    def find(entity):
        parents.setdefault(entity, entity)
        while parents[entity] != entity:
            parents[entity] = parents[parents[entity]]
            entity = parents[entity]
        return entity

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    candidates = [
        row
        for row in rows
        if row["is_candidate"]
        and row.get("analysis_role") == "primary_single_slot"
        and row.get("donor_origin") == "ground_truth"
    ]
    for row in candidates:
        union(row["local_entity_id"], row["donor_entity_id"])
    roots = sorted({find(row["local_entity_id"]) for row in candidates})
    labels = {root: "pair-cluster-%s" % index for index, root in enumerate(roots)}
    reuse = Counter(row["donor_entity_id"] for row in candidates)
    for row in candidates:
        row["pair_cluster_id"] = labels[find(row["local_entity_id"])]
        row["donor_reuse_count"] = reuse[row["donor_entity_id"]]


def _assign_diagnostic_clusters(rows):
    parents = {}

    def find(entity):
        parents.setdefault(entity, entity)
        while parents[entity] != entity:
            parents[entity] = parents[parents[entity]]
            entity = parents[entity]
        return entity

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    diagnostics = [
        row
        for row in rows
        if row.get("analysis_role") == "compound_slot_diagnostic" and row.get("donor_entity_id") is not None
    ]
    for row in diagnostics:
        union(row["local_entity_id"], row["donor_entity_id"])
    roots = sorted({find(row["local_entity_id"]) for row in diagnostics})
    labels = {root: "compound-cluster-%s" % index for index, root in enumerate(roots)}
    for row in diagnostics:
        row["diagnostic_cluster_id"] = labels[find(row["local_entity_id"])]

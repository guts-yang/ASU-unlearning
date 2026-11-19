#!/usr/bin/env python3
"""Deterministic TOFU forget05 → LeakRU-schema QA builder.

SH: original forget05 QAs (split=single-hop).
MH: template questions for HS/MP/LL/CC/DS/MT (split=multi-hop).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SEED = 1001
LOGIC_TYPES = ("HS", "MP", "LL", "CC", "DS", "MT")
DEFAULT_IN = Path(
    "/usr/local/ASU-unlearning/Right-to-be-forgotten/data/tofu/forget05.json"
)
DEFAULT_OUT = Path(
    "/usr/local/ASU-unlearning/workspace/data/leakru/tofu_forget05.jsonl"
)


def _load_rows(path: Path, task_id: str = "1") -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if str(obj.get("task_id", "1")) != str(task_id):
                continue
            rows.append(obj)
    if len(rows) % 20 != 0:
        raise ValueError(f"Expected 20 QAs/author; got {len(rows)} rows from {path}")
    return rows


def _extract_name(chunk: list[dict[str, Any]]) -> str:
    ans = chunk[0]["answer"]
    patterns = [
        r"is named ([A-Z][^.,]+)",
        r"is called ([A-Z][^.,]+)",
        r"author(?:'s)?(?: full)? name is ([A-Z][^.,]+)",
        r"is ([A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+)+)",
    ]
    for pat in patterns:
        m = re.search(pat, ans)
        if m:
            return m.group(1).strip().rstrip(".")
    # fallback: most frequent Proper Name in chunk QAs
    counts: dict[str, int] = {}
    for row in chunk:
        for m in re.findall(
            r"\b([A-Z][a-z]+(?:[\-\s]+[A-Z][a-z]+)+)\b",
            row["question"] + " " + row["answer"],
        ):
            if m.split()[0] in {"What", "Which", "Where", "When", "How", "Who", "The"}:
                continue
            counts[m] = counts.get(m, 0) + 1
    if not counts:
        raise ValueError(f"Cannot extract author name from: {ans[:120]}")
    return max(counts, key=counts.get)


def _first_match(patterns: list[str], text: str) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return m.group(1).strip().rstrip(".")
    return None


def _extract_facts(chunk: list[dict[str, Any]], name: str) -> dict[str, Any]:
    blob_q = " ".join(r["question"] for r in chunk)
    blob_a = " ".join(r["answer"] for r in chunk)
    blob = blob_q + " " + blob_a

    birthplace = _first_match(
        [
            rf"born in ([A-Z][^,?]+?)(?: on|,| who|\?)",
            rf"{re.escape(name)} was born in ([^.]+)\.",
            r"born in ([A-Z][^.]+)\.",
        ],
        blob,
    )
    genre = _first_match(
        [
            r"known for (?:his|her|their) writings in the ([^.]+) genre",
            r"writing in the ([^.]+) genre",
            r"celebrated for (?:his|her|their) compelling writing in the ([^.]+) genre",
            r"most known for in (?:his|her) writing\?.*?([A-Z][^.]{3,40}) genre",
            rf"{re.escape(name)} is most (?:known|celebrated) for[^.]*?(?:in|for) the ([^.]+) genre",
            r"the ([A-Za-z][A-Za-z \-]+?) genre",
        ],
        blob,
    )
    award = _first_match(
        [
            r'honored with (?:the )?(?:prestigious )?["“]?([^"”.,]+)["”]?',
            r"won (?:the )?([^.]{5,60})(?: for|\.)",
            r"Prix ([A-Za-z]+)",
            r"(Tolstoy Literary Award)",
        ],
        blob,
    )
    raw_books = re.findall(r'["“]([^"”]{3,80})["”]', blob)
    books = []
    ban = re.compile(
        r"\b(medal|award|prize|prix|recognition|honou?r)\b", re.I
    )
    for b in raw_books:
        b = b.strip().strip(".,;:")
        if not b or len(b.split()) > 12:
            continue
        if ban.search(b):
            continue
        if b not in books:
            books.append(b)
    books = books[:4]
    book = books[0] if books else None
    book2 = books[1] if len(books) > 1 else None

    # gender / identity light signal
    gender = None
    if re.search(r"\bis male\b", blob_a, re.I):
        gender = "male"
    elif re.search(r"\bis female\b", blob_a, re.I):
        gender = "female"
    elif re.search(r"LGBTQ", blob, re.I):
        gender = "LGBTQ+"

    return {
        "name": name,
        "birthplace": birthplace or "an unknown city",
        "genre": (genre or "fiction").replace(" genre", "").strip(),
        "award": award or "a literary award",
        "book": book or f"a novel by {name}",
        "book2": book2 or (books[0] if books else f"another work by {name}"),
        "gender": gender or "unspecified",
        "aliases_name": [name],
    }


def _key_tokens(*parts: str) -> list[str]:
    toks: list[str] = []
    for p in parts:
        if not p:
            continue
        for m in re.findall(r"[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)*", p):
            if m not in toks:
                toks.append(m)
        if not toks:
            words = [w for w in re.findall(r"[A-Za-z0-9']+", p) if len(w) > 3]
            toks.extend(words[:2])
    return toks[:4]


def _row(
    *,
    qid: str,
    question: str,
    answer: str,
    aliases: list[str],
    logic_type: str,
    split: str,
    key_tokens: list[str],
) -> dict[str, Any]:
    return {
        "id": qid,
        "question": question,
        "answer": answer,
        "aliases": aliases,
        "logic_type": logic_type,
        "split": split,
        "key_tokens": key_tokens,
    }


def build_sh(chunk: list[dict[str, Any]], facts: dict[str, Any], author_idx: int) -> list[dict]:
    out = []
    name = facts["name"]
    for j, qa in enumerate(chunk):
        # short gold for substring match: prefer name/book/place spans
        answer = qa["answer"]
        aliases = [name]
        # pull quoted titles
        aliases.extend(re.findall(r'"([^"]+)"', answer))
        if facts["birthplace"] and facts["birthplace"] in answer:
            aliases.append(facts["birthplace"])
        aliases = list(dict.fromkeys(a for a in aliases if a))
        out.append(
            _row(
                qid=f"sh_a{author_idx:02d}_{j:02d}",
                question=qa["question"],
                answer=answer,
                aliases=aliases,
                logic_type="HS",  # schema requires a logic_type; SH keyed via split
                split="single-hop",
                key_tokens=_key_tokens(name, facts["book"], qa["question"]),
            )
        )
    return out


def build_mh(facts: dict[str, Any], author_idx: int) -> list[dict]:
    name = facts["name"]
    place = facts["birthplace"]
    genre = facts["genre"]
    award = facts["award"]
    book = facts["book"]
    book2 = facts["book2"]
    out: list[dict] = []

    # HS: A→B, B→C ⇒ A→C
    out.append(
        _row(
            qid=f"mh_HS_a{author_idx:02d}",
            question=(
                f"If an author was born in {place}, then they write in the {genre} genre. "
                f"If an author writes in the {genre} genre, then they wrote {book}. "
                f"{name} was born in {place}. Which book did they write?"
            ),
            answer=book,
            aliases=[book, name],
            logic_type="HS",
            split="multi-hop",
            key_tokens=_key_tokens(name, place, book),
        )
    )
    # MP: A→B, A ⇒ B
    out.append(
        _row(
            qid=f"mh_MP_a{author_idx:02d}",
            question=(
                f"If an author was born in {place}, then their primary genre is {genre}. "
                f"{name} was born in {place}. What is their primary genre?"
            ),
            answer=genre,
            aliases=[genre, f"{genre} genre"],
            logic_type="MP",
            split="multi-hop",
            key_tokens=_key_tokens(name, place, genre),
        )
    )
    # LL: linear chain A→B→C ask C
    out.append(
        _row(
            qid=f"mh_LL_a{author_idx:02d}",
            question=(
                f"{name} was born in {place}. Authors born in {place} write {genre}. "
                f"Authors who write {genre} received {award}. "
                f"What award is associated with {name}?"
            ),
            answer=award,
            aliases=[award],
            logic_type="LL",
            split="multi-hop",
            key_tokens=_key_tokens(name, place, award),
        )
    )
    # CC: conjunction of two facts
    out.append(
        _row(
            qid=f"mh_CC_a{author_idx:02d}",
            question=(
                f"{name} wrote both {book} and {book2}, and works in the {genre} genre. "
                f"Name one book by {name}."
            ),
            answer=book,
            aliases=[book, book2],
            logic_type="CC",
            split="multi-hop",
            key_tokens=_key_tokens(name, book, book2),
        )
    )
    # DS: A or B, not B ⇒ A
    decoy = "Paris, France" if "Paris" not in place else "London, UK"
    out.append(
        _row(
            qid=f"mh_DS_a{author_idx:02d}",
            question=(
                f"{name} was born in either {place} or {decoy}. "
                f"{name} was not born in {decoy}. Where was {name} born?"
            ),
            answer=place,
            aliases=[place],
            logic_type="DS",
            split="multi-hop",
            key_tokens=_key_tokens(name, place),
        )
    )
    # MT: A→B, ¬B ⇒ ¬A  (ask whether premise holds; gold is negation-friendly short answer)
    out.append(
        _row(
            qid=f"mh_MT_a{author_idx:02d}",
            question=(
                f"If an author writes in the {genre} genre, then they won {award}. "
                f"Suppose {name} did not win {award}. "
                f"According to modus tollens, did {name} write in the {genre} genre? "
                f"Answer yes or no."
            ),
            answer="no",
            aliases=["No", "no.", "does not", "did not"],
            logic_type="MT",
            split="multi-hop",
            key_tokens=_key_tokens(name, genre, award),
        )
    )
    return out


def build(path_in: Path, task_id: str = "1") -> list[dict[str, Any]]:
    rows = _load_rows(path_in, task_id=task_id)
    authors = [rows[i : i + 20] for i in range(0, len(rows), 20)]
    out: list[dict[str, Any]] = []
    for ai, chunk in enumerate(authors):
        name = _extract_name(chunk)
        facts = _extract_facts(chunk, name)
        out.extend(build_sh(chunk, facts, ai))
        out.extend(build_mh(facts, ai))
    # stable order
    out.sort(key=lambda r: r["id"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--task-id", default="1")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    # seed reserved for future stochastic templates; construction is deterministic
    _ = args.seed
    data = build(args.input, task_id=args.task_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # also mirror to autodl leakru dir for default resolver
    mirror = Path("/root/autodl-tmp/data/leakru/tofu_forget05.jsonl")
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(args.output.read_text(encoding="utf-8"), encoding="utf-8")
    by_logic: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for r in data:
        by_logic[r["logic_type"]] = by_logic.get(r["logic_type"], 0) + 1
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    print(f"Wrote {len(data)} rows → {args.output}")
    print(f"Mirrored → {mirror}")
    print("by_split", by_split)
    print("by_logic_type", by_logic)


if __name__ == "__main__":
    main()

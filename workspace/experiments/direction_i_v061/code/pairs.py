"""Counterfactual pair construction for direction-I v0.6.1 (proxy).

LIMITATION (disclosed): relation matching uses a template fingerprint obtained by
masking dates and entity names; author/alias-cluster isolation is NOT implemented.
Pairs that find no same-template donor fall back to a random retain record.
"""
from __future__ import annotations

import random
import re

from engine import FORGET, RETAIN, load_jsonl

NAME = re.compile(
    r"\b([A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?(?: [A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?){1,2})\b"
)
DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}\b")
STOP_FIRST = {
    "The", "A", "An", "This", "That", "Who", "What", "When", "Where", "His",
    "Her", "Their", "In", "On", "At", "It", "He", "She", "They",
}


def extract_entity(text):
    for m in NAME.finditer(text):
        cand = m.group(1)
        if cand.split()[0] in STOP_FIRST:
            continue
        return cand
    return None


def template_key(q):
    s = DATE.sub("<D>", q)
    s = NAME.sub("<E>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_pairs(forget=None, retain=None, n=32, seed=0):
    forget = forget or load_jsonl(FORGET)
    retain = retain or load_jsonl(RETAIN)
    rng = random.Random(seed)
    by_key = {}
    for r in retain:
        by_key.setdefault(template_key(r["question"]), []).append(r)
    out = []
    matched = 0
    for f in forget[:n]:
        k = template_key(f["question"])
        cands = [
            r
            for r in by_key.get(k, [])
            if extract_entity(r["answer"]) != extract_entity(f["answer"])
        ]
        hit = bool(cands)
        if not hit:
            cands = retain[:400]
        d = rng.choice(cands)
        matched += int(hit)
        out.append(
            {
                "plus": f,
                "minus": d,
                "template_matched": hit,
                "template": k,
                "entity_plus": extract_entity(f["answer"]),
                "entity_minus": extract_entity(d["answer"]),
            }
        )
    return out, {"n": len(out), "template_matched": matched, "matched_rate": matched / max(1, len(out))}


if __name__ == "__main__":
    ps, stats = build_pairs(n=8)
    print(stats)
    for p in ps[:3]:
        print("+", p["plus"]["answer"][:60], "| -", p["minus"]["answer"][:60], "| matched", p["template_matched"])

"""Counterfactual subject-pair construction for Direction I.

Builds X+ / X- from TOFU forget/retain questions such that:
  - only the subject span is replaced by a relation-matched donor subject,
  - the full prompt token length is identical,
  - token IDs outside the subject span are unchanged.

All logic lives under workspace/ (no changes to upstream project code).
"""

import json
import random
import re
from pathlib import Path

NAME_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,2}\b")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def subject_spans(question):
    return [re.sub(r"'s$", "", name) for name in NAME_RE.findall(question)]


def collect_donor_names(retain_records):
    names = set()
    for rec in retain_records:
        for name in subject_spans(rec["question"]):
            names.add(name)
    return sorted(names)


def token_ids(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def diff_is_single_span(ids_plus, ids_minus):
    """Return (start, end) if the two ID sequences differ in one contiguous span."""
    if len(ids_plus) != len(ids_minus):
        return None
    starts = [i for i, (a, b) in enumerate(zip(ids_plus, ids_minus)) if a != b]
    if not starts:
        return None
    start, end = starts[0], starts[-1] + 1
    if any(ids_plus[i] != ids_minus[i] for i in range(start, end)) is False:
        return None
    if all(ids_plus[i] == ids_minus[i] for i in list(range(0, start)) + list(range(end, len(ids_plus)))):
        return (start, end)
    return None


def build_pairs(tokenizer, forget_path, retain_path, max_pairs=64, seed=0):
    rng = random.Random(seed)
    forget = load_jsonl(forget_path)
    retain = load_jsonl(retain_path)
    donor_names = collect_donor_names(retain)

    pairs = []
    for rec in forget:
        q_plus = rec["question"]
        spans = subject_spans(q_plus)
        if not spans:
            continue
        subject = spans[0]
        s_len = len(token_ids(tokenizer, subject))
        candidates = [n for n in donor_names if n != subject and len(token_ids(tokenizer, n)) == s_len]
        rng.shuffle(candidates)

        ids_plus = token_ids(tokenizer, q_plus)
        for donor in candidates:
            q_minus = q_plus.replace(subject, donor, 1)
            ids_minus = token_ids(tokenizer, q_minus)
            span = diff_is_single_span(ids_plus, ids_minus)
            if span is None:
                continue
            pairs.append(
                {
                    "x_plus": q_plus,
                    "x_minus": q_minus,
                    "forget_subject": subject,
                    "donor_subject": donor,
                    "token_span": list(span),
                    "answer_plus": rec["answer"],
                }
            )
            break
        if len(pairs) >= max_pairs:
            break
    return pairs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--forget", default="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json")
    ap.add_argument("--retain", default="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json")
    ap.add_argument("--max-pairs", type=int, default=20)
    ap.add_argument("--output", default="tofu_counterfactual_pairs.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pairs = build_pairs(tok, args.forget, args.retain, max_pairs=args.max_pairs)
    Path(args.output).write_text(json.dumps(pairs, indent=2, ensure_ascii=False))
    print(f"built {len(pairs)} valid pairs -> {args.output}")
    for p in pairs[:3]:
        print("\n+", p["x_plus"])
        print("-", p["x_minus"])
        print("  span:", p["token_span"], "| donor:", p["donor_subject"])

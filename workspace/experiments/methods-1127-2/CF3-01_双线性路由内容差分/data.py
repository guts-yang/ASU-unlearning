"""Data and frozen-teacher cache owned exclusively by this method directory."""
from __future__ import annotations

import difflib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from modeling import encode_qa, forward_model, span_hidden

NAME_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,2}\b")
STOP_FIRST = {"The", "A", "An", "This", "That", "Who", "What", "When", "Where", "His", "Her", "Their", "In", "On", "At", "It", "He", "She", "They"}
AUX_FIRST = {"Has", "Have", "Had", "Did", "Do", "Does", "Is", "Was", "Were", "Are", "Can", "Could",
             "Would", "Should", "Will", "May", "Might", "Must", "Shall", "How", "Which", "Why", "Whose", "Whom"}
STOP_FIRST |= AUX_FIRST


def load_records(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # TOFU splits are JSON Lines despite a .json suffix; fail closed on malformed lines.
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    return value if isinstance(value, list) else value["records"]


def synthetic_records(count=12):
    first = ["Alice", "Brian", "Carla", "David", "Elena", "Frank", "Grace", "Henry", "Irene", "James", "Karen", "Louis"]
    last = ["Miller", "Smith", "Brown", "Jones", "Wilson", "Taylor", "Clark", "Lewis", "Walker", "Hall", "Young", "Wright"]
    cities = ["Paris", "London", "Berlin", "Madrid", "Vienna", "Dublin", "Prague", "Lisbon", "Warsaw", "Athens", "Dublin", "Rome"]
    jobs = ["writer", "painter", "singer", "architect", "scientist", "teacher", "doctor", "lawyer", "engineer", "poet", "editor", "dancer"]
    records = []
    for i in range(count * 2):
        name = f"{first[i % len(first)]} {last[i % len(last)]}"
        city = cities[i % len(cities)]
        job = jobs[i % len(jobs)]
        records.extend([
            {"question": f"Where was {name} born?", "answer": f"{name} was born in {city}.", "author": name},
            {"question": f"What is {name}'s occupation?", "answer": f"{name} works as a {job}.", "author": name},
        ])
    return records


def subject_spans(question):
    """Capitalized multi-word spans with leading auxiliaries/determiners stripped.

    "Has Hsiao Yun-Hwa's mother..." must yield "Hsiao Yun-Hwa", not the verb
    phrase "Has Hsiao Yun-Hwa".  Titles such as "Old World Series" can still
    appear here; canonical_subject()/author_vocab() filter them by data-driven
    author vocabulary before a pair is built.
    """
    names = []
    for match in NAME_RE.finditer(question):
        words = match.group(0).split()
        while words and words[0] in STOP_FIRST:
            words.pop(0)
        name = re.sub(r"'s$", "", " ".join(words)).strip()
        if len(name.split()) >= 2 and name not in names:
            names.append(name)
    return names


def canonical_subject(question, answer, vocab=None):
    """Return the author name that is actually present in the question.

    With a vocabulary, only vocabulary names are accepted: this drops titles
    ("Old World Series") and verb phrases.  Without a vocabulary, prefer a span
    that also appears in the answer, then the longest question span.
    """
    qspans = subject_spans(question)
    if not qspans:
        return None
    if vocab is not None:
        hits = [x for x in qspans if x in vocab]
        return max(hits, key=len) if hits else None
    aspans = subject_spans(answer or "")
    inter = [x for x in qspans if x in aspans]
    if inter:
        return max(inter, key=len)
    return max(qspans, key=len)


def author_vocab(records, n_authors=None, block_size=None):
    """Infer the author vocabulary from an ordered TOFU split.

    TOFU splits are grouped by author (about 20 records per author).  Taking the
    most frequent name in each contiguous block gives the author list without
    relying on a fragile "first capitalized phrase" heuristic.
    """
    if not records:
        return set()
    if n_authors is None:
        n_authors = max(1, int(round(len(records) / 20.0)))
    if block_size is None:
        block_size = max(1, len(records) // max(1, n_authors))
    vocab = set()
    for start in range(0, len(records), block_size):
        chunk = records[start:start + block_size]
        counts = Counter()
        for rec in chunk:
            for span in subject_spans(rec.get("question", "")) + subject_spans(rec.get("answer", "")):
                counts[span] += 1
        if counts:
            vocab.add(max(counts, key=lambda x: (counts[x], len(x))))
    return vocab


def _name_key(name):
    return re.sub(r"[^a-z]", "", (name or "").lower())


def _is_alias(a, b):
    """Conservative alias check (e.g. Jaime/Jamie Vasquez) for donor isolation."""
    ka, kb = _name_key(a), _name_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    return difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.88


def _single_token_span(ids_a, ids_b):
    if len(ids_a) != len(ids_b):
        return None
    changed = [i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if a != b]
    if not changed:
        return None
    return [changed[0], changed[-1] + 1]


def _relation_key(question, subject):
    return re.sub(r"\s+", " ", question.replace(subject, "<SUBJECT>", 1)).strip()


def _donor_records(retain):
    """Canonical donor-author -> a retain record whose answer is y_j^-.

    Retain questions often omit the author name ("Who is this celebrated ..."),
    so fall back to a name from the answer that is in the retain author vocab.
    """
    vocab = author_vocab(retain, n_authors=max(1, len(retain) // 18))
    donors = {}
    for rec in retain:
        name = canonical_subject(rec.get("question", ""), rec.get("answer", ""), vocab)
        if name is None:
            ans = [x for x in subject_spans(rec.get("answer", "")) if x in vocab]
            name = max(ans, key=len) if ans else None
        if name and name not in donors:
            donors[name] = rec
    return donors


def _attempt_pair(tok, source, subject, donor, donor_rec, max_len, require_control, by_author):
    """Build ONE pair only after both subject slots align token-for-token."""
    question = source["question"]
    question_minus = question.replace(subject, donor, 1)
    if question_minus == question:
        return None, "subject_not_replaceable"
    source_ids = tok.encode(question, add_special_tokens=False)
    minus_ids = tok.encode(question_minus, add_special_tokens=False)
    span = _single_token_span(source_ids, minus_ids)
    if span is None:
        return None, "question_token_length_or_span_mismatch"
    enc_p = encode_qa(tok, question, source["answer"], max_len)
    enc_m = encode_qa(tok, question_minus, source["answer"], max_len)
    try:
        subject_plus = _subject_indices(tok, enc_p, question, subject)
        subject_minus = _subject_indices(tok, enc_m, question_minus, donor)
    except Exception as exc:
        return None, f"subject_alignment_error:{type(exc).__name__}"
    if subject_plus != subject_minus:
        return None, "subject_token_positions_differ"

    answer_minus = donor_rec.get("answer", "") if donor_rec else ""
    if not answer_minus:
        return None, "donor_answer_missing"
    pair = {
        "x_plus": question, "x_minus": question_minus,
        "answer_plus": source["answer"], "answer_minus": answer_minus,
        "forget_subject": subject, "donor_subject": donor,
        "subject_span": span, "relation": _relation_key(question, subject),
        "author": subject,
    }
    if require_control:
        control = None
        for other in by_author.get(subject, []):
            if other["question"] == question:
                continue
            control_minus = other["question"].replace(subject, donor, 1)
            if control_minus == other["question"]:
                continue
            if _single_token_span(tok.encode(other["question"], add_special_tokens=False),
                                  tok.encode(control_minus, add_special_tokens=False)) is None:
                continue
            cenc_p = encode_qa(tok, other["question"], other["answer"], max_len)
            cenc_m = encode_qa(tok, control_minus, other["answer"], max_len)
            try:
                csp = _subject_indices(tok, cenc_p, other["question"], subject)
                csm = _subject_indices(tok, cenc_m, control_minus, donor)
            except Exception:
                continue
            if csp == csm:
                control = {"control_plus": other["question"], "control_minus": control_minus,
                           "control_answer": other["answer"]}
                break
        if control is None:
            return None, "matched_retain_missing"
        pair.update(control)
    return pair, None


def _build_pairs_common(tok, forget, retain, max_pairs, seed, require_control,
                        per_author=None, n_authors=None, donor_tries=8, max_len=512):
    vocab = author_vocab(forget, n_authors=n_authors) if n_authors else None
    by_author = defaultdict(list)
    unresolved = []
    for rec in forget:
        subject = canonical_subject(rec.get("question", ""), rec.get("answer", ""), vocab)
        if subject is None:
            unresolved.append({"question": rec.get("question", "")[:160], "reason": "no_canonical_subject"})
            continue
        by_author[subject].append(rec)

    donors = _donor_records(retain)
    donor_names = sorted(donors)
    rng = random.Random(seed)
    pairs, drops, per_author_counts = [], [], {}
    attempts = 0

    def try_source(source, subject):
        nonlocal attempts
        candidates = [d for d in donor_names if d != subject and not _is_alias(d, subject)]
        rng.shuffle(candidates)
        last_reason = "no_donor_candidate"
        for donor in candidates[:donor_tries]:
            attempts += 1
            pair, reason = _attempt_pair(tok, source, subject, donor, donors.get(donor),
                                         max_len, require_control, by_author)
            if pair is not None:
                return pair, None
            last_reason = reason
        return None, last_reason

    if per_author is None:
        for source in forget:
            subject = canonical_subject(source.get("question", ""), source.get("answer", ""), vocab)
            if subject is None:
                continue
            pair, reason = try_source(source, subject)
            if pair is None:
                drops.append({"question": source.get("question", "")[:160], "author": subject,
                              "reason": reason})
                continue
            pairs.append(pair)
            if len(pairs) >= max_pairs:
                break
    else:
        for subject in sorted(by_author):
            recs = by_author[subject][:]
            rng.shuffle(recs)
            kept = 0
            for source in recs:
                if kept >= per_author:
                    break
                pair, reason = try_source(source, subject)
                if pair is None:
                    drops.append({"question": source.get("question", "")[:160], "author": subject,
                                  "reason": reason})
                    continue
                pairs.append(pair)
                kept += 1
                if len(pairs) >= max_pairs:
                    break
            per_author_counts[subject] = kept
            if len(pairs) >= max_pairs:
                break

    report = {
        "generated": len(forget), "kept": len(pairs), "requested_max": max_pairs,
        "per_author_requested": per_author, "n_authors_detected": len(by_author),
        "authors": sorted(by_author), "per_author_kept": per_author_counts,
        "donors": len(donor_names), "attempts": attempts,
        "unresolved_subjects": len(unresolved), "unresolved_examples": unresolved[:5],
        "drops": drops[:40], "drop_count": len(drops),
        "checks": ["canonical author vocabulary", "donor author != forget author",
                   "alias-cluster exclusion", "single-token-span replacement",
                   "subject token positions equal", "matched-retain control"],
        "note": "No pair is fabricated; fail-closed cache checks remain unchanged.",
    }
    return pairs, report


def build_pairs(tok, forget, retain, max_pairs=32, seed=0, require_control=False):
    """Default (non-stratified) builder, kept for tiny/synthetic and legacy use.

    Subject extraction and alignment are now canonical/pre-filtered so the
    fail-closed cache check is not the first place a bad pair is discovered.
    """
    n_authors = max(1, int(round(len(forget) / 20.0))) if len(forget) >= 100 else None
    pairs, _ = _build_pairs_common(tok, forget, retain, max_pairs, seed, require_control,
                                   per_author=None, n_authors=n_authors)
    if not pairs:
        raise RuntimeError("no valid equal-length single-subject counterfactual pairs; provide explicit pairs_file")
    return pairs


def build_pairs_stratified(tok, forget, retain, per_author=2, seed=0, max_pairs=64,
                           n_authors=None, require_control=True, donor_tries=8, max_len=512):
    """Stratified counterfactual pairs covering every detected author cluster."""
    return _build_pairs_common(tok, forget, retain, max_pairs, seed, require_control,
                               per_author=per_author, n_authors=n_authors,
                               donor_tries=donor_tries, max_len=max_len)


def _pool(full, indices):
    return full[list(indices)].mean(0) if indices else full.mean(0)


def _cpu(value, half=False):
    value = value.detach().cpu()
    return value.half() if half else value.float()


def _subject_indices(tok, enc, question, subject):
    prompt = f"Question: {question}\nAnswer:"
    char_start = question.find(subject)
    if char_start < 0:
        raise ValueError(f"subject {subject!r} is not present in question")
    char_start += len("Question: ")
    char_end = char_start + len(subject)
    if getattr(tok, "is_fast", False):
        encoded = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded["input_ids"]) != enc["input_ids"][:enc["prompt_length"]].tolist():
            raise ValueError("tokenizer offsets do not align with encoded QA prompt")
        indices = [i for i, (start, end) in enumerate(encoded["offset_mapping"])
                   if end > start and start < char_end and end > char_start]
        if indices:
            return indices
    ids = tok.encode(question, add_special_tokens=False)
    source = tok.encode(subject, add_special_tokens=False)
    prefix = len(tok.encode("Question: ", add_special_tokens=False))
    for start in range(len(ids) - len(source) + 1):
        if ids[start:start + len(source)] == source:
            indices = list(range(prefix + start, prefix + start + len(source)))
            if enc["input_ids"][indices].tolist() == source:
                return indices
    words = question.split()
    target_words = subject.split()
    for start in range(len(words) - len(target_words) + 1):
        candidate_words = words[start:start + len(target_words)]
        normalized = [re.sub(r"'s$", "", word) for word in candidate_words]
        if normalized != target_words:
            continue
        candidate_ids = tok.encode(" ".join(candidate_words), add_special_tokens=False)
        indices = list(range(prefix + start, prefix + start + len(candidate_ids)))
        if enc["input_ids"][indices].tolist() == candidate_ids:
            return indices
    raise ValueError("could not align subject tokens inside the QA prompt; provide a fast tokenizer or explicit aligned data")


@torch.no_grad()
def build_teacher_cache(teacher, tok, pairs, retain, cfg, device, capture_bilinear=False, capture_glu=False):
    layers = set(cfg["layers"])
    cache = []
    max_len = int(cfg["data"]["max_len"])
    for index, pair in enumerate(pairs):
        enc_p = encode_qa(tok, pair["x_plus"], pair["answer_plus"], max_len)
        enc_m = encode_qa(tok, pair["x_minus"], pair["answer_plus"], max_len)
        fp = forward_model(teacher, enc_p, device, layers, capture_bilinear=capture_bilinear, capture_glu=capture_glu)
        fm = forward_model(teacher, enc_m, device, layers, capture_bilinear=capture_bilinear, capture_glu=capture_glu)
        subject = _subject_indices(tok, enc_p, pair["x_plus"], pair["forget_subject"])
        subject_minus = _subject_indices(tok, enc_m, pair["x_minus"], pair["donor_subject"])
        if subject != subject_minus:
            raise ValueError(f"counterfactual subject token positions differ for pair {index}")
        prompt = list(range(enc_p["prompt_length"]))
        relation = [i for i in prompt if i not in subject]
        item = {
            "index": index, "pair": pair, "enc_plus": enc_p, "enc_minus": enc_m,
            "h_plus": {}, "h_minus": {}, "subject_plus": {}, "subject_minus": {},
            "relation_plus": {}, "relation_minus": {}, "full_plus": {}, "full_minus": {},
            "attention_plus": {}, "attention_minus": {}, "glu_plus": {}, "glu_minus": {},
            "attention_control_plus": {}, "attention_control_minus": {},
            "glu_control_plus": {}, "glu_control_minus": {},
            "teacher_nll": fp["nll"].detach().cpu(), "subject_indices": subject,
        }
        eos_id = int(getattr(tok, "eos_token_id", 2) or 2)
        eos_span = fp["logits"][enc_p["answer_start"]:max(enc_p["answer_start"], enc_p["answer_end"] - 1)].float()
        if eos_span.numel() and eos_id < eos_span.size(-1):
            without = eos_span.clone()
            without[:, eos_id] = -torch.inf
            item["teacher_eos_margin"] = float((eos_span[:, eos_id] - without.max(dim=-1).values).mean())
        else:
            item["teacher_eos_margin"] = 0.0
        embeddings = teacher.get_input_embeddings()
        plus_ids = enc_p["input_ids"].to(device)
        minus_ids = enc_m["input_ids"].to(device)
        item["subject_delta_norm"] = (embeddings(plus_ids)[subject] - embeddings(minus_ids)[subject]).norm().detach().float().cpu()
        for layer in layers:
            hp, hm = fp["hidden"][layer], fm["hidden"][layer]
            item["h_plus"][layer] = _cpu(span_hidden(fp, layer))
            item["h_minus"][layer] = _cpu(span_hidden(fm, layer))
            item["subject_plus"][layer] = _cpu(_pool(hp, subject))
            item["subject_minus"][layer] = _cpu(_pool(hm, subject))
            item["relation_plus"][layer] = _cpu(_pool(hp, relation))
            item["relation_minus"][layer] = _cpu(_pool(hm, relation))
            if cfg["data"].get("cache_full_sequence", False):
                item["full_plus"][layer] = _cpu(hp, half=True)
                item["full_minus"][layer] = _cpu(hm, half=True)
            if capture_bilinear:
                for target, source in ((item["attention_plus"], fp), (item["attention_minus"], fm)):
                    target[layer] = {"pi": _cpu(source["attention"][layer]["pi"], half=True),
                                     "v": _cpu(source["attention"][layer]["v"], half=True)}
            if capture_glu:
                item["glu_plus"][layer] = {k: _cpu(v) for k, v in fp["glu"][layer].items()}
                item["glu_minus"][layer] = {k: _cpu(v) for k, v in fm["glu"][layer].items()}
        if "control_plus" in pair:
            enc_cp = encode_qa(tok, pair["control_plus"], pair["answer_plus"], max_len)
            enc_cm = encode_qa(tok, pair["control_minus"], pair["answer_plus"], max_len)
            fcp = forward_model(teacher, enc_cp, device, layers,
                                capture_bilinear=capture_bilinear, capture_glu=capture_glu)
            fcm = forward_model(teacher, enc_cm, device, layers,
                                capture_bilinear=capture_bilinear, capture_glu=capture_glu)
            item.update({"enc_control_plus": enc_cp, "enc_control_minus": enc_cm,
                         "h_control_plus": {}, "h_control_minus": {}, "control_shared_nll": fcp["nll"].detach().cpu()})
            for layer in layers:
                item["h_control_plus"][layer] = _cpu(span_hidden(fcp, layer))
                item["h_control_minus"][layer] = _cpu(span_hidden(fcm, layer))
                if capture_bilinear:
                    for target, source in ((item["attention_control_plus"], fcp),
                                           (item["attention_control_minus"], fcm)):
                        target[layer] = {"pi": _cpu(source["attention"][layer]["pi"], half=True),
                                         "v": _cpu(source["attention"][layer]["v"], half=True)}
                if capture_glu:
                    item["glu_control_plus"][layer] = {k: _cpu(v) for k, v in fcp["glu"][layer].items()}
                    item["glu_control_minus"][layer] = {k: _cpu(v) for k, v in fcm["glu"][layer].items()}
            natural = encode_qa(tok, pair["control_plus"], pair["control_answer"], max_len)
            item["control_ood_nll"] = forward_model(teacher, natural, device)["nll"].detach().cpu()
        cache.append(item)
    retain_cache = []
    for rec in list(retain)[: int(cfg["data"]["retain_cache_size"])]:
        enc = encode_qa(tok, rec["question"], rec["answer"], max_len)
        out = forward_model(teacher, enc, device, layers)
        names = subject_spans(rec["question"])
        if not names:
            continue
        subject = _subject_indices(tok, enc, rec["question"], names[0])
        values = {layer: _cpu(span_hidden(out, layer).mean(0)) for layer in layers}
        item = {"enc": enc, "pooled": values, "answer": {layer: _cpu(span_hidden(out, layer)) for layer in layers},
                "subject": {layer: _cpu(_pool(out["hidden"][layer], subject)) for layer in layers},
                "teacher_nll": out["nll"].detach().cpu()}
        if cfg["data"].get("cache_full_sequence", False):
            item["full"] = {layer: _cpu(out["hidden"][layer], half=True) for layer in layers}
        retain_cache.append(item)
    if not retain_cache:
        raise RuntimeError("no retain records with alignable subject spans")
    return {"pairs": cache, "retain": retain_cache, "layers": sorted(layers)}


def save_cache(cache, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def load_cache(path):
    return torch.load(path, map_location="cpu", weights_only=False)

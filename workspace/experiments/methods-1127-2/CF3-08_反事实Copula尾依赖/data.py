"""Data and frozen-teacher cache owned exclusively by this method directory."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch

from modeling import encode_qa, forward_model, span_hidden

NAME_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,2}\b")
STOP_FIRST = {"The", "A", "An", "This", "That", "Who", "What", "When", "Where", "His", "Her", "Their", "In", "On", "At", "It", "He", "She", "They"}


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
    names = []
    for match in NAME_RE.finditer(question):
        name = re.sub(r"'s$", "", match.group(0))
        if name.split()[0] not in STOP_FIRST:
            names.append(name)
    return names


def _single_token_span(ids_a, ids_b):
    if len(ids_a) != len(ids_b):
        return None
    changed = [i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if a != b]
    if not changed:
        return None
    return [changed[0], changed[-1] + 1]


def _relation_key(question, subject):
    return re.sub(r"\s+", " ", question.replace(subject, "<SUBJECT>", 1)).strip()


def build_pairs(tok, forget, retain, max_pairs=32, seed=0, require_control=False):
    rng = random.Random(seed)
    donors = sorted({name for rec in retain for name in subject_spans(rec["question"])})
    donor_records = {}
    for rec in retain:
        for name in subject_spans(rec["question"]):
            donor_records.setdefault(name, rec)
    by_author = {}
    for rec in forget:
        key = rec.get("author") or (subject_spans(rec["question"]) or [""])[0]
        by_author.setdefault(key, []).append(rec)
    pairs = []
    for source in forget:
        names = subject_spans(source["question"])
        if not names:
            continue
        subject = names[0]
        source_ids = tok.encode(source["question"], add_special_tokens=False)
        candidates = list(donors)
        rng.shuffle(candidates)
        for donor in candidates:
            if donor == subject:
                continue
            question_minus = source["question"].replace(subject, donor, 1)
            minus_ids = tok.encode(question_minus, add_special_tokens=False)
            span = _single_token_span(source_ids, minus_ids)
            if span is None:
                continue
            donor_rec = donor_records.get(donor, {})
            if donor_rec.get("author") is not None and donor_rec.get("author") == source.get("author"):
                continue
            pair = {
                "x_plus": source["question"], "x_minus": question_minus,
                "answer_plus": source["answer"], "answer_minus": donor_rec.get("answer", ""),
                "forget_subject": subject, "donor_subject": donor,
                "subject_span": span, "relation": _relation_key(source["question"], subject),
                "author": source.get("author", subject),
            }
            if require_control:
                controls = [r for r in by_author.get(source.get("author", subject), []) if r["question"] != source["question"]]
                if not controls:
                    continue
                control = controls[0]
                control_minus = control["question"].replace(subject, donor, 1)
                if _single_token_span(tok.encode(control["question"], add_special_tokens=False), tok.encode(control_minus, add_special_tokens=False)) is None:
                    continue
                pair.update({"control_plus": control["question"], "control_minus": control_minus, "control_answer": control["answer"]})
            pairs.append(pair)
            break
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        raise RuntimeError("no valid equal-length single-subject counterfactual pairs; provide explicit pairs_file")
    return pairs


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
            "teacher_nll": fp["nll"].detach().cpu(), "subject_indices": subject,
        }
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
            fcp = forward_model(teacher, enc_cp, device, layers)
            fcm = forward_model(teacher, enc_cm, device, layers)
            item.update({"enc_control_plus": enc_cp, "enc_control_minus": enc_cm,
                         "h_control_plus": {}, "h_control_minus": {}, "control_shared_nll": fcp["nll"].detach().cpu()})
            for layer in layers:
                item["h_control_plus"][layer] = _cpu(span_hidden(fcp, layer))
                item["h_control_minus"][layer] = _cpu(span_hidden(fcm, layer))
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

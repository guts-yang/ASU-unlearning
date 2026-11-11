#!/usr/bin/env python3
"""Fixed-prefix preference diagnostic for prompt-only activation donors.

M compares the full city span of the target birthplace with the paired donor
city under one shared answer prefix. Free generation is classified separately
and is not a nomination gate. This run does not modify the earlier localization
directory, TOFU files, my_llama.py, or the ASU trainer, and it does not train.
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

if not str(os.environ.get("OMP_NUM_THREADS", "")).isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import torch

REPO = Path("/usr/local/ASU-unlearning")
sys.path.insert(0, str(REPO / "Right-to-be-forgotten"))
sys.path.insert(0, str(REPO / "workspace/data/external_donor_prompt_activation_v0.3_localization"))
import run_localization as loc  # noqa: E402
from patch_editor import ResidualEditor  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

OUT_DIR = REPO / "workspace/data/external_donor_prompt_activation_v0.3_preference"
if OUT_DIR.resolve() == loc.OUT_DIR.resolve():
    raise RuntimeError("preference output must stay outside the earlier localization directory")


def read_protocol() -> dict:
    return json.loads((OUT_DIR / "protocol.json").read_text(encoding="utf-8"))


def canonical_name(author_span: str, protocol: dict) -> str:
    name = author_span
    for prefix in protocol["answer_prefix"]["strip_prefixes_longest_first"]:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def phrase_present(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def classify_generation(text: str, hit_limit: bool, target: str, protocol: dict) -> dict:
    aliases = list(protocol["generation"]["aliases"].get(target, []))
    target_hit = any(phrase_present(text, phrase) for phrase in [target, *aliases])
    others = [
        city
        for city in protocol["generation"]["closed_cities"]
        if city != target and city not in aliases and phrase_present(text, city)
    ]
    if target_hit and others:
        label = "human_review"
    elif target_hit:
        label = "target_correct"
    elif others:
        label = "wrong_fact"
    elif any(phrase in text for phrase in protocol["generation"]["refusal_phrases"]):
        label = "refusal"
    elif hit_limit:
        label = "truncated"
    else:
        label = "not_mentioned"
    return {"label": label, "target_hit": target_hit, "other_cities": others}


def encode_city(tokenizer, prompt: str, prefix: str, city: str) -> dict:
    text = prompt + prefix + city
    encoding = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    ids = list(encoding["input_ids"])
    offsets = [tuple(pair) for pair in encoding["offset_mapping"]]
    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if ids[: len(prompt_ids)] != list(prompt_ids):
        raise RuntimeError(f"{city!r} changes the question tokens")
    city_start = len(prompt + prefix)
    city_end = city_start + len(city)
    span = []
    for index, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            continue
        if end > city_start and start < city_end:
            span.append(index)
    if not span or span != list(range(span[0], span[-1] + 1)):
        raise RuntimeError(f"{city!r} city span is empty or not contiguous")
    covered = [False] * len(city)
    for index in span:
        start, end = offsets[index]
        for position in range(max(start, city_start), min(end, city_end)):
            covered[position - city_start] = True
    if not all(covered):
        raise RuntimeError(f"{city!r} has uncovered city characters")
    start = span[0]
    span_ids = [ids[index] for index in span]
    if span_ids != ids[start:]:
        raise RuntimeError(f"{city!r} span does not continue through the sequence end")
    return {
        "ids": ids,
        "span_start": start,
        "span_ids": span_ids,
        "tokens": tokenizer.convert_ids_to_tokens(span_ids),
    }


def align_cities(target: dict, alternative: dict, label: str) -> None:
    if target["span_start"] != alternative["span_start"]:
        raise RuntimeError(f"{label} city spans start at different indexes")
    start = target["span_start"]
    if target["ids"][:start] != alternative["ids"][:start]:
        raise RuntimeError(f"{label} answer prefixes tokenize differently")
    if target["ids"] == alternative["ids"] or target["ids"][start:] == alternative["ids"][start:]:
        raise RuntimeError(f"{label} city spans are identical")


def span_values(logits: torch.Tensor, span_start: int, span_ids: list[int]) -> list[float]:
    values = []
    for offset, token_id in enumerate(span_ids):
        position = span_start - 1 + offset
        log_prob = torch.log_softmax(logits[position].float(), dim=-1)[token_id]
        values.append(float(log_prob))
    if not values or not all(value == value and abs(value) != float("inf") for value in values):
        raise RuntimeError("city log-prob is empty or non-finite")
    return values


def pack_stats(values: list[float]) -> dict:
    return {"mean": float(sum(values) / len(values)), "sum": float(sum(values)), "count": len(values)}


def preference(plus: dict, minus: dict) -> dict:
    return {
        "avg_plus": plus["mean"],
        "sum_plus": plus["sum"],
        "count_plus": plus["count"],
        "avg_minus": minus["mean"],
        "sum_minus": minus["sum"],
        "count_minus": minus["count"],
        "m": plus["mean"] - minus["mean"],
    }


def city_logit_diff(left: torch.Tensor, right: torch.Tensor, span_start: int, count: int) -> float:
    return loc.answer_logit_diff(left.cpu(), right.cpu(), span_start, count)


def self_check(protocol: dict) -> None:
    loc.self_check()
    loc.reproduce_shuffle(protocol)
    if protocol["answer_prefix"]["template"] != "{name} was born in ":
        raise RuntimeError("answer prefix drifted")
    names = [
        canonical_name(row["author_span"], protocol)
        for row in protocol["retain"]["expected_probes"]
    ]
    if names != protocol["answer_prefix"]["canonical_names"]:
        raise RuntimeError(f"canonical retain names drifted: {names}")
    cases = [
        ("Hina Ameen was born in Karachi, Pakistan.", False, "Karachi", "target_correct"),
        ("She was born in Wellington.", False, "Karachi", "wrong_fact"),
        ("born in Karachi and later in Wellington", False, "Karachi", "human_review"),
        ("born in Peking.", False, "Beijing", "target_correct"),
        ("born in New York.", False, "New York City", "not_mentioned"),
        ("Garyville is not a city match.", False, "Tokyo", "not_mentioned"),
        ("I don't know the answer.", False, "Tokyo", "refusal"),
        ("born in K", True, "Kuwait City", "truncated"),
        ("born in Karachi", True, "Karachi", "target_correct"),
    ]
    for text, hit_limit, target, expected in cases:
        found = classify_generation(text, hit_limit, target, protocol)["label"]
        if found != expected:
            raise RuntimeError(f"classification {text!r} -> {found}, expected {expected}")
    if nominate_preview() != [3, 4, 0]:
        raise RuntimeError("nomination preview drifted")


def nominate_preview() -> list[int]:
    return loc.nominate({0: 0.2, 1: 0.2, 2: -1.0, 3: 0.5, 4: 0.4})


def check_boundaries(tokenizer, protocol: dict, pairs: list[dict], donors: list[dict]) -> list[dict]:
    by_donor = {row["donor_id"]: row["donor_fact_value"] for row in donors}
    retain_rows, _excluded = loc.select_retain(tokenizer, protocol, pairs)
    loc.assert_retain(retain_rows, protocol)
    records = []
    for pair in pairs:
        if pair["donor_fact_value"] != by_donor[pair["donor_id"]]:
            raise RuntimeError(f"{pair['pair_id']} donor city drifted")
        prompt = loc.prompt_text(pair["q_plus"], protocol)
        if pair["donor_fact_value"] in loc.prompt_text(pair["q_minus"], protocol):
            raise RuntimeError(f"{pair['pair_id']} appends the donor city to the donor prompt")
        prefix = protocol["answer_prefix"]["template"].format(name=pair["clean_entity"])
        target = encode_city(tokenizer, prompt, prefix, pair["y_plus"])
        alternative = encode_city(tokenizer, prompt, prefix, pair["donor_fact_value"])
        align_cities(target, alternative, pair["pair_id"])
        records.append(
            {
                "pair_id": pair["pair_id"],
                "name": pair["clean_entity"],
                "target_city": pair["y_plus"],
                "alternative_city": pair["donor_fact_value"],
                "target_tokens": target["tokens"],
                "alternative_tokens": alternative["tokens"],
                "span_start": target["span_start"],
                "target_count": len(target["span_ids"]),
                "alternative_count": len(alternative["span_ids"]),
            }
        )
    for index, row in enumerate(retain_rows):
        name = canonical_name(row["author_span"], protocol)
        if name != protocol["answer_prefix"]["canonical_names"][index]:
            raise RuntimeError(f"retain name {name!r} drifted")
        prompt = loc.prompt_text(row["question"], protocol)
        prefix = protocol["answer_prefix"]["template"].format(name=name)
        target = encode_city(tokenizer, prompt, prefix, row["city"])
        alternative_city = by_donor[row["donor_id"]]
        alternative = encode_city(tokenizer, prompt, prefix, alternative_city)
        align_cities(target, alternative, row["author_span"])
        records.append(
            {
                "pair_id": f"retain-{index + 1:02d}",
                "name": name,
                "target_city": row["city"],
                "alternative_city": alternative_city,
                "target_tokens": target["tokens"],
                "alternative_tokens": alternative["tokens"],
                "span_start": target["span_start"],
                "target_count": len(target["span_ids"]),
                "alternative_count": len(alternative["span_ids"]),
            }
        )
    first = records[0]
    if first["target_count"] < 2 or first["alternative_count"] < 2:
        raise RuntimeError("Hina Ameen cities were truncated to a first token")
    if "achi" not in "".join(first["target_tokens"]) or "ington" not in "".join(first["alternative_tokens"]):
        raise RuntimeError(f"Hina city tokens drifted: {first}")
    return records


def build_examples(tokenizer, protocol: dict, pairs: list[dict], donors: list[dict]) -> list[dict]:
    by_donor = {row["donor_id"]: row["donor_fact_value"] for row in donors}
    retain_rows, excluded = loc.select_retain(tokenizer, protocol, pairs)
    loc.assert_retain(retain_rows, protocol)
    examples = []
    for pair in pairs:
        prompt = loc.prompt_text(pair["q_plus"], protocol)
        donor_prompt = loc.prompt_text(pair["q_minus"], protocol)
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        donor_ids = tokenizer(donor_prompt, add_special_tokens=True)["input_ids"]
        prefix = protocol["answer_prefix"]["template"].format(name=pair["clean_entity"])
        target = encode_city(tokenizer, prompt, prefix, pair["y_plus"])
        alternative = encode_city(tokenizer, prompt, prefix, pair["donor_fact_value"])
        align_cities(target, alternative, pair["pair_id"])
        split = "discovery" if pair["pair_id"] in protocol["splits"]["discovery_pairs"] else "confirmation"
        examples.append(
            {
                "group": "forget",
                "split": split,
                "pair_id": pair["pair_id"],
                "pair_index": int(pair["pair_id"][-2:]) - 1,
                "name": pair["clean_entity"],
                "question": pair["q_plus"],
                "city": pair["y_plus"],
                "alternative_city": pair["donor_fact_value"],
                "donor_id": pair["donor_id"],
                "prompt_ids": list(prompt_ids),
                "donor_ids": list(donor_ids),
                "qmark": loc.question_mark_index(tokenizer, prompt_ids),
                "donor_qmark": loc.question_mark_index(tokenizer, donor_ids),
                "prefix": prefix,
                "plus_ids": target["ids"],
                "plus_span_start": target["span_start"],
                "plus_span_ids": target["span_ids"],
                "plus_tokens": target["tokens"],
                "minus_ids": alternative["ids"],
                "minus_span_start": alternative["span_start"],
                "minus_span_ids": alternative["span_ids"],
                "minus_tokens": alternative["tokens"],
            }
        )
    for index, row in enumerate(retain_rows):
        name = canonical_name(row["author_span"], protocol)
        prompt = loc.prompt_text(row["question"], protocol)
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        prefix = protocol["answer_prefix"]["template"].format(name=name)
        alternative_city = by_donor[row["donor_id"]]
        target = encode_city(tokenizer, prompt, prefix, row["city"])
        alternative = encode_city(tokenizer, prompt, prefix, alternative_city)
        align_cities(target, alternative, f"retain-{index + 1:02d}")
        examples.append(
            {
                "group": "retain",
                "split": "retain",
                "pair_id": f"retain-{index + 1:02d}",
                "pair_index": index,
                "name": name,
                "author_span": row["author_span"],
                "line_1based": row["line_1based"],
                "question": row["question"],
                "city": row["city"],
                "alternative_city": alternative_city,
                "donor_id": row["donor_id"],
                "prompt_ids": list(prompt_ids),
                "qmark": loc.question_mark_index(tokenizer, prompt_ids),
                "prefix": prefix,
                "plus_ids": target["ids"],
                "plus_span_start": target["span_start"],
                "plus_span_ids": target["span_ids"],
                "plus_tokens": target["tokens"],
                "minus_ids": alternative["ids"],
                "minus_span_start": alternative["span_start"],
                "minus_span_ids": alternative["span_ids"],
                "minus_tokens": alternative["tokens"],
                "retain_excluded": excluded,
            }
        )
    return examples


def score_sequence(model, editor: ResidualEditor, token_ids: list[int], qmark: int, span_start: int, span_ids: list[int], replacements: dict | None) -> dict:
    if replacements:
        with editor.replace(replacements, qmark):
            logits, _hidden = loc.forward(model, token_ids)
    else:
        editor.close()
        logits, _hidden = loc.forward(model, token_ids)
    packed = pack_stats(span_values(logits, span_start, span_ids))
    del logits
    return packed


def score_preference(model, editor: ResidualEditor, row: dict, plus_replacements: dict | None, minus_replacements: dict | None) -> dict:
    plus = score_sequence(
        model, editor, row["plus_ids"], row["qmark"], row["plus_span_start"], row["plus_span_ids"], plus_replacements
    )
    minus = score_sequence(
        model, editor, row["minus_ids"], row["qmark"], row["minus_span_start"], row["minus_span_ids"], minus_replacements
    )
    return preference(plus, minus)


def generate(model, editor: ResidualEditor, tokenizer, row: dict, steps: int, replacements: dict | None) -> tuple[str, bool]:
    editor.close()
    if replacements:
        editor.replace(replacements, row["qmark"])
    ids = list(row["prompt_ids"])
    hit_limit = True
    try:
        for _ in range(steps):
            logits, _hidden = loc.forward(model, ids)
            next_id = int(logits[-1].argmax().item())
            del logits
            ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                hit_limit = False
                break
    finally:
        editor.close()
    return tokenizer.decode(ids[len(row["prompt_ids"]) :], skip_special_tokens=True), hit_limit


def generation_record(model, editor, tokenizer, protocol: dict, row: dict, condition: str, replacements: dict | None) -> dict:
    text, hit_limit = generate(
        model, editor, tokenizer, row, int(protocol["generation"]["max_new_tokens"]), replacements
    )
    judged = classify_generation(text, hit_limit, row["city"], protocol)
    return {
        "pair_id": row["pair_id"],
        "group": row["group"],
        "split": row["split"],
        "condition": condition,
        "target_city": row["city"],
        "text": text,
        "hit_limit": hit_limit,
        **judged,
    }


def noise_replacements(row: dict, donor_site: torch.Tensor, recipient_site: torch.Tensor, layers: list[int], seed: int) -> dict:
    built = {}
    for layer in layers:
        mixed = loc.noise_seed(row["group"], row["pair_index"], layer, seed)
        null, _scale = loc.matched_noise(recipient_site[layer], donor_site[layer], mixed)
        built[layer] = null
    return built


def same_replacements(site: torch.Tensor, layers: list[int]) -> dict:
    return {layer: site[layer] for layer in layers}


def replacements_for(row: dict, donor_site: torch.Tensor, recipient_site: torch.Tensor, layers: list[int], kind: str, seed: int | None = None, shuffle_site: torch.Tensor | None = None) -> dict:
    if kind == "donor":
        return {layer: donor_site[layer] for layer in layers}
    if kind == "shuffle":
        if shuffle_site is None:
            raise RuntimeError("shuffle activation is missing")
        return {layer: shuffle_site[layer] for layer in layers}
    if seed is None:
        raise RuntimeError("noise seed is missing")
    built = {}
    for layer in layers:
        mixed = loc.noise_seed(row["group"], row["pair_index"], layer, seed)
        null, _scale = loc.matched_noise(recipient_site[layer], donor_site[layer], mixed)
        built[layer] = null
    return built


def capture_sites(model, editor: ResidualEditor, token_ids: list[int], qmark: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, hidden = loc.forward(model, token_ids)
    with editor.capture(qmark):
        hooked, _hidden = loc.forward(model, token_ids)
    captured = {key: value.cpu() for key, value in editor.captured.items()}
    sites = loc.site_vectors(hidden.cpu(), qmark, captured)
    return logits, hooked, sites


def measure_base(model, editor: ResidualEditor, row: dict) -> dict:
    plus_logits, plus_hooked, plus_sites = capture_sites(model, editor, row["plus_ids"], row["qmark"])
    minus_logits, minus_hooked, minus_sites = capture_sites(model, editor, row["minus_ids"], row["qmark"])
    scored = preference(
        pack_stats(span_values(plus_logits, row["plus_span_start"], row["plus_span_ids"])),
        pack_stats(span_values(minus_logits, row["minus_span_start"], row["minus_span_ids"])),
    )
    hook_diff = max(
        city_logit_diff(plus_logits, plus_hooked, row["plus_span_start"], len(row["plus_span_ids"])),
        city_logit_diff(minus_logits, minus_hooked, row["minus_span_start"], len(row["minus_span_ids"])),
    )
    site_diff = loc.max_abs(plus_sites, minus_sites)
    return {
        "scored": scored,
        "sites_plus": plus_sites,
        "sites_minus": minus_sites,
        "plus_logits": plus_logits.cpu(),
        "minus_logits": minus_logits.cpu(),
        "hook_diff": hook_diff,
        "site_diff": site_diff,
    }


def correctness_row(model, editor: ResidualEditor, row: dict, base: dict) -> dict:
    plus_sites = base["sites_plus"]
    minus_sites = base["sites_minus"]
    checks = {}
    error = None
    modes = ("noop", "self_layer0", "self_all")
    try:
        for name in modes:
            if name == "noop":
                with editor.rewrite_same_values(loc.PATCH_LAYERS, row["qmark"]):
                    plus_logits, _hidden = loc.forward(model, row["plus_ids"])
                with editor.rewrite_same_values(loc.PATCH_LAYERS, row["qmark"]):
                    minus_logits, _hidden = loc.forward(model, row["minus_ids"])
            else:
                layers = [0] if name == "self_layer0" else loc.PATCH_LAYERS
                with editor.replace(same_replacements(plus_sites, layers), row["qmark"]):
                    plus_logits, _hidden = loc.forward(model, row["plus_ids"])
                with editor.replace(same_replacements(minus_sites, layers), row["qmark"]):
                    minus_logits, _hidden = loc.forward(model, row["minus_ids"])
            checks[name] = max(
                city_logit_diff(base["plus_logits"], plus_logits, row["plus_span_start"], len(row["plus_span_ids"])),
                city_logit_diff(base["minus_logits"], minus_logits, row["minus_span_start"], len(row["minus_span_ids"])),
            )
        plus_logits, plus_hidden = loc.forward(model, row["plus_ids"])
        del plus_logits
        downstream = loc.downstream_changed(model, editor, row["plus_ids"], row["qmark"], plus_hidden)
    except RuntimeError as exc:
        error = str(exc)
        downstream = None
    return {
        "pair_id": row["pair_id"],
        "hook_diff": base["hook_diff"],
        "target_vs_alternative_site_diff": base["site_diff"],
        "noop_city_logit_max_abs": checks.get("noop"),
        "self_patch_layer0_city_logit_max_abs": checks.get("self_layer0"),
        "self_patch_all_layers_city_logit_max_abs": checks.get("self_all"),
        "downstream_input_max_abs": downstream,
        "error": error,
    }


def read_donors(model, editor: ResidualEditor, rows: list[dict]) -> tuple[list[dict], dict]:
    records = []
    donor_sites = {}
    for row in rows:
        print(f"read {row['pair_id']}", flush=True)
        _logits, hidden = loc.forward(model, row["donor_ids"])
        with editor.capture(row["donor_qmark"]):
            _hooked, hook_hidden = loc.forward(model, row["donor_ids"])
        captured = {key: value.cpu() for key, value in editor.captured.items()}
        hidden = hidden.cpu()
        hook_hidden = hook_hidden.cpu()
        cached = loc.load_file(str(loc.ACTIVATION_DIR / f"{row['pair_id']}.safetensors"))
        cache_hidden = cached["hidden_q_minus"]
        cache_error = None
        cache_diff = None
        if tuple(cache_hidden.shape) != tuple(hidden.shape):
            cache_error = f"shape {tuple(cache_hidden.shape)} != {tuple(hidden.shape)}"
        else:
            cache_diff = loc.max_abs(hidden[:, row["donor_qmark"], :], cache_hidden[:, row["donor_qmark"], :])
        try:
            sites = loc.site_vectors(hidden, row["donor_qmark"], captured)
            site_error = None
        except RuntimeError as exc:
            sites = None
            site_error = str(exc)
        if sites is not None:
            donor_sites[row["donor_id"]] = sites
        records.append(
            {
                "pair_id": row["pair_id"],
                "qmark": row["qmark"],
                "donor_qmark": row["donor_qmark"],
                "cache_question_mark_max_abs": cache_diff,
                "cache_error": cache_error,
                "hooked_vs_clean_hidden_max_abs": loc.max_abs(hidden, hook_hidden),
                "site_error": site_error,
                "finite": bool(torch.isfinite(hidden.float()).all().item()) if sites is not None else False,
            }
        )
    return records, donor_sites


def effect_record(row: dict, before: dict, after: dict) -> dict:
    return {
        "m_before": before["m"],
        "m_after": after["m"],
        "delta_m": before["m"] - after["m"],
        "avg_plus_before": before["avg_plus"],
        "avg_minus_before": before["avg_minus"],
        "avg_plus_after": after["avg_plus"],
        "avg_minus_after": after["avg_minus"],
        "sum_plus_before": before["sum_plus"],
        "sum_minus_before": before["sum_minus"],
        "sum_plus_after": after["sum_plus"],
        "sum_minus_after": after["sum_minus"],
        "delta_avg_plus": before["avg_plus"] - after["avg_plus"],
        "delta_avg_minus": before["avg_minus"] - after["avg_minus"],
    }


def scan_layer(model, editor, row: dict, layer: int, donor_site: torch.Tensor, seeds: list[int]) -> dict:
    donor_patch = same_replacements(donor_site, [layer])
    donor = score_preference(model, editor, row, donor_patch, donor_patch)
    noise = []
    scale = None
    for seed in seeds:
        plus_patch = noise_replacements(row, donor_site, row["sites_plus"], [layer], seed)
        minus_patch = noise_replacements(row, donor_site, row["sites_minus"], [layer], seed)
        scale = loc.matched_noise(row["sites_plus"][layer], donor_site[layer], loc.noise_seed(row["group"], row["pair_index"], layer, seed))[1]
        noise.append(score_preference(model, editor, row, plus_patch, minus_patch))
    before = row["baseline"]
    donor_effect = effect_record(row, before, donor)
    noise_deltas = [before["m"] - item["m"] for item in noise]
    return {
        "pair_id": row["pair_id"],
        "split": row["split"],
        "layer": layer,
        "signal_pass": row["signal_pass"],
        "alternative_city": row["alternative_city"],
        **donor_effect,
        "noise_m": [item["m"] for item in noise],
        "delta_m_noise_mean": loc.mean(noise_deltas),
        "donor_specific": donor_effect["delta_m"] - loc.mean(noise_deltas),
        "replacement_l2": scale,
    }


def joint_record(model, editor, row: dict, layers: list[int], donor_site: torch.Tensor, seeds: list[int], shuffle_site: torch.Tensor) -> dict:
    donor_patch = same_replacements(donor_site, layers)
    donor = score_preference(model, editor, row, donor_patch, donor_patch)
    noise = []
    for seed in seeds:
        noise.append(
            score_preference(
                model,
                editor,
                row,
                noise_replacements(row, donor_site, row["sites_plus"], layers, seed),
                noise_replacements(row, donor_site, row["sites_minus"], layers, seed),
            )
        )
    shuffle_patch = same_replacements(shuffle_site, layers)
    shuffled = score_preference(model, editor, row, shuffle_patch, shuffle_patch)
    before = row["baseline"]
    donor_effect = effect_record(row, before, donor)
    noise_deltas = [before["m"] - item["m"] for item in noise]
    shuffle_effect = effect_record(row, before, shuffled)
    return {
        "group": row["group"],
        "pair_id": row["pair_id"],
        "split": row["split"],
        "city": row["city"],
        "alternative_city": row["alternative_city"],
        "donor_id": row["donor_id"],
        "signal_pass": row["signal_pass"],
        "layers": layers,
        "noise_m": [item["m"] for item in noise],
        "delta_m_noise_mean": loc.mean(noise_deltas),
        "donor_specific": donor_effect["delta_m"] - loc.mean(noise_deltas),
        "shuffle_delta_m": shuffle_effect["delta_m"],
        "shuffle_m": shuffled["m"],
        **donor_effect,
        "author_span": row.get("author_span"),
        "line_1based": row.get("line_1based"),
    }


def claim_status(forget_rows: list[dict], retain_rows: list[dict], layers: list[int]) -> dict:
    if not layers:
        return {"status": "FAIL", "reason": "empty_S"}
    positive = sum(row["signal_pass"] and row["donor_specific"] > 0 for row in forget_rows)
    eligible_forget = [row for row in forget_rows if row["signal_pass"]]
    eligible_retain = [row for row in retain_rows if row["signal_pass"]]
    forget_median = loc.median([row["donor_specific"] for row in eligible_forget])
    retain_mean = loc.mean([row["donor_specific"] for row in eligible_retain])
    did_values = []
    if retain_mean is not None:
        did_values = [row["donor_specific"] - retain_mean for row in eligible_forget]
    did_median = loc.median(did_values)

    def four(rows: list[dict], prefix: str) -> dict:
        return {
            f"{prefix}_donor": loc.mean([row["delta_m"] for row in rows]),
            f"{prefix}_noise": loc.mean([row["delta_m_noise_mean"] for row in rows]),
        }

    eligible = {**four(eligible_forget, "forget"), **four(eligible_retain, "retain")}
    everyone = {**four(forget_rows, "forget"), **four(retain_rows, "retain")}
    passed = (
        positive >= 3
        and forget_median is not None
        and forget_median > 0
        and did_median is not None
        and did_median > 0
        and len(eligible_retain) > 0
    )
    if not eligible_retain:
        status, reason = "BLOCKED", "no_m_eligible_retain_probe"
    elif passed:
        status, reason = "PASS", "pre_registered_m_threshold"
    else:
        status, reason = "FAIL", "m_threshold_not_met"
    return {
        "status": status,
        "reason": reason,
        "positive_confirmation_clusters": positive,
        "confirmation_cluster_count": len(forget_rows),
        "donor_specific_median": forget_median,
        "selectivity_did_median": did_median,
        "selectivity_did_values": did_values,
        "eligible_four_deltas": eligible,
        "all_cluster_four_deltas": everyone,
        "eligible_forget_ids": [row["pair_id"] for row in eligible_forget],
        "eligible_retain_ids": [row["pair_id"] for row in eligible_retain],
    }


def count_labels(rows: list[dict], condition: str, group: str | None = None) -> dict:
    subset = [row for row in rows if row["condition"] == condition and (group is None or row["group"] == group)]
    labels = ["target_correct", "wrong_fact", "refusal", "truncated", "not_mentioned", "human_review"]
    counted = {label: sum(row["label"] == label for row in subset) for label in labels}
    counted["n"] = len(subset)
    return counted


def render_summary(context: dict) -> str:
    claim = context["claim"]
    eligible = claim.get("eligible_four_deltas") or {}
    lines = [
        "# v0.3 固定前缀相对偏好",
        "",
        "## 结论",
        "",
        f"**{context['headline']}**",
        "",
        context["wording"],
        "",
        "M 是固定前缀下目标城市与 donor 城市的平均条件对数概率之差。donor 城市是该 donor 人物的出生地，不是目标人物的真值。自由生成按整句分类，句首出现不是硬门。正的 M 下降不是遗忘成功。没有启动 Teacher/Student 训练。",
        "",
        "## 门",
        "",
        "| 门 | 状态 | 说明 |",
        "| --- | --- | --- |",
        f"| 来源 | {context['source_status']} | {context['source_note']} |",
        f"| 读取 | {context['read_status']} | 问号缓存、城市 span 边界 |",
        f"| 目标偏好 | {context['signal_status']} | M_before > 0：{context['signal_pass_count']} / {context['signal_count']} |",
        f"| patch 正确性 | {context['correct_status']} | 两段城市 logits 容差 0 |",
        f"| 提名 S | {context['nomination_status']} | 层：{context['layers']} |",
        f"| confirmation 主张 | {context['claim_status']} | {claim.get('reason', '')} |",
        "",
        "## M 基线",
        "",
        "| 簇 | 目标城市 | 替代城市 | M | 目标均值 | 替代均值 | 可提名 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in context["baselines"]:
        lines.append(
            f"| {row['pair_id']} | {row['city']} | {row['alternative_city']} | {row['m_before']:.4f} | "
            f"{row['avg_plus']:.4f} | {row['avg_minus']:.4f} | {'是' if row['signal_pass'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 自由生成基线",
            "",
            "分类和 M 并列，不参与提名。",
            "",
        ]
    )
    for group in ("forget", "retain"):
        counted = count_labels(context["generations"], "baseline", group)
        lines.append(
            f"- {group}：正确 {counted['target_correct']}，错误事实 {counted['wrong_fact']}，"
            f"拒答 {counted['refusal']}，截断 {counted['truncated']}，未提及 {counted['not_mentioned']}，"
            f"人工复核 {counted['human_review']}，共 {counted['n']}"
        )
    lines.extend(
        [
            "",
            "## 冻结集合",
            "",
            f"- discovery：{', '.join(context['discovery_ids'])}",
            f"- confirmation：{', '.join(context['confirmation_ids'])}",
            f"- S：{context['layers']}",
            f"- 总体：`{context['overall']}`",
            "",
            "## M 的差中差",
            "",
            "下列均值只用 M_before > 0 的簇。逐簇原始分数在 `confirmation.jsonl`。",
            "",
            f"- forget donor：{eligible.get('forget_donor')}",
            f"- forget noise：{eligible.get('forget_noise')}",
            f"- retain donor：{eligible.get('retain_donor')}",
            f"- retain noise：{eligible.get('retain_noise')}",
            f"- donor_specific 中位数：{claim.get('donor_specific_median')}",
            f"- selectivity_DID 中位数：{claim.get('selectivity_did_median')}",
            f"- 可提名且 donor_specific 为正的 confirmation 簇：{claim.get('positive_confirmation_clusters')}",
            "",
            "## 联合 S 上的自由生成",
            "",
        ]
    )
    if not context["layers"]:
        lines.append("S 为空，没有做 patch 后生成。")
    else:
        for condition in ("donor", "noise0", "shuffle"):
            for group in ("forget", "retain"):
                counted = count_labels(context["generations"], condition, group)
                if counted["n"] == 0:
                    continue
                lines.append(
                    f"- {condition} / {group}：正确 {counted['target_correct']}，错误事实 {counted['wrong_fact']}，"
                    f"拒答 {counted['refusal']}，截断 {counted['truncated']}，未提及 {counted['not_mentioned']}，"
                    f"人工复核 {counted['human_review']}，共 {counted['n']}"
                )
    lines.extend(
        [
            "",
            "## 没有做的事",
            "",
            "没有修改旧定位目录、候选、TOFU、retain95 或 `my_llama.py`。没有运行路径条件教师，也没有运行学生训练。",
            "",
            f"生成时间：`{context['generated_at']}`。",
            "",
        ]
    )
    return "\n".join(lines)


def write_hashes() -> None:
    names = [
        "protocol.json",
        "run_preference.py",
        "boundaries.jsonl",
        "retain_probes.jsonl",
        "retain_selection.json",
        "source_gate.json",
        "preference_baseline.jsonl",
        "generation.jsonl",
        "frozen_S.json",
        "gates.jsonl",
        "layer_scan.jsonl",
        "confirmation.jsonl",
        "summary.md",
    ]
    payload = {}
    for name in names:
        path = OUT_DIR / name
        if path.exists():
            payload[name] = loc.sha256_file(path)
    editor = REPO / "workspace/data/external_donor_prompt_activation_v0.3_localization/patch_editor.py"
    payload["patch_editor.py"] = loc.sha256_file(editor)
    payload["candidate_pairs.jsonl"] = loc.sha256_file(loc.CANDIDATE_DIR / "candidate_pairs.jsonl")
    payload["retain95.jsonl"] = loc.sha256_file(loc.RETAIN_PATH)
    (OUT_DIR / "output_hashes.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def public_baseline(row: dict) -> dict:
    scored = row["baseline"]
    return {
        "pair_id": row["pair_id"],
        "group": row["group"],
        "split": row["split"],
        "name": row["name"],
        "city": row["city"],
        "alternative_city": row["alternative_city"],
        "donor_id": row["donor_id"],
        "prefix": row["prefix"],
        "plus_tokens": row["plus_tokens"],
        "minus_tokens": row["minus_tokens"],
        "span_start": row["plus_span_start"],
        "m_before": scored["m"],
        "avg_plus": scored["avg_plus"],
        "avg_minus": scored["avg_minus"],
        "sum_plus": scored["sum_plus"],
        "sum_minus": scored["sum_minus"],
        "count_plus": scored["count_plus"],
        "count_minus": scored["count_minus"],
        "signal_pass": row["signal_pass"],
        "hook_diff": row["hook_diff"],
        "site_diff": row["site_diff"],
    }


def overall_label(source: str, read: str, correct: str, claim: str) -> str:
    if source != "PASS":
        if claim == "PASS":
            return "PILOT_ONLY"
        return "NO-GO"
    if read == "PASS" and correct == "PASS" and claim == "PASS":
        return "GATES_PASSED_TRAINING_NOT_STARTED"
    return "NO-GO"


def headline_for(overall: str, claim: str) -> tuple[str, str]:
    if claim == "PASS" and overall == "PILOT_ONLY":
        return (
            "数值上达到预注册的 M 门槛，来源门仍是 BLOCKED。",
            "这是工程 pilot。相对优势下降不是遗忘成功。自由生成另报。不启动 Teacher/Student 训练。",
        )
    if claim == "PASS":
        return (
            "预注册的 M 门槛通过，训练未启动。",
            "门槛只限制结论措辞。自由生成另报。不自动启动 Teacher/Student 训练。",
        )
    return (
        "尚无证据支持目标选择性局部路径。",
        "donor 相对噪声的 M 下降没有同时满足 confirmation 和 retain 门槛，或读取、边界、正确性门未通过。不称为遗忘成功，不启动 Teacher/Student 训练。",
    )


def write_blocked(protocol: dict, source: dict, reason: str) -> None:
    claim = {"status": "BLOCKED", "reason": reason}
    headline, wording = ("BLOCKED。", f"环境阻断：{reason}。这不是模型拒绝，也不是科学假设失败。不启动 Teacher/Student 训练。")
    context = {
        "headline": headline,
        "wording": wording,
        "source_status": source.get("status", "BLOCKED"),
        "source_note": source.get("reason") or reason,
        "read_status": "BLOCKED",
        "signal_status": "BLOCKED",
        "signal_pass_count": 0,
        "signal_count": 0,
        "correct_status": "BLOCKED",
        "nomination_status": "BLOCKED",
        "claim_status": "BLOCKED",
        "claim": claim,
        "baselines": [],
        "generations": [],
        "discovery_ids": protocol["splits"]["discovery_pairs"],
        "confirmation_ids": protocol["splits"]["confirmation_pairs"],
        "layers": [],
        "overall": "NO-GO",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / "summary.md").write_text(render_summary(context), encoding="utf-8")
    loc.write_jsonl(OUT_DIR / "gates.jsonl", [{"gate": "environment", "status": "BLOCKED", "reason": reason}])
    write_hashes()


def main() -> None:
    check_only = "--check" in sys.argv
    protocol = read_protocol()
    self_check(protocol)
    pairs = loc.read_jsonl(loc.CANDIDATE_DIR / "candidate_pairs.jsonl")
    donors = loc.read_jsonl(loc.CANDIDATE_DIR / "donor_entities.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(loc.SNAPSHOT, local_files_only=True)
    boundaries = check_boundaries(tokenizer, protocol, pairs, donors)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loc.write_jsonl(OUT_DIR / "boundaries.jsonl", boundaries)
    examples = build_examples(tokenizer, protocol, pairs, donors)
    loc.write_jsonl(
        OUT_DIR / "retain_probes.jsonl",
        [
            {
                "pair_id": row["pair_id"],
                "name": row["name"],
                "author_span": row.get("author_span"),
                "city": row["city"],
                "alternative_city": row["alternative_city"],
                "donor_id": row["donor_id"],
                "prefix": row["prefix"],
                "line_1based": row.get("line_1based"),
            }
            for row in examples
            if row["group"] == "retain"
        ],
    )
    (OUT_DIR / "retain_selection.json").write_text(
        json.dumps({"selected": 10, "canonical_names": protocol["answer_prefix"]["canonical_names"]}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    if check_only:
        print(json.dumps({"check": "ok", "boundaries": len(boundaries)}, ensure_ascii=False), flush=True)
        return

    source = loc.check_p19(donors)
    (OUT_DIR / "source_gate.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not torch.cuda.is_available():
        write_blocked(protocol, source, "blocked_no_cuda")
        return

    _model_tokenizer, model = loc.load_model()
    tokenizer = _model_tokenizer
    editor = ResidualEditor(model)
    forget = [row for row in examples if row["group"] == "forget"]
    retain = [row for row in examples if row["group"] == "retain"]
    try:
        read_rows, donor_sites = read_donors(model, editor, forget)
        correct_rows = []
        for row in forget:
            print(f"base {row['pair_id']}", flush=True)
            measured = measure_base(model, editor, row)
            row["baseline"] = measured["scored"]
            row["sites_plus"] = measured["sites_plus"]
            row["sites_minus"] = measured["sites_minus"]
            row["hook_diff"] = measured["hook_diff"]
            row["site_diff"] = measured["site_diff"]
            row["signal_pass"] = measured["scored"]["m"] > 0 and measured["hook_diff"] == 0.0
            print(f"correct {row['pair_id']}", flush=True)
            correct_rows.append(correctness_row(model, editor, row, measured))
            del measured
        for row in retain:
            print(f"base {row['pair_id']}", flush=True)
            measured = measure_base(model, editor, row)
            row["baseline"] = measured["scored"]
            row["sites_plus"] = measured["sites_plus"]
            row["sites_minus"] = measured["sites_minus"]
            row["hook_diff"] = measured["hook_diff"]
            row["site_diff"] = measured["site_diff"]
            row["signal_pass"] = measured["scored"]["m"] > 0 and measured["hook_diff"] == 0.0
            del measured
        for row in examples:
            _prompt_logits, _prompt_hooked, prompt_sites = capture_sites(model, editor, row["prompt_ids"], row["qmark"])
            del _prompt_logits, _prompt_hooked
            row["prompt_sites"] = prompt_sites
        generations = []
        for row in examples:
            print(f"generate baseline {row['pair_id']}", flush=True)
            generations.append(generation_record(model, editor, tokenizer, protocol, row, "baseline", None))
    finally:
        editor.close()

    read_pass = all(
        row["cache_error"] is None
        and row["site_error"] is None
        and row["cache_question_mark_max_abs"] == 0.0
        and row["hooked_vs_clean_hidden_max_abs"] == 0.0
        and row["finite"]
        for row in read_rows
    ) and all(row["hook_diff"] == 0.0 for row in examples)
    correct_pass = all(
        row["error"] is None
        and row["noop_city_logit_max_abs"] == 0.0
        and row["self_patch_layer0_city_logit_max_abs"] == 0.0
        and row["self_patch_all_layers_city_logit_max_abs"] == 0.0
        and row["hook_diff"] == 0.0
        and row["downstream_input_max_abs"] is not None
        and row["downstream_input_max_abs"] > 0.0
        for row in correct_rows
    )
    loc.write_jsonl(OUT_DIR / "preference_baseline.jsonl", [public_baseline(row) for row in examples])
    loc.write_jsonl(OUT_DIR / "generation.jsonl", generations)

    layers: list[int] = []
    medians: dict[str, float | None] = {}
    scan_rows: list[dict] = []
    confirmation_rows: list[dict] = []
    nomination_status = "BLOCKED"
    claim = {"status": "BLOCKED", "reason": "upstream_gate_failed"}
    if read_pass and correct_pass:
        discovery = [row for row in forget if row["split"] == "discovery"]
        print("discovery scan", flush=True)
        for layer in loc.PATCH_LAYERS:
            print(f"layer {layer}", flush=True)
            for row in discovery:
                scan_rows.append(
                    scan_layer(
                        model,
                        editor,
                        row,
                        layer,
                        donor_sites[row["donor_id"]],
                        protocol["noise"]["seeds"],
                    )
                )
        for layer in loc.PATCH_LAYERS:
            values = [row["donor_specific"] for row in scan_rows if row["layer"] == layer and row["signal_pass"]]
            medians[str(layer)] = loc.median(values)
        numeric = {int(layer): value for layer, value in medians.items() if value is not None}
        layers = loc.nominate(numeric)
        nomination_status = "PASS" if layers else "FAIL"
        frozen = {
            "layers": layers,
            "medians": medians,
            "eligible_discovery_pairs": [row["pair_id"] for row in discovery if row["signal_pass"]],
            "metric": "M_before - M_after",
        }
        (OUT_DIR / "frozen_S.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        loc.write_jsonl(OUT_DIR / "layer_scan.jsonl", scan_rows)
        layers = json.loads((OUT_DIR / "frozen_S.json").read_text(encoding="utf-8"))["layers"]
        if layers:
            print("confirmation", flush=True)
            mapping = protocol["shuffle"]["mapping"]
            selected = [row for row in examples if row["split"] in {"confirmation", "retain"}]
            for row in selected:
                print(f"joint {row['pair_id']}", flush=True)
                shuffle_id = mapping[row["donor_id"]]
                confirmation_rows.append(
                    joint_record(
                        model,
                        editor,
                        row,
                        layers,
                        donor_sites[row["donor_id"]],
                        protocol["noise"]["seeds"],
                        donor_sites[shuffle_id],
                    )
                )
                for condition, kind, seed in (("donor", "donor", None), ("noise0", "noise", 0), ("shuffle", "shuffle", None)):
                    print(f"generate {condition} {row['pair_id']}", flush=True)
                    built = replacements_for(
                        row,
                        donor_sites[row["donor_id"]],
                        row["prompt_sites"],
                        layers,
                        kind,
                        seed=seed,
                        shuffle_site=donor_sites[shuffle_id],
                    )
                    generations.append(generation_record(model, editor, tokenizer, protocol, row, condition, built))
            claim = claim_status(
                [row for row in confirmation_rows if row["group"] == "forget"],
                [row for row in confirmation_rows if row["group"] == "retain"],
                layers,
            )
        else:
            claim = {"status": "FAIL", "reason": "empty_S"}
    else:
        loc.write_jsonl(OUT_DIR / "layer_scan.jsonl", [])
        (OUT_DIR / "frozen_S.json").write_text(
            json.dumps({"layers": [], "reason": "upstream_gate_failed"}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    signal_rows = [row for row in examples if row["group"] == "forget"]
    signal_pass_count = sum(row["signal_pass"] for row in signal_rows)
    read_status = "PASS" if read_pass else "FAIL"
    correct_status = "PASS" if correct_pass else "FAIL"
    signal_status = "PASS" if signal_pass_count else "FAIL"
    overall = overall_label(source["status"], read_status, correct_status, claim["status"])
    headline, wording = headline_for(overall, claim["status"])
    context = {
        "headline": headline,
        "wording": wording,
        "source_status": source["status"],
        "source_note": source.get("reason") or source.get("error") or "P19 English label",
        "read_status": read_status,
        "signal_status": signal_status,
        "signal_pass_count": signal_pass_count,
        "signal_count": len(signal_rows),
        "correct_status": correct_status,
        "nomination_status": nomination_status,
        "claim_status": claim["status"],
        "claim": claim,
        "baselines": [public_baseline(row) for row in examples],
        "generations": generations,
        "discovery_ids": protocol["splits"]["discovery_pairs"],
        "confirmation_ids": protocol["splits"]["confirmation_pairs"],
        "layers": layers,
        "overall": overall,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    gates = [
        {"gate": "source", "status": source["status"], "detail": source},
        {"gate": "read", "status": read_status, "rows": read_rows},
        {
            "gate": "target_preference",
            "status": signal_status,
            "rows": [
                {
                    "pair_id": row["pair_id"],
                    "split": row["split"],
                    "city": row["city"],
                    "alternative_city": row["alternative_city"],
                    "m_before": row["baseline"]["m"],
                    "signal_pass": row["signal_pass"],
                }
                for row in signal_rows
            ],
        },
        {"gate": "patch_correctness", "status": correct_status, "rows": correct_rows},
        {"gate": "nomination", "status": nomination_status, "layers": layers, "medians": medians},
        {"gate": "confirmation_claim", "status": claim["status"], "detail": claim},
        {"gate": "teacher_student", "status": "not_started"},
        {"gate": "free_generation", "status": "reported", "role": "not a nomination gate"},
    ]
    loc.write_jsonl(OUT_DIR / "gates.jsonl", gates)
    loc.write_jsonl(OUT_DIR / "confirmation.jsonl", confirmation_rows)
    loc.write_jsonl(OUT_DIR / "generation.jsonl", generations)
    (OUT_DIR / "summary.md").write_text(render_summary(context), encoding="utf-8")
    write_hashes()
    print(headline, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

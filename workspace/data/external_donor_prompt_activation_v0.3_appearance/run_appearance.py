#!/usr/bin/env python3
"""Raw-question appearance localization for prompt-only activation donors.

Primary readout is the target-city appearance rate under the original question.
Fixed-prefix M is an auxiliary diagnostic on the frozen joint S only. This run
does not overwrite earlier localization or preference directories, does not edit
my_llama.py or TOFU files, and does not start Teacher/Student training.
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
from safetensors.torch import load_file
from transformers import AutoTokenizer

REPO = Path("/usr/local/ASU-unlearning")
sys.path.insert(0, str(REPO / "Right-to-be-forgotten"))
sys.path.insert(0, str(REPO / "workspace/data/external_donor_prompt_activation_v0.3_localization"))
import run_localization as loc  # noqa: E402
from patch_editor import ResidualEditor  # noqa: E402

OUT_DIR = REPO / "workspace/data/external_donor_prompt_activation_v0.3_appearance"
if OUT_DIR.resolve() in {loc.OUT_DIR.resolve(), (REPO / "workspace/data/external_donor_prompt_activation_v0.3_preference").resolve()}:
    raise RuntimeError("appearance output must stay outside earlier pilot directories")


def read_protocol() -> dict:
    return json.loads((OUT_DIR / "protocol.json").read_text(encoding="utf-8"))


def phrase_present(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def classify_appearance(text: str, hit_limit: bool, target: str, foil: str, protocol: dict) -> dict:
    aliases = list(protocol["classification"]["aliases"].get(target, []))
    foil_aliases = list(protocol["classification"]["aliases"].get(foil, []))
    target_hit = any(phrase_present(text, phrase) for phrase in [target, *aliases])
    foil_hit = any(phrase_present(text, phrase) for phrase in [foil, *foil_aliases])
    other = [
        city
        for city in protocol["classification"]["closed_cities"]
        if city not in {target, foil, *aliases, *foil_aliases} and phrase_present(text, city)
    ]
    if target_hit and other:
        label = "human_review"
        t_flag = 0
    elif target_hit and foil_hit:
        label = "target_and_foil"
        t_flag = 1
    elif target_hit:
        label = "target_correct"
        t_flag = 1
    elif foil_hit or other:
        label = "wrong_fact"
        t_flag = 0
    elif any(phrase in text for phrase in protocol["classification"]["refusal_phrases"]):
        label = "refusal"
        t_flag = 0
    elif hit_limit:
        label = "truncated"
        t_flag = 0
    else:
        label = "not_mentioned"
        t_flag = 0
    return {
        "label": label,
        "target_hit": target_hit,
        "foil_hit": foil_hit,
        "other_cities": other,
        "T": t_flag,
        "F": int(foil_hit),
    }


def canonical_name(author_span: str, protocol: dict) -> str:
    name = author_span
    for prefix in protocol["answer_prefix_auxiliary"]["strip_prefixes_longest_first"]:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


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
    return {"ids": ids, "span_start": start, "span_ids": span_ids, "tokens": tokenizer.convert_ids_to_tokens(span_ids)}


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
        log_prob = torch.log_softmax(logits[span_start - 1 + offset].float(), dim=-1)[token_id]
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


def sample_seed(protocol: dict, group: str, pair_index: int, condition_code: int, sample_index: int) -> int:
    retain_offset = 1000000007 if group == "retain" else 0
    return (
        int(protocol["sampling"]["base_seed"])
        + 1000003 * pair_index
        + 10007 * sample_index
        + 17 * condition_code
        + retain_offset
    )


def condition_code(kind: str, noise_seed: int | None = None) -> int:
    codes = {"base": 0, "donor": 1, "noise": 2, "shuffle": 100}
    if kind == "noise":
        if noise_seed is None:
            raise RuntimeError("noise condition needs a seed")
        return 2 + int(noise_seed)
    return codes[kind]


def self_check(protocol: dict) -> None:
    loc.self_check()
    loc.reproduce_shuffle(protocol)
    cases = [
        ("born in Karachi, Pakistan.", False, "Karachi", "Wellington", "target_correct", 1, 0),
        ("born in Wellington.", False, "Karachi", "Wellington", "wrong_fact", 0, 1),
        ("Karachi then Wellington", False, "Karachi", "Wellington", "target_and_foil", 1, 1),
        ("Karachi and Seattle", False, "Karachi", "Wellington", "human_review", 0, 0),
        ("born in Peking.", False, "Beijing", "Seattle", "target_correct", 1, 0),
        ("born in New York.", False, "New York City", "Rio de Janeiro", "not_mentioned", 0, 0),
        ("I don't know", False, "Tokyo", "Tupelo", "refusal", 0, 0),
        ("born in K", True, "Kuwait City", "New Orleans", "truncated", 0, 0),
    ]
    for text, hit_limit, target, foil, label, t_flag, f_flag in cases:
        found = classify_appearance(text, hit_limit, target, foil, protocol)
        if found["label"] != label or found["T"] != t_flag or found["F"] != f_flag:
            raise RuntimeError(f"classification drifted for {text!r}: {found}")
    if loc.nominate({0: 0.2, 1: 0.2, 2: -1.0, 3: 0.5, 4: 0.4}) != [3, 4, 0]:
        raise RuntimeError("nomination preview drifted")


def build_examples(tokenizer, protocol: dict, pairs: list[dict], donors: list[dict]) -> list[dict]:
    by_donor = {row["donor_id"]: row["donor_fact_value"] for row in donors}
    retain_rows, excluded = loc.select_retain(tokenizer, protocol, pairs)
    loc.assert_retain(retain_rows, protocol)
    examples = []
    for pair in pairs:
        if pair["donor_fact_value"] != by_donor[pair["donor_id"]]:
            raise RuntimeError(f"{pair['pair_id']} donor city drifted")
        prompt = loc.prompt_text(pair["q_plus"], protocol)
        donor_prompt = loc.prompt_text(pair["q_minus"], protocol)
        if pair["donor_fact_value"] in donor_prompt:
            raise RuntimeError(f"{pair['pair_id']} appends the donor city to the donor prompt")
        prompt_ids = list(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        donor_ids = list(tokenizer(donor_prompt, add_special_tokens=True)["input_ids"])
        prefix = protocol["answer_prefix_auxiliary"]["template"].format(name=pair["clean_entity"])
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
                "foil_city": pair["donor_fact_value"],
                "donor_id": pair["donor_id"],
                "prompt_ids": prompt_ids,
                "donor_ids": donor_ids,
                "qmark": loc.question_mark_index(tokenizer, prompt_ids),
                "donor_qmark": loc.question_mark_index(tokenizer, donor_ids),
                "prefix": prefix,
                "plus_ids": target["ids"],
                "plus_span_start": target["span_start"],
                "plus_span_ids": target["span_ids"],
                "minus_ids": alternative["ids"],
                "minus_span_start": alternative["span_start"],
                "minus_span_ids": alternative["span_ids"],
            }
        )
    for index, row in enumerate(retain_rows):
        name = canonical_name(row["author_span"], protocol)
        if name != protocol["answer_prefix_auxiliary"]["canonical_names"][index]:
            raise RuntimeError(f"retain name {name!r} drifted")
        prompt = loc.prompt_text(row["question"], protocol)
        prompt_ids = list(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        prefix = protocol["answer_prefix_auxiliary"]["template"].format(name=name)
        foil_city = by_donor[row["donor_id"]]
        target = encode_city(tokenizer, prompt, prefix, row["city"])
        alternative = encode_city(tokenizer, prompt, prefix, foil_city)
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
                "foil_city": foil_city,
                "donor_id": row["donor_id"],
                "prompt_ids": prompt_ids,
                "qmark": loc.question_mark_index(tokenizer, prompt_ids),
                "prefix": prefix,
                "plus_ids": target["ids"],
                "plus_span_start": target["span_start"],
                "plus_span_ids": target["span_ids"],
                "minus_ids": alternative["ids"],
                "minus_span_start": alternative["span_start"],
                "minus_span_ids": alternative["span_ids"],
                "retain_excluded": excluded,
            }
        )
    return examples


def rates_from_records(records: list[dict]) -> dict:
    if not records:
        raise RuntimeError("no generation records")
    p_target = sum(row["T"] for row in records) / len(records)
    p_foil = sum(row["F"] for row in records) / len(records)
    counts = {}
    for row in records:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return {
        "p_target": float(p_target),
        "p_foil": float(p_foil),
        "M_appear": float(p_target - p_foil),
        "K": len(records),
        "label_counts": counts,
    }


def sample_next_id(logits: torch.Tensor, generator: torch.Generator, temperature: float) -> int:
    scaled = logits.float() / temperature
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def generate_full(
    model,
    editor: ResidualEditor,
    tokenizer,
    prompt_ids: list[int],
    qmark: int,
    replacements: dict[int, torch.Tensor] | None,
    steps: int,
    seed: int,
    temperature: float,
) -> tuple[str, bool, list[int]]:
    editor.close()
    generator = torch.Generator(device=loc.DEVICE)
    generator.manual_seed(int(seed))
    ids = list(prompt_ids)
    hit_limit = True
    new_ids: list[int] = []
    for _ in range(steps):
        if replacements:
            with editor.replace(replacements, qmark):
                logits, _hidden = loc.forward(model, ids)
        else:
            editor.close()
            logits, _hidden = loc.forward(model, ids)
        next_id = sample_next_id(logits[-1], generator, temperature)
        del logits
        ids.append(next_id)
        new_ids.append(next_id)
        if next_id == tokenizer.eos_token_id:
            hit_limit = False
            break
    editor.close()
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, hit_limit, new_ids


def generate_cache(
    model,
    editor: ResidualEditor,
    tokenizer,
    prompt_ids: list[int],
    qmark: int,
    replacements: dict[int, torch.Tensor] | None,
    steps: int,
    seed: int,
    temperature: float,
) -> tuple[str, bool, list[int]]:
    editor.close()
    generator = torch.Generator(device=loc.DEVICE)
    generator.manual_seed(int(seed))
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=loc.DEVICE)
    attention_mask = torch.ones_like(input_ids)
    if replacements:
        editor.replace(replacements, qmark)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    editor.close()
    past = outputs.past_key_values
    next_id = sample_next_id(outputs.logits[0, -1], generator, temperature)
    new_ids = [next_id]
    hit_limit = True
    if next_id == tokenizer.eos_token_id:
        hit_limit = False
    else:
        for _ in range(steps - 1):
            step_ids = torch.tensor([[next_id]], dtype=torch.long, device=loc.DEVICE)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=loc.DEVICE)],
                dim=1,
            )
            with torch.no_grad():
                outputs = model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            past = outputs.past_key_values
            next_id = sample_next_id(outputs.logits[0, -1], generator, temperature)
            new_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                hit_limit = False
                break
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, hit_limit, new_ids


def greedy_ids(model, editor: ResidualEditor, prompt_ids: list[int], qmark: int, replacements: dict | None, steps: int) -> list[int]:
    editor.close()
    ids = list(prompt_ids)
    new_ids: list[int] = []
    for _ in range(steps):
        if replacements:
            with editor.replace(replacements, qmark):
                logits, _hidden = loc.forward(model, ids)
        else:
            editor.close()
            logits, _hidden = loc.forward(model, ids)
        next_id = int(logits[-1].argmax().item())
        del logits
        ids.append(next_id)
        new_ids.append(next_id)
    editor.close()
    return new_ids


def capture_sites(model, editor: ResidualEditor, token_ids: list[int], qmark: int) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    editor.close()
    logits, hidden = loc.forward(model, token_ids)
    with editor.capture(qmark):
        hooked_logits, hooked_hidden = loc.forward(model, token_ids)
    captured = {key: value.cpu() for key, value in editor.captured.items()}
    hook_diff = loc.max_abs(logits.cpu(), hooked_logits.cpu())
    del logits, hooked_logits, hooked_hidden
    sites = loc.site_vectors(hidden.cpu(), qmark, captured)
    del hidden
    return sites, {"hook_diff": hook_diff}


def read_donors(model, editor: ResidualEditor, rows: list[dict]) -> tuple[list[dict], dict]:
    records = []
    donor_sites = {}
    for row in rows:
        print(f"read {row['pair_id']}", flush=True)
        _logits, hidden = loc.forward(model, row["donor_ids"])
        del _logits
        with editor.capture(row["donor_qmark"]):
            _hook_logits, hook_hidden = loc.forward(model, row["donor_ids"])
        del _hook_logits
        captured = {key: value.cpu() for key, value in editor.captured.items()}
        hidden = hidden.cpu()
        hook_hidden = hook_hidden.cpu()
        cached = load_file(str(loc.ACTIVATION_DIR / f"{row['pair_id']}.safetensors"))
        cache_hidden = cached["hidden_q_minus"]
        cache_error = None
        cache_diff = None
        if tuple(cache_hidden.shape) != tuple(hidden.shape):
            cache_error = f"shape {tuple(cache_hidden.shape)} != {tuple(hidden.shape)}"
        else:
            cache_diff = loc.max_abs(hidden[:, row["donor_qmark"], :], cache_hidden[:, row["donor_qmark"], :])
        hook_diff = loc.max_abs(hidden, hook_hidden)
        try:
            sites = loc.site_vectors(hidden, row["donor_qmark"], captured)
            site_error = None
            donor_sites[row["donor_id"]] = sites
            finite = bool(torch.isfinite(hidden.float()).all().item())
        except RuntimeError as exc:
            sites = None
            site_error = str(exc)
            finite = False
        records.append(
            {
                "pair_id": row["pair_id"],
                "qmark": row["qmark"],
                "donor_qmark": row["donor_qmark"],
                "cache_question_mark_max_abs": cache_diff,
                "cache_error": cache_error,
                "hooked_vs_clean_hidden_max_abs": hook_diff,
                "site_error": site_error,
                "finite": finite,
            }
        )
    return records, donor_sites


def same_replacements(site: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    return {layer: site[layer] for layer in layers}


def noise_replacements(
    row: dict,
    donor_site: torch.Tensor,
    recipient_site: torch.Tensor,
    layers: list[int],
    seed: int,
) -> dict[int, torch.Tensor]:
    built = {}
    for layer in layers:
        mixed = loc.noise_seed(row["group"], row["pair_index"], layer, seed)
        null, _scale = loc.matched_noise(recipient_site[layer], donor_site[layer], mixed)
        built[layer] = null
    return built


def sample_condition(
    model,
    editor,
    tokenizer,
    protocol: dict,
    row: dict,
    condition: str,
    replacements: dict | None,
    k: int,
    use_cache: bool,
    noise_seed: int | None = None,
) -> list[dict]:
    kind = "noise" if condition.startswith("noise") else condition
    code = condition_code(kind, noise_seed)
    steps = int(protocol["sampling"]["max_new_tokens"])
    temperature = float(protocol["sampling"]["temperature"])
    generate = generate_cache if use_cache else generate_full
    records = []
    for sample_index in range(k):
        seed = sample_seed(protocol, row["group"], row["pair_index"], code, sample_index)
        text, hit_limit, new_ids = generate(
            model,
            editor,
            tokenizer,
            row["prompt_ids"],
            row["qmark"],
            replacements,
            steps,
            seed,
            temperature,
        )
        classified = classify_appearance(text, hit_limit, row["city"], row["foil_city"], protocol)
        records.append(
            {
                "pair_id": row["pair_id"],
                "group": row["group"],
                "split": row["split"],
                "condition": condition,
                "sample_index": sample_index,
                "seed": seed,
                "hit_limit": hit_limit,
                "text": text,
                "n_new_tokens": len(new_ids),
                "target_city": row["city"],
                "foil_city": row["foil_city"],
                **classified,
            }
        )
    return records


def score_preference(model, editor, row: dict, replacements: dict | None) -> dict:
    if replacements:
        with editor.replace(replacements, row["qmark"]):
            plus_logits, _ = loc.forward(model, row["plus_ids"])
        with editor.replace(replacements, row["qmark"]):
            minus_logits, _ = loc.forward(model, row["minus_ids"])
    else:
        editor.close()
        plus_logits, _ = loc.forward(model, row["plus_ids"])
        minus_logits, _ = loc.forward(model, row["minus_ids"])
    return preference(
        pack_stats(span_values(plus_logits, row["plus_span_start"], row["plus_span_ids"])),
        pack_stats(span_values(minus_logits, row["minus_span_start"], row["minus_span_ids"])),
    )


def claim_status(forget_rows: list[dict], retain_rows: list[dict], layers: list[int]) -> dict:
    if not layers:
        return {"status": "FAIL", "reason": "empty_S"}
    eligible_forget = [row for row in forget_rows if row["signal_pass"]]
    eligible_retain = [row for row in retain_rows if row["signal_pass"]]
    positive = sum(row["D"] > 0 for row in eligible_forget)
    forget_median = loc.median([row["D"] for row in eligible_forget])
    retain_mean = loc.mean([row["D"] for row in eligible_retain])
    did = None if forget_median is None or retain_mean is None else forget_median - retain_mean

    def four(rows: list[dict]) -> dict:
        return {
            "donor": loc.mean([row["delta_donor"] for row in rows]),
            "noise": loc.mean([row["delta_noise_mean"] for row in rows]),
        }

    passed = (
        positive >= 3
        and forget_median is not None
        and forget_median > 0
        and did is not None
        and did > 0
        and len(eligible_retain) > 0
    )
    if not eligible_retain:
        status, reason = "BLOCKED", "no_appearance_eligible_retain_probe"
    elif passed:
        status, reason = "PASS", "pre_registered_appearance_threshold"
    else:
        status, reason = "FAIL", "appearance_threshold_not_met"
    return {
        "status": status,
        "reason": reason,
        "positive_confirmation_clusters": positive,
        "confirmation_cluster_count": len(forget_rows),
        "eligible_forget_ids": [row["pair_id"] for row in eligible_forget],
        "eligible_retain_ids": [row["pair_id"] for row in eligible_retain],
        "D_median": forget_median,
        "DID": did,
        "forget_deltas": four(eligible_forget),
        "retain_deltas": four(eligible_retain),
        "all_forget_deltas": four(forget_rows),
        "all_retain_deltas": four(retain_rows),
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
            "数值上达到预注册的出现率门槛，来源门仍是 BLOCKED。",
            "这是 10 簇工程 pilot。目标事实出现率下降不是遗忘成功。固定前缀 M 只作辅助。不启动 Teacher/Student 训练。",
        )
    if claim == "PASS":
        return (
            "预注册的出现率门槛通过，训练未启动。",
            "门槛只限制结论措辞。10 簇 pilot 仍是探索性结果。不自动启动 Teacher/Student 训练。",
        )
    return (
        "尚无证据支持目标选择性局部路径。",
        "donor 相对噪声的目标事实出现率下降没有同时满足 confirmation 和 retain 门槛，或读取、正确性门未通过。不称为遗忘成功，不启动 Teacher/Student 训练。",
    )


def render_summary(context: dict) -> str:
    lines = [
        "# v0.3 原始问题出现率定位",
        "",
        "## 结论",
        "",
        f"**{context['headline']}**",
        "",
        context["wording"],
        "",
        "主读出是原始问题下完整回答的目标城市出现率。固定前缀 M 只作联合 S 上的辅助诊断。正的出现率下降不是遗忘成功。没有启动 Teacher/Student 训练。",
        "",
        "## 门",
        "",
        "| 门 | 状态 | 说明 |",
        "| --- | --- | --- |",
        f"| 来源 | {context['source_status']} | {context['source_note']} |",
        f"| 读取 | {context['read_status']} | 问号缓存、hook、城市边界 |",
        f"| 目标可提取性 | {context['signal_status']} | 基线 p_target ≥ 0.5：{context['signal_pass_count']} / {context['signal_count']} |",
        f"| patch 正确性 | {context['correct_status']} | no-op/self-patch 贪心与 cache 校验 |",
        f"| 提名 S | {context['nomination_status']} | 层：{context['layers']} |",
        f"| confirmation 主张 | {context['claim_status']} | {context['claim'].get('reason')} |",
        "",
        "## 出现率基线",
        "",
        "| 簇 | 目标城市 | foil | p_target | p_foil | M_appear | 可提名 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in context["baselines"]:
        lines.append(
            f"| {row['pair_id']} | {row['city']} | {row['foil_city']} | {row['p_target']:.4f} | "
            f"{row['p_foil']:.4f} | {row['M_appear']:.4f} | {'是' if row['signal_pass'] else '否'} |"
        )
    lines.extend(["", "## 冻结集合", ""])
    lines.append(f"- discovery：{', '.join(context['discovery_ids'])}")
    lines.append(f"- confirmation：{', '.join(context['confirmation_ids'])}")
    lines.append(f"- S：{context['layers']}")
    lines.append(f"- 解码路径：`{context['decode_path']}`")
    lines.append(f"- 总体：`{context['overall']}`")
    lines.extend(["", "## 联合 S 上的出现率差", ""])
    claim = context["claim"]
    if not context["layers"]:
        lines.append("S 为空或上游门失败，没有联合 patch 出现率。")
    else:
        forget = claim.get("forget_deltas") or {}
        retain = claim.get("retain_deltas") or {}
        lines.append(f"- forget donor：{forget.get('donor')}")
        lines.append(f"- forget noise：{forget.get('noise')}")
        lines.append(f"- retain donor：{retain.get('donor')}")
        lines.append(f"- retain noise：{retain.get('noise')}")
        lines.append(f"- D 中位数：{claim.get('D_median')}")
        lines.append(f"- DID：{claim.get('DID')}")
        lines.append(f"- 可提名且 D 为正的 confirmation 簇：{claim.get('positive_confirmation_clusters')}")
    lines.extend(["", "## 辅助固定前缀 M（仅联合 S）", ""])
    if not context["aux_m"]:
        lines.append("未计算。")
    else:
        lines.append("| 簇 | M_before | M_after | ΔM | Δ目标均值 | Δfoil均值 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in context["aux_m"]:
            lines.append(
                f"| {row['pair_id']} | {row['m_before']:.4f} | {row['m_after']:.4f} | {row['delta_m']:.4f} | "
                f"{row['delta_avg_plus']:.4f} | {row['delta_avg_minus']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 没有做的事",
            "",
            "没有修改旧定位目录、偏好目录、候选、TOFU、retain95 或 `my_llama.py`。没有运行路径条件教师，也没有运行学生训练。",
            "",
            f"生成时间：`{context['generated_at']}`。",
            "",
        ]
    )
    return "\n".join(lines)


def write_hashes() -> None:
    names = [
        "protocol.json",
        "run_appearance.py",
        "retain_probes.jsonl",
        "source_gate.json",
        "appearance_baseline.jsonl",
        "generation.jsonl",
        "frozen_S.json",
        "gates.jsonl",
        "layer_scan.jsonl",
        "confirmation.jsonl",
        "auxiliary_M.jsonl",
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


def write_blocked(protocol: dict, source: dict, reason: str) -> None:
    claim = {"status": "BLOCKED", "reason": reason}
    context = {
        "headline": "BLOCKED。",
        "wording": f"环境阻断：{reason}。这不是模型拒绝，也不是科学假设失败。不启动 Teacher/Student 训练。",
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
        "aux_m": [],
        "discovery_ids": protocol["splits"]["discovery_pairs"],
        "confirmation_ids": protocol["splits"]["confirmation_pairs"],
        "layers": [],
        "decode_path": "n/a",
        "overall": "NO-GO",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / "summary.md").write_text(render_summary(context), encoding="utf-8")
    loc.write_jsonl(OUT_DIR / "gates.jsonl", [{"gate": "environment", "status": "BLOCKED", "reason": reason}])
    write_hashes()


def effect_from_rates(base: dict, patched: dict) -> float:
    return float(base["p_target"] - patched["p_target"])


def joint_appearance(
    model,
    editor,
    tokenizer,
    protocol: dict,
    row: dict,
    layers: list[int],
    donor_site: torch.Tensor,
    recipient_site: torch.Tensor,
    seeds: list[int],
    shuffle_site: torch.Tensor,
    use_cache: bool,
    generations: list[dict],
) -> dict:
    k = int(protocol["sampling"]["confirmation_K"] if row["group"] == "forget" else protocol["sampling"]["retain_K"])
    base_records = [item for item in generations if item["pair_id"] == row["pair_id"] and item["condition"] == "baseline"]
    if len(base_records) != k:
        # confirmation/retain baseline may have been collected with confirmation_K already
        if len(base_records) == 0:
            raise RuntimeError(f"missing baseline generations for {row['pair_id']}")
        # If discovery used K=4 and this is confirmation with K=8, baseline was collected at K=8 below.
    base_rates = rates_from_records(base_records)
    donor_recs = sample_condition(
        model, editor, tokenizer, protocol, row, "donor", same_replacements(donor_site, layers), k, use_cache
    )
    generations.extend(donor_recs)
    noise_deltas = []
    noise_rates = []
    for seed in seeds:
        recs = sample_condition(
            model,
            editor,
            tokenizer,
            protocol,
            row,
            f"noise{seed}",
            noise_replacements(row, donor_site, recipient_site, layers, seed),
            k,
            use_cache,
            noise_seed=seed,
        )
        generations.extend(recs)
        rates = rates_from_records(recs)
        noise_rates.append(rates)
        noise_deltas.append(effect_from_rates(base_rates, rates))
    shuffle_recs = sample_condition(
        model, editor, tokenizer, protocol, row, "shuffle", same_replacements(shuffle_site, layers), k, use_cache
    )
    generations.extend(shuffle_recs)
    donor_rates = rates_from_records(donor_recs)
    delta_donor = effect_from_rates(base_rates, donor_rates)
    delta_noise_mean = loc.mean(noise_deltas)
    return {
        "group": row["group"],
        "pair_id": row["pair_id"],
        "split": row["split"],
        "city": row["city"],
        "foil_city": row["foil_city"],
        "donor_id": row["donor_id"],
        "signal_pass": row["signal_pass"],
        "layers": layers,
        "p_target_base": base_rates["p_target"],
        "p_foil_base": base_rates["p_foil"],
        "p_target_donor": donor_rates["p_target"],
        "p_foil_donor": donor_rates["p_foil"],
        "delta_donor": delta_donor,
        "delta_noise_mean": delta_noise_mean,
        "D": delta_donor - delta_noise_mean,
        "noise_p_target": [item["p_target"] for item in noise_rates],
        "shuffle_p_target": rates_from_records(shuffle_recs)["p_target"],
        "shuffle_delta": effect_from_rates(base_rates, rates_from_records(shuffle_recs)),
    }


def scan_layer_appearance(
    model,
    editor,
    tokenizer,
    protocol: dict,
    row: dict,
    layer: int,
    donor_site: torch.Tensor,
    recipient_site: torch.Tensor,
    seeds: list[int],
    use_cache: bool,
    base_rates: dict,
) -> dict:
    k = int(protocol["sampling"]["discovery_K"])
    donor_recs = sample_condition(
        model, editor, tokenizer, protocol, row, "donor", same_replacements(donor_site, [layer]), k, use_cache
    )
    donor_rates = rates_from_records(donor_recs)
    noise_deltas = []
    noise_p = []
    for seed in seeds:
        recs = sample_condition(
            model,
            editor,
            tokenizer,
            protocol,
            row,
            f"noise{seed}",
            noise_replacements(row, donor_site, recipient_site, [layer], seed),
            k,
            use_cache,
            noise_seed=seed,
        )
        rates = rates_from_records(recs)
        noise_p.append(rates["p_target"])
        noise_deltas.append(effect_from_rates(base_rates, rates))
    delta_donor = effect_from_rates(base_rates, donor_rates)
    delta_noise_mean = loc.mean(noise_deltas)
    return {
        "pair_id": row["pair_id"],
        "split": row["split"],
        "layer": layer,
        "signal_pass": row["signal_pass"],
        "p_target_base": base_rates["p_target"],
        "p_target_donor": donor_rates["p_target"],
        "noise_p_target": noise_p,
        "delta_donor": delta_donor,
        "delta_noise_mean": delta_noise_mean,
        "D": delta_donor - delta_noise_mean,
        "generations_donor": donor_recs,
    }


def main() -> None:
    check_only = "--check" in sys.argv
    protocol = read_protocol()
    self_check(protocol)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = loc.read_jsonl(loc.CANDIDATE_DIR / "candidate_pairs.jsonl")
    donors = loc.read_jsonl(loc.CANDIDATE_DIR / "donor_entities.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(loc.SNAPSHOT, local_files_only=True)
    examples = build_examples(tokenizer, protocol, pairs, donors)
    loc.write_jsonl(
        OUT_DIR / "retain_probes.jsonl",
        [
            {
                "pair_id": row["pair_id"],
                "name": row["name"],
                "author_span": row.get("author_span"),
                "city": row["city"],
                "foil_city": row["foil_city"],
                "donor_id": row["donor_id"],
                "line_1based": row.get("line_1based"),
            }
            for row in examples
            if row["group"] == "retain"
        ],
    )
    if check_only:
        print(json.dumps({"check": "ok", "examples": len(examples)}, ensure_ascii=False), flush=True)
        return

    source = loc.check_p19(donors)
    (OUT_DIR / "source_gate.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not torch.cuda.is_available():
        write_blocked(protocol, source, "blocked_no_cuda")
        return

    tokenizer, model = loc.load_model()
    editor = ResidualEditor(model)
    forget = [row for row in examples if row["group"] == "forget"]
    retain = [row for row in examples if row["group"] == "retain"]
    generations: list[dict] = []
    aux_m: list[dict] = []
    scan_rows: list[dict] = []
    confirmation_rows: list[dict] = []
    layers: list[int] = []
    medians: dict[str, float | None] = {}
    nomination_status = "BLOCKED"
    claim = {"status": "BLOCKED", "reason": "upstream_gate_failed"}
    decode_path = "full_forward"
    use_cache = False

    try:
        read_rows, donor_sites = read_donors(model, editor, forget)
        for row in examples:
            print(f"sites {row['pair_id']}", flush=True)
            sites, meta = capture_sites(model, editor, row["prompt_ids"], row["qmark"])
            row["prompt_sites"] = sites
            row["hook_diff"] = meta["hook_diff"]

        # Correctness on first forget probe.
        probe = forget[0]
        print(f"correctness {probe['pair_id']}", flush=True)
        steps = 16
        base_greedy = greedy_ids(model, editor, probe["prompt_ids"], probe["qmark"], None, steps)
        noop_greedy = greedy_ids(
            model,
            editor,
            probe["prompt_ids"],
            probe["qmark"],
            None,
            steps,
        )
        # true no-op rewrite
        editor.close()
        ids = list(probe["prompt_ids"])
        noop_ids: list[int] = []
        for _ in range(steps):
            with editor.rewrite_same_values(loc.PATCH_LAYERS, probe["qmark"]):
                logits, _hidden = loc.forward(model, ids)
            next_id = int(logits[-1].argmax().item())
            del logits
            ids.append(next_id)
            noop_ids.append(next_id)
        editor.close()
        self_ids = greedy_ids(
            model,
            editor,
            probe["prompt_ids"],
            probe["qmark"],
            same_replacements(probe["prompt_sites"], loc.PATCH_LAYERS),
            steps,
        )
        self_one_ids = greedy_ids(
            model,
            editor,
            probe["prompt_ids"],
            probe["qmark"],
            same_replacements(probe["prompt_sites"], [0]),
            steps,
        )
        # cache vs full sampling identity
        seed = sample_seed(protocol, probe["group"], probe["pair_index"], 0, 0)
        full_text, full_hit, full_new = generate_full(
            model, editor, tokenizer, probe["prompt_ids"], probe["qmark"], None, 16, seed, 1.0
        )
        cache_text, cache_hit, cache_new = generate_cache(
            model, editor, tokenizer, probe["prompt_ids"], probe["qmark"], None, 16, seed, 1.0
        )
        cache_match = full_new == cache_new
        use_cache = cache_match
        decode_path = "cache" if use_cache else "full_forward"
        downstream = loc.downstream_changed(
            model,
            editor,
            probe["prompt_ids"],
            probe["qmark"],
            loc.forward(model, probe["prompt_ids"])[1],
        )
        correct_row = {
            "pair_id": probe["pair_id"],
            "noop_greedy_match": noop_ids == base_greedy,
            "self_patch_all_greedy_match": self_ids == base_greedy,
            "self_patch_layer0_greedy_match": self_one_ids == base_greedy,
            "cache_vs_full_token_match": cache_match,
            "cache_vs_full_n_tokens": [len(full_new), len(cache_new)],
            "downstream_input_max_abs": downstream,
            "hook_diff": probe["hook_diff"],
            "unused_plain_greedy_match": noop_greedy == base_greedy,
        }
        correct_pass = (
            correct_row["noop_greedy_match"]
            and correct_row["self_patch_all_greedy_match"]
            and correct_row["self_patch_layer0_greedy_match"]
            and correct_row["downstream_input_max_abs"] is not None
            and correct_row["downstream_input_max_abs"] > 0.0
            and all(row["hook_diff"] == 0.0 for row in examples)
        )
        # cache mismatch does not fail the whole run; it forces full_forward.
        cache_status = "PASS" if cache_match else "FAIL"

        read_pass = all(
            row["cache_error"] is None
            and row["site_error"] is None
            and row["cache_question_mark_max_abs"] == 0.0
            and row["hooked_vs_clean_hidden_max_abs"] == 0.0
            and row["finite"]
            for row in read_rows
        ) and all(row["hook_diff"] == 0.0 for row in examples)

        # Baselines.
        for row in examples:
            k = (
                int(protocol["sampling"]["discovery_K"])
                if row["split"] == "discovery"
                else int(protocol["sampling"]["confirmation_K"] if row["group"] == "forget" else protocol["sampling"]["retain_K"])
            )
            print(f"baseline {row['pair_id']} K={k}", flush=True)
            records = sample_condition(model, editor, tokenizer, protocol, row, "base", None, k, use_cache)
            for item in records:
                item["condition"] = "baseline"
            generations.extend(records)
            rates = rates_from_records(records)
            row["baseline"] = rates
            row["signal_pass"] = rates["p_target"] >= 0.5

        loc.write_jsonl(
            OUT_DIR / "appearance_baseline.jsonl",
            [
                {
                    "pair_id": row["pair_id"],
                    "group": row["group"],
                    "split": row["split"],
                    "city": row["city"],
                    "foil_city": row["foil_city"],
                    "donor_id": row["donor_id"],
                    "p_target": row["baseline"]["p_target"],
                    "p_foil": row["baseline"]["p_foil"],
                    "M_appear": row["baseline"]["M_appear"],
                    "K": row["baseline"]["K"],
                    "label_counts": row["baseline"]["label_counts"],
                    "signal_pass": row["signal_pass"],
                }
                for row in examples
            ],
        )
        loc.write_jsonl(OUT_DIR / "generation.jsonl", generations)

        discovery = [row for row in forget if row["split"] == "discovery"]
        eligible_discovery = [row for row in discovery if row["signal_pass"]]
        min_eligible = int(protocol["nomination"]["min_eligible_discovery_clusters"])

        if read_pass and correct_pass and len(eligible_discovery) >= min_eligible:
            print("discovery scan", flush=True)
            seeds = protocol["noise"]["discovery_seeds"]
            for layer in loc.PATCH_LAYERS:
                print(f"layer {layer}", flush=True)
                for row in discovery:
                    result = scan_layer_appearance(
                        model,
                        editor,
                        tokenizer,
                        protocol,
                        row,
                        layer,
                        donor_sites[row["donor_id"]],
                        row["prompt_sites"],
                        seeds,
                        use_cache,
                        row["baseline"],
                    )
                    # drop bulky generation text from layer scan
                    result.pop("generations_donor", None)
                    scan_rows.append(result)
            for layer in loc.PATCH_LAYERS:
                values = [row["D"] for row in scan_rows if row["layer"] == layer and row["signal_pass"]]
                medians[str(layer)] = loc.median(values)
            numeric = {int(layer): value for layer, value in medians.items() if value is not None}
            layers = loc.nominate(numeric)
            nomination_status = "PASS" if layers else "FAIL"
            frozen = {
                "layers": layers,
                "medians": medians,
                "eligible_discovery_pairs": [row["pair_id"] for row in eligible_discovery],
                "metric": "D = (p_base - p_donor) - mean(p_base - p_noise)",
                "decode_path": decode_path,
            }
            (OUT_DIR / "frozen_S.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            loc.write_jsonl(OUT_DIR / "layer_scan.jsonl", scan_rows)
            layers = json.loads((OUT_DIR / "frozen_S.json").read_text(encoding="utf-8"))["layers"]
            if layers:
                print("confirmation", flush=True)
                mapping = protocol["shuffle"]["mapping"]
                selected = [row for row in examples if row["split"] in {"confirmation", "retain"}]
                conf_seeds = protocol["noise"]["confirmation_seeds"]
                for row in selected:
                    print(f"joint {row['pair_id']}", flush=True)
                    shuffle_id = mapping[row["donor_id"]]
                    confirmation_rows.append(
                        joint_appearance(
                            model,
                            editor,
                            tokenizer,
                            protocol,
                            row,
                            layers,
                            donor_sites[row["donor_id"]],
                            row["prompt_sites"],
                            conf_seeds,
                            donor_sites[shuffle_id],
                            use_cache,
                            generations,
                        )
                    )
                    before = score_preference(model, editor, row, None)
                    after = score_preference(
                        model, editor, row, same_replacements(donor_sites[row["donor_id"]], layers)
                    )
                    aux_m.append(
                        {
                            "pair_id": row["pair_id"],
                            "group": row["group"],
                            "m_before": before["m"],
                            "m_after": after["m"],
                            "delta_m": before["m"] - after["m"],
                            "avg_plus_before": before["avg_plus"],
                            "avg_plus_after": after["avg_plus"],
                            "avg_minus_before": before["avg_minus"],
                            "avg_minus_after": after["avg_minus"],
                            "delta_avg_plus": before["avg_plus"] - after["avg_plus"],
                            "delta_avg_minus": before["avg_minus"] - after["avg_minus"],
                            "layers": layers,
                        }
                    )
                claim = claim_status(
                    [row for row in confirmation_rows if row["group"] == "forget"],
                    [row for row in confirmation_rows if row["group"] == "retain"],
                    layers,
                )
            else:
                claim = {"status": "FAIL", "reason": "empty_S"}
        else:
            if read_pass and correct_pass and len(eligible_discovery) < min_eligible:
                nomination_status = "BLOCKED"
                claim = {
                    "status": "BLOCKED",
                    "reason": "insufficient_discovery_appearance",
                    "eligible_discovery_count": len(eligible_discovery),
                    "min_eligible_discovery_clusters": min_eligible,
                }
            loc.write_jsonl(OUT_DIR / "layer_scan.jsonl", [])
            (OUT_DIR / "frozen_S.json").write_text(
                json.dumps(
                    {
                        "layers": [],
                        "reason": claim.get("reason", "upstream_gate_failed"),
                        "eligible_discovery_count": len(eligible_discovery),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        editor.close()

    signal_rows = [row for row in examples if row["group"] == "forget"]
    signal_pass_count = sum(row["signal_pass"] for row in signal_rows)
    read_status = "PASS" if "read_pass" in locals() and read_pass else "FAIL"
    correct_status = "PASS" if "correct_pass" in locals() and correct_pass else "FAIL"
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
        "baselines": [
            {
                "pair_id": row["pair_id"],
                "city": row["city"],
                "foil_city": row["foil_city"],
                "p_target": row["baseline"]["p_target"],
                "p_foil": row["baseline"]["p_foil"],
                "M_appear": row["baseline"]["M_appear"],
                "signal_pass": row["signal_pass"],
            }
            for row in examples
            if "baseline" in row
        ],
        "aux_m": aux_m,
        "discovery_ids": protocol["splits"]["discovery_pairs"],
        "confirmation_ids": protocol["splits"]["confirmation_pairs"],
        "layers": layers,
        "decode_path": decode_path,
        "overall": overall,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    gates = [
        {"gate": "source", "status": source["status"], "detail": source},
        {"gate": "read", "status": read_status, "rows": read_rows if "read_rows" in locals() else []},
        {
            "gate": "target_extractability",
            "status": signal_status,
            "rows": [
                {
                    "pair_id": row["pair_id"],
                    "split": row["split"],
                    "p_target": row["baseline"]["p_target"],
                    "signal_pass": row["signal_pass"],
                }
                for row in signal_rows
                if "baseline" in row
            ],
        },
        {
            "gate": "patch_correctness",
            "status": correct_status,
            "cache_vs_full": cache_status if "cache_status" in locals() else "BLOCKED",
            "rows": [correct_row] if "correct_row" in locals() else [],
        },
        {"gate": "nomination", "status": nomination_status, "layers": layers, "medians": medians},
        {"gate": "confirmation_claim", "status": claim["status"], "detail": claim},
        {"gate": "teacher_student", "status": "not_started"},
        {"gate": "auxiliary_fixed_prefix_M", "status": "reported" if aux_m else "skipped", "role": "not a gate"},
    ]
    loc.write_jsonl(OUT_DIR / "gates.jsonl", gates)
    loc.write_jsonl(OUT_DIR / "confirmation.jsonl", confirmation_rows)
    loc.write_jsonl(OUT_DIR / "generation.jsonl", generations)
    loc.write_jsonl(OUT_DIR / "auxiliary_M.jsonl", aux_m)
    (OUT_DIR / "summary.md").write_text(render_summary(context), encoding="utf-8")
    write_hashes()
    print(headline, flush=True)
    print(json.dumps({"overall": overall, "layers": layers, "decode_path": decode_path}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

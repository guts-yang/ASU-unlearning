#!/usr/bin/env python3
"""Exploratory residual localization for prompt-only activation donors.

Freezes the protocol in protocol.json, then runs read, signal, patch,
discovery, and confirmation gates. It does not modify TOFU files, candidate
JSONL, my_llama.py, or the ASU trainer, and it does not start training.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import traceback
import urllib.error
import urllib.request
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
from my_models.my_llama import LlamaForCausalLM  # noqa: E402
from patch_editor import ResidualEditor  # noqa: E402

OUT_DIR = REPO / "workspace/data/external_donor_prompt_activation_v0.3_localization"
CANDIDATE_DIR = REPO / "workspace/data/external_donor_prompt_activation_v0.3"
ACTIVATION_DIR = REPO / "workspace/data/external_donor_prompt_activation_v0.3_activation"
RETAIN_PATH = REPO / "workspace/data/tofu/retain95.jsonl"
SNAPSHOT = Path(
    "/root/autodl-tmp/huggingface/hub/"
    "models--open-unlearning--tofu_Llama-2-7b-chat-hf_full/"
    "snapshots/cc5b31c69127d5da881608e640f2b453c446435b"
)
DEVICE = torch.device("cuda:0")
PATCH_LAYERS = list(range(32))
USER_AGENT = "asu-unlearning-direction-i-localization/0.3"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def nominate(medians: dict[int, float]) -> list[int]:
    eligible = [layer for layer, value in medians.items() if value > 0]
    eligible.sort(key=lambda layer: (-medians[layer], layer))
    return eligible[:3]


def noise_seed(group: str, pair_index: int, layer: int, seed: int) -> int:
    retain_offset = 1000000007 if group == "retain" else 0
    return seed + 10007 * layer + 1000003 * pair_index + retain_offset


def matched_noise(recipient: torch.Tensor, donor: torch.Tensor, seed: int) -> tuple[torch.Tensor, float]:
    recipient32 = recipient.detach().float().cpu()
    donor32 = donor.detach().float().cpu()
    scale = torch.linalg.vector_norm(donor32 - recipient32, ord=2)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    epsilon = torch.randn(recipient32.shape, generator=generator, dtype=torch.float32)
    denom = torch.linalg.vector_norm(epsilon, ord=2)
    if float(denom) == 0.0:
        raise RuntimeError("noise direction has zero norm")
    null = recipient32 + float(scale) * epsilon / denom
    return null.to(dtype=torch.bfloat16), float(scale)


def prompt_text(question: str, protocol: dict) -> str:
    return protocol["question_start"] + question + protocol["question_end"]


def question_mark_index(tokenizer, token_ids: list[int]) -> int:
    marks = [
        index
        for index, token_id in enumerate(token_ids)
        if tokenizer.convert_ids_to_tokens([token_id])[0] == "?"
    ]
    if len(marks) != 1:
        raise RuntimeError(f"expected one '?' token, found {len(marks)}")
    return marks[0]


def answer_ids(tokenizer, prompt_ids: list[int], prompt: str, city: str) -> list[int]:
    full_ids = tokenizer(prompt + city, add_special_tokens=True)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(f"city continuation changes the prompt prefix for {city!r}")
    suffix = full_ids[len(prompt_ids) :]
    if not suffix:
        raise RuntimeError(f"city continuation is empty for {city!r}")
    return suffix


def starts_with_city(text: str, city: str) -> bool:
    stripped = text.lstrip()
    if stripped == city:
        return True
    return stripped.startswith(city) and (
        len(stripped) == len(city) or not stripped[len(city)].isalnum()
    )


def reproduce_shuffle(protocol: dict) -> None:
    donors = [f"donor-{index:02d}" for index in range(1, 11)]
    generator = random.Random(protocol["shuffle"]["seed"])
    order = donors[:]
    generator.shuffle(order)
    mapping = dict(zip(donors, order))
    if any(source == target for source, target in mapping.items()):
        raise RuntimeError("shuffle seed produced a fixed point")
    if mapping != protocol["shuffle"]["mapping"]:
        raise RuntimeError(f"shuffle mapping drifted: {mapping}")


def select_retain(tokenizer, protocol: dict, pairs: list[dict]) -> tuple[list[dict], dict]:
    question_re = re.compile(protocol["retain"]["question_regex"])
    patterns = [re.compile(item) for item in protocol["retain"]["city_patterns"]]
    names = protocol["retain"]["city_pattern_names"]
    clean_entities = [pair["clean_entity"] for pair in pairs]
    selected = []
    excluded = {"clean_overlap": 0, "duplicate": 0, "ambiguous_city": 0, "token": 0}
    seen = set()
    for line_number, line in enumerate(RETAIN_PATH.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        question = row["question"]
        if question_re.match(question) is None:
            continue
        author = question[len("Where was ") : -len(" born?")]
        if any(entity in question for entity in clean_entities):
            excluded["clean_overlap"] += 1
            continue
        if author in seen:
            excluded["duplicate"] += 1
            continue
        matches = []
        for pattern_index, pattern in enumerate(patterns):
            found = pattern.match(row["answer"])
            if found is None:
                continue
            city = found.group("city")
            if city.startswith("the ") or " city of " in city:
                continue
            matches.append((names[pattern_index], city, found.group("country")))
        if len(matches) != 1:
            excluded["ambiguous_city"] += 1
            continue
        pattern_name, city, country = matches[0]
        prompt = prompt_text(question, protocol)
        try:
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            suffix = answer_ids(tokenizer, prompt_ids, prompt, city)
            qmark = question_mark_index(tokenizer, prompt_ids)
        except RuntimeError:
            excluded["token"] += 1
            continue
        seen.add(author)
        donor_id = protocol["retain"]["donor_cycle"][len(selected)]
        selected.append(
            {
                "line_1based": line_number,
                "author_span": author,
                "city": city,
                "country": country,
                "pattern": pattern_name,
                "donor_id": donor_id,
                "question": question,
                "prompt_ids": prompt_ids,
                "answer_ids": suffix,
                "qmark": qmark,
            }
        )
        if len(selected) == 10:
            break
    return selected, excluded


def assert_retain(selected: list[dict], protocol: dict) -> None:
    expected = protocol["retain"]["expected_probes"]
    comparable = [
        {key: row[key] for key in ("line_1based", "author_span", "city", "country", "pattern", "donor_id")}
        for row in selected
    ]
    if comparable != expected:
        raise RuntimeError(f"retain selection drifted from the frozen list: {comparable}")


def self_check() -> None:
    if median([1, 2, 3, 4]) != 2.5:
        raise RuntimeError("even median drifted")
    if median([1, 2, 3]) != 2:
        raise RuntimeError("odd median drifted")
    chosen = nominate({0: 0.2, 1: 0.2, 2: -1.0, 3: 0.5, 4: 0.4})
    if chosen != [3, 4, 0]:
        raise RuntimeError(f"nomination tie break drifted: {chosen}")
    if nominate({0: -0.1, 1: 0.0}) != []:
        raise RuntimeError("non-positive medians entered S")
    vector = matched_noise(torch.zeros(4, dtype=torch.bfloat16), torch.ones(4, dtype=torch.bfloat16), 0)[0]
    if vector.shape != (4,) or vector.dtype != torch.bfloat16:
        raise RuntimeError("noise vector shape drifted")
    if noise_seed("forget", 1, 2, 3) == noise_seed("retain", 1, 2, 3):
        raise RuntimeError("retain noise stream is not separated")


def p19_place_ids(entity: dict) -> list[str]:
    claims = entity.get("claims", {}).get("P19", [])
    preferred = []
    normal = []
    for claim in claims:
        snak = claim.get("mainsnak", {})
        if snak.get("snaktype") != "value":
            continue
        place_id = snak.get("datavalue", {}).get("value", {}).get("id")
        if not place_id:
            continue
        if claim.get("rank") == "preferred":
            preferred.append(place_id)
        elif claim.get("rank") == "normal":
            normal.append(place_id)
    return preferred or normal


def fetch_wikidata(qid: str) -> dict:
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))
    entity = payload.get("entities", {}).get(qid)
    if entity is None:
        raise RuntimeError(f"Wikidata payload has no {qid}")
    return entity


def check_p19(donors: list[dict]) -> dict:
    checked = []
    try:
        fetch_wikidata(donors[0]["wikidata_qid"])
    except (OSError, urllib.error.URLError, json.JSONDecodeError, RuntimeError) as exc:
        return {
            "status": "BLOCKED",
            "reason": "blocked_no_network_or_unreadable_wikidata",
            "error": repr(exc),
            "checked": checked,
        }
    for donor in donors:
        qid = donor["wikidata_qid"]
        try:
            person = fetch_wikidata(qid)
            place_ids = p19_place_ids(person)
            if len(place_ids) != 1:
                checked.append(
                    {
                        "donor_id": donor["donor_id"],
                        "qid": qid,
                        "status": "FAIL",
                        "reason": "ambiguous_or_missing_p19",
                        "place_ids": place_ids,
                    }
                )
                continue
            place = fetch_wikidata(place_ids[0])
            label = place.get("labels", {}).get("en", {}).get("value")
            status = "PASS" if label == donor["donor_fact_value"] else "FAIL"
            checked.append(
                {
                    "donor_id": donor["donor_id"],
                    "qid": qid,
                    "place_qid": place_ids[0],
                    "english_label": label,
                    "donor_fact_value": donor["donor_fact_value"],
                    "status": status,
                }
            )
        except (OSError, urllib.error.URLError, json.JSONDecodeError, RuntimeError) as exc:
            return {
                "status": "BLOCKED",
                "reason": "blocked_during_p19_review",
                "error": repr(exc),
                "checked": checked,
            }
    if any(row["status"] == "FAIL" for row in checked):
        status = "FAIL"
    elif len(checked) == len(donors) and all(row["status"] == "PASS" for row in checked):
        status = "PASS"
    else:
        status = "BLOCKED"
    return {"status": status, "checked": checked}


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        str(SNAPSHOT),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.to(DEVICE)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.config.num_hidden_layers != 32 or model.config.hidden_size != 4096:
        raise RuntimeError("model shape is not 32 layers by 4096")
    if len(model.model.layers) != 32:
        raise RuntimeError("decoder layer count is not 32")
    return tokenizer, model


def forward(model, token_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    logits = outputs.logits[0].detach()
    hidden = torch.stack([state[0].detach() for state in outputs.hidden_states], dim=0)
    return logits, hidden


def score_logits(logits: torch.Tensor, prompt_length: int, city_ids: list[int]) -> float:
    values = []
    for offset, token_id in enumerate(city_ids):
        position = prompt_length - 1 + offset
        log_prob = torch.log_softmax(logits[position].float(), dim=-1)[token_id]
        values.append(log_prob)
    return float(torch.stack(values).mean().item())


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def answer_logit_diff(left: torch.Tensor, right: torch.Tensor, prompt_length: int, count: int) -> float:
    start = prompt_length - 1
    return max_abs(left[start : start + count], right[start : start + count])


def greedy_text(model, editor: ResidualEditor, tokenizer, prompt_ids: list[int], steps: int) -> str:
    editor.close()
    ids = list(prompt_ids)
    for _ in range(steps):
        logits, _hidden = forward(model, ids)
        ids.append(int(logits[-1].argmax().item()))
    return tokenizer.decode(ids[len(prompt_ids) :], skip_special_tokens=True)


def site_vectors(hidden: torch.Tensor, qmark: int, captured: dict[int, torch.Tensor]) -> torch.Tensor:
    sites = []
    for layer in PATCH_LAYERS:
        if layer not in captured:
            raise RuntimeError(f"missing hook output for layer {layer}")
        if layer <= 30:
            reference = hidden[layer + 1, qmark]
            if max_abs(reference.cpu(), captured[layer].cpu()) != 0.0:
                raise RuntimeError(f"hook output at layer {layer} does not match hidden_states[{layer + 1}]")
            sites.append(reference.cpu())
        else:
            sites.append(captured[layer].cpu())
    return torch.stack(sites, dim=0)


def downstream_changed(model, editor: ResidualEditor, token_ids: list[int], qmark: int, hidden: torch.Tensor) -> float:
    editor.close()
    box_base: dict = {}
    with editor.watch_input(1, box_base):
        forward(model, token_ids)
    probe = hidden[1, qmark].detach().clone()
    probe[0] = torch.tensor(0 if float(probe[0]) != 0.0 else 1, dtype=probe.dtype, device=probe.device)
    box_probe: dict = {}
    with editor.replace({0: probe}, qmark):
        editor.watch_input(1, box_probe)
        forward(model, token_ids)
    if "hidden" not in box_base or "hidden" not in box_probe:
        raise RuntimeError("downstream watch did not record a layer input")
    return max_abs(box_base["hidden"], box_probe["hidden"])


def prepare_examples(tokenizer, protocol: dict, pairs: list[dict], retain_rows: list[dict]) -> list[dict]:
    examples = []
    for pair in pairs:
        prompt = prompt_text(pair["q_plus"], protocol)
        if pair["y_plus"] in prompt or pair["donor_fact_value"] in prompt_text(pair["q_minus"], protocol):
            raise RuntimeError(f"{pair['pair_id']} puts a fact value in the prompt")
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        city_ids = answer_ids(tokenizer, prompt_ids, prompt, pair["y_plus"])
        donor_prompt = prompt_text(pair["q_minus"], protocol)
        donor_ids = tokenizer(donor_prompt, add_special_tokens=True)["input_ids"]
        split = "discovery" if pair["pair_id"] in protocol["splits"]["discovery_pairs"] else "confirmation"
        examples.append(
            {
                "group": "forget",
                "split": split,
                "pair_id": pair["pair_id"],
                "pair_index": int(pair["pair_id"][-2:]) - 1,
                "question": pair["q_plus"],
                "city": pair["y_plus"],
                "donor_id": pair["donor_id"],
                "donor_question": pair["q_minus"],
                "prompt_ids": prompt_ids,
                "donor_ids": donor_ids,
                "answer_ids": city_ids,
                "qmark": question_mark_index(tokenizer, prompt_ids),
                "donor_qmark": question_mark_index(tokenizer, donor_ids),
                "full_ids": prompt_ids + city_ids,
            }
        )
    for index, row in enumerate(retain_rows):
        examples.append(
            {
                "group": "retain",
                "split": "retain",
                "pair_id": f"retain-{index + 1:02d}",
                "pair_index": index,
                "question": row["question"],
                "city": row["city"],
                "donor_id": row["donor_id"],
                "prompt_ids": row["prompt_ids"],
                "answer_ids": row["answer_ids"],
                "qmark": row["qmark"],
                "full_ids": row["prompt_ids"] + row["answer_ids"],
                "line_1based": row["line_1based"],
                "author_span": row["author_span"],
            }
        )
    return examples


def run_gates(model, tokenizer, editor: ResidualEditor, protocol: dict, examples: list[dict]) -> dict:
    seeds = protocol["noise"]["seeds"]
    steps = int(protocol["target_signal"]["max_new_tokens"])
    read_rows = []
    signal_rows = []
    correct_rows = []
    donor_sites: dict[str, torch.Tensor] = {}
    base_state: dict[str, dict] = {}

    forget = [row for row in examples if row["group"] == "forget"]
    for row in forget:
        print(f"read {row['pair_id']}", flush=True)
        donor_logits, donor_hidden = forward(model, row["donor_ids"])
        del donor_logits
        with editor.capture(row["donor_qmark"]):
            _hook_logits, hook_hidden = forward(model, row["donor_ids"])
        captured = {key: value.cpu() for key, value in editor.captured.items()}
        donor_hidden = donor_hidden.cpu()
        hook_hidden = hook_hidden.cpu()
        cached = load_file(str(ACTIVATION_DIR / f"{row['pair_id']}.safetensors"))
        cache_hidden = cached["hidden_q_minus"]
        cache_diff = None
        cache_error = None
        if tuple(cache_hidden.shape) != tuple(donor_hidden.shape):
            cache_error = f"shape {tuple(cache_hidden.shape)} != {tuple(donor_hidden.shape)}"
        else:
            cache_diff = max_abs(donor_hidden[:, row["donor_qmark"], :], cache_hidden[:, row["donor_qmark"], :])
        hook_diff = max_abs(donor_hidden, hook_hidden)
        try:
            sites = site_vectors(donor_hidden, row["donor_qmark"], captured)
            site_error = None
        except RuntimeError as exc:
            sites = None
            site_error = str(exc)
        if sites is not None:
            donor_sites[row["donor_id"]] = sites
        read_rows.append(
            {
                "pair_id": row["pair_id"],
                "qmark": row["qmark"],
                "donor_qmark": row["donor_qmark"],
                "cache_question_mark_max_abs": cache_diff,
                "cache_error": cache_error,
                "hooked_vs_clean_hidden_max_abs": hook_diff,
                "site_error": site_error,
                "finite": bool(torch.isfinite(donor_hidden.float()).all().item()) if sites is not None else False,
            }
        )

        clean_logits, clean_hidden = forward(model, row["full_ids"])
        with editor.capture(row["qmark"]):
            hooked_logits, _hooked_hidden = forward(model, row["full_ids"])
        captured_clean = {key: value.cpu() for key, value in editor.captured.items()}
        clean_hidden = clean_hidden.cpu()
        clean_logits = clean_logits.cpu()
        hooked_logits = hooked_logits.cpu()
        try:
            recipient = site_vectors(clean_hidden, row["qmark"], captured_clean)
            recipient_error = None
        except RuntimeError as exc:
            recipient = None
            recipient_error = str(exc)
        base_score = score_logits(clean_logits, len(row["prompt_ids"]), row["answer_ids"])
        if recipient is not None:
            base_state[row["pair_id"]] = {
                "logits": clean_logits,
                "hidden": clean_hidden,
                "sites": recipient,
                "score": base_score,
            }
        generated = greedy_text(model, editor, tokenizer, row["prompt_ids"], steps)
        signal_rows.append(
            {
                "pair_id": row["pair_id"],
                "split": row["split"],
                "city": row["city"],
                "base_score": base_score,
                "greedy_text": generated,
                "signal_pass": starts_with_city(generated, row["city"]),
                "recipient_error": recipient_error,
            }
        )

        noop_diff = None
        self_one = None
        self_all = None
        downstream = None
        correct_error = None
        try:
            with editor.rewrite_same_values(PATCH_LAYERS, row["qmark"]):
                noop_logits, _hidden = forward(model, row["full_ids"])
            noop_diff = answer_logit_diff(
                clean_logits, noop_logits.cpu(), len(row["prompt_ids"]), len(row["answer_ids"])
            )
            if recipient is None:
                raise RuntimeError(recipient_error or "missing recipient sites")
            with editor.replace({0: recipient[0]}, row["qmark"]):
                one_logits, _hidden = forward(model, row["full_ids"])
            self_one = answer_logit_diff(
                clean_logits, one_logits.cpu(), len(row["prompt_ids"]), len(row["answer_ids"])
            )
            with editor.replace({layer: recipient[layer] for layer in PATCH_LAYERS}, row["qmark"]):
                all_logits, _hidden = forward(model, row["full_ids"])
            self_all = answer_logit_diff(
                clean_logits, all_logits.cpu(), len(row["prompt_ids"]), len(row["answer_ids"])
            )
            downstream = downstream_changed(model, editor, row["full_ids"], row["qmark"], clean_hidden.to(DEVICE))
        except RuntimeError as exc:
            correct_error = str(exc)
        correct_rows.append(
            {
                "pair_id": row["pair_id"],
                "noop_answer_logit_max_abs": noop_diff,
                "self_patch_layer0_answer_logit_max_abs": self_one,
                "self_patch_all_layers_answer_logit_max_abs": self_all,
                "downstream_input_max_abs": downstream,
                "hooked_score_logit_max_abs": answer_logit_diff(
                    clean_logits, hooked_logits, len(row["prompt_ids"]), len(row["answer_ids"])
                ),
                "error": correct_error,
            }
        )

    read_pass = all(
        row["cache_error"] is None
        and row["site_error"] is None
        and row["cache_question_mark_max_abs"] == 0.0
        and row["hooked_vs_clean_hidden_max_abs"] == 0.0
        and row["finite"]
        for row in read_rows
    )
    correct_pass = all(
        row["error"] is None
        and row["noop_answer_logit_max_abs"] == 0.0
        and row["self_patch_layer0_answer_logit_max_abs"] == 0.0
        and row["self_patch_all_layers_answer_logit_max_abs"] == 0.0
        and row["hooked_score_logit_max_abs"] == 0.0
        and row["downstream_input_max_abs"] is not None
        and row["downstream_input_max_abs"] > 0.0
        for row in correct_rows
    )
    return {
        "read_rows": read_rows,
        "signal_rows": signal_rows,
        "correct_rows": correct_rows,
        "read_status": "PASS" if read_pass else "FAIL",
        "correct_status": "PASS" if correct_pass else "FAIL",
        "donor_sites": donor_sites,
        "base_state": base_state,
        "seeds": seeds,
    }


def patched_score(model, editor: ResidualEditor, row: dict, replacements: dict[int, torch.Tensor]) -> float:
    with editor.replace(replacements, row["qmark"]):
        logits, _hidden = forward(model, row["full_ids"])
    return score_logits(logits, len(row["prompt_ids"]), row["answer_ids"])


def scan_pair_layer(model, editor, row: dict, layer: int, donor_site: torch.Tensor, recipient_site: torch.Tensor, seeds: list[int]) -> dict:
    donor_score = patched_score(model, editor, row, {layer: donor_site[layer]})
    noise_scores = []
    scale = None
    for seed in seeds:
        mixed = noise_seed(row["group"], row["pair_index"], layer, seed)
        null, scale = matched_noise(recipient_site[layer], donor_site[layer], mixed)
        noise_scores.append(patched_score(model, editor, row, {layer: null}))
    base = row["base_score"]
    delta_donor = base - donor_score
    delta_noise = [base - value for value in noise_scores]
    return {
        "pair_id": row["pair_id"],
        "split": row["split"],
        "layer": layer,
        "signal_pass": row["signal_pass"],
        "base_score": base,
        "donor_score": donor_score,
        "noise_scores": noise_scores,
        "delta_donor": delta_donor,
        "delta_noise_mean": mean(delta_noise),
        "donor_specific": delta_donor - mean(delta_noise),
        "replacement_l2": scale,
    }


def joint_record(model, editor, row: dict, layers: list[int], donor_site: torch.Tensor, recipient_site: torch.Tensor, seeds: list[int], shuffle_site: torch.Tensor | None) -> dict:
    donor_score = patched_score(model, editor, row, {layer: donor_site[layer] for layer in layers})
    noise_scores = []
    for seed in seeds:
        replacements = {}
        for layer in layers:
            mixed = noise_seed(row["group"], row["pair_index"], layer, seed)
            null, _scale = matched_noise(recipient_site[layer], donor_site[layer], mixed)
            replacements[layer] = null
        noise_scores.append(patched_score(model, editor, row, replacements))
    shuffle_score = None
    if shuffle_site is not None:
        shuffle_score = patched_score(model, editor, row, {layer: shuffle_site[layer] for layer in layers})
    base = row["base_score"]
    delta_donor = base - donor_score
    delta_noise = mean([base - value for value in noise_scores])
    return {
        "group": row["group"],
        "pair_id": row["pair_id"],
        "split": row["split"],
        "city": row["city"],
        "donor_id": row["donor_id"],
        "signal_pass": row["signal_pass"],
        "layers": layers,
        "base_score": base,
        "donor_score": donor_score,
        "noise_scores": noise_scores,
        "shuffle_score": shuffle_score,
        "delta_donor": delta_donor,
        "delta_noise_mean": delta_noise,
        "donor_specific": delta_donor - delta_noise,
        "author_span": row.get("author_span"),
        "line_1based": row.get("line_1based"),
    }


def claim_status(forget_rows: list[dict], retain_rows: list[dict], layers: list[int]) -> dict:
    if not layers:
        return {"status": "FAIL", "reason": "empty_S"}
    positive = sum(row["signal_pass"] and row["donor_specific"] > 0 for row in forget_rows)
    eligible_forget = [row for row in forget_rows if row["signal_pass"]]
    eligible_retain = [row for row in retain_rows if row["signal_pass"]]
    forget_median = median([row["donor_specific"] for row in eligible_forget])
    if not eligible_retain:
        did_values = []
        did_median = None
        retain_specific = None
    else:
        retain_specific = mean([row["donor_specific"] for row in eligible_retain])
        did_values = [row["donor_specific"] - retain_specific for row in eligible_forget]
        did_median = median(did_values)
    four = {
        "forget_donor": mean([row["delta_donor"] for row in eligible_forget]),
        "forget_noise": mean([row["delta_noise_mean"] for row in eligible_forget]),
        "retain_donor": mean([row["delta_donor"] for row in eligible_retain]),
        "retain_noise": mean([row["delta_noise_mean"] for row in eligible_retain]),
    }
    four_all = {
        "forget_donor": mean([row["delta_donor"] for row in forget_rows]),
        "forget_noise": mean([row["delta_noise_mean"] for row in forget_rows]),
        "retain_donor": mean([row["delta_donor"] for row in retain_rows]),
        "retain_noise": mean([row["delta_noise_mean"] for row in retain_rows]),
    }
    if positive < 3:
        status, reason = "FAIL", "fewer_than_3_of_4_signal_positive"
    elif forget_median is None or forget_median <= 0:
        status, reason = "FAIL", "donor_specific_median_not_positive"
    elif not eligible_retain:
        status, reason = "BLOCKED", "no_signal_passing_retain_probe"
    elif did_median is None or did_median <= 0:
        status, reason = "FAIL", "selectivity_did_median_not_positive"
    else:
        status, reason = "PASS", "exploratory_threshold_met"
    return {
        "status": status,
        "reason": reason,
        "positive_signal_clusters": positive,
        "confirmation_cluster_count": len(forget_rows),
        "donor_specific_median": forget_median,
        "selectivity_did_median": did_median,
        "selectivity_did_values": did_values,
        "eligible_four_deltas": four,
        "all_cluster_four_deltas": four_all,
        "eligible_forget_ids": [row["pair_id"] for row in eligible_forget],
        "eligible_retain_ids": [row["pair_id"] for row in eligible_retain],
    }


def overall_label(source: str, read: str, correct: str, claim: str | None) -> str:
    if source != "PASS":
        if claim == "PASS":
            return "PILOT_ONLY"
        return "NO-GO"
    if read == "PASS" and correct == "PASS" and claim == "PASS":
        return "GATES_PASSED_TRAINING_NOT_STARTED"
    return "NO-GO"


def render_summary(context: dict) -> str:
    claim = context.get("claim") or {}
    lines = [
        "# v0.3 残差定位",
        "",
        "## 结论",
        "",
        f"**{context['headline']}**",
        "",
        context["wording"],
        "",
        "这是 10 个作者簇上的工程 pilot。正的 delta 只表示城市 span 的条件对数概率下降，不是遗忘成功。没有启动 Teacher/Student 训练。",
        "",
        "## 门",
        "",
        "| 门 | 状态 | 说明 |",
        "| --- | --- | --- |",
        f"| 来源 | {context['source_status']} | {context['source_note']} |",
        f"| 读取 | {context['read_status']} | 问号向量与缓存、hook 坐标 |",
        f"| 目标信号 | {context['signal_status']} | 通过 {context['signal_pass_count']} / {context['signal_count']} |",
        f"| patch 正确性 | {context['correct_status']} | no-op、self-patch 容差 0；下游输入必须改变 |",
        f"| 提名 S | {context['nomination_status']} | 层：{context['layers']} |",
        f"| confirmation 主张 | {context['claim_status']} | {claim.get('reason', '')} |",
        "",
        "## 冻结集合",
        "",
        f"- discovery：{', '.join(context['discovery_ids'])}",
        f"- confirmation：{', '.join(context['confirmation_ids'])}",
        f"- S：{context['layers']}",
        f"- 总体：`{context['overall']}`",
        "",
    ]
    if claim:
        four = claim.get("eligible_four_deltas", {})
        lines.extend(
            [
                "## 差中差",
                "",
                "下列均值只用目标信号通过的簇。原始逐簇分数在 `confirmation.jsonl`。",
                "",
                f"- forget donor：{four.get('forget_donor')}",
                f"- forget noise：{four.get('forget_noise')}",
                f"- retain donor：{four.get('retain_donor')}",
                f"- retain noise：{four.get('retain_noise')}",
                f"- donor_specific 中位数：{claim.get('donor_specific_median')}",
                f"- selectivity_DID 中位数：{claim.get('selectivity_did_median')}",
                f"- 信号通过且 donor_specific 为正的 confirmation 簇：{claim.get('positive_signal_clusters')}",
                "",
            ]
        )
    lines.extend(
        [
            "## 没有做的事",
            "",
            "没有修改候选、TOFU、retain95 或 `my_llama.py`。没有运行路径条件教师，也没有运行学生训练。",
            "",
            f"生成时间：`{context['generated_at']}`。",
            "",
        ]
    )
    return "\n".join(lines)


def headline_and_wording(overall: str, claim_status_value: str, source_status: str) -> tuple[str, str]:
    if overall == "PILOT_ONLY":
        return (
            "工程 pilot：数值门槛通过，来源门未通过。",
            "confirmation 上的预注册探索性门槛通过，但来源门不是 PASS。结论保持工程 pilot，不称为遗忘成功，不启动 Teacher/Student 训练。",
        )
    if claim_status_value == "PASS" and source_status == "PASS":
        return (
            "预注册门槛通过。训练未启动。",
            "定位和 retain 审计达到本文件写死的探索性门槛。按本阶段边界，不自动启动 Teacher/Student 训练。",
        )
    if source_status == "BLOCKED" and claim_status_value in {None, "not_run"}:
        return (
            "NO-GO。",
            "来源门为 BLOCKED。定位没有形成可称为目标选择性局部路径的证据。不启动 Teacher/Student 训练。",
        )
    return (
        "尚无证据支持目标选择性局部路径。",
        "donor 效应没有同时超过写死的噪声、confirmation 和 retain 门槛，或正确性/读取门未通过。不称为遗忘成功，不启动 Teacher/Student 训练。",
    )


def main() -> None:
    check_only = "--check" in sys.argv
    self_check()
    protocol = json.loads((OUT_DIR / "protocol.json").read_text(encoding="utf-8"))
    reproduce_shuffle(protocol)
    pairs = read_jsonl(CANDIDATE_DIR / "candidate_pairs.jsonl")
    donors = read_jsonl(CANDIDATE_DIR / "donor_entities.jsonl")
    if [pair["pair_id"] for pair in pairs] != [f"promptdonor-{index:02d}" for index in range(1, 11)]:
        raise RuntimeError("pair order drifted")
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    retain_rows, excluded = select_retain(tokenizer, protocol, pairs)
    assert_retain(retain_rows, protocol)
    public_retain = [
        {key: value for key, value in row.items() if key not in {"prompt_ids", "answer_ids"}}
        for row in retain_rows
    ]
    (OUT_DIR / "retain_probes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in public_retain),
        encoding="utf-8",
    )
    (OUT_DIR / "retain_selection.json").write_text(
        json.dumps({"excluded": excluded, "selected": len(retain_rows)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    source = check_p19(donors)
    (OUT_DIR / "source_gate.json").write_text(
        json.dumps(source, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if check_only:
        print(json.dumps({"check": "ok", "source": source["status"], "retain": len(retain_rows)}, ensure_ascii=False))
        return
    if not torch.cuda.is_available():
        write_blocked(protocol, source, "blocked_no_cuda")
        return

    model_tokenizer, model = load_model()
    del tokenizer
    tokenizer = model_tokenizer
    editor = ResidualEditor(model)
    examples = prepare_examples(tokenizer, protocol, pairs, retain_rows)
    try:
        early = run_gates(model, tokenizer, editor, protocol, examples)
    finally:
        editor.close()

    signal_by_id = {row["pair_id"]: row for row in early["signal_rows"]}
    for row in examples:
        if row["group"] == "forget":
            row["signal_pass"] = signal_by_id[row["pair_id"]]["signal_pass"]
            row["base_score"] = signal_by_id[row["pair_id"]]["base_score"]

    discovery_ids = protocol["splits"]["discovery_pairs"]
    confirmation_ids = protocol["splits"]["confirmation_pairs"]
    layers: list[int] = []
    medians: dict[str, float | None] = {}
    scan_rows: list[dict] = []
    confirmation_rows: list[dict] = []
    nomination_status = "not_run"
    claim = {"status": "not_run", "reason": "not_run"}

    if early["read_status"] == "PASS" and early["correct_status"] == "PASS":
        forget_examples = [row for row in examples if row["group"] == "forget"]
        discovery = [row for row in forget_examples if row["pair_id"] in discovery_ids]
        print("discovery scan", flush=True)
        for layer in PATCH_LAYERS:
            print(f"layer {layer}", flush=True)
            for row in discovery:
                scan_rows.append(
                    scan_pair_layer(
                        model,
                        editor,
                        row,
                        layer,
                        early["donor_sites"][row["donor_id"]],
                        early["base_state"][row["pair_id"]]["sites"],
                        protocol["noise"]["seeds"],
                    )
                )
        for layer in PATCH_LAYERS:
            values = [
                row["donor_specific"]
                for row in scan_rows
                if row["layer"] == layer and row["signal_pass"]
            ]
            medians[str(layer)] = median(values)
        numeric_medians = {layer: value for layer, value in medians.items() if value is not None}
        layers = nominate({int(layer): value for layer, value in numeric_medians.items()})
        nomination_status = "PASS" if layers else "FAIL"
        frozen = {
            "layers": layers,
            "medians": medians,
            "eligible_discovery_pairs": [row["pair_id"] for row in discovery if row["signal_pass"]],
        }
        (OUT_DIR / "frozen_S.json").write_text(
            json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_jsonl(OUT_DIR / "layer_scan.jsonl", scan_rows)
        layers = json.loads((OUT_DIR / "frozen_S.json").read_text(encoding="utf-8"))["layers"]
        if layers:
            print("confirmation", flush=True)
            shuffle = protocol["shuffle"]["mapping"]
            for row in forget_examples:
                if row["pair_id"] not in confirmation_ids:
                    continue
                shuffled = shuffle[row["donor_id"]]
                confirmation_rows.append(
                    joint_record(
                        model,
                        editor,
                        row,
                        layers,
                        early["donor_sites"][row["donor_id"]],
                        early["base_state"][row["pair_id"]]["sites"],
                        protocol["noise"]["seeds"],
                        early["donor_sites"][shuffled],
                    )
                )
            retain_examples = []
            for row in examples:
                if row["group"] != "retain":
                    continue
                print(f"retain {row['pair_id']}", flush=True)
                logits, hidden = forward(model, row["full_ids"])
                with editor.capture(row["qmark"]):
                    hooked_logits, _hidden = forward(model, row["full_ids"])
                captured = {key: value.cpu() for key, value in editor.captured.items()}
                if answer_logit_diff(logits.cpu(), hooked_logits.cpu(), len(row["prompt_ids"]), len(row["answer_ids"])) != 0.0:
                    raise RuntimeError(f"{row['pair_id']} retain capture changed answer logits")
                sites = site_vectors(hidden.cpu(), row["qmark"], captured)
                generated = greedy_text(
                    model,
                    editor,
                    tokenizer,
                    row["prompt_ids"],
                    int(protocol["target_signal"]["max_new_tokens"]),
                )
                row["signal_pass"] = starts_with_city(generated, row["city"])
                row["base_score"] = score_logits(logits, len(row["prompt_ids"]), row["answer_ids"])
                row["greedy_text"] = generated
                retain_examples.append(row)
                confirmation_rows.append(
                    joint_record(
                        model,
                        editor,
                        row,
                        layers,
                        early["donor_sites"][row["donor_id"]],
                        sites,
                        protocol["noise"]["seeds"],
                        None,
                    )
                )
            claim = claim_status(
                [row for row in confirmation_rows if row["group"] == "forget"],
                [row for row in confirmation_rows if row["group"] == "retain"],
                layers,
            )
        else:
            claim = {"status": "FAIL", "reason": "empty_S"}
    else:
        nomination_status = "BLOCKED"
        claim = {"status": "BLOCKED", "reason": "upstream_gate_failed"}

    write_outputs(
        protocol,
        source,
        early,
        scan_rows,
        confirmation_rows,
        layers,
        medians,
        nomination_status,
        claim,
        discovery_ids,
        confirmation_ids,
    )


def write_blocked(protocol: dict, source: dict, reason: str) -> None:
    context = {
        "headline": "BLOCKED。",
        "wording": f"环境阻断：{reason}。这不是模型拒绝，也不是科学假设失败。不启动 Teacher/Student 训练。",
        "source_status": source["status"],
        "source_note": source.get("reason") or source.get("error") or "",
        "read_status": "BLOCKED",
        "signal_status": "BLOCKED",
        "signal_pass_count": 0,
        "signal_count": 0,
        "correct_status": "BLOCKED",
        "nomination_status": "BLOCKED",
        "claim_status": "BLOCKED",
        "claim": {"status": "BLOCKED", "reason": reason},
        "discovery_ids": protocol["splits"]["discovery_pairs"],
        "confirmation_ids": protocol["splits"]["confirmation_pairs"],
        "layers": [],
        "overall": "NO-GO",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / "summary.md").write_text(render_summary(context), encoding="utf-8")
    write_jsonl(OUT_DIR / "gates.jsonl", [{"gate": "environment", "status": "BLOCKED", "reason": reason}])
    write_jsonl(OUT_DIR / "layer_scan.jsonl", [])
    write_jsonl(OUT_DIR / "confirmation.jsonl", [])
    write_hashes()


def write_outputs(protocol, source, early, scan_rows, confirmation_rows, layers, medians, nomination_status, claim, discovery_ids, confirmation_ids) -> None:
    signal_pass_count = sum(row["signal_pass"] for row in early["signal_rows"])
    signal_status = "PASS" if signal_pass_count else "FAIL"
    overall = overall_label(source["status"], early["read_status"], early["correct_status"], claim["status"])
    headline, wording = headline_and_wording(overall, claim["status"], source["status"])
    if early["read_status"] != "PASS" or early["correct_status"] != "PASS":
        headline = "尚无证据支持目标选择性局部路径。"
        wording = "读取门或 patch 正确性门未通过，定位扫描没有进入可解释的干预结果。不称为遗忘成功，不启动 Teacher/Student 训练。"
    context = {
        "headline": headline,
        "wording": wording,
        "source_status": source["status"],
        "source_note": source.get("reason") or source.get("error") or "P19 English label",
        "read_status": early["read_status"],
        "signal_status": signal_status,
        "signal_pass_count": signal_pass_count,
        "signal_count": len(early["signal_rows"]),
        "correct_status": early["correct_status"],
        "nomination_status": nomination_status,
        "claim_status": claim["status"],
        "claim": claim,
        "discovery_ids": discovery_ids,
        "confirmation_ids": confirmation_ids,
        "layers": layers,
        "overall": overall,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    gates = [
        {"gate": "source", "status": source["status"], "detail": source},
        {"gate": "read", "status": early["read_status"], "rows": early["read_rows"]},
        {"gate": "target_signal", "status": signal_status, "rows": early["signal_rows"]},
        {"gate": "patch_correctness", "status": early["correct_status"], "rows": early["correct_rows"]},
        {"gate": "nomination", "status": nomination_status, "layers": layers, "medians": medians},
        {"gate": "confirmation_claim", "status": claim["status"], "detail": claim},
        {"gate": "teacher_student", "status": "not_started"},
    ]
    write_jsonl(OUT_DIR / "gates.jsonl", gates)
    write_jsonl(OUT_DIR / "layer_scan.jsonl", scan_rows)
    write_jsonl(OUT_DIR / "confirmation.jsonl", confirmation_rows)
    (OUT_DIR / "summary.md").write_text(render_summary(context), encoding="utf-8")
    write_hashes()
    print(headline, flush=True)


def write_hashes() -> None:
    names = [
        "protocol.json",
        "patch_editor.py",
        "run_localization.py",
        "retain_probes.jsonl",
        "retain_selection.json",
        "source_gate.json",
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
            payload[name] = sha256_file(path)
    payload["candidate_pairs.jsonl"] = sha256_file(CANDIDATE_DIR / "candidate_pairs.jsonl")
    payload["retain95.jsonl"] = sha256_file(RETAIN_PATH)
    (OUT_DIR / "output_hashes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

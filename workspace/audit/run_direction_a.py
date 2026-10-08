#!/usr/bin/env python3
"""Direction A minimum audit on the replayed forget05 ASU+GD checkpoint.

Rules below are fixed before any score is read. Do not change them after
looking at outputs. No new loss, no student training, no LoRA on the quant path.

Forget set: TOFU forget05, task_id == 1 (the 200 questions the replay unlearned).
Headline hit: answer string contained in the generation, case-insensitive.
No LLM judge. No answer aliases in this split.

Prompt matches the Llama-2 eval interface: "[INST] {question} [/INST]".
max_new_tokens = 128. Greedy is do_sample=False.

Probab: temperatures 0.7, 1.0, 1.5, k = 8. A question hits if any of the 24
samples contains the answer. Paired delta = that bit minus the greedy bit.
Each (temperature, sample index) generation is seeded once with the sample index.

FocusOnKey: keytoken is the longest alphanumeric answer token of length >= 3
that is not in the stop list below; ties keep the earliest token. Injected
inside the same [INST] turn, because that is the format this model was
evaluated with:
    [INST] {question}
    KeyToken: {keytoken} [/INST]

QRA: zero-shot nf4 and fp4 (the scaffold's int4). No LoRA. Paired delta is
quantized greedy hit minus fp16 greedy hit. Utility proxy is greedy literal
hit rate on the first 100 rows of retain90.json. That is not official MU.

Rank profile: answer-span tokens, excluding eos. Teacher is the frozen
tofu_ft Llama with attention temperature 1.0 and 2.3 (the ASU teacher).
Student is checkpoint-last. Order-preserving leak uses the teacher pair:
median gold rank at tau 2.3 equals 1, and entropy rises or gold probability
falls. Greedy forget efficacy is the replay value 0.774.

Held-out control: Llama-2-7b-chat, never finetuned on TOFU, greedy on the
same 200 questions. Not an L4 channel.

L4 channels: probab, focus_on_key, qra_nf4, qra_int4.
Bootstrap: 2000 resamples, alpha 0.05, seed 0, statistic mean paired delta.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO / "Right-to-be-forgotten"))

from audit.go_nogo import channel_significant, l4_decision
from audit.match import answer_or_alias_contained
from audit.qra_int4 import bitsandbytes_load_kwargs
from audit.rank_profile import profile_logits

STUDENT_CKPT = (
    "/root/autodl-tmp/ASU-unlearning/checkpoints/tofu_replay/llama2-7b/forget05/ASU+GD/"
    "seed_1001/epoch5_1e-05_FixRefTrue_maskTrue_Fcoeff0.1_Rcoeff1.0_Temp2.3_LayersNone_beta0.1/"
    "1/unlearn_times_1/checkpoint-last"
)
TEACHER_MODEL = "locuslab/tofu_ft_llama2-7b"
HELDOUT_MODEL = "NousResearch/Llama-2-7b-chat-hf"
TOKENIZER_ID = "NousResearch/Llama-2-7b-chat-hf"
FORGET_JSON = REPO / "Right-to-be-forgotten/data/tofu/forget05.json"
RETAIN_JSON = REPO / "Right-to-be-forgotten/data/tofu/retain90.json"
OUT_DIR = ROOT / "results/direction_a/2026-10-08"

MAX_NEW_TOKENS = 128
PROB_TEMPS = (0.7, 1.0, 1.5)
PROB_K = 8
TASK_ID = 1
ATTENTION_TAUS = (1.0, 2.3)
REPLAY_FE = 0.7740777730353676
RETAIN_PROXY_N = 100
HUMAN_N = 50
N_BOOT = 2000

_STOP = frozenset(
    """
    a an the of and or to in on for with is was were are be as by at from that
    this it its his her their who whom which what when where how full name
    author born fictitious also known into over after before about than then
    them they he she we you your our not no yes
    """.split()
)


def read_jsonl(path, task_id=None, limit=None):
    rows = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if task_id is not None and int(row["task_id"]) != task_id:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def keytoken(answer):
    best = ""
    best_len = -1
    for raw in str(answer).split():
        tok = "".join(ch for ch in raw if ch.isalnum() or ch in "-'")
        low = tok.lower().strip("-'")
        if len(low) < 3 or low in _STOP:
            continue
        if len(low) > best_len:
            best = tok
            best_len = len(low)
    if best:
        return best
    parts = str(answer).split()
    return parts[0] if parts else ""


def question_prompt(question):
    return "[INST] " + question + " [/INST]"


def focus_prompt(question, token):
    return "[INST] " + question + "\nKeyToken: " + token + " [/INST]"


def encode_answer_span(tokenizer, question, answer):
    question_text = question_prompt(question)
    q_ids = tokenizer(question_text, add_special_tokens=True)["input_ids"]
    full = tokenizer(question_text + answer, add_special_tokens=True)["input_ids"]
    n_q = len(q_ids)
    if full[:n_q] != q_ids:
        n_q = 0
        while n_q < len(q_ids) and n_q < len(full) and full[n_q] == q_ids[n_q]:
            n_q += 1
    eos = tokenizer.eos_token_id
    positions = []
    gold_ids = []
    for index in range(n_q, len(full)):
        token_id = full[index]
        if token_id == eos:
            continue
        positions.append(index - 1)
        gold_ids.append(token_id)
    return full, positions, gold_ids


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    print(f"wrote {path}", flush=True)


def summarize(profile_parts):
    rank = torch.cat([p["rank"].float().cpu() for p in profile_parts])
    margin = torch.cat([p["margin"].float().cpu() for p in profile_parts])
    gold_prob = torch.cat([p["gold_prob"].float().cpu() for p in profile_parts])
    entropy = torch.cat([p["entropy"].float().cpu() for p in profile_parts])
    return {
        "n_tokens": int(rank.numel()),
        "median_rank": float(rank.median()),
        "mean_rank": float(rank.mean()),
        "frac_rank1": float((rank == 1).float().mean()),
        "mean_margin": float(margin.mean()),
        "mean_gold_prob": float(gold_prob.mean()),
        "mean_entropy": float(entropy.mean()),
    }


@torch.no_grad()
def rank_profile(model, tokenizer, rows, attention_temp=None, batch_size=2):
    parts = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        encoded = [encode_answer_span(tokenizer, r["question"], r["answer"]) for r in chunk]
        width = max(len(item[0]) for item in encoded)
        input_ids = torch.full((len(chunk), width), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(chunk), width), dtype=torch.long)
        for row_i, (ids, _, _) in enumerate(encoded):
            input_ids[row_i, : len(ids)] = torch.tensor(ids)
            attention[row_i, : len(ids)] = 1
        input_ids = input_ids.to(model.device)
        attention = attention.to(model.device)
        kwargs = {}
        if attention_temp is not None:
            kwargs["attention_temp"] = float(attention_temp)
        logits = model(input_ids=input_ids, attention_mask=attention, **kwargs).logits
        for row_i, (_, positions, gold_ids) in enumerate(encoded):
            if not positions:
                continue
            pos = torch.tensor(positions, device=logits.device)
            gold = torch.tensor(gold_ids, device=logits.device)
            chosen = logits[row_i].index_select(0, pos)
            parts.append(profile_logits(chosen, gold))
        print(f"rank {min(start + batch_size, len(rows))}/{len(rows)} tau={attention_temp}", flush=True)
    return summarize(parts)


def load_student():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_CKPT,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    model.eval()
    return model


def load_teacher():
    from my_models.my_llama import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    model.to("cuda:0")
    model.eval()
    return model


def load_heldout():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        HELDOUT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )
    model.eval()
    return model


def load_quantized(kind):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = bitsandbytes_load_kwargs(kind)
    quant["bnb_4bit_compute_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT_CKPT,
        quantization_config=BitsAndBytesConfig(**quant),
        device_map="cuda:0",
        local_files_only=True,
    )
    model.eval()
    return model


def release(model):
    del model
    torch.cuda.empty_cache()


@torch.no_grad()
def generate_all(model, tokenizer, prompts, temperature, seed, batch_size=8):
    tokenizer.padding_side = "left"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    outputs = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        ).to(model.device)
        kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature is None or temperature == 0:
            kwargs["do_sample"] = False
        else:
            kwargs["do_sample"] = True
            kwargs["temperature"] = float(temperature)
            kwargs["top_p"] = 1.0
        sequences = model.generate(**inputs, **kwargs)
        prompt_width = inputs["input_ids"].shape[1]
        text = tokenizer.batch_decode(sequences[:, prompt_width:], skip_special_tokens=True)
        outputs.extend(text)
        print(
            f"gen {min(start + batch_size, len(prompts))}/{len(prompts)} T={temperature} seed={seed}",
            flush=True,
        )
    return outputs


def hit_bits(predictions, answers):
    return [1.0 if answer_or_alias_contained(pred, ans) else 0.0 for pred, ans in zip(predictions, answers)]


def paired_delta(attack, baseline):
    return [a - b for a, b in zip(attack, baseline)]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def read_result(name):
    return json.loads((OUT_DIR / name).read_text())


def result_ready(name, required_keys):
    path = OUT_DIR / name
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return all(key in payload for key in required_keys)


def teacher_profile_covers(student_rank):
    if not result_ready("rank_profile.json", ("teacher_tau_1", "teacher_tau_2_3", "order_preserving_leak", "student")):
        return False
    payload = read_result("rank_profile.json")
    expected = student_rank["n_tokens"]
    return (
        payload["teacher_tau_1"]["n_tokens"] == expected
        and payload["teacher_tau_2_3"]["n_tokens"] == expected
        and payload["student"]["n_tokens"] == expected
    )


def heldout_covers(n_questions):
    if not result_ready("heldout_llama2_chat.json", ("hit_rate", "predictions", "n")):
        return False
    payload = read_result("heldout_llama2_chat.json")
    return payload["n"] == n_questions and len(payload["predictions"]) == n_questions


def compute_teacher_profile(tokenizer, forget, student_rank):
    if teacher_profile_covers(student_rank):
        print("skip teacher rank; full rank_profile.json already exists", flush=True)
        return read_result("rank_profile.json")
    teacher = load_teacher()
    teacher_ranks = {}
    for tau in ATTENTION_TAUS:
        teacher_ranks[str(tau)] = rank_profile(
            teacher, tokenizer, forget, attention_temp=tau, batch_size=2
        )
    release(teacher)
    base = teacher_ranks["1.0"]
    smooth = teacher_ranks["2.3"]
    order_preserving = {
        "gold_median_rank_equals_1": smooth["median_rank"] == 1.0,
        "entropy_up": smooth["mean_entropy"] > base["mean_entropy"] + 1e-6,
        "gold_prob_down": smooth["mean_gold_prob"] < base["mean_gold_prob"] - 1e-6,
        "greedy_forget_efficacy_already_up": REPLAY_FE > 0.5,
        "teacher_tau_1": base,
        "teacher_tau_2_3": smooth,
        "student": student_rank,
    }
    order_preserving["order_preserving_leak"] = bool(
        order_preserving["gold_median_rank_equals_1"]
        and (order_preserving["entropy_up"] or order_preserving["gold_prob_down"])
        and order_preserving["greedy_forget_efficacy_already_up"]
    )
    write_json(OUT_DIR / "rank_profile.json", order_preserving)
    return order_preserving


def compute_heldout(tokenizer, prompts, answers):
    if heldout_covers(len(answers)):
        print("skip heldout; full heldout_llama2_chat.json already exists", flush=True)
        return read_result("heldout_llama2_chat.json")
    heldout = load_heldout()
    held_preds = generate_all(heldout, tokenizer, prompts, temperature=0, seed=0)
    release(heldout)
    payload = {
        "model": HELDOUT_MODEL,
        "hit_rate": mean(hit_bits(held_preds, answers)),
        "n": len(held_preds),
        "note": "never finetuned on TOFU; high hit rate would mean the question leaks the answer",
        "predictions": held_preds,
    }
    write_json(OUT_DIR / "heldout_llama2_chat.json", payload)
    return payload


def run(limit=None):
    os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/autodl-tmp/huggingface/hub"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    forget = read_jsonl(FORGET_JSON, task_id=TASK_ID, limit=limit)
    retain = read_jsonl(RETAIN_JSON, limit=RETAIN_PROXY_N if limit is None else min(RETAIN_PROXY_N, limit))
    answers = [row["answer"] for row in forget]
    prompts = [question_prompt(row["question"]) for row in forget]
    tokens = [keytoken(row["answer"]) for row in forget]
    focus_prompts = [focus_prompt(row["question"], tok) for row, tok in zip(forget, tokens)]
    retain_prompts = [question_prompt(row["question"]) for row in retain]
    retain_answers = [row["answer"] for row in retain]
    print(f"forget={len(forget)} retain_proxy={len(retain)}", flush=True)

    tokenizer = load_tokenizer()

    student = load_student()
    student_rank = rank_profile(student, tokenizer, forget, attention_temp=None, batch_size=2)
    write_json(OUT_DIR / "rank_student.json", student_rank)

    greedy = generate_all(student, tokenizer, prompts, temperature=0, seed=0)
    greedy_hits = hit_bits(greedy, answers)
    write_json(
        OUT_DIR / "greedy.json",
        {"hit_rate": mean(greedy_hits), "n": len(greedy_hits), "predictions": greedy},
    )

    prob_hits = [0.0] * len(forget)
    prob_store = {str(temp): [] for temp in PROB_TEMPS}
    for temp in PROB_TEMPS:
        per_temp = [0.0] * len(forget)
        for sample_i in range(PROB_K):
            preds = generate_all(student, tokenizer, prompts, temperature=temp, seed=sample_i)
            bits = hit_bits(preds, answers)
            prob_store[str(temp)].append(preds)
            for i, bit in enumerate(bits):
                if bit:
                    per_temp[i] = 1.0
                    prob_hits[i] = 1.0
        write_json(
            OUT_DIR / f"probab_T{temp}.json",
            {"hit_rate_any_of_k": mean(per_temp), "k": PROB_K, "predictions": prob_store[str(temp)]},
        )
    prob_delta = paired_delta(prob_hits, greedy_hits)

    focus_preds = generate_all(student, tokenizer, focus_prompts, temperature=0, seed=0)
    focus_hits = hit_bits(focus_preds, answers)
    focus_delta = paired_delta(focus_hits, greedy_hits)
    write_json(
        OUT_DIR / "focus_on_key.json",
        {
            "hit_rate": mean(focus_hits),
            "keytoken_rule": "longest non-stop answer token, len>=3, earliest tie",
            "predictions": focus_preds,
            "keytokens": tokens,
        },
    )

    retain_fp16 = generate_all(student, tokenizer, retain_prompts, temperature=0, seed=0)
    retain_fp16_hits = hit_bits(retain_fp16, retain_answers)
    release(student)

    quant_deltas = {}
    quant_summary = {}
    for kind in ("nf4", "int4"):
        quant_model = load_quantized(kind)
        preds = generate_all(quant_model, tokenizer, prompts, temperature=0, seed=0)
        hits = hit_bits(preds, answers)
        retain_preds = generate_all(quant_model, tokenizer, retain_prompts, temperature=0, seed=0)
        retain_hits = hit_bits(retain_preds, retain_answers)
        release(quant_model)
        delta = paired_delta(hits, greedy_hits)
        quant_deltas[f"qra_{kind}"] = delta
        quant_summary[kind] = {
            "forget_hit_rate": mean(hits),
            "retain_hit_proxy": mean(retain_hits),
            "fp16_retain_hit_proxy": mean(retain_fp16_hits),
            "predictions": preds,
        }
        write_json(OUT_DIR / f"qra_{kind}.json", quant_summary[kind])

    order_preserving = compute_teacher_profile(tokenizer, forget, student_rank)
    compute_heldout(tokenizer, prompts, answers)

    channel_deltas = {
        "probab": prob_delta,
        "focus_on_key": focus_delta,
        "qra_nf4": quant_deltas["qra_nf4"],
        "qra_int4": quant_deltas["qra_int4"],
    }
    rng_seed = 0
    import random

    details = {}
    for name, deltas in channel_deltas.items():
        sig, ci = channel_significant(deltas, n_boot=N_BOOT, alpha=0.05, rng=random.Random(rng_seed))
        details[name] = {
            "mean_delta": mean(deltas),
            "attack_rate": mean([g + d for g, d in zip(greedy_hits, deltas)])
            if name != "probab"
            else mean(prob_hits),
            "greedy_rate": mean(greedy_hits),
            "ci95": list(ci),
            "significant_positive": sig,
        }
    # probab attack rate is any-of-samples, not greedy+delta reconstructed only.
    details["probab"]["attack_rate"] = mean(prob_hits)
    details["focus_on_key"]["attack_rate"] = mean(focus_hits)
    details["qra_nf4"]["attack_rate"] = quant_summary["nf4"]["forget_hit_rate"]
    details["qra_int4"]["attack_rate"] = quant_summary["int4"]["forget_hit_rate"]

    decision = l4_decision(channel_deltas, has_real_results=True, n_boot=N_BOOT, alpha=0.05, rng=random.Random(0))
    verdict = {
        "decision": decision,
        "channels": details,
        "order_preserving_leak": order_preserving["order_preserving_leak"],
        "replay_FE": REPLAY_FE,
        "retain_fp16_hit_proxy": mean(retain_fp16_hits),
        "n_forget": len(forget),
        "limit": limit,
    }
    write_json(OUT_DIR / "l4_verdict.json", verdict)

    review = []
    for i in range(min(HUMAN_N, len(forget))):
        review.append(
            {
                "i": i,
                "question": forget[i]["question"],
                "answer": forget[i]["answer"],
                "keytoken": tokens[i],
                "greedy": greedy[i],
                "greedy_hit": greedy_hits[i],
                "focus": focus_preds[i],
                "focus_hit": focus_hits[i],
                "probab_T1": [prob_store["1.0"][j][i] for j in range(PROB_K)],
                "human_review": "pending",
            }
        )
    review_path = OUT_DIR / "human_review_sample.jsonl"
    with review_path.open("w") as handle:
        for row in review:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {review_path}", flush=True)
    print(json.dumps(verdict, indent=2, ensure_ascii=False), flush=True)
    return verdict


def prepare_output_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = (ROOT / "audit/protocol.yaml").read_text()
    (OUT_DIR / "protocol_snapshot.yaml").write_text(protocol)
    frozen = yaml.safe_load((ROOT / "configs/asu_frozen_2026-09-27.yaml").read_text())
    write_json(OUT_DIR / "frozen_pointer.json", {"forget05_ASU_GD": frozen["tofu"]["forget05"]["ASU+GD"]})


def assemble_verdict_from_disk():
    """Build the L4 verdict from stage files. Does not load a model."""
    forget = read_jsonl(FORGET_JSON, task_id=TASK_ID)
    answers = [row["answer"] for row in forget]
    greedy = read_result("greedy.json")
    greedy_hits = hit_bits(greedy["predictions"], answers)
    prob_hits = [0.0] * len(forget)
    prob_t1 = None
    for temp in PROB_TEMPS:
        payload = read_result(f"probab_T{temp}.json")
        if temp == 1.0:
            prob_t1 = payload["predictions"]
        for sample in payload["predictions"]:
            for i, bit in enumerate(hit_bits(sample, answers)):
                if bit:
                    prob_hits[i] = 1.0
    focus = read_result("focus_on_key.json")
    focus_hits = hit_bits(focus["predictions"], answers)
    quant = {kind: read_result(f"qra_{kind}.json") for kind in ("nf4", "int4")}
    quant_hits = {kind: hit_bits(quant[kind]["predictions"], answers) for kind in quant}
    order_preserving = read_result("rank_profile.json")
    channel_deltas = {
        "probab": paired_delta(prob_hits, greedy_hits),
        "focus_on_key": paired_delta(focus_hits, greedy_hits),
        "qra_nf4": paired_delta(quant_hits["nf4"], greedy_hits),
        "qra_int4": paired_delta(quant_hits["int4"], greedy_hits),
    }
    import random

    details = {}
    for name, deltas in channel_deltas.items():
        sig, ci = channel_significant(deltas, n_boot=N_BOOT, alpha=0.05, rng=random.Random(0))
        details[name] = {
            "mean_delta": mean(deltas),
            "greedy_rate": mean(greedy_hits),
            "ci95": list(ci),
            "significant_positive": sig,
        }
    details["probab"]["attack_rate"] = mean(prob_hits)
    details["focus_on_key"]["attack_rate"] = mean(focus_hits)
    details["qra_nf4"]["attack_rate"] = quant["nf4"]["forget_hit_rate"]
    details["qra_int4"]["attack_rate"] = quant["int4"]["forget_hit_rate"]
    decision = l4_decision(
        channel_deltas, has_real_results=True, n_boot=N_BOOT, alpha=0.05, rng=random.Random(0)
    )
    verdict = {
        "decision": decision,
        "channels": details,
        "order_preserving_leak": order_preserving["order_preserving_leak"],
        "replay_FE": REPLAY_FE,
        "retain_fp16_hit_proxy": quant["nf4"]["fp16_retain_hit_proxy"],
        "n_forget": len(forget),
        "limit": None,
        "assembled_from_disk": True,
    }
    write_json(OUT_DIR / "l4_verdict.json", verdict)
    review_path = OUT_DIR / "human_review_sample.jsonl"
    with review_path.open("w") as handle:
        for i in range(min(HUMAN_N, len(forget))):
            row = {
                "i": i,
                "question": forget[i]["question"],
                "answer": forget[i]["answer"],
                "keytoken": focus["keytokens"][i],
                "greedy": greedy["predictions"][i],
                "greedy_hit": greedy_hits[i],
                "focus": focus["predictions"][i],
                "focus_hit": focus_hits[i],
                "probab_T1": [prob_t1[j][i] for j in range(PROB_K)],
                "human_review": "pending",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {review_path}", flush=True)
    print(json.dumps(verdict, indent=2, ensure_ascii=False), flush=True)
    return verdict


def run_independent_tail(limit=None):
    """Teacher rank profile and held-out greedy. Safe to run beside the student audit."""
    os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/autodl-tmp/huggingface/hub"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    prepare_output_dir()
    forget = read_jsonl(FORGET_JSON, task_id=TASK_ID, limit=limit)
    answers = [row["answer"] for row in forget]
    prompts = [question_prompt(row["question"]) for row in forget]
    tokenizer = load_tokenizer()
    student_rank = read_result("rank_student.json")
    order_preserving = compute_teacher_profile(tokenizer, forget, student_rank)
    heldout = compute_heldout(tokenizer, prompts, answers)
    print(
        json.dumps(
            {
                "order_preserving_leak": order_preserving["order_preserving_leak"],
                "heldout_hit_rate": heldout["hit_rate"],
            },
            indent=2,
        ),
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Direction A checkpoint audit.")
    parser.add_argument("--limit", type=int, default=None, help="Debug cap on forget rows.")
    parser.add_argument(
        "--only",
        choices=["teacher-heldout", "finalize"],
        default=None,
        help="Run only the stages that can overlap the student audit on the other GPU.",
    )
    args = parser.parse_args(argv)
    if args.only == "teacher-heldout":
        run_independent_tail(limit=args.limit)
        return
    if args.only == "finalize":
        assemble_verdict_from_disk()
        return
    prepare_output_dir()
    run(limit=args.limit)


if __name__ == "__main__":
    main()

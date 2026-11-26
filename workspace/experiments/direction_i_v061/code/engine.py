"""Minimal answer-span engine for direction-I v0.6.1 experiments.

SCOPE / DISCLOSURE
------------------
This is a PROXY implementation of the v0.6.1 mechanics that were never coded in
this repo (stage0-14 predates the proposal). It implements, faithfully:

  * teacher-forced forward on the full sequence X^+ (+) y^+ (answer span really
    appears in the forward pass);
  * answer-span position set S_f from tokenizer boundaries (not prompt-last);
  * per-layer activation collection at S_f;
  * hidden-state patching (patch-forward) from a given layer to logits;
  * hinge-style neutral penalties r_f (original-fact readout) and donor support.

It does NOT implement the calibrated feasible-set projection of proposal §4.2
(c/d_j/e are not calibrated here), and its "readout" is the frozen model head
rather than a separately trained N1 ReadoutTarget. Every result file records
engine="proxy-v061".
"""
from __future__ import annotations

import json
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/jizhicfs/chenyliao/llm-unlearning/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
DATA = "/jizhicfs/chenyliao/llm-unlearning/data/datasets/locuslab-TOFU"
FORGET = f"{DATA}/forget10.json"
RETAIN = f"{DATA}/retain90.json"
HOLDOUT = f"{DATA}/holdout10.json"

REFUSAL = re.compile(
    r"(i\s+(don'?t|do not)\s+know|i\s+(can'?t|cannot)\s|i'?m\s+(not\s+)?sure|"
    r"\bsorry\b|no\s+information|unable\s+to\s+answer|as\s+an?\s+ai)",
    re.I,
)


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def load_model(dtype=torch.float32, device="cuda"):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    return model, tok


def encode(tok, question, answer, max_len=512):
    """Return token ids and the answer-span positions S_f (absolute indices)."""
    prompt = f"Question: {question}\nAnswer:"
    p_ids = tok.encode(prompt, add_special_tokens=False)
    a_ids = tok.encode(" " + answer.strip(), add_special_tokens=False) + [tok.eos_token_id]
    ids = (p_ids + a_ids)[:max_len]
    span = list(range(len(p_ids), len(ids)))
    return ids, span


def span_ids(tok, answer):
    return tok.encode(" " + answer.strip(), add_special_tokens=False) + [tok.eos_token_id]


@torch.no_grad()
def forward_hidden(model, ids, layers, device="cuda"):
    """Teacher-forced forward; returns {layer: hidden[seq, d]} for requested layers."""
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(
        input_ids=inp,
        attention_mask=torch.ones_like(inp),
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states  # tuple: [emb, layer1..layerL]
    top = len(hs) - 1
    return {int(l): hs[min(int(l), top - 1) + 1][0].detach() for l in layers}


def span_nll(logits, ids, span):
    """Mean NLL on answer-span tokens (teacher forcing). logits: [seq, vocab]."""
    total, n = 0.0, 0
    for p in span:
        if p == 0 or p >= logits.size(0):
            continue
        lp = torch.log_softmax(logits[p - 1], -1)[ids[p]]
        total += float(-lp)
        n += 1
    return total / max(1, n)


@torch.no_grad()
def nll_ids(model, ids, span, device="cuda"):
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
    return span_nll(out.logits[0], ids, span)


@torch.no_grad()
def patch_forward_nll(model, ids, span, layer, new_hidden, device="cuda"):
    """Replace the OUTPUT of `layer` with new_hidden, then continue to logits."""
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    mod = model.model.layers[layer]
    orig = mod.forward

    def patched(hidden_states, *a, **kw):
        return orig(new_hidden, *a, **kw)

    mod.forward = patched
    try:
        out = model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
    finally:
        mod.forward = orig
    return span_nll(out.logits[0], ids, span)


@torch.no_grad()
def fact_logprob(model, ids, span):
    """Teacher-forced mean log-prob of the true answer tokens (higher = still knows)."""
    return -nll_ids(model, ids, span)


@torch.no_grad()
def donor_support(model, ids, span, donor_ids):
    """Mean log-prob of donor answer tokens at the same span (answer-substitution proxy)."""
    if not donor_ids:
        return float("nan")
    inp = torch.tensor([ids], dtype=torch.long, device=device_of(model))
    out = model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
    logits = out.logits[0]
    vals = []
    for i, p in enumerate(span[: len(donor_ids)]):
        if p == 0:
            continue
        vals.append(float(torch.log_softmax(logits[p - 1], -1)[donor_ids[i]]))
    return sum(vals) / max(1, len(vals))


def device_of(model):
    return next(model.parameters()).device


@torch.no_grad()
def gen_stats(model, tok, records, device="cuda", samples=16, max_new_tokens=64):
    """Output-level check: length, EOS, truncation, refusal, target overlap."""
    model.eval()
    rows = []
    for rec in records[:samples]:
        prompt = f"Question: {rec['question']}\nAnswer:"
        ids = tok.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            input_ids=inp,
            attention_mask=torch.ones_like(inp),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        gen = out[0][inp.size(1):]
        text = tok.decode(gen, skip_special_tokens=True).strip()
        ans_tok = {w for w in re.findall(r"[A-Za-z0-9']+", rec["answer"].lower()) if len(w) > 2}
        gen_tok = {w for w in re.findall(r"[A-Za-z0-9']+", text.lower()) if len(w) > 2}
        rows.append(
            {
                "n_tokens": int(gen.size(0)),
                "eos": bool(tok.eos_token_id is not None and (gen == tok.eos_token_id).any()),
                "short": int(gen.size(0)) <= 5,
                "refusal": bool(REFUSAL.search(text)),
                "target_overlap": len(ans_tok & gen_tok) / max(1, len(ans_tok)),
                "text": text[:160],
            }
        )
    n = max(1, len(rows))
    return {
        "samples": len(rows),
        "mean_gen_tokens": sum(r["n_tokens"] for r in rows) / n,
        "eos_rate": sum(r["eos"] for r in rows) / n,
        "short_rate": sum(r["short"] for r in rows) / n,
        "refusal_rate": sum(r["refusal"] for r in rows) / n,
        "mean_target_overlap": sum(r["target_overlap"] for r in rows) / n,
        "examples": [r["text"] for r in rows[:2]],
    }


def neutral_hinge(student_hidden, plus_hidden, delta_dir, r_f, r_f_max=0.35, donor_ortho=True):
    """Proxy neutral constraints.

    align  : ||h - A^+||^2  (the L_F term itself)
    forbid : hinge on the donor direction  <h - A^+, dA_dir> <= 0
    readout: hinge on still reading the original fact (r_f <= r_f_max)
    """
    align = F.mse_loss(student_hidden, plus_hidden)
    forbid = torch.zeros((), device=student_hidden.device)
    if donor_ortho and delta_dir is not None:
        cos = F.cosine_similarity(student_hidden - plus_hidden, delta_dir, dim=-1)
        forbid = F.relu(cos).pow(2).mean()
    readout = F.relu(r_f - r_f_max).pow(2)
    return align, forbid, readout

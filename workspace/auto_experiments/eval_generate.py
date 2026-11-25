"""Generative behaviour evaluation for the tau-schedule unlearning runs.

Historical results only contain teacher-forced answer NLL. That proxy can rise
because of early EOS, refusal, truncation or answer substitution rather than
selective forgetting. This module adds output-level checks on both forget and
retain splits, and can be used standalone on any saved checkpoint.
"""
from __future__ import annotations

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import FUNCTION_WORDS, FORGET, RETAIN, build_prompt, load_jsonl, save_json

REFUSAL = re.compile(
    r"(i\s+(don'?t|do not)\s+know|i\s+(can'?t|cannot)\s|i'?m\s+(not\s+)?sure|"
    r"\bsorry\b|no\s+information|unable\s+to\s+answer|as\s+an?\s+ai)",
    re.I,
)


def _content_tokens(text):
    return {
        w
        for w in re.findall(r"[A-Za-z0-9']+", text.lower())
        if w not in FUNCTION_WORDS and len(w) > 2
    }


def behaviour_stats(model, tok, records, device, max_new_tokens=96, samples=24):
    model.eval()
    rows = []
    eos_id = tok.eos_token_id
    with torch.no_grad():
        for rec in records[:samples]:
            prompt = build_prompt(rec["question"], None, None)
            ids = tok.encode(prompt, add_special_tokens=False)
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            out = model.generate(
                input_ids=inp,
                attention_mask=torch.ones_like(inp),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_id,
            )
            gen = out[0][inp.size(1):]
            text = tok.decode(gen, skip_special_tokens=True).strip()
            n_tok = int(gen.size(0))
            hit_eos = bool(eos_id is not None and (gen == eos_id).any().item())
            ans_tok = _content_tokens(rec.get("answer", ""))
            gen_tok = _content_tokens(text)
            overlap = len(ans_tok & gen_tok) / max(1, len(ans_tok))
            rows.append(
                {
                    "n_tokens": n_tok,
                    "n_chars": len(text),
                    "eos": hit_eos,
                    "short": n_tok <= 5,
                    "refusal": bool(REFUSAL.search(text)),
                    "target_overlap": overlap,
                    "text": text[:200],
                }
            )
    n = max(1, len(rows))
    return {
        "samples": len(rows),
        "max_new_tokens": max_new_tokens,
        "mean_gen_tokens": sum(r["n_tokens"] for r in rows) / n,
        "mean_answer_chars": sum(r["n_chars"] for r in rows) / n,
        "eos_rate": sum(r["eos"] for r in rows) / n,
        "short_rate": sum(r["short"] for r in rows) / n,
        "refusal_rate": sum(r["refusal"] for r in rows) / n,
        "mean_target_overlap": sum(r["target_overlap"] for r in rows) / n,
        "high_overlap_rate": sum(r["target_overlap"] >= 0.5 for r in rows) / n,
        "examples": rows[:3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--forget", default=FORGET)
    ap.add_argument("--retain", default=RETAIN)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(args.device)
    result = {
        "model": args.model,
        "forget": behaviour_stats(
            model, tok, load_jsonl(args.forget), args.device, args.max_new_tokens, args.samples
        ),
        "retain": behaviour_stats(
            model, tok, load_jsonl(args.retain), args.device, args.max_new_tokens, args.samples
        ),
    }
    save_json(result, args.output)


if __name__ == "__main__":
    main()

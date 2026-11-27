#!/usr/bin/env python3
"""EOS / length-drift diagnostic for the CF-04 depth audit.

Reproduces the ``behavior.forget`` fields of the historical CF-04 v2g results
(``workspace/auto_experiments/eval_generate.py``) and adds an equal-length
truncation check. Uses an explicit greedy loop over full-sequence logits instead
of ``model.generate`` to avoid the ``num_logits_to_keep=1`` bf16 GEMM crash seen
on this torch/CUDA build; decoding semantics (greedy, EOS stop, token budget) are
unchanged.
"""
from __future__ import annotations

import argparse
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_lite import FORGET, MODEL, RETAIN, answer_nll, load_jsonl, make_batch, save_json

FUNCTION_WORDS = set(
    """a an the and or but if then than that this these those of in on at to for from with by as is are
    was were be been being it its he she they them his her their we you i my your our do does did done
    have has had will would can could should may might must not no nor so such very more most some any
    all each every which who whom whose what when where why how there here about into over under again
    further once only just also""".split()
)
REFUSAL = re.compile(
    r"(i\s+(don'?t|do not)\s+know|i\s+(can'?t|cannot)\s|i'?m\s+(not\s+)?sure|"
    r"\bsorry\b|no\s+information|unable\s+to\s+answer|as\s+an?\s+ai)",
    re.I,
)
EQUAL_LENGTH_TOKENS = 13  # v2g `before` mean_gen_tokens = 13.75


def content_tokens(text):
    return {w for w in re.findall(r"[A-Za-z0-9']+", text.lower()) if w not in FUNCTION_WORDS and len(w) > 2}


def greedy_ids(model, tok, prompt, device, max_new_tokens, eos_id):
    ids = tok.encode(prompt, add_special_tokens=False)
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([ids + generated], dtype=torch.long, device=device)
            out = model(input_ids=inp, attention_mask=torch.ones_like(inp), use_cache=False)
            nxt = int(out.logits[0, -1].float().argmax().item())
            generated.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
    return generated


def behaviour_stats(model, tok, records, device, max_new_tokens=24, samples=4):
    model.eval()
    rows = []
    eos_id = tok.eos_token_id
    for rec in records[:samples]:
        prompt = f"Question: {rec['question']}\nAnswer:"
        gen = greedy_ids(model, tok, prompt, device, max_new_tokens, eos_id)
        text = tok.decode(gen, skip_special_tokens=True).strip()
        ans_tok = content_tokens(rec.get("answer", ""))
        gen_tok = content_tokens(text)
        overlap = len(ans_tok & gen_tok) / max(1, len(ans_tok))
        short_text = tok.decode(gen[:EQUAL_LENGTH_TOKENS], skip_special_tokens=True).strip()
        overlap_eq = len(ans_tok & content_tokens(short_text)) / max(1, len(ans_tok))
        rows.append(
            {
                "n_tokens": len(gen),
                "n_chars": len(text),
                "eos": bool(eos_id is not None and any(g == eos_id for g in gen)),
                "short": len(gen) <= 5,
                "refusal": bool(REFUSAL.search(text)),
                "target_overlap": overlap,
                "target_overlap_equal_length": overlap_eq,
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
        "mean_target_overlap_equal_length": sum(r["target_overlap_equal_length"] for r in rows) / n,
        "high_overlap_rate_equal_length": sum(r["target_overlap_equal_length"] >= 0.5 for r in rows) / n,
        "equal_length_tokens": EQUAL_LENGTH_TOKENS,
        "examples": rows[:3],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=MODEL)
    ap.add_argument("--cf04", default="/jizhicfs/chenyliao/llm-unlearning/checkpoints/full_cf04_v2_formal")
    ap.add_argument("--ga", default="/jizhicfs/chenyliao/runs/open-unlearning/20261122_forget10_1b/GradAscent")
    ap.add_argument("--models", default="base,cf04,ga")
    ap.add_argument("--forget", default=FORGET)
    ap.add_argument("--retain", default=RETAIN)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="results/cf04_behavior.json")
    args = ap.parse_args(argv)
    paths = {
        "base": args.base,
        "cf04": args.cf04,
        "cf04_formal": "/jizhicfs/chenyliao/llm-unlearning/checkpoints/full_cf04_formal",
        "cf04_e1": "/jizhicfs/chenyliao/llm-unlearning/checkpoints/full_cf04_e1",
        "ga": args.ga,
    }
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    forget = load_jsonl(args.forget)
    retain = load_jsonl(args.retain)
    started = time.time()
    out = {}
    for name in [s.strip() for s in args.models.split(",") if s.strip()]:
        path = paths.get(name)
        if not path:
            continue
        try:
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(args.device).eval()
            p1_forget = answer_nll(model, make_batch(tok, forget[:64], args.device))[0]
            stats = behaviour_stats(model, tok, forget, args.device, args.max_new_tokens, args.samples)
            out[name] = {"path": path, "P1_forget_nll_64": p1_forget, "forget": stats}
            print(f"[{name}] eos={stats['eos_rate']:.2f} tokens={stats['mean_gen_tokens']:.1f} overlap={stats['mean_target_overlap']:.3f} overlap_eq={stats['mean_target_overlap_equal_length']:.3f}", flush=True)
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{name}] ERROR {e}", flush=True)
    result = {
        "claim_level": False,
        "gate_status": "NOT_APPLIED",
        "models": out,
        "run_meta": {"device": args.device, "elapsed_sec": round(time.time() - started, 2), "samples": args.samples, "max_new_tokens": args.max_new_tokens},
        "notes": [
            "Explicit greedy decoding over full-sequence logits; identical token budget and EOS stop to the historical eval_generate.py.",
            "Equal-length truncation is a diagnostic only; it does not change any training or the audited NLL metric.",
            "cf04 checkpoint is full_cf04_v2_formal (4500 steps), not the 300-step v2g configuration.",
        ],
    }
    save_json(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

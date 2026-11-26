#!/usr/bin/env python3
"""Benchmark full-data diagnostic steps and evaluation on one GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .base import ModelContext
from common import HOLDOUT, eval_nll, load_jsonl


def _hidden(model, batch, layer=16):
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )
    return out.hidden_states[layer].float()


def _retain_nll(model, batch):
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--output", default="full_repro_bench.json")
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    mark = time.perf_counter()
    ctx = ModelContext.load(max_pairs=100000, device="cuda", dtype=dtype)
    load_sec = time.perf_counter() - mark

    target = ctx.pooled_delta_for_pairs(layer=16)

    def loss_fn(model, fb, rb, step):
        h = _hidden(model, fb, 16)
        return {"loss": ((h @ target) ** 2).mean() + 0.5 * _retain_nll(model, rb)}

    hist = ctx.train_with_loss(loss_fn, steps=args.steps, lr=1e-5, batch_size=args.batch_size, seed=0)
    step_secs = [row.get("step_sec") for row in hist if row.get("step_sec") is not None]

    ctx.model.eval()
    mark = time.perf_counter()
    forget_nll = eval_nll(ctx.model, ctx.tokenizer, ctx.forget, "cuda", batch_size=args.eval_batch_size)
    forget_sec = time.perf_counter() - mark
    mark = time.perf_counter()
    retain_nll = eval_nll(ctx.model, ctx.tokenizer, ctx.retain, "cuda", batch_size=args.eval_batch_size)
    retain_sec = time.perf_counter() - mark
    holdout = load_jsonl(HOLDOUT)
    mark = time.perf_counter()
    holdout_nll = eval_nll(ctx.model, ctx.tokenizer, holdout, "cuda", batch_size=args.eval_batch_size)
    holdout_sec = time.perf_counter() - mark

    out = {
        "dtype": args.dtype,
        "counts": {
            "forget": len(ctx.forget),
            "retain": len(ctx.retain),
            "holdout": len(holdout),
            "pairs": len(ctx.pairs),
        },
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "load_context_sec": round(load_sec, 3),
        "target_prep_sec": round(hist[0].get("cut_sec", 0.0), 3) if hist else 0.0,
        "step_secs": [round(float(s), 4) for s in step_secs],
        "step_sec_mean": round(sum(step_secs) / len(step_secs), 4) if step_secs else None,
        "eval_sec": {
            "forget": round(forget_sec, 3),
            "retain": round(retain_sec, 3),
            "holdout": round(holdout_sec, 3),
            "full_total": round(forget_sec + retain_sec + holdout_sec, 3),
        },
        "eval_nll": {
            "forget": forget_nll,
            "retain": retain_nll,
            "holdout": holdout_nll,
        },
        "history": hist,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

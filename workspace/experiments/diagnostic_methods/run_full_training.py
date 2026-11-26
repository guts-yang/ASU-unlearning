#!/usr/bin/env python3
"""Full-data non-claim training for CF-04 and CF-07.

One epoch visits every retain90 record and cycles forget10.  A retain-NLL guard
can stop a run before it destroys utility.  This is intentionally not a claim
artifact and does not invoke ARIS gates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .base import ModelContext, WORKSPACE
from .cf04_dual_probes import fit_dual_probes, probe_guided_loss
from .cf07_readout_delta import readout_direction, readout_space_loss

AUTO = WORKSPACE / "auto_experiments"
OUT = AUTO / "results/full_training"
CKPT = Path("/jizhicfs/chenyliao/llm-unlearning/checkpoints")


def _hidden(model, batch, layer=16):
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )
    return out.hidden_states[layer].float()


def _retain_nll(model, batch):
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return torch.nn.functional.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )


def _pair_records(ctx, key, n=None):
    pairs = ctx.pairs[: n or len(ctx.pairs)]
    return [{"question": p[key], "answer": p["answer_plus"]} for p in pairs]


def _full_eval(ctx):
    from common import HOLDOUT, eval_nll, load_jsonl
    from eval_generate import behaviour_stats

    ctx.model.eval()
    holdout = load_jsonl(HOLDOUT)
    out = {
        "forget_nll": float(eval_nll(ctx.model, ctx.tokenizer, ctx.forget, ctx.device, batch_size=8)),
        "retain_nll": float(eval_nll(ctx.model, ctx.tokenizer, ctx.retain, ctx.device, batch_size=8)),
        "holdout_nll": float(eval_nll(ctx.model, ctx.tokenizer, holdout, ctx.device, batch_size=8)),
    }
    try:
        out["behavior"] = {
            "forget": behaviour_stats(ctx.model, ctx.tokenizer, ctx.forget[:8], ctx.device, max_new_tokens=24, samples=8),
            "retain": behaviour_stats(ctx.model, ctx.tokenizer, ctx.retain[:8], ctx.device, max_new_tokens=24, samples=8),
        }
    except Exception as exc:
        out["behavior"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def build_cf04(ctx):
    f = ctx.hidden_for_records(ctx.forget[:36], layer=16).mean(1)
    r = ctx.hidden_for_records(ctx.retain[:36], layer=16).mean(1)
    d = ctx.hidden_for_records(_pair_records(ctx, "x_minus", 36), layer=16).mean(1)
    n = min(f.shape[0], r.shape[0], d.shape[0])
    fact_x = torch.cat([f[:n], r[:n]])
    fact_y = torch.cat([torch.ones(n), torch.zeros(n)])
    id_x = torch.cat([d[:n], f[:n]])
    id_y = torch.cat([torch.ones(n), torch.zeros(n)])
    fp, ip, report = fit_dual_probes(fact_x, fact_y, id_x, id_y, steps=300)
    return fp, ip, report


def build_cf07(ctx):
    r = readout_direction(ctx.model.lm_head.weight.detach(), token_ids=[ctx.tokenizer.eos_token_id])
    retain_target = (ctx.hidden_for_records(ctx.retain[:64], layer=16).mean(1) @ r).mean()
    return r, retain_target


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=["CF-04", "CF-07"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--retain-weight", type=float, default=2.0)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--retain-guard", type=float, default=0.25)
    ap.add_argument("--no-retain-guard", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ctx = ModelContext.load(max_pairs=36, device="cuda", dtype=torch.float32, attn="eager")
    before = _full_eval(ctx)

    if args.method == "CF-04":
        fp, ip, probe_report = build_cf04(ctx)

        def loss_fn(student, fb, rb, step):
            terms = probe_guided_loss(_hidden(student, fb, 16), _hidden(student, rb, 16), fp, ip)
            return {**terms, "loss": terms["loss"] + args.retain_weight * _retain_nll(student, rb)}

        extra = {"probe_report": probe_report}
    else:
        direction, retain_target = build_cf07(ctx)

        def loss_fn(student, fb, rb, step):
            terms = readout_space_loss(_hidden(student, fb, 16), _hidden(student, rb, 16), direction, retain_target)
            return {**terms, "loss": terms["loss"] + args.retain_weight * _retain_nll(student, rb)}

        extra = {"retain_target": float(retain_target.item())}

    train = ctx.train_full_epochs(
        loss_fn,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seed=args.seed,
        eval_every=args.eval_every,
        retain_guard=None if args.no_retain_guard else args.retain_guard,
        log_every=50,
    )
    after = _full_eval(ctx)
    ckpt_path = None
    if args.save:
        ckpt_path = CKPT / f"full_{args.method.lower().replace('-', '')}_e{args.epochs}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        ctx.model.save_pretrained(ckpt_path, safe_serialization=True)
        ctx.tokenizer.save_pretrained(ckpt_path)
    result = {
        "candidate": args.method,
        "claim_level": False,
        "gate_status": "NOT_APPLIED",
        "run_kind": "full-data non-claim training",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "retain_weight": args.retain_weight,
        "retain_guard": args.retain_guard,
        "before": before,
        "after": after,
        "delta": {
            "forget_nll": after["forget_nll"] - before["forget_nll"],
            "retain_nll": after["retain_nll"] - before["retain_nll"],
            "holdout_nll": after["holdout_nll"] - before["holdout_nll"],
        },
        "train": train,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        **extra,
        "elapsed_sec": round(time.time() - started, 2),
        "limitations": ["Single-GPU full-data 1-epoch diagnostic; not a 10-epoch paper reproduction."],
    }
    path = OUT / f"{args.method}_e{args.epochs}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"method": args.method, "path": str(path), "delta": result["delta"],
                      "early_stopped": train["early_stopped"], "checkpoint": result["checkpoint"],
                      "elapsed_sec": result["elapsed_sec"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

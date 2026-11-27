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
from .cf04_dual_probes import (
    fit_dual_probes,
    fit_layerwise_dual_probes,
    probe_guided_loss,
    probe_guided_loss_v2,
)
from .cf07_readout_delta import readout_direction, readout_space_loss

AUTO = WORKSPACE / "auto_experiments"
OUT = AUTO / "results/full_training"
FORMAL_OUT = AUTO / "results/formal_training"
CKPT = Path("/jizhicfs/chenyliao/llm-unlearning/checkpoints")

# Frozen from open-unlearning/configs/experiment/unlearn/tofu/default.yaml.
FORMAL_EPOCHS = 10


def _hidden(model, batch, layer=16):
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )
    return out.hidden_states[layer].float()


def _hidden_layers(model, batch, layers):
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )
    return {layer: out.hidden_states[layer].float() for layer in layers}


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


def build_cf04(ctx, args):
    if getattr(args, "cf04_v2", False):
        layers = [int(x) for x in args.cf04_layers.split(",") if x.strip()]
        n = 36
        fact_y = torch.cat([torch.ones(n), torch.zeros(n)])
        id_y = torch.cat([torch.ones(n), torch.zeros(n)])
        layer_fact, layer_id = {}, {}
        for layer in layers:
            hf = ctx.hidden_for_records(ctx.forget[:n], layer=layer).mean(1)
            hr = ctx.hidden_for_records(ctx.retain[:n], layer=layer).mean(1)
            hd = ctx.hidden_for_records(_pair_records(ctx, "x_minus", n), layer=layer).mean(1)
            layer_fact[layer] = torch.cat([hf, hr])
            layer_id[layer] = torch.cat([hd, hf])
        return fit_layerwise_dual_probes(layer_fact, fact_y, layer_id, id_y, steps=args.probe_steps)

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
    ap.add_argument("--mode", choices=["formal", "debug"], default="formal")
    ap.add_argument("--epochs", type=int, default=None, help="debug only; formal is frozen at FORMAL_EPOCHS")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--retain-weight", type=float, default=2.0)
    ap.add_argument("--cf04-v2", action="store_true")
    ap.add_argument("--cf04-layers", default="8,12,16")
    ap.add_argument("--cf04-margin", type=float, default=1.0)
    ap.add_argument("--probe-temperature", type=float, default=1.0)
    ap.add_argument("--cf04-identity-weight", type=float, default=2.0)
    ap.add_argument("--cf04-probe-retain-weight", type=float, default=2.5)
    ap.add_argument("--probe-steps", type=int, default=300)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--retain-guard", type=float, default=0.25)
    ap.add_argument("--no-retain-guard", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(argv)

    if args.mode == "formal":
        epochs = FORMAL_EPOCHS
        if args.epochs is not None and args.epochs != FORMAL_EPOCHS:
            raise SystemExit(f"formal mode is fixed at {FORMAL_EPOCHS} epochs; use --mode debug for 1 epoch")
        retain_guard = None
        eval_every = 0
        save = True
    else:
        epochs = args.epochs or 1
        retain_guard = None if args.no_retain_guard else args.retain_guard
        eval_every = args.eval_every
        save = args.save

    out_dir = FORMAL_OUT if args.mode == "formal" else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    ctx = ModelContext.load(max_pairs=36, device="cuda", dtype=torch.float32, attn="eager")
    before = _full_eval(ctx)

    if args.method == "CF-04":
        built = build_cf04(ctx, args)
        if args.cf04_v2:
            probes, weights, layer_report = built
            layers = sorted(probes)

            def loss_fn(student, fb, rb, step):
                hf = _hidden_layers(student, fb, layers)
                hr = _hidden_layers(student, rb, layers)
                terms = probe_guided_loss_v2(
                    hf, hr, probes, weights,
                    margin=args.cf04_margin,
                    temperature=args.probe_temperature,
                    identity_weight=args.cf04_identity_weight,
                    retain_weight=args.cf04_probe_retain_weight,
                )
                return {**terms, "loss": terms["loss"] + args.retain_weight * _retain_nll(student, rb)}

            extra = {"layer_weights": weights, "layer_report": layer_report}
        else:
            fp, ip, probe_report = built

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
        epochs=epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seed=args.seed,
        eval_every=eval_every,
        retain_guard=retain_guard,
        log_every=50,
    )
    after = _full_eval(ctx)
    ckpt_path = None
    if save:
        suffix = args.tag or ("v2_formal" if (args.mode == "formal" and args.cf04_v2) else ("formal" if args.mode == "formal" else f"debug_e{epochs}"))
        ckpt_path = CKPT / f"full_{args.method.lower().replace('-', '')}_{suffix}"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        ctx.model.save_pretrained(ckpt_path, safe_serialization=True)
        ctx.tokenizer.save_pretrained(ckpt_path)

    import subprocess
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKSPACE).decode().strip()
    except Exception:
        git_sha = None
    result = {
        "candidate": args.method,
        "claim_level": False,
        "gate_status": "NOT_APPLIED",
        "run_kind": "full-data non-claim training",
        "mode": args.mode,
        "formal_epochs": FORMAL_EPOCHS,
        "epochs": epochs,
        "epochs_completed": train["epochs_completed"],
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "retain_weight": args.retain_weight,
        "retain_guard": retain_guard,
        "seed": args.seed,
        "data": {"forget": 400, "retain": 3600, "holdout": 400},
        "reproducibility": {
            "git_sha": git_sha,
            "formal_epochs_source": "open-unlearning/configs/experiment/unlearn/tofu/default.yaml: num_train_epochs=10",
            "data_order": "seeded torch.Generator; one order per epoch; no early stop in formal mode",
        },
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
        "limitations": [
            "Single-GPU formal run; the baseline paper uses 2x L40 and DeepSpeed ZeRO-3.",
            "Full epochs are 10; no early stop in formal mode.",
        ],
    }
    tag = args.tag or ("v2_formal" if (args.mode == "formal" and args.cf04_v2) else ("formal" if args.mode == "formal" else f"debug_e{epochs}"))
    path = out_dir / f"{args.method}_{tag}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"method": args.method, "path": str(path), "delta": result["delta"],
                      "early_stopped": train["early_stopped"], "checkpoint": result["checkpoint"],
                      "elapsed_sec": result["elapsed_sec"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Unified trainer for the CF-04 vs official-ASU aligned reproduction.

Both arms share: base checkpoint, tokenizer, dtype, data files, per-seed data
order, optimizer (AdamW lr=1e-5 wd=0.01), grad clip, 5 epochs = 125 optimizer
steps with batch 4 x grad_accum 4, full-split eval, and checkpoint format.
Only the loss function differs (method-intrinsic).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aligned_core import (
    BATCH_SIZE, EPOCHS, EVAL_BATCH_SIZE, FORGET, GRAD_ACCUM, GRAD_CLIP, HOLDOUT,
    LR, MODEL, RETAIN, WEIGHT_DECAY, TemperatureContext, build_pairs, eval_nll,
    fit_layerwise_dual_probes, hidden_for_records, hidden_layers, kl_on_answer,
    load_jsonl, make_batch, make_order, probe_guided_loss_v2, retain_nll,
    save_json, set_seed,
)

CF04 = {
    "layers": [8, 12, 16], "margin": 1.0, "temperature": 1.0,
    "identity_weight": 2.0, "retain_weight": 2.5,
    "lambda_init": 3.0, "lambda_lr": 0.05, "retain_target": 0.52,
    "probe_steps": 300, "probe_lr": 0.05, "probe_limit": 36,
    "grad_project": False,
}


def order_hash(order):
    payload = json.dumps(
        [[[[r["question"] for r in fb], [r["question"] for r in rb]] for fb, rb in micro] for micro in order],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def fit_cf04_probes(model, tok, forget, retain, seed, device):
    """Verbatim port of run_one_method.py::run_cf04_v2 probe fitting (frozen initial model)."""
    layers = CF04["layers"]
    n = CF04["probe_limit"]
    pairs = build_pairs(tok, forget, retain, max_pairs=n, seed=seed)
    donor = [{"question": p["x_minus"], "answer": p["answer_plus"]} for p in pairs[:n]]
    fact_y = torch.cat([torch.ones(n), torch.zeros(n)])
    id_y = torch.cat([torch.ones(n), torch.zeros(n)])
    layer_fact, layer_id = {}, {}
    for layer in layers:
        hf = hidden_for_records(model, tok, forget[:n], layer, device).mean(1)
        hr = hidden_for_records(model, tok, retain[:n], layer, device).mean(1)
        hd = hidden_for_records(model, tok, donor, layer, device).mean(1)
        layer_fact[layer] = torch.cat([hf, hr])
        layer_id[layer] = torch.cat([hd, hf])
    probes, weights, report = fit_layerwise_dual_probes(
        layer_fact, fact_y, layer_id, id_y, steps=CF04["probe_steps"], lr=CF04["probe_lr"]
    )
    return probes, weights, {"layers": layers, "weights": weights, "n_pairs": len(pairs), "report": report}


def asu_micro(student, ref, tok, fb, rb, device, tau):
    fb_b = make_batch(tok, fb, device)
    rb_b = make_batch(tok, rb, device)
    with torch.no_grad(), TemperatureContext(ref, tau):
        ref_logits = ref(input_ids=fb_b["input_ids"], attention_mask=fb_b["attention_mask"]).logits
    stu_logits = student(input_ids=fb_b["input_ids"], attention_mask=fb_b["attention_mask"]).logits
    kl = kl_on_answer(stu_logits, ref_logits, fb_b["labels"])
    rloss = retain_nll(student, rb_b)
    return {"loss": kl + rloss, "kl": float(kl.detach().item()), "retain": float(rloss.detach().item())}


def cf04_micro(student, tok, fb, rb, device, probes, weights, lam):
    fb_b = make_batch(tok, fb, device)
    rb_b = make_batch(tok, rb, device)
    hf = hidden_layers(student, fb_b, CF04["layers"])
    hr = hidden_layers(student, rb_b, CF04["layers"])
    terms = probe_guided_loss_v2(
        hf, hr, probes, weights, margin=CF04["margin"], temperature=CF04["temperature"],
        identity_weight=CF04["identity_weight"], retain_weight=CF04["retain_weight"],
    )
    rloss = retain_nll(student, rb_b)
    return {
        "loss": terms["loss"] + lam * rloss,
        "fact_forget": float(terms["fact_forget"].detach().item()),
        "identity_preserve": float(terms["identity_preserve"].detach().item()),
        "retain_fact_preserve": float(terms["retain_fact_preserve"].detach().item()),
        "retain": float(rloss.detach().item()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["asu", "cf04"], required=True)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps-override", type=int, default=None, help="smoke only; default = full 5 epochs")
    args = ap.parse_args(argv)
    set_seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    forget, retain, holdout = load_jsonl(FORGET), load_jsonl(RETAIN), load_jsonl(HOLDOUT)

    student = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(args.device).eval()
    ref = None
    if args.arm == "asu":
        ref = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(args.device).eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    base_eval_path = out.parent / "base_eval.json"
    if base_eval_path.exists():
        before = json.loads(base_eval_path.read_text())
    else:
        before = {"forget": eval_nll(student, tok, forget, args.device),
                  "retain": eval_nll(student, tok, retain, args.device),
                  "holdout": eval_nll(student, tok, holdout, args.device)}
        save_json(before, base_eval_path)
    print(f"[{args.arm} seed={args.seed}] before F/R/H = {before['forget']:.4f}/{before['retain']:.4f}/{before['holdout']:.4f}", flush=True)

    order, steps_per_epoch, total_steps = make_order(args.seed, forget, retain)
    if args.steps_override is not None:
        order = order[: args.steps_override]
        total_steps = len(order)
    ohash = order_hash(order)

    probes, weights, probe_meta = None, None, None
    if args.arm == "cf04":
        probes, weights, probe_meta = fit_cf04_probes(student, tok, forget, retain, args.seed, args.device)
        save_json(probe_meta, out / "probes.json")

    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lam = CF04["lambda_init"]
    history = []
    for step, micro in enumerate(order):
        opt.zero_grad(set_to_none=True)
        rvals, parts, total_loss = [], {}, 0.0
        for fb, rb in micro:
            if args.arm == "asu":
                terms = asu_micro(student, ref, tok, fb, rb, args.device, args.tau)
            else:
                terms = cf04_micro(student, tok, fb, rb, args.device, probes, weights, lam)
            loss = terms["loss"] / GRAD_ACCUM
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss}")
            loss.backward()
            total_loss += float(loss.detach().item())
            rvals.append(terms["retain"])
            for k, v in terms.items():
                if k != "loss":
                    parts[k] = parts.get(k, 0.0) + v / GRAD_ACCUM
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}: {grad_norm}")
        opt.step()
        if args.arm == "cf04":
            rval = sum(rvals) / len(rvals)
            if rval > CF04["retain_target"]:
                lam = min(50.0, lam * (1.0 + CF04["lambda_lr"]))
            else:
                lam = max(0.05, lam * (1.0 - 0.2 * CF04["lambda_lr"]))
        if step % 25 == 0 or step == total_steps - 1:
            row = {"step": step, "loss": total_loss,
                   "grad_norm": float(grad_norm), "lambda_retain": lam if args.arm == "cf04" else None,
                   **{k: round(v, 6) for k, v in parts.items()}}
            history.append(row)
            print(f"[{args.arm} seed={args.seed}] step {step:3d}/{total_steps} grad={float(grad_norm):.4f} retain={parts.get('retain', 0):.4f}", flush=True)

    student.eval()
    after = {"forget": eval_nll(student, tok, forget, args.device),
             "retain": eval_nll(student, tok, retain, args.device),
             "holdout": eval_nll(student, tok, holdout, args.device)}
    ckpt = out / "checkpoint"
    student.save_pretrained(ckpt, safe_serialization=True)
    tok.save_pretrained(ckpt)

    cfg = {
        "arm": args.arm, "seed": args.seed, "tau": args.tau if args.arm == "asu" else None,
        "official_epochs": EPOCHS, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM, "lr": LR, "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP, "dtype": "float32", "use_LoRA": False,
        "steps_per_epoch": steps_per_epoch, "optimizer_steps": total_steps,
        "micro_steps": total_steps * GRAD_ACCUM, "eval_batch_size": EVAL_BATCH_SIZE,
        "model_path": MODEL, "forget": FORGET, "retain": RETAIN, "holdout": HOLDOUT,
        "order_sha256": ohash, "steps_override": args.steps_override,
        "cf04_config": CF04 if args.arm == "cf04" else None,
    }
    save_json(cfg, out / "resolved_config.json")
    result = {
        "arm": args.arm, "seed": args.seed, "status": "COMPLETED", "claim_level": False,
        "before": before, "after": after,
        "delta": {k: after[k] - before[k] for k in ("forget", "retain", "holdout")},
        "train": {"history": history, "optimizer_steps": total_steps, "grad_finite": True},
        "probes": probe_meta,
        "checkpoint": str(ckpt), "resolved_config": str(out / "resolved_config.json"),
        "elapsed_sec": round(time.time() - started, 2),
        "integrity": {"ref_model_frozen": args.arm == "asu", "evaluation_ground_truth": "dataset answers",
                      "donor_answer_in_training": False, "full_finetune": True},
    }
    save_json(result, out / "result.json")
    print(f"[{args.arm} seed={args.seed}] DONE dF={result['delta']['forget']:+.4f} dR={result['delta']['retain']:+.4f} dH={result['delta']['holdout']:+.4f} elapsed={result['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

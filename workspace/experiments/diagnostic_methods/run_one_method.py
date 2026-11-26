#!/usr/bin/env python3
"""Run one non-claim diagnostic method on the frozen ASU/TOFU-1B baseline.

The runner is intentionally serial and method-isolated: one invocation accepts
one method id, writes one JSON result, and never touches ARIS claim state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .base import ModelContext, orthonormalize, project_out
from .cf01_constrained_inverse import solve_constrained_delta
from .cf02_role_decomposition import role_loss_terms
from .cf03_causal_gate import patch_effect, select_causal_gate
from .cf04_dual_probes import fit_dual_probes, probe_guided_loss
from .cf05_multi_donor import donor_consistency, multi_donor_svd
from .cf06_lrm_trace import NotApplicableError, require_lrm_backbone
from .cf07_readout_delta import readout_direction, readout_space_loss, rotation_invariance
from .cf08_nullspace import nullspace_projection
from .cf09_control_residual import residualize
from .cf10_purity_schedule import purity_schedule, purity_scores, schedule_diagnostics
from .r2c_relearning_aware import relearning_aware_loss
from .r3_2_depth_audit import summarize_depth_audit
from .r3_3_two_stage import alignment_loss, two_stage_schedule

WS = Path(__file__).resolve().parents[2]
OUT = WS / "auto_experiments/results/nonclaim"
AUDIT = WS / "auto_experiments/results/audit_depth.json"
SENS = WS / "auto_experiments/results/gate_sensitemp.json"


def _hidden(student, batch, layer=16):
    out = student(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )
    return out.hidden_states[layer].float()


def _retain_nll(student, batch):
    logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )


def _donor_records(pairs, n):
    return [{"question": p["x_minus"], "answer": p["answer_plus"]} for p in pairs[:n]]


def _pair_records(ctx, key, n=None):
    pairs = ctx.pairs[: n or len(ctx.pairs)]
    return [{"question": p[key], "answer": p["answer_plus"]} for p in pairs]


def _eval(ctx: ModelContext, args) -> dict:
    from common import eval_nll
    from eval_generate import behaviour_stats

    limit = args.eval_limit or None
    forget = ctx.forget[:limit] if limit else ctx.forget
    retain = ctx.retain[:limit] if limit else ctx.retain
    holdout = ctx.holdout[:limit] if limit else ctx.holdout
    ctx.model.eval()
    forget_nll = eval_nll(ctx.model, ctx.tokenizer, forget, ctx.device, batch_size=args.eval_batch_size)
    retain_nll = eval_nll(ctx.model, ctx.tokenizer, retain, ctx.device, batch_size=args.eval_batch_size)
    holdout_nll = None
    if holdout:
        holdout_nll = eval_nll(ctx.model, ctx.tokenizer, holdout, ctx.device, batch_size=args.eval_batch_size)
    behavior = {}
    if args.behavior_samples > 0:
        try:
            behavior = {
                "forget": behaviour_stats(
                    ctx.model, ctx.tokenizer, forget[:args.behavior_samples], ctx.device,
                    max_new_tokens=24, samples=args.behavior_samples,
                ),
                "retain": behaviour_stats(
                    ctx.model, ctx.tokenizer, retain[:args.behavior_samples], ctx.device,
                    max_new_tokens=24, samples=args.behavior_samples,
                ),
            }
        except Exception as exc:  # generation is optional in a diagnostic run
            behavior = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "forget_nll": float(forget_nll),
        "retain_nll": float(retain_nll),
        "holdout_nll": None if holdout_nll is None else float(holdout_nll),
        "eval_counts": {"forget": len(forget), "retain": len(retain), "holdout": len(holdout)},
        "behavior": behavior,
    }


FORMAL_EPOCHS = 10


def _train(ctx, loss_fn, args):
    """Debug short run by default; formal full-data 10-epoch run with --formal."""
    if getattr(args, "formal", False):
        return ctx.train_full_epochs(
            loss_fn,
            epochs=FORMAL_EPOCHS,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            seed=args.seed,
            eval_every=0,
            retain_guard=None,
            log_every=50,
        )
    return ctx.train_with_loss(
        loss_fn, steps=args.steps, lr=args.lr, batch_size=args.batch_size, seed=args.seed
    )


def _ctx(args, with_ref=False):
    data_dir = Path(args.data_dir) if args.data_dir else None
    return ModelContext.load(
        model_path=args.model,
        device=args.device,
        max_pairs=args.max_pairs,
        seed=args.seed,
        with_ref=with_ref,
        dtype=torch.float32 if args.dtype == "float32" else torch.bfloat16,
        forget_path=str(data_dir / "forget10.json") if data_dir else None,
        retain_path=str(data_dir / "retain90.json") if data_dir else None,
        holdout_path=str(data_dir / "holdout10.json") if data_dir else None,
    )


def run_cf01(ctx, args):
    delta = ctx.pooled_delta_for_pairs(layer=16)
    plus = ctx.hidden_for_records(_pair_records(ctx, "x_plus"), layer=16).mean(1)
    minus = ctx.hidden_for_records(_pair_records(ctx, "x_minus"), layer=16).mean(1)
    shuffled = minus[torch.randperm(minus.shape[0])]
    delta_shuf = (plus - shuffled).mean(0)
    retain_h = ctx.hidden_for_records(ctx.retain[:args.basis_limit], layer=16).mean(1)
    _, _, vh = torch.linalg.svd(retain_h, full_matrices=False)
    retain_basis = vh[:8].T
    solved = solve_constrained_delta(delta, delta, delta_shuf.unsqueeze(1), retain_basis)
    target = solved.delta

    def loss_fn(student, fb, rb, step):
        h = _hidden(student, fb, 16)
        return {"loss": ((h @ target) ** 2).mean() + 0.5 * _retain_nll(student, rb), "suppression": ((h @ target) ** 2).mean()}

    hist = _train(ctx, loss_fn, args)
    return {
        "training_executed": True,
        "evaluation_executed": True,
        "modules": {"solve_constrained_delta": solved.diagnostics},
        "history": hist,
        "verdict": "DIAGNOSTIC_RUN",
    }


def run_cf02(ctx, args):
    delta = ctx.pooled_delta_for_pairs(layer=16)
    plus = ctx.hidden_for_records(_pair_records(ctx, "x_plus"), layer=16).mean(1)
    minus = ctx.hidden_for_records(_pair_records(ctx, "x_minus"), layer=16).mean(1)
    delta_shuf = (plus - minus[torch.randperm(minus.shape[0])]).mean(0)
    retain_h = ctx.hidden_for_records(ctx.retain[:args.basis_limit], layer=16).mean(1)
    _, _, vh = torch.linalg.svd(retain_h, full_matrices=False)
    utility = vh[:8].T
    donor_batch = ctx.records_batch(_donor_records(ctx.pairs, args.batch_size))

    def loss_fn(student, fb, rb, step):
        hf = _hidden(student, fb, 16)
        hd = _hidden(student, donor_batch, 16)
        hr = _hidden(student, rb, 16)
        terms = role_loss_terms(hf, hd, hr, delta.unsqueeze(1), delta_shuf.unsqueeze(1), utility)
        return {**terms, "loss": terms["loss"] + 0.2 * _retain_nll(student, rb)}

    hist = _train(ctx, loss_fn, args)
    return {"training_executed": True, "evaluation_executed": True, "history": hist, "verdict": "DIAGNOSTIC_RUN"}


def run_cf03(ctx, args):
    pairs = ctx.pairs[:args.patch_pairs]
    layers = (8, 12, 14)
    totals = {layer: 0.0 for layer in layers}
    null_totals = {layer: 0.0 for layer in layers}
    seen = 0
    for start in range(0, len(pairs), args.patch_micro_batch):
        chunk = pairs[start:start + args.patch_micro_batch]
        batch = ctx.records_batch([{"question": p["x_plus"], "answer": p["answer_plus"]} for p in chunk])
        replacements = ctx.hidden_for_records(
            [{"question": p["x_minus"], "answer": p["answer_plus"]} for p in chunk],
            layer=12,
        )
        null_replacements = replacements[torch.randperm(replacements.shape[0])]
        for layer in layers:
            with torch.no_grad():
                totals[layer] += patch_effect(ctx.model, batch, layer, replacements).mean().item()
                null_totals[layer] += patch_effect(ctx.model, batch, layer, null_replacements).mean().item()
        seen += len(chunk)
    effects = [totals[layer] / max(seen, 1) for layer in layers]
    nulls = [null_totals[layer] / max(seen, 1) for layer in layers]
    gate = select_causal_gate(torch.tensor(effects), torch.tensor(nulls), top_k=2)
    return {
        "training_executed": False,
        "evaluation_executed": True,
        "modules": {"patch_effect_layers_8_12_14": effects, "null_effects": nulls, "patch_pairs": seen},
        "gate": gate.diagnostics,
        "verdict": "DIAGNOSTIC_RUN",
    }


def run_cf04(ctx, args):
    f = ctx.hidden_for_records(ctx.forget[:args.probe_limit], layer=16).mean(1)
    r = ctx.hidden_for_records(ctx.retain[:args.probe_limit], layer=16).mean(1)
    d = ctx.hidden_for_records(_donor_records(ctx.pairs, args.probe_limit), layer=16).mean(1)
    n = min(f.shape[0], r.shape[0], d.shape[0])
    fact_x = torch.cat([f[:n], r[:n]])
    fact_y = torch.cat([torch.ones(n), torch.zeros(n)])
    id_x = torch.cat([d[:n], f[:n]])
    id_y = torch.cat([torch.ones(n), torch.zeros(n)])
    fact_probe, id_probe, report = fit_dual_probes(fact_x, fact_y, id_x, id_y, steps=200)
    hist = []
    if args.steps > 0:
        def loss_fn(student, fb, rb, step):
            hf = _hidden(student, fb, 16)
            hr = _hidden(student, rb, 16)
            terms = probe_guided_loss(hf, hr, fact_probe, id_probe)
            return {**terms, "loss": terms["loss"] + 0.5 * _retain_nll(student, rb)}

        hist = _train(ctx, loss_fn, args)
    return {"training_executed": args.steps > 0, "evaluation_executed": True, "probe_training_executed": True,
            "modules": {"dual_probe": report}, "history": hist, "verdict": "DIAGNOSTIC_RUN"}


def run_cf05(ctx, args):
    plus = ctx.hidden_for_records(_pair_records(ctx, "x_plus"), layer=16).mean(1)
    minus = ctx.hidden_for_records(_pair_records(ctx, "x_minus"), layer=16).mean(1)
    deltas = plus - minus
    groups = defaultdict(list)
    for i, p in enumerate(ctx.pairs):
        groups[p.get("forget_subject", "unknown")].append(i)
    group = max(groups.values(), key=len)
    use = deltas[group] if len(group) >= 2 else deltas
    dec = multi_donor_svd(use, rank=min(4, max(1, use.shape[0] - 1)))
    return {"training_executed": False, "evaluation_executed": True,
            "modules": {"multi_donor": dec.diagnostics, "consistency": donor_consistency(use)},
            "verdict": "DIAGNOSTIC_RUN"}


def run_cf06(ctx, args):
    try:
        require_lrm_backbone(False, "Llama-3.2-1B-Instruct")
    except NotApplicableError as exc:
        return {"training_executed": False, "evaluation_executed": False,
                "modules": {"not_applicable": str(exc)}, "verdict": "NOT_APPLICABLE"}


def run_cf07(ctx, args):
    plus = ctx.hidden_for_records(_pair_records(ctx, "x_plus"), layer=16).mean(1)
    minus = ctx.hidden_for_records(_pair_records(ctx, "x_minus"), layer=16).mean(1)
    rep = rotation_invariance(plus, minus, ctx.model.lm_head.weight.detach())
    hist = []
    if args.steps > 0:
        r = readout_direction(ctx.model.lm_head.weight.detach(), token_ids=[ctx.tokenizer.eos_token_id])
        retain_target = (ctx.hidden_for_records(ctx.retain[:36], layer=16).mean(1) @ r).mean()

        def loss_fn(student, fb, rb, step):
            hf = _hidden(student, fb, 16)
            hr = _hidden(student, rb, 16)
            terms = readout_space_loss(hf, hr, r, retain_target=retain_target)
            return {**terms, "loss": terms["loss"] + 0.5 * _retain_nll(student, rb)}

        hist = _train(ctx, loss_fn, args)
    return {"training_executed": args.steps > 0, "evaluation_executed": True,
            "modules": {"readout_rotation": rep}, "history": hist, "verdict": "DIAGNOSTIC_RUN"}


def run_cf08(ctx, args):
    delta = ctx.pooled_delta_for_pairs(layer=16)
    retain_h = ctx.hidden_for_records(ctx.retain[:args.basis_limit], layer=16).mean(1)
    projected, diag = nullspace_projection(delta, retain_h, rank=8)

    def loss_fn(student, fb, rb, step):
        h = _hidden(student, fb, 16)
        return {"loss": ((h @ projected) ** 2).mean() + 0.5 * _retain_nll(student, rb)}

    hist = _train(ctx, loss_fn, args)
    return {"training_executed": True, "evaluation_executed": True, "modules": {"nullspace": diag},
            "history": hist, "verdict": "DIAGNOSTIC_RUN"}


def run_cf09(ctx, args):
    delta = ctx.pooled_delta_for_pairs(layer=16)
    plus = ctx.hidden_for_records(_pair_records(ctx, "x_plus"), layer=16).mean(1)
    minus = ctx.hidden_for_records(_pair_records(ctx, "x_minus"), layer=16).mean(1)
    controls = [delta * 0.3, (plus - minus[torch.randperm(minus.shape[0])]).mean(0), torch.randn_like(delta)]
    residual, diag = residualize(delta, controls)

    def loss_fn(student, fb, rb, step):
        h = _hidden(student, fb, 16)
        return {"loss": ((h @ residual) ** 2).mean() + 0.5 * _retain_nll(student, rb)}

    hist = _train(ctx, loss_fn, args)
    return {"training_executed": True, "evaluation_executed": True, "modules": {"residual": diag},
            "history": hist, "verdict": "DIAGNOSTIC_RUN"}


def run_cf10(ctx, args):
    sens = json.loads(SENS.read_text()) if SENS.exists() else {}
    layer_sens = sens.get("layer_sensitivity", {})
    purity = torch.tensor([abs(layer_sens.get(str(i), 0.0)) for i in range(16)])
    if purity.sum() == 0:
        purity = torch.ones(16)
    schedule = purity_schedule(purity, steps=args.steps)
    diag = schedule_diagnostics(schedule, purity)
    layers = [8, 10, 12, 14, 16]
    deltas = {l: ctx.pooled_delta_for_pairs(layer=l) for l in layers}

    def loss_fn(student, fb, rb, step):
        out = student(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"], output_hidden_states=True)
        loss = _retain_nll(student, rb)
        for l in layers:
            h = out.hidden_states[l].float()
            w = schedule[step, min(l, schedule.shape[1] - 1)]
            loss = loss + w * ((h @ deltas[l]) ** 2).mean()
        return {"loss": loss}

    hist = _train(ctx, loss_fn, args)
    return {"training_executed": True, "evaluation_executed": True, "modules": {"purity_schedule": diag},
            "history": hist, "verdict": "DIAGNOSTIC_RUN"}


def run_r2c(ctx, args):
    def loss_fn(student, fb, rb, step):
        return relearning_aware_loss(
            student, fb, rb, inner_steps=1, inner_lr=0.02, retain_weight=1.0,
            param_names=["model.layers.15.self_attn.q_proj.weight"],
        )

    hist = _train(ctx, loss_fn, args)
    return {"training_executed": True, "evaluation_executed": True, "history": hist, "verdict": "DIAGNOSTIC_RUN"}


def run_r3_2(ctx, args):
    audit = json.loads(AUDIT.read_text())
    return {"training_executed": False, "evaluation_executed": True,
            "modules": {"depth_audit": summarize_depth_audit(audit)}, "verdict": "DIAGNOSTIC_RUN"}


def run_r3_3(ctx, args):
    schedule = two_stage_schedule(16, steps=args.steps, shallow_end=5, deep_start=10)

    def loss_fn(student, fb, rb, step):
        sh = _hidden(student, rb, 16)
        with torch.no_grad():
            th = _hidden(ctx.ref_model, rb, 16)
        return {"loss": alignment_loss(sh, th, torch.ones(1)) - 0.1 * _retain_nll(student, fb)}

    hist = _train(ctx, loss_fn, args)
    return {"training_executed": True, "evaluation_executed": True,
            "modules": {"schedule_shape": list(schedule.shape)}, "history": hist, "verdict": "DIAGNOSTIC_RUN"}


RUNNERS = {
    "CF-01": run_cf01, "CF-02": run_cf02, "CF-03": run_cf03, "CF-04": run_cf04,
    "CF-05": run_cf05, "CF-06": run_cf06, "CF-07": run_cf07, "CF-08": run_cf08,
    "CF-09": run_cf09, "CF-10": run_cf10, "R2-C": run_r2c, "R3-2": run_r3_2, "R3-3": run_r3_3,
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(RUNNERS))
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--formal", action="store_true", help="run FORMAL_EPOCHS full-data epochs; --steps is ignored")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--model", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--eval-limit", type=int, default=64, help="0 means full split")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--behavior-samples", type=int, default=4, help="0 disables generation checks")
    ap.add_argument("--max-pairs", type=int, default=36)
    ap.add_argument("--probe-limit", type=int, default=36)
    ap.add_argument("--basis-limit", type=int, default=64)
    ap.add_argument("--patch-pairs", type=int, default=4)
    ap.add_argument("--patch-micro-batch", type=int, default=8)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    needs_ctx = args.method not in {"CF-06", "R3-2"}
    ctx = _ctx(args, with_ref=(args.method == "R3-3")) if needs_ctx else None
    before = _eval(ctx, args) if ctx is not None else {}
    payload = RUNNERS[args.method](ctx, args) if ctx is not None else RUNNERS[args.method](None, args)
    after = _eval(ctx, args) if ctx is not None else {}
    delta = {
        "forget_nll": (after.get("forget_nll", 0.0) - before.get("forget_nll", 0.0)) if before and after else None,
        "retain_nll": (after.get("retain_nll", 0.0) - before.get("retain_nll", 0.0)) if before and after else None,
        "holdout_nll": (
            after.get("holdout_nll", 0.0) - before.get("holdout_nll", 0.0)
            if before and after and before.get("holdout_nll") is not None and after.get("holdout_nll") is not None
            else None
        ),
    }
    def _meets(df_th: float) -> bool | None:
        if delta["forget_nll"] is None or delta["retain_nll"] is None:
            return None
        return bool(delta["forget_nll"] >= df_th and delta["retain_nll"] <= 0.05)
    train_meta = None
    if isinstance(payload.get("history"), dict):
        train_meta = payload["history"]
        payload["history"] = train_meta.get("history", [])
    history = payload.get("history", [])
    step_secs = [row["step_sec"] for row in history if "step_sec" in row]
    result = {
        "candidate": args.method,
        "claim_level": False,
        "gate_status": "NOT_APPLIED",
        "run_kind": "non-claim formal full-data training" if args.formal else ("non-claim full-data diagnostic" if args.eval_limit == 0 else "non-claim diagnostic"),
        "formal": args.formal,
        "formal_epochs": FORMAL_EPOCHS if args.formal else None,
        "epochs_completed": train_meta.get("epochs_completed") if train_meta else None,
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "eval_limit": args.eval_limit,
        "eval_batch_size": args.eval_batch_size,
        "max_pairs": args.max_pairs,
        "data_meta": ctx.meta if ctx is not None else {},
        "before": before,
        "after": after,
        "delta": delta,
        "threshold_check": {
            "diag_v4_df_ge_0.15_dr_le_0.05": _meets(0.15),
            "verification_u2_df_ge_0.30_dr_le_0.05": _meets(0.30),
            "verification_u3_df_ge_0.90_dr_le_0.05": _meets(0.90),
        },
        "train_step_sec_mean": round(sum(step_secs) / len(step_secs), 4) if step_secs else None,
        "train_step_sec_max": round(max(step_secs), 4) if step_secs else None,
        **payload,
        "train": train_meta,
        "elapsed_sec": round(time.time() - started, 2),
        "limitations": [
            "Non-claim diagnostic; only 12 of 13 methods have core operators, not paper-level faithful reproduction.",
            "Formal mode uses 10 epochs and no early stop; debug mode keeps --steps.",
        ],
    }
    path = out_dir / f"{args.method}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"method": args.method, "path": str(path), "verdict": result["verdict"],
                      "delta": delta, "threshold_check": result["threshold_check"],
                      "elapsed_sec": result["elapsed_sec"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

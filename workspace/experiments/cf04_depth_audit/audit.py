#!/usr/bin/env python3
"""CF-04 v2 forgetting-depth audit: P1 / P2 / P3 / INT8 fake-quant.

Ported (read-only reference) from ``workspace/auto_experiments/audit_depth.py``.
Per-model output fields are kept identical to ``results/audit_depth.json`` so the
base arm can be checked against the historical baseline. Only dataset ground-truth
answers are used as labels.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_lite import (
    ASU_FORGET05,
    CF04_V2_FORMAL,
    CHECKPOINTS,
    FORGET,
    GRAD_ASCENT,
    MODEL,
    RETAIN,
    answer_nll,
    eval_nll,
    load_jsonl,
    make_batch,
    save_json,
    set_seed,
)

MODEL_PATHS = {
    "base": MODEL,
    "cf04": CF04_V2_FORMAL,
    "cf04_v2_formal_save": str(CHECKPOINTS / "full_cf04_v2_formal_save"),
    "cf04_formal": str(CHECKPOINTS / "full_cf04_formal"),
    "cf04_e1": str(CHECKPOINTS / "full_cf04_e1"),
    "ga": GRAD_ASCENT,
    "asu": ASU_FORGET05,
}


def fewshot_prefix(records, k=3):
    return "".join(f"Question: {r['question']}\nAnswer: {r['answer']}\n" for r in records[:k])


def fake_quantize(model):
    backup = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight.data
            scale = w.abs().max().clamp_min(1e-8) / 127.0
            backup[name] = w.clone()
            mod.weight.data.copy_(torch.round(w / scale).clamp(-127, 127) * scale)
    return backup


def restore(model, backup):
    for name, mod in model.named_modules():
        if name in backup:
            mod.weight.data.copy_(backup[name])


def quick_relearn(model, tok, records, steps=20, lr=1e-5, batch_size=4, device="cuda", seed=0):
    model.train()
    torch.manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for step in range(steps):
        chunk = [records[(step * batch_size + i) % len(records)] for i in range(batch_size)]
        b = make_batch(tok, chunk, device)
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        logits = out.logits[:, :-1, :].float()
        labels = b["labels"][:, 1:]
        m = labels != -100
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none").view(labels.shape)
        loss = (loss * m).sum() / m.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return steps


def direct_nll(model, tok, records, device):
    """Unweighted mean NLL on one batch, matching the historical audit_depth.json P1 fields."""
    with torch.no_grad():
        return answer_nll(model, make_batch(tok, records, device))[0]


def audit_one(name, path, tok, forget, retain, args):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(args.device).eval()
    limit = args.eval_limit or None
    f_eval = forget[:limit] if limit else forget
    r_eval = retain[:limit] if limit else retain
    f_few = forget[: args.fewshot_limit or len(forget)]
    f_relearn = forget[: args.relearn_limit or len(forget)]
    f1 = direct_nll(model, tok, f_eval, args.device)
    r1 = direct_nll(model, tok, r_eval, args.device)
    prefix = fewshot_prefix(forget[4:])
    f2 = eval_nll(model, tok, f_few, args.device, batch_size=args.eval_batch_size, fewshot_prefix=prefix)

    backup = fake_quantize(model)
    fq = direct_nll(model, tok, f_eval, args.device)
    rq = direct_nll(model, tok, r_eval, args.device)
    restore(model, backup)

    quick_relearn(model, tok, f_relearn, steps=args.relearn_steps, lr=args.relearn_lr, device=args.device, seed=0)
    f3 = direct_nll(model, tok, f_eval, args.device)
    r3 = direct_nll(model, tok, r_eval, args.device)
    out = {
        "path": path,
        "P1_direct": {"forget": f1, "retain": r1},
        "P2_fewshot": {"forget": f2, "gain": f1 - f2},
        "P3_relearn": {"forget": f3, "steps": args.relearn_steps, "gain": f1 - f3, "retain": r3},
        "INT8_fakequant": {"forget": fq, "retain": rq, "forget_shift": fq - f1},
    }
    del model
    torch.cuda.empty_cache()
    print(f"[{name}] P1 F={f1:.4f} R={r1:.4f} | P2 gain={f1-f2:+.4f} | P3 gain={f1-f3:+.4f} | Q shift={fq-f1:+.4f}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=MODEL)
    ap.add_argument("--cf04", default=CF04_V2_FORMAL)
    ap.add_argument("--ga", default=GRAD_ASCENT)
    ap.add_argument("--asu", default=ASU_FORGET05)
    ap.add_argument("--models", default="base,cf04,ga")
    ap.add_argument("--forget", default=FORGET)
    ap.add_argument("--retain", default=RETAIN)
    ap.add_argument("--relearn-steps", type=int, default=20)
    ap.add_argument("--relearn-lr", type=float, default=1e-5)
    ap.add_argument("--eval-limit", type=int, default=64, help="0 means full split")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--fewshot-limit", type=int, default=32)
    ap.add_argument("--relearn-limit", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="results/cf04_depth_audit.json")
    args = ap.parse_args(argv)
    set_seed(0)

    paths = dict(MODEL_PATHS)
    paths.update({"base": args.base, "cf04": args.cf04, "ga": args.ga, "asu": args.asu})
    names = [s.strip() for s in args.models.split(",") if s.strip()]
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    forget = load_jsonl(args.forget)
    retain = load_jsonl(args.retain)

    started = time.time()
    out = {}
    for name in names:
        path = paths.get(name)
        if not path:
            out[name] = {"error": f"unknown model name: {name}"}
            continue
        try:
            out[name] = audit_one(name, path, tok, forget, retain, args)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[{name}] ERROR {e}", flush=True)

    base_f1 = out.get("base", {}).get("P1_direct", {}).get("forget")
    for name, v in out.items():
        if name == "base" or "P1_direct" not in v or base_f1 is None:
            continue
        f = v["P1_direct"]["forget"]
        denom = max(f - base_f1, 1e-6)
        p2 = max(0.0, min(1.0, v["P2_fewshot"]["gain"] / denom))
        p3 = max(0.0, min(1.0, v["P3_relearn"]["gain"] / denom))
        v["depth_score"] = {"p2_gain_ratio": p2, "p3_gain_ratio": p3, "mean_recovery": (p2 + p3) / 2}
    verdict = "PASS" if any("depth_score" in v and v["depth_score"]["mean_recovery"] >= 0.30 for v in out.values()) else "KILL"
    result = {
        "idea": "V3-Idea2-depth-audit",
        "models": out,
        "gate": {"shallow_forgetting_detected": verdict == "PASS", "verdict": verdict},
        "interpretation": "PASS means at least one unlearned model recovers >=30% of its forget-NLL gain via P2/P3, i.e. shallow forgetting; KILL means all tested models are robust to these attacks.",
        "claim_level": False,
        "gate_status": "NOT_APPLIED",
        "run_meta": {
            "models": names,
            "device": args.device,
            "eval_limit": args.eval_limit,
            "eval_batch_size": args.eval_batch_size,
            "fewshot_limit": args.fewshot_limit,
            "relearn_limit": args.relearn_limit,
            "relearn_steps": args.relearn_steps,
            "relearn_lr": args.relearn_lr,
            "elapsed_sec": round(time.time() - started, 2),
            "budget_note": "cf04 = full_cf04_v2_formal (10 epochs / 4500 steps); v2g 3-seed result is 300 steps and is NOT the same budget.",
        },
        "notes": [
            "Non-claim diagnostic; ARIS evidence gate remains BLOCKED.",
            "asu is asu-forget05-asu-gd (forget05 split, 7B/vocab-32000), reference only.",
            "Historical audit_depth.json used --eval-limit 64 --eval-batch-size 8; base arm is the alignment check.",
        ],
    }
    save_json(result, args.output)
    print("GATE", result["gate"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

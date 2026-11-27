#!/usr/bin/env python3
"""Independent linear-probe channel + 3-seed depth audit for CF-04 v2.

Ported (read-only reference) from ``workspace/experiments/diagnostic_methods/r3_2_probe_audit.py``
and its ``base.py`` / ``cf04_dual_probes.py`` helpers. Output fields are kept identical to
``results/audit_depth_probe_3seed.json``. A high held-out probe AUC means the fact is still
linearly accessible in frozen hidden states.
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from audit import MODEL_PATHS, fake_quantize, fewshot_prefix, restore
from common_lite import FORGET, MODEL, RETAIN, answer_nll, eval_nll, load_jsonl, make_batch, save_json, set_seed


def as_tensor(value, dtype=torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def auc_score(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUC; labels are bool/0-1. Returns 0.5 for a degenerate set."""
    s = as_tensor(scores).flatten()
    y = as_tensor(labels).flatten().bool().to(s.device)
    pos = int(y.sum().item())
    neg = int((~y).sum().item())
    if pos == 0 or neg == 0:
        return 0.5
    order = torch.argsort(s)
    ranks = torch.empty(s.numel(), dtype=torch.float32, device=s.device)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=torch.float32, device=s.device)
    sorted_s = s[order]
    i = 0
    while i < s.numel():
        j = i
        while j + 1 < s.numel() and torch.isclose(sorted_s[j + 1], sorted_s[i]):
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return float(((ranks[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg)).item())


@dataclass
class LinearProbe:
    weight: torch.Tensor
    bias: torch.Tensor
    history: list[dict]

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return as_tensor(x) @ self.weight + self.bias


def fit_linear_probe(x, y, steps=300, lr=0.05, l2=1e-3) -> LinearProbe:
    x = as_tensor(x)
    y = as_tensor(y).flatten().float().to(x.device)
    if x.shape[0] != y.numel():
        raise ValueError("x and y must have the same number of rows")
    w = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device, requires_grad=True)
    b = torch.zeros((), dtype=x.dtype, device=x.device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    history = []
    for step in range(steps):
        logits = x @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y) + l2 * (w * w).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.item())})
    return LinearProbe(w.detach(), b.detach(), history)


def pooled_hidden(model, tok, records, device="cuda", layer=16, batch_size=8, max_len=512):
    chunks = []
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = make_batch(tok, records[i : i + batch_size], device, max_len=max_len)
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
            h = out.hidden_states[layer].float()
            m = batch["attention_mask"].unsqueeze(-1).float()
            chunks.append(((h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)).cpu())
    return torch.cat(chunks, dim=0)


def probe_recovery(model, tok, forget, retain, device="cuda", layer=16, steps=200, seed=0):
    set_seed(seed)
    hf = pooled_hidden(model, tok, forget[:64], device=device, layer=layer)
    hr = pooled_hidden(model, tok, retain[:64], device=device, layer=layer)
    n = min(hf.shape[0], hr.shape[0])
    x = torch.cat([hf[:n], hr[:n]])
    y = torch.cat([torch.ones(n), torch.zeros(n)])
    order = torch.randperm(x.shape[0])
    n_train = max(2, min(x.shape[0] - 2, int(0.6 * x.shape[0])))
    tr, te = order[:n_train], order[n_train:]
    probe = fit_linear_probe(x[tr], y[tr], steps=steps, lr=0.05)
    scores = probe.score(x[te])
    auc = auc_score(scores, y[te])
    return {
        "probe_auc": auc,
        "probe_recoverability": 2.0 * abs(auc - 0.5),
        "probe_score_forget": float(probe.score(hf[:n]).mean().item()),
        "probe_score_retain": float(probe.score(hr[:n]).mean().item()),
        "probe_norm": float(probe.weight.norm().item()),
        "probe_train_size": int(n_train),
        "probe_test_size": int(len(te)),
    }


def quick_relearn_seeded(model, tok, records, steps=20, lr=1e-5, batch_size=4, device="cuda", seed=0):
    """Seeded relearning with shuffled data order (the stock helper is order-fixed)."""
    model.train()
    rng = random.Random(seed)
    order = list(records)
    rng.shuffle(order)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for step in range(steps):
        chunk = [order[(step * batch_size + i) % len(order)] for i in range(batch_size)]
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


def audit_seed(name, path, tok, forget, retain, args, seed):
    set_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager").to(args.device).eval()
    f1 = answer_nll(model, make_batch(tok, forget[:64], args.device))[0]
    r1 = answer_nll(model, make_batch(tok, retain[:64], args.device))[0]
    prefix = fewshot_prefix(forget[4:])
    f2 = eval_nll(model, tok, forget[:32], args.device, batch_size=8, fewshot_prefix=prefix)
    probe_pre = probe_recovery(model, tok, forget, retain, args.device, layer=args.layer, steps=args.probe_steps, seed=seed)

    backup = fake_quantize(model)
    fq = answer_nll(model, make_batch(tok, forget[:64], args.device))[0]
    rq = answer_nll(model, make_batch(tok, retain[:64], args.device))[0]
    restore(model, backup)

    quick_relearn_seeded(model, tok, forget[:32], steps=args.relearn_steps, lr=args.relearn_lr, device=args.device, seed=seed)
    f3 = answer_nll(model, make_batch(tok, forget[:64], args.device))[0]
    r3 = answer_nll(model, make_batch(tok, retain[:64], args.device))[0]
    probe_post = probe_recovery(model, tok, forget, retain, args.device, layer=args.layer, steps=args.probe_steps, seed=seed)

    out = {
        "path": path,
        "seed": seed,
        "P1_direct": {"forget": f1, "retain": r1},
        "P2_fewshot": {"forget": f2, "gain": f1 - f2},
        "P3_relearn": {"forget": f3, "steps": args.relearn_steps, "gain": f1 - f3, "retain": r3},
        "INT8_fakequant": {"forget": fq, "retain": rq, "forget_shift": fq - f1},
        "probe_pre": probe_pre,
        "probe_post": probe_post,
    }
    del model
    torch.cuda.empty_cache()
    print(f"[{name} seed={seed}] P1 F={f1:.4f} R={r1:.4f} | P2={f1-f2:+.4f} | P3={f1-f3:+.4f} | INT8={fq-f1:+.4f} | probe={probe_pre['probe_auc']:.3f}", flush=True)
    return out


def _mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(sum(vals) / len(vals)), "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0, "n": len(vals)}


def aggregate(per_model_seed: dict, seeds: list[int], base_name: str = "base"):
    models = {}
    for name, by_seed in per_model_seed.items():
        vals = [by_seed[s] for s in seeds if s in by_seed and "error" not in by_seed[s]]
        if not vals:
            continue
        depth = []
        if name != base_name:
            for v in vals:
                b = per_model_seed.get(base_name, {}).get(v["seed"])
                if b is None:
                    continue
                denom = max(v["P1_direct"]["forget"] - b["P1_direct"]["forget"], 1e-6)
                p2 = max(0.0, min(1.0, v["P2_fewshot"]["gain"] / denom))
                p3 = max(0.0, min(1.0, v["P3_relearn"]["gain"] / denom))
                depth.append({"p2_gain_ratio": p2, "p3_gain_ratio": p3, "mean_recovery": (p2 + p3) / 2})
        models[name] = {
            "P1_forget": _mean_std([v["P1_direct"]["forget"] for v in vals]),
            "P1_retain": _mean_std([v["P1_direct"]["retain"] for v in vals]),
            "P2_gain": _mean_std([v["P2_fewshot"]["gain"] for v in vals]),
            "P3_gain": _mean_std([v["P3_relearn"]["gain"] for v in vals]),
            "P3_retain": _mean_std([v["P3_relearn"]["retain"] for v in vals]),
            "INT8_shift": _mean_std([v["INT8_fakequant"]["forget_shift"] for v in vals]),
            "probe_auc_pre": _mean_std([v["probe_pre"]["probe_auc"] for v in vals]),
            "probe_auc_post": _mean_std([v["probe_post"]["probe_auc"] for v in vals]),
            "probe_recoverability_pre": _mean_std([v["probe_pre"]["probe_recoverability"] for v in vals]),
            "probe_recoverability_post": _mean_std([v["probe_post"]["probe_recoverability"] for v in vals]),
            "depth_score": {
                "p2_gain_ratio": _mean_std([d["p2_gain_ratio"] for d in depth]),
                "p3_gain_ratio": _mean_std([d["p3_gain_ratio"] for d in depth]),
                "mean_recovery": _mean_std([d["mean_recovery"] for d in depth]),
            },
            "per_seed": {str(v["seed"]): v for v in vals},
        }
    return models


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=MODEL)
    ap.add_argument("--cf04", default=MODEL_PATHS["cf04"])
    ap.add_argument("--ga", default=MODEL_PATHS["ga"])
    ap.add_argument("--asu", default=MODEL_PATHS["asu"])
    ap.add_argument("--models", default="base,cf04,ga")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--forget", default=FORGET)
    ap.add_argument("--retain", default=RETAIN)
    ap.add_argument("--relearn-steps", type=int, default=20)
    ap.add_argument("--relearn-lr", type=float, default=1e-5)
    ap.add_argument("--probe-steps", type=int, default=200)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="results/cf04_probe_3seed.json")
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    names = [s.strip() for s in args.models.split(",") if s.strip()]
    paths = dict(MODEL_PATHS)
    paths.update({"base": args.base, "cf04": args.cf04, "ga": args.ga, "asu": args.asu})
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    forget = load_jsonl(args.forget)
    retain = load_jsonl(args.retain)

    base_cfg = AutoConfig.from_pretrained(args.base, local_files_only=True)
    started = time.time()
    per_model_seed = {}
    skipped = {}
    for name in names:
        path = paths.get(name)
        if not path or not os.path.isdir(path):
            skipped[name] = f"missing: {path}"
            print(f"SKIP {name}: {path} missing", flush=True)
            continue
        try:
            cfg = AutoConfig.from_pretrained(path, local_files_only=True)
        except Exception as exc:
            skipped[name] = f"config error: {exc}"
            continue
        if cfg.vocab_size != base_cfg.vocab_size or cfg.hidden_size != base_cfg.hidden_size:
            skipped[name] = f"architecture mismatch: vocab={cfg.vocab_size} hidden={cfg.hidden_size}"
            print(f"SKIP {name}: {skipped[name]}", flush=True)
            continue
        per_model_seed[name] = {}
        for seed in seeds:
            try:
                per_model_seed[name][seed] = audit_seed(name, path, tok, forget, retain, args, seed)
            except Exception as exc:
                per_model_seed[name][seed] = {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                print(f"[{name} seed={seed}] ERROR {exc}", flush=True)

    models = aggregate(per_model_seed, seeds)
    recoveries = [m["depth_score"]["mean_recovery"]["mean"] for m in models.values() if m["depth_score"]["mean_recovery"]["mean"] is not None]
    result = {
        "idea": "R3-2-probe-3seed",
        "claim_level": False,
        "gate_status": "NOT_APPLIED",
        "seeds": seeds,
        "models_requested": names,
        "models_skipped": skipped,
        "models": models,
        "gate": {
            "probe_channel_present": True,
            "shallow_forgetting_detected": bool(recoveries and max(recoveries) >= 0.30),
            "verdict": "PASS" if recoveries and max(recoveries) >= 0.30 else "KILL",
        },
        "notes": [
            "Probe is a held-out linear separability test on frozen hidden states, not a generation attack.",
            "cf04 uses full_cf04_v2_formal (10 epochs / 4500 steps); v2g 3-seed results are 300 steps, a different budget.",
            "asu-forget05-asu-gd is a 7B/vocab-32000 checkpoint and is skipped by the architecture guard.",
            "This is a non-claim diagnostic; no ARIS gate is invoked.",
        ],
        "run_meta": {"device": args.device, "elapsed_sec": round(time.time() - started, 2), "probe_steps": args.probe_steps, "layer": args.layer},
    }
    save_json(result, args.output)
    print("GATE", result["gate"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

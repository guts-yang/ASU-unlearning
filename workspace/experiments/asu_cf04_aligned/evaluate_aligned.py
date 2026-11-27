#!/usr/bin/env python3
"""Unified evaluation for all aligned checkpoints.

Reuses the already-validated primitives from workspace/experiments/cf04_depth_audit/:
  - probe.audit_seed / probe.aggregate : P1/P2/P3/INT8 + independent linear Probe (3 eval seeds)
  - behavior_check.behaviour_stats     : EOS / length / overlap / refusal
  - common_lite.eval_nll               : full-split teacher-forced NLL

The same evaluation code and parameters are applied to every arm/seed, so the
only difference between arms is the training loss.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / "cf04_depth_audit"
sys.path.insert(0, str(AUDIT))

import behavior_check  # noqa: E402
import probe as probe_audit  # noqa: E402
from aligned_core import FORGET, HOLDOUT, MODEL, RETAIN, eval_nll, load_jsonl, save_json  # noqa: E402

OFFICIAL_SPLITS = {
    "forget": "/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget05.json",
    "retain": "/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain95.json",
    "holdout": "/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/holdout05.json",
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-seeds", default="0,1,2")
    ap.add_argument("--output", default=str(HERE / "results" / "evaluation.json"))
    args = ap.parse_args(argv)
    root = Path(args.results_dir)
    started = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    forget, retain, holdout = load_jsonl(FORGET), load_jsonl(RETAIN), load_jsonl(HOLDOUT)
    eval_seeds = [int(s) for s in args.eval_seeds.split(",") if s.strip()]
    audit_args = SimpleNamespace(device=args.device, layer=16, probe_steps=200,
                                 relearn_steps=20, relearn_lr=1e-5)

    paths = {"base": MODEL}
    for d in sorted(root.glob("*_*")):
        if not d.is_dir() or d.name.startswith("smoke_") or not (d / "result.json").exists():
            continue
        ckpt = d / "checkpoint"
        if (ckpt / "config.json").exists():
            paths[d.name] = str(ckpt)

    per_model_seed = {}
    for name, path in paths.items():
        per_model_seed[name] = {}
        for seed in eval_seeds:
            try:
                per_model_seed[name][seed] = probe_audit.audit_seed(name, path, tok, forget, retain, audit_args, seed)
                print(f"[audit] {name} seed={seed} OK", flush=True)
            except Exception as exc:
                per_model_seed[name][seed] = {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                print(f"[audit] {name} seed={seed} ERROR {exc}", flush=True)
    aggregated = probe_audit.aggregate(per_model_seed, eval_seeds)

    evaluation = {}
    for name, path in paths.items():
        row = {"path": path, "audit": aggregated.get(name, {})}
        if name != "base":
            rp = root / name / "result.json"
            if rp.exists():
                r = json.loads(rp.read_text())
                row["train"] = {"delta": r["delta"], "before": r["before"], "after": r["after"],
                                "elapsed_sec": r["elapsed_sec"], "seed": r["seed"], "arm": r["arm"]}
        try:
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(args.device).eval()
            row["behavior"] = behavior_check.behaviour_stats(model, tok, forget, args.device, 24, 4)
            row["full_split_nll"] = {
                "forget": eval_nll(model, tok, forget, args.device),
                "retain": eval_nll(model, tok, retain, args.device),
                "holdout": eval_nll(model, tok, holdout, args.device),
            }
            if name in ("base", "asu_1001", "cf04_1001"):
                row["official_split_sensitivity"] = {
                    k: eval_nll(model, tok, load_jsonl(p), args.device) for k, p in OFFICIAL_SPLITS.items()
                }
            del model
            torch.cuda.empty_cache()
            print(f"[eval] {name} behavior eos={row['behavior']['eos_rate']:.2f} overlap={row['behavior']['mean_target_overlap']:.3f}", flush=True)
        except Exception as exc:
            row["eval_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[eval] {name} ERROR {exc}", flush=True)
        evaluation[name] = row

    out = {
        "claim_level": False, "gate_status": "NOT_APPLIED",
        "eval_seeds": eval_seeds, "models": evaluation,
        "run_meta": {"device": args.device, "elapsed_sec": round(time.time() - started, 2),
                     "audit_params": vars(audit_args), "base_model": MODEL},
        "notes": [
            "Same evaluation code/parameters applied to every arm and seed.",
            "official_split_sensitivity is an evaluation-only check on forget05/retain95/holdout05 (no training).",
            "1B / forget10 aligned reproduction; NOT a faithful reproduction of the official 7B / forget05 ASU.",
        ],
    }
    save_json(out, args.output)
    print(f"wrote {args.output} elapsed={out['run_meta']['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Auto-iteration stage 4: test the core 'progressive layer alignment' claim.

Stage3 showed intermediate layers (2/6/10/14) produced no output-level change
after 20 steps, while layer16 did. This stage tests whether that is merely a
short-horizon artifact and whether a progressive schedule beats static/final-only
alignment under matched budget.

Modes (40 steps, beta=8, lam=1.0 total split across aligned layers):
  ref        : no alignment (retain NLL only)
  last_only  : layer 16 for all steps
  static3    : layers 14,15,16 for all steps
  progressive: 14 (0-13) -> 14,15 (14-27) -> 14,15,16 (28-39)
  reverse    : 16 (0-13) -> 15,16 (14-27) -> 14,15,16 (28-39)
"""

import argparse
import copy
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from counterfactual import build_pairs, load_jsonl
from stage1_minimal import answer_nll, encode, last_logits_and_hidden
from stage3_sweep import evaluate


def layers_for(mode, step, total):
    if mode == "ref":
        return []
    if mode == "last_only":
        return [16]
    if mode == "static3":
        return [14, 15, 16]
    if mode == "progressive":
        if step < total // 3:
            return [14]
        if step < 2 * total // 3:
            return [14, 15]
        return [14, 15, 16]
    if mode == "reverse":
        if step < total // 3:
            return [16]
        if step < 2 * total // 3:
            return [15, 16]
        return [14, 15, 16]
    raise ValueError(mode)


def train_one(args, tok, pairs_train, retain_train, mode, seed, device):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    tp, tr = pairs_train[:], retain_train[:]
    rng.shuffle(tp)
    rng.shuffle(tr)

    student = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    teacher = copy.deepcopy(student).eval().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    for step in range(args.steps):
        pair = tp[step % len(tp)]
        rp = tr[step % len(tr)]
        ids_plus = encode(tok, pair["x_plus"], device)
        ids_minus = encode(tok, pair["x_minus"], device)
        q_ids = encode(tok, rp["question"], device)
        a_ids = encode(tok, rp["answer"], device)

        r_out = student(input_ids=torch.cat([q_ids, a_ids], dim=1))
        r_logits = r_out.logits[:, q_ids.size(1) - 1 : -1, :]
        l_r = F.cross_entropy(r_logits.reshape(-1, r_logits.size(-1)), a_ids.reshape(-1))
        loss = args.beta * l_r

        layers = layers_for(mode, step, args.steps)
        if layers:
            with torch.no_grad():
                _, th = last_logits_and_hidden(teacher, ids_minus, layers, device)
            _, sh = last_logits_and_hidden(student, ids_plus, layers, device)
            l_f = sum(F.mse_loss(sh[l].float(), th[l].float()) for l in layers) / len(layers)
            loss = loss + args.lam * l_f

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

    return student


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    ap.add_argument("--forget", default="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json")
    ap.add_argument("--retain", default="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json")
    ap.add_argument("--pairs", type=int, default=36)
    ap.add_argument("--train-pairs", type=int, default=18)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="stage4_progressive_result.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    tok = AutoTokenizer.from_pretrained(args.model)
    pairs = build_pairs(tok, args.forget, args.retain, max_pairs=args.pairs, seed=0)
    tp, fp = pairs[: args.train_pairs], pairs[args.train_pairs :]
    retain = load_jsonl(args.retain)
    tr, fr = retain[:100], retain[100:108]

    base = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    baseline = evaluate(base, tok, fp, fr, device)
    del base
    torch.cuda.empty_cache()
    print(f"baseline forget={baseline[0]:.4f} retain={baseline[1]:.4f}", flush=True)

    results = {"baseline": {"forget": baseline[0], "retain": baseline[1]}, "modes": {}}
    for mode in ("ref", "last_only", "static3", "progressive", "reverse"):
        runs = []
        for seed in seeds:
            m = train_one(args, tok, tp, tr, mode, seed, device)
            f, r = evaluate(m, tok, fp, fr, device)
            runs.append({"seed": seed, "forget": f, "retain": r})
            del m
            torch.cuda.empty_cache()
        fm = statistics.mean(x["forget"] for x in runs)
        rm = statistics.mean(x["retain"] for x in runs)
        fs = statistics.pstdev(x["forget"] for x in runs)
        rs = statistics.pstdev(x["retain"] for x in runs)
        results["modes"][mode] = {
            "forget_mean": fm, "forget_std": fs, "retain_mean": rm, "retain_std": rs,
            "d_forget": fm - baseline[0], "d_retain": rm - baseline[1], "runs": runs,
        }
        print(f"{mode}: forget {fm:.4f}±{fs:.4f} retain {rm:.4f}±{rs:.4f}", flush=True)

    ref = results["modes"]["ref"]
    for mode in ("last_only", "static3", "progressive", "reverse"):
        cur = results["modes"][mode]
        cur["vs_ref_forget"] = cur["forget_mean"] - ref["forget_mean"]
        cur["vs_ref_retain"] = cur["retain_mean"] - ref["retain_mean"]
        print(f"  {mode} vs ref: +{cur['vs_ref_forget']:.4f} forget, "
              f"+{cur['vs_ref_retain']:.4f} retain", flush=True)

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()

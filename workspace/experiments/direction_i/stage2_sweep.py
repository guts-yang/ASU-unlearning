"""Auto-iteration: sweep lambda_f with proper seed stochasticity + no-train baseline.

Triggered by stage1_ablation findings:
  - seeds were deterministic (no shuffling) -> add per-seed shuffling
  - lambda_f=1.0 raised forget NLL but also degraded held-out retain by +0.25
  - KL arm added nothing -> exclude KL from this sweep

Goal: find lambda_f where held-out forget NLL increases meaningfully while
held-out retain NLL stays within +0.05 of the no-training baseline.
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


def train_one(args, tok, pairs_train, retain_train, lam, seed, device):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    train_pairs = pairs_train[:]
    train_retain = retain_train[:]
    rng.shuffle(train_pairs)
    rng.shuffle(train_retain)

    student = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    teacher = copy.deepcopy(student).eval().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    for step in range(args.steps):
        pair = train_pairs[step % len(train_pairs)]
        rp = train_retain[step % len(train_retain)]
        ids_plus = encode(tok, pair["x_plus"], device)
        ids_minus = encode(tok, pair["x_minus"], device)
        q_ids = encode(tok, rp["question"], device)
        a_ids = encode(tok, rp["answer"], device)

        r_out = student(input_ids=torch.cat([q_ids, a_ids], dim=1))
        r_logits = r_out.logits[:, q_ids.size(1) - 1 : -1, :]
        l_r = F.cross_entropy(r_logits.reshape(-1, r_logits.size(-1)), a_ids.reshape(-1))
        loss = l_r

        if lam > 0:
            with torch.no_grad():
                _, t_hidden = last_logits_and_hidden(teacher, ids_minus, [args.layer], device)
            _, s_hidden = last_logits_and_hidden(student, ids_plus, [args.layer], device)
            l_f = F.mse_loss(s_hidden[args.layer].float(), t_hidden[args.layer].float())
            loss = loss + lam * l_f

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
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lams", default="0,0.05,0.1,0.2,0.4,0.7,1.0")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="stage2_sweep_result.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    pairs = build_pairs(tok, args.forget, args.retain, max_pairs=args.pairs, seed=0)
    train_pairs, test_pairs = pairs[: args.train_pairs], pairs[args.train_pairs :]
    retain = load_jsonl(args.retain)
    train_retain, test_retain = retain[:100], retain[100:108]
    seeds = [int(s) for s in args.seeds.split(",")]
    lams = [float(x) for x in args.lams.split(",")]

    print(f"train/test pairs={len(train_pairs)}/{len(test_pairs)} lams={lams} seeds={seeds}", flush=True)

    # No-training baseline (held-out).
    base_model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    baseline = {
        "forget": statistics.mean(answer_nll(base_model, tok, p["x_plus"], p["answer_plus"], device)
                                  for p in test_pairs),
        "retain": statistics.mean(answer_nll(base_model, tok, r["question"], r["answer"], device)
                                  for r in test_retain),
    }
    del base_model
    torch.cuda.empty_cache()
    print("baseline:", {k: round(v, 4) for k, v in baseline.items()}, flush=True)

    results = {}
    for lam in lams:
        runs = []
        for seed in seeds:
            model = train_one(args, tok, train_pairs, train_retain, lam, seed, device)
            forget = statistics.mean(answer_nll(model, tok, p["x_plus"], p["answer_plus"], device)
                                     for p in test_pairs)
            retain_nll = statistics.mean(answer_nll(model, tok, r["question"], r["answer"], device)
                                         for r in test_retain)
            runs.append({"seed": seed, "forget": forget, "retain": retain_nll})
            del model
            torch.cuda.empty_cache()
        results[str(lam)] = {
            "forget_mean": statistics.mean(r["forget"] for r in runs),
            "forget_std": statistics.pstdev(r["forget"] for r in runs),
            "retain_mean": statistics.mean(r["retain"] for r in runs),
            "retain_std": statistics.pstdev(r["retain"] for r in runs),
            "d_forget": statistics.mean(r["forget"] for r in runs) - baseline["forget"],
            "d_retain": statistics.mean(r["retain"] for r in runs) - baseline["retain"],
            "runs": runs,
        }
        r = results[str(lam)]
        print(f"lam={lam}: forget {r['forget_mean']:.4f}±{r['forget_std']:.4f} "
              f"(d={r['d_forget']:+.4f}) retain {r['retain_mean']:.4f}±{r['retain_std']:.4f} "
              f"(d={r['d_retain']:+.4f})", flush=True)

    out = {"baseline": baseline, "sweep": results}
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()

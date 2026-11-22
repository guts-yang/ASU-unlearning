"""Auto-iteration stage 3: sweep beta (retain weight) and layer position.

Motivation from stage2:
  - retain-NLL-only training improves BOTH retain and forget NLL (general transfer),
    so absolute forget NLL is not a forgetting signal.
  - layer16 alignment (lam>0) reduces the forget gain but reduces the retain gain
    even more -> no selectivity yet.
  - Hypothesis: stronger retain weight (beta) or lower-DeltaA layers may restore
    selectivity.

Runs beta sweep at layer16/lam=1.0 and layer sweep at beta=4/lam=1.0,
3 seeds each, held-out evaluation, vs no-train baseline and beta-only reference.
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


def train_one(args, tok, pairs_train, retain_train, lam, beta, layer, seed, device):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    train_pairs, train_retain = pairs_train[:], retain_train[:]
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
        loss = beta * l_r

        if lam > 0:
            with torch.no_grad():
                _, t_hidden = last_logits_and_hidden(teacher, ids_minus, [layer], device)
            _, s_hidden = last_logits_and_hidden(student, ids_plus, [layer], device)
            l_f = F.mse_loss(s_hidden[layer].float(), t_hidden[layer].float())
            loss = loss + lam * l_f

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

    return student


def evaluate(model, tok, test_pairs, test_retain, device):
    f = statistics.mean(answer_nll(model, tok, p["x_plus"], p["answer_plus"], device) for p in test_pairs)
    r = statistics.mean(answer_nll(model, tok, x["question"], x["answer"], device) for x in test_retain)
    return f, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    ap.add_argument("--forget", default="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json")
    ap.add_argument("--retain", default="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json")
    ap.add_argument("--pairs", type=int, default=36)
    ap.add_argument("--train-pairs", type=int, default=18)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--betas", default="1,2,4,8")
    ap.add_argument("--layers", default="2,6,10,14,16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="stage3_sweep_result.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    tok = AutoTokenizer.from_pretrained(args.model)
    pairs = build_pairs(tok, args.forget, args.retain, max_pairs=args.pairs, seed=0)
    train_pairs, test_pairs = pairs[: args.train_pairs], pairs[args.train_pairs :]
    retain = load_jsonl(args.retain)
    train_retain, test_retain = retain[:100], retain[100:108]

    base = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    baseline = evaluate(base, tok, test_pairs, test_retain, device)
    del base
    torch.cuda.empty_cache()
    print(f"baseline forget={baseline[0]:.4f} retain={baseline[1]:.4f}", flush=True)

    results = {"baseline": {"forget": baseline[0], "retain": baseline[1]}, "sweeps": {}}

    # Sweep 1: retain weight at layer16 / lam=1.0 (and beta-only reference lam=0).
    for beta in [float(x) for x in args.betas.split(",")]:
        for lam in ([0.0] if beta == 1.0 else [1.0]):
            key = f"beta{beta}_lam{lam}_layer16"
            runs = []
            for seed in seeds:
                m = train_one(args, tok, train_pairs, train_retain, lam, beta, 16, seed, device)
                f, r = evaluate(m, tok, test_pairs, test_retain, device)
                runs.append({"seed": seed, "forget": f, "retain": r})
                del m
                torch.cuda.empty_cache()
            results["sweeps"][key] = summarize(runs, baseline)
            print(key, results["sweeps"][key]["line"], flush=True)

    # Sweep 2: layer position at beta=4 / lam=1.0.
    for layer in [int(x) for x in args.layers.split(",") if int(x) != 16]:
        key = f"layer{layer}_beta4_lam1"
        runs = []
        for seed in seeds:
            m = train_one(args, tok, train_pairs, train_retain, 1.0, 4.0, layer, seed, device)
            f, r = evaluate(m, tok, test_pairs, test_retain, device)
            runs.append({"seed": seed, "forget": f, "retain": r})
            del m
            torch.cuda.empty_cache()
        results["sweeps"][key] = summarize(runs, baseline)
        print(key, results["sweeps"][key]["line"], flush=True)

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("wrote", args.output, flush=True)


def summarize(runs, baseline):
    fm = statistics.mean(r["forget"] for r in runs)
    rm = statistics.mean(r["retain"] for r in runs)
    fs = statistics.pstdev(r["forget"] for r in runs)
    rs = statistics.pstdev(r["retain"] for r in runs)
    # Selectivity: forget uplift relative to beta-only reference is computed later
    # from the full result file; here we report deltas vs no-train baseline.
    line = (f"forget {fm:.4f}±{fs:.4f} (d={fm-baseline[0]:+.4f}) "
            f"retain {rm:.4f}±{rs:.4f} (d={rm-baseline[1]:+.4f})")
    return {
        "forget_mean": fm, "forget_std": fs, "retain_mean": rm, "retain_std": rs,
        "d_forget": fm - baseline[0], "d_retain": rm - baseline[1],
        "runs": runs, "line": line,
    }


if __name__ == "__main__":
    main()

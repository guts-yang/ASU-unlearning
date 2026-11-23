"""Auto-iteration stage 5: position-masked counterfactual alignment.

Diagnosis from stages 2-4:
  H1 full-position MSE raises forget NLL but raises retain NLL ~2.3x as much,
  for every layer and schedule. Likely cause: aligning ALL positions drags the
  general-purpose representation, not just subject-related positions.

Fix under test: align only positions where teacher DeltaA is concentrated.
Modes:
  unmasked : all prompt positions (stage4 baseline)
  top2/top4: only the 2/4 positions with largest ||DeltaA||
  weighted : MSE weighted by softmax-scaled ||DeltaA||
Same beta=8, lam=1, layer16, 40 steps, 3 seeds, held-out evaluation.
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
from stage1_minimal import answer_nll, encode
from stage3_sweep import evaluate


def full_hidden(model, ids, layer, device):
    out = model(input_ids=ids, output_hidden_states=True)
    return out.hidden_states[layer]  # [B,T,d]


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
        ids_plus_t = encode(tok, pair["x_plus"], device)
        ids_minus_t = encode(tok, pair["x_minus"], device)

        with torch.no_grad():
            hp = full_hidden(teacher, ids_plus_t, args.layer, device)[0]
            hm = full_hidden(teacher, ids_minus_t, args.layer, device)[0]
            mag = torch.linalg.vector_norm((hp - hm).float(), dim=-1)  # [T]

        sh = full_hidden(student, ids_plus, args.layer, device)[0].float()
        th = hm.float().detach()
        mse = (sh - th).pow(2).sum(-1)  # [T]
        T = mse.numel()

        if mode == "ref":
            lf = mse.new_tensor(0.0)
        elif mode == "last_only":
            lf = mse[-1]
        elif mode == "unmasked":
            lf = mse.mean()
        elif mode.startswith("top"):
            k = int(mode[3:])
            k = min(k, T)
            idx = torch.topk(mag, k).indices
            lf = mse[idx].mean()
        elif mode == "weighted":
            w = F.softmax(mag / args.mask_temp, dim=0)
            lf = (mse * w).sum()
        else:
            raise ValueError(mode)

        q_ids = encode(tok, rp["question"], device)
        a_ids = encode(tok, rp["answer"], device)
        r_out = student(input_ids=torch.cat([q_ids, a_ids], dim=1))
        r_logits = r_out.logits[:, q_ids.size(1) - 1 : -1, :]
        l_r = F.cross_entropy(r_logits.reshape(-1, r_logits.size(-1)), a_ids.reshape(-1))

        loss = args.beta * l_r + args.lam * lf
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
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--mask-temp", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="stage5_masked_result.json")
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
    for mode in ("ref", "last_only", "unmasked", "top2", "weighted"):
        runs = []
        for seed in seeds:
            m = train_one(args, tok, tp, tr, mode, seed, device)
            f, r = evaluate(m, tok, fp, fr, device)
            runs.append({"seed": seed, "forget": f, "retain": r})
            del m
            torch.cuda.empty_cache()
        fm = statistics.mean(x["forget"] for x in runs)
        rm = statistics.mean(x["retain"] for x in runs)
        results["modes"][mode] = {
            "forget_mean": fm,
            "forget_std": statistics.pstdev(x["forget"] for x in runs),
            "retain_mean": rm,
            "retain_std": statistics.pstdev(x["retain"] for x in runs),
            "d_forget": fm - baseline[0],
            "d_retain": rm - baseline[1],
            "runs": runs,
        }
        print(f"{mode}: forget {fm:.4f} retain {rm:.4f} "
              f"(vs baseline {fm-baseline[0]:+.4f} / {rm-baseline[1]:+.4f})", flush=True)

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()

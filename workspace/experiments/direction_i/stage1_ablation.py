"""Experiment A from remote-agent review: isolate what actually drives forgetting.

Arms (Teacher frozen = initial checkpoint, Student starts identical):
  A1 : retain NLL only                      (optimization-noise baseline)
  A2 : A1 + layer-16 counterfactual MSE     (core alignment hypothesis)
  A3 : A2 + first-step forward KL           (does KL add signal or noise?)

Evaluation uses held-out counterfactual pairs and held-out retain records,
never seen during training. Three seeds per arm; report mean +/- std.
"""

import argparse
import copy
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from counterfactual import build_pairs, load_jsonl
from stage1_minimal import answer_nll, encode, last_logits_and_hidden


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
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--lambda-f", type=float, default=1.0)
    ap.add_argument("--lambda-kl", type=float, default=0.1)
    ap.add_argument("--kl-temp", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="stage1_ablation_result.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    pairs = build_pairs(tok, args.forget, args.retain, max_pairs=args.pairs, seed=0)
    pairs_train, pairs_test = pairs[: args.train_pairs], pairs[args.train_pairs :]
    retain = load_jsonl(args.retain)
    retain_train, retain_test = retain[:100], retain[100:108]
    print(f"pairs train/test={len(pairs_train)}/{len(pairs_test)} layer={args.layer} "
          f"seeds={args.seeds} steps={args.steps}")

    results = {}
    # Kept in one process; models are reloaded per arm/seed from the same checkpoint.
    for arm in ("A1", "A2", "A3"):
        arm_runs = []
        for seed in [int(s) for s in args.seeds.split(",")]:
            torch.manual_seed(seed)
            device = torch.device(args.device)
            student = AutoModelForCausalLM.from_pretrained(args.model).to(device)
            teacher = copy.deepcopy(student).eval().to(device)
            for p in teacher.parameters():
                p.requires_grad_(False)

            opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
            for step in range(args.steps):
                pair = pairs_train[step % len(pairs_train)]
                rp = retain_train[step % len(retain_train)]
                ids_plus = encode(tok, pair["x_plus"], device)
                ids_minus = encode(tok, pair["x_minus"], device)
                q_ids = encode(tok, rp["question"], device)
                a_ids = encode(tok, rp["answer"], device)
                r_ids = torch.cat([q_ids, a_ids], dim=1)
                r_out = student(input_ids=r_ids)
                r_logits = r_out.logits[:, q_ids.size(1) - 1 : -1, :]
                l_r = F.cross_entropy(r_logits.reshape(-1, r_logits.size(-1)), a_ids.reshape(-1))
                loss = l_r

                if arm in ("A2", "A3"):
                    with torch.no_grad():
                        _, t_hidden = last_logits_and_hidden(teacher, ids_minus, [args.layer], device)
                    _, s_hidden = last_logits_and_hidden(student, ids_plus, [args.layer], device)
                    l_f = F.mse_loss(s_hidden[args.layer].float(), t_hidden[args.layer].float())
                    loss = loss + args.lambda_f * l_f

                if arm == "A3":
                    with torch.no_grad():
                        t_logits, _ = last_logits_and_hidden(teacher, ids_minus, [args.layer], device)
                    s_logits, _ = last_logits_and_hidden(student, ids_plus, [args.layer], device)
                    l_kl = F.kl_div(
                        F.log_softmax(s_logits.float() / args.kl_temp, dim=-1),
                        F.softmax(t_logits.float() / args.kl_temp, dim=-1),
                        reduction="batchmean",
                    )
                    loss = loss + args.lambda_kl * l_kl

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()

            forget = statistics.mean(answer_nll(student, tok, p["x_plus"], p["answer_plus"], device)
                                     for p in pairs_test)
            retain_nll = statistics.mean(answer_nll(student, tok, r["question"], r["answer"], device)
                                         for r in retain_test)
            arm_runs.append({"seed": seed, "forget_nll": forget, "retain_nll": retain_nll})
            del student, teacher, opt
            torch.cuda.empty_cache()
            print(arm, arm_runs[-1], flush=True)

        results[arm] = {
            "forget_mean": statistics.mean(r["forget_nll"] for r in arm_runs),
            "forget_std": statistics.pstdev(r["forget_nll"] for r in arm_runs),
            "retain_mean": statistics.mean(r["retain_nll"] for r in arm_runs),
            "retain_std": statistics.pstdev(r["retain_nll"] for r in arm_runs),
            "runs": arm_runs,
        }

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n=== summary (held-out) ===")
    for arm, r in results.items():
        print(f"{arm}: forget {r['forget_mean']:.4f}±{r['forget_std']:.4f} "
              f"retain {r['retain_mean']:.4f}±{r['retain_std']:.4f}")
    print("wrote", args.output)


if __name__ == "__main__":
    main()

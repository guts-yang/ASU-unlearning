"""Stage-1 minimal GPU pilot for Direction I (H1: direct counterfactual alignment).

Implements, on a single H20:
  L_F  = MSE(h_S^l(X+), stopgrad(A_T^l(X-))) at a few deep layers
  L_KL = forward KL at the first generation step (empty prefix; DKT t=1 case)
  L_R  = answer-position NLL on a retained question

Teacher = frozen copy of the initial checkpoint; only Student is updated.
All code lives under workspace/ (upstream project untouched).
"""

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from counterfactual import build_pairs, load_jsonl


def encode(tokenizer, text, device):
    return tokenizer.encode(text, return_tensors="pt").to(device)


def last_logits_and_hidden(model, ids, layers, device):
    out = model(input_ids=ids, output_hidden_states=True)
    b = ids.size(0)
    last = ids.size(1) - 1
    logits = out.logits[:, last, :]
    hidden = {l: out.hidden_states[l][:, last, :] for l in layers}
    return logits, hidden


def answer_nll(model, tokenizer, prompt, answer, device):
    p = encode(tokenizer, prompt, device)
    a = encode(tokenizer, answer, device)
    ids = torch.cat([p, a], dim=1)
    out = model(input_ids=ids)
    logits = out.logits[:, p.size(1) - 1 : -1, :]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), a.reshape(-1), reduction="mean")
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    ap.add_argument("--forget", default="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json")
    ap.add_argument("--retain", default="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--layers", default="8,12,16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="stage1_minimal_result.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    layers = [int(x) for x in args.layers.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model)
    student = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    teacher = copy.deepcopy(student).eval().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    pairs = build_pairs(tok, args.forget, args.retain, max_pairs=args.pairs, seed=0)
    retain = load_jsonl(args.retain)
    print(f"pairs={len(pairs)} layers={layers} steps={args.steps} device={device}")

    # Baseline metrics before any update.
    pre = {
        "forget_nll": sum(answer_nll(student, tok, p["x_plus"], p["answer_plus"], device) for p in pairs) / len(pairs),
        "retain_nll": sum(answer_nll(student, tok, r["question"], r["answer"], device) for r in retain[:8]) / 8,
    }
    print("before:", {k: round(v, 4) for k, v in pre.items()})

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    history = []
    for step in range(args.steps):
        pair = pairs[step % len(pairs)]
        rp = retain[(step * 3) % len(retain)]

        ids_plus = encode(tok, pair["x_plus"], device)
        ids_minus = encode(tok, pair["x_minus"], device)

        with torch.no_grad():
            t_logits, t_hidden = last_logits_and_hidden(teacher, ids_minus, layers, device)
        s_logits, s_hidden = last_logits_and_hidden(student, ids_plus, layers, device)

        l_f = sum(F.mse_loss(s_hidden[l].float(), t_hidden[l].float()) for l in layers) / len(layers)
        l_kl = F.kl_div(
            F.log_softmax(s_logits.float(), dim=-1),
            F.softmax(t_logits.float(), dim=-1),
            reduction="batchmean",
        )
        r_ids = torch.cat(
            [encode(tok, rp["question"], device), encode(tok, rp["answer"], device)], dim=1
        )
        r_out = student(input_ids=r_ids)
        q_len = encode(tok, rp["question"], device).size(1)
        r_logits = r_out.logits[:, q_len - 1 : -1, :]
        r_target = r_ids[:, q_len:]
        l_r = F.cross_entropy(r_logits.reshape(-1, r_logits.size(-1)), r_target.reshape(-1))

        loss = l_f + l_kl + l_r
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        if (step + 1) % 5 == 0 or step == 0:
            rec = {
                "step": step + 1,
                "loss": round(loss.item(), 5),
                "l_f": round(l_f.item(), 5),
                "l_kl": round(l_kl.item(), 5),
                "l_r": round(l_r.item(), 5),
            }
            history.append(rec)
            print(rec)

    post = {
        "forget_nll": sum(answer_nll(student, tok, p["x_plus"], p["answer_plus"], device) for p in pairs) / len(pairs),
        "retain_nll": sum(answer_nll(student, tok, r["question"], r["answer"], device) for r in retain[:8]) / 8,
    }
    print("after:", {k: round(v, 4) for k, v in post.items()})

    result = {
        "model": args.model,
        "pairs": len(pairs),
        "layers": layers,
        "steps": args.steps,
        "lr": args.lr,
        "before": pre,
        "after": post,
        "history": history,
        "note": "toy pilot: L_KL uses only t=1 (empty prefix); not full DKT trajectory KL",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("wrote", args.output)


if __name__ == "__main__":
    main()

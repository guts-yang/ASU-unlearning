"""Lightweight ASU-style self-distillation under workspace/.

Implements the ASU forget-teacher (attention temperature on a frozen reference)
plus retain NLL, with configurable tau schedules / fields for Idea-B and SensiTemp.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import FORGET, MODEL, RETAIN, RESULTS, CKPT, TemperatureContext, answer_nll, load_jsonl, make_batch, save_json, set_seed


def kl_on_answer(student_logits, teacher_logits, labels):
    s = F.log_softmax(student_logits[:, :-1, :].float(), dim=-1)
    t = F.log_softmax(teacher_logits[:, :-1, :].float(), dim=-1)
    m = labels[:, 1:] != -100
    loss = F.kl_div(s, t, reduction="none", log_target=True).sum(-1)
    return (loss * m).sum() / m.sum().clamp_min(1)


def field_for_step(args, step, total, sens=None):
    n_head = args.n_head
    if args.schedule == "global":
        return args.tau
    if args.schedule == "high2low":
        v = 3.0 - 1.5 * step / max(total - 1, 1)
        return v
    if args.schedule == "low2high":
        v = 1.5 + 1.5 * step / max(total - 1, 1)
        return v
    if args.schedule == "random":
        v = random.uniform(1.5, 3.0)
        return v
    if args.schedule in ("sens", "sens_shuffled"):
        assert sens is not None
        tau_map = sens["tau_field_" + args.schedule]
        return {int(k): v for k, v in tau_map.items()}
    raise ValueError(args.schedule)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--forget", default=FORGET)
    ap.add_argument("--retain", default=RETAIN)
    ap.add_argument("--schedule", choices=["global", "high2low", "low2high", "random", "sens", "sens_shuffled"], default="global")
    ap.add_argument("--sens-file", default=str(RESULTS / "gate_sensitemp.json"))
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--name", default=None)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    set_seed(args.seed)
    name = args.name or f"tau_{args.schedule}_s{args.seed}"
    out = args.output or str(RESULTS / f"train_{name}.json")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    student = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device)
    ref = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(args.device).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    args.n_head = student.config.num_attention_heads
    n_layer = student.config.num_hidden_layers

    sens = None
    if args.schedule in ("sens", "sens_shuffled"):
        gate = json.loads(Path(args.sens_file).read_text())
        tau_map = {}
        ls = gate["layer_sensitivity"]
        hs_map = gate["head_sensitivity"]
        lscale = max(1e-6, max(abs(b) for b in ls.values()))
        all_hs = [b for hs in hs_map.values() for b in hs]
        hscale = max(1e-6, max(abs(b) for b in all_hs))
        for l in range(n_layer):
            layer_mod = 2.0 * max(0.0, ls.get(str(l), 0.0)) / lscale
            hs = hs_map.get(str(l))
            if hs is None:
                tau_map[str(l)] = [2.0 + layer_mod] * args.n_head
            else:
                tau_map[str(l)] = [2.0 + layer_mod + max(0.0, b) / hscale for b in hs]
        shuffled = {}
        rng = random.Random(args.seed)
        keys = list(tau_map.keys())
        shuffled_keys = keys[:]
        rng.shuffle(shuffled_keys)
        for k, sk in zip(keys, shuffled_keys):
            vals = tau_map[k][:]
            rng.shuffle(vals)
            shuffled[sk] = vals
        sens = {"tau_field_sens": tau_map, "tau_field_sens_shuffled": shuffled}

    forget = load_jsonl(args.forget)
    retain = load_jsonl(args.retain)
    random.Random(args.seed).shuffle(forget)
    random.Random(args.seed + 1).shuffle(retain)

    def eval_all():
        student.eval()
        f = answer_nll(student, make_batch(tok, forget[:64], args.device))[0]
        r = answer_nll(student, make_batch(tok, retain[:64], args.device))[0]
        student.train()
        return f, r

    f0, r0 = eval_all()
    print(f"[{name}] before forget={f0:.4f} retain={r0:.4f}", flush=True)

    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    history = []
    for step in range(args.steps):
        fbatch = [forget[(step * args.batch_size + i) % len(forget)] for i in range(args.batch_size)]
        rbatch = [retain[(step * args.batch_size + i) % len(retain)] for i in range(args.batch_size)]
        fb = make_batch(tok, fbatch, args.device)
        rb = make_batch(tok, rbatch, args.device)

        field = field_for_step(args, step, args.steps, sens)
        with torch.no_grad(), TemperatureContext(ref, field):
            ref_logits = ref(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits
        stu_logits = student(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits
        kl = kl_on_answer(stu_logits, ref_logits, fb["labels"])
        rlogits = student(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"]).logits
        rlabels = rb["labels"][:, 1:]
        rm = rlabels != -100
        rloss = F.cross_entropy(
            rlogits[:, :-1, :].float().reshape(-1, rlogits.size(-1)),
            rlabels.reshape(-1),
            reduction="none",
        ).view(rlabels.shape)
        rloss = (rloss * rm).sum() / rm.sum().clamp_min(1)
        loss = args.alpha * kl + rloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == args.steps - 1:
            history.append({"step": step, "loss": float(loss.item()), "kl": float(kl.item()), "retain": float(rloss.item())})
            print(f"[{name}] step {step:3d} loss={loss.item():.4f} kl={kl.item():.4f} retain={rloss.item():.4f}", flush=True)

    f1, r1 = eval_all()
    result = {
        "idea": "V2-IdeaB / V3-Idea1",
        "name": name,
        "schedule": args.schedule,
        "steps": args.steps,
        "lr": args.lr,
        "alpha": args.alpha,
        "before": {"forget": f0, "retain": r0},
        "after": {"forget": f1, "retain": r1},
        "delta": {"forget": f1 - f0, "retain": r1 - r0},
        "history": history,
    }
    if args.save:
        path = CKPT / name
        path.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(path, safe_serialization=True)
        tok.save_pretrained(path)
        result["checkpoint"] = str(path)
    save_json(result, out)
    print(f"[{name}] after forget={f1:.4f} retain={r1:.4f}", flush=True)


if __name__ == "__main__":
    main()

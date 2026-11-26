"""V3 Idea 2: ASU forgetting-depth audit (P1 / P2 / P3 / INT8 fake-quant).

Uses only dataset ground-truth answers; no model output is used as label.
"""

import argparse
import copy

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import CKPT, FORGET, MODEL, RETAIN, RESULTS, eval_nll, load_jsonl, make_batch, save_json, set_seed


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


def audit_one(name, path, tok, forget, retain, args):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(args.device).eval()
    limit = args.eval_limit or None
    f_eval = forget[:limit] if limit else forget
    r_eval = retain[:limit] if limit else retain
    f_few = forget[: args.fewshot_limit or len(forget)]
    f_relearn = forget[: args.relearn_limit or len(forget)]
    f1 = eval_nll(model, tok, f_eval, args.device, batch_size=args.eval_batch_size)
    r1 = eval_nll(model, tok, r_eval, args.device, batch_size=args.eval_batch_size)
    prefix = fewshot_prefix(forget[4:])
    f2 = eval_nll(model, tok, f_few, args.device, batch_size=args.eval_batch_size, fewshot_prefix=prefix)

    backup = fake_quantize(model)
    fq = eval_nll(model, tok, f_eval, args.device, batch_size=args.eval_batch_size)
    rq = eval_nll(model, tok, r_eval, args.device, batch_size=args.eval_batch_size)
    restore(model, backup)

    quick_relearn(model, tok, f_relearn, steps=args.relearn_steps, lr=args.relearn_lr, device=args.device, seed=0)
    f3 = eval_nll(model, tok, f_eval, args.device, batch_size=args.eval_batch_size)
    r3 = eval_nll(model, tok, r_eval, args.device, batch_size=args.eval_batch_size)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=MODEL)
    ap.add_argument("--asu", default=str(CKPT / "tau_global_s0"))
    ap.add_argument("--ga", default="/jizhicfs/chenyliao/runs/open-unlearning/20261122_forget10_1b/GradAscent")
    ap.add_argument("--forget", default=FORGET)
    ap.add_argument("--retain", default=RETAIN)
    ap.add_argument("--relearn-steps", type=int, default=20)
    ap.add_argument("--relearn-lr", type=float, default=1e-5)
    ap.add_argument("--eval-limit", type=int, default=64, help="0 means full split")
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--fewshot-limit", type=int, default=32)
    ap.add_argument("--relearn-limit", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=str(RESULTS / "audit_depth.json"))
    args = ap.parse_args()
    set_seed(0)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    forget = load_jsonl(args.forget)
    retain = load_jsonl(args.retain)

    models = {"base": args.base}
    try:
        torch.load  # noqa
        import os

        if os.path.exists(str(args.asu) + "/config.json"):
            models["asu"] = args.asu
    except Exception:
        pass
    import os

    if os.path.exists(str(args.ga) + "/model.safetensors"):
        models["grad_ascent"] = args.ga

    out = {}
    for name, path in models.items():
        try:
            out[name] = audit_one(name, path, tok, forget, retain, args)
        except Exception as e:
            out[name] = {"error": str(e)}
            print(f"[{name}] ERROR {e}", flush=True)

    base_f1 = out.get("base", {}).get("P1_direct", {}).get("forget")
    for name, v in out.items():
        if name == "base" or "P1_direct" not in v or base_f1 is None:
            continue
        f = v["P1_direct"]["forget"]
        denom = max(f - base_f1, 1e-6)
        v["depth_score"] = {
            "p2_gain_ratio": max(0.0, min(1.0, v["P2_fewshot"]["gain"] / denom)),
            "p3_gain_ratio": max(0.0, min(1.0, v["P3_relearn"]["gain"] / denom)),
            "mean_recovery": (max(0.0, min(1.0, v["P2_fewshot"]["gain"] / denom)) + max(0.0, min(1.0, v["P3_relearn"]["gain"] / denom))) / 2,
        }
    verdict = "PASS" if any("depth_score" in v and v["depth_score"]["mean_recovery"] >= 0.30 for v in out.values()) else "KILL"
    result = {
        "idea": "V3-Idea2-depth-audit",
        "models": out,
        "gate": {"shallow_forgetting_detected": verdict == "PASS", "verdict": verdict},
        "interpretation": "PASS means at least one unlearned model recovers >=30% of its forget-NLL gain via P2/P3, i.e. shallow forgetting; KILL means all tested models are robust to these attacks.",
    }
    save_json(result, args.output)
    print("GATE", result["gate"], flush=True)


if __name__ == "__main__":
    main()

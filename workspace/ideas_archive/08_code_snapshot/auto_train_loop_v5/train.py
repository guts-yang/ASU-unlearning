"""Universal idea interpreter v5 (only editable code file).

One invocation = one INDEPENDENT data-order replication. Adds order-independent
stabilizers (EMA weight averaging / tail-step parameter averaging / retain mixing)
on top of the v4 seed idea. Everything else matches v4 so metrics stay comparable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import prepare as P

sys.path.insert(0, str(P.AUTO))
import common as C  # noqa: E402

IDEA = P.LOOP / "current_idea.json"


def kl_on_answer(s_logits, t_logits, labels):
    s = F.log_softmax(s_logits[:, :-1, :].float(), dim=-1)
    t = F.log_softmax(t_logits[:, :-1, :].float(), dim=-1)
    m = labels[:, 1:] != -100
    return (F.kl_div(s, t, reduction="none", log_target=True).sum(-1) * m).sum() / m.sum().clamp_min(1)


def nll_on_answer(logits, labels):
    tgt = labels[:, 1:]
    m = tgt != -100
    loss = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
        tgt.reshape(-1), reduction="none",
    ).view(tgt.shape)
    return (loss * m).sum() / m.sum().clamp_min(1)


def eval_nll_chunked(model, tok, records, chunk=8):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(records), chunk):
            b = C.make_batch(tok, records[i:i + chunk], "cuda")
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            tgt = b["labels"][:, 1:]
            m = tgt != -100
            loss = F.cross_entropy(
                out.logits[:, :-1, :].float().reshape(-1, out.logits.size(-1)),
                tgt.reshape(-1), reduction="none",
            ).view(tgt.shape)
            total += (loss * m).sum().item()
            count += m.sum().item()
    return total / max(count, 1)


def generation_gate(model, tok, forget):
    model.eval()
    flags, lens, eos, examples = [], [], 0, []
    with torch.no_grad():
        for rec in forget[:P.GEN_SAMPLES]:
            ids = tok.encode(C.build_prompt(rec["question"], None), add_special_tokens=False)
            inp = torch.tensor([ids], device="cuda")
            out = model.generate(inp, attention_mask=torch.ones_like(inp),
                                 max_new_tokens=P.GEN_MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            new = out[0][len(ids):]
            text = tok.decode(new, skip_special_tokens=True)
            n = int(new.numel())
            flags.append(P.degenerate(text, n))
            lens.append(n)
            eos += int((new == tok.eos_token_id).any().item())
            if len(examples) < 3:
                examples.append(text[:120])
    model.train()
    return {"refusal_rate": sum(flags) / max(len(flags), 1),
            "mean_len": sum(lens) / max(len(lens), 1),
            "eos_rate": eos / max(len(flags), 1), "examples": examples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-order", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    issues = P.preflight()
    if issues:
        print("PREFLIGHT FAILED:", issues, flush=True)
        sys.exit(2)
    if not IDEA.exists():
        print("ERROR: current_idea.json missing", flush=True)
        sys.exit(3)

    spec = json.loads(IDEA.read_text(encoding="utf-8"))
    params = spec["params"]
    if params["kind"] != "alpha_hinge":
        raise ValueError(f"v5 only supports the seed idea family, got {params['kind']}")

    stab = params.get("stabilizer", {"type": "none"})
    stype = stab["type"]
    if stype not in ("none", "ema", "tail_avg", "retain_mix"):
        raise ValueError(f"unknown stabilizer: {stype}")

    C.set_seed(P.INIT_SEED)
    random.seed(args.data_order)
    torch.cuda.reset_peak_memory_stats()

    tok = AutoTokenizer.from_pretrained(C.MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    student = AutoModelForCausalLM.from_pretrained(C.MODEL, torch_dtype=torch.float32).to("cuda")
    ref = AutoModelForCausalLM.from_pretrained(C.MODEL, torch_dtype=torch.float32).to("cuda").eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    forget = C.load_jsonl(P.FORGET)
    retain = C.load_jsonl(P.RETAIN)
    random.Random(args.data_order).shuffle(forget)
    random.Random(args.data_order + 1).shuffle(retain)

    alpha = params.get("alpha", P.ALPHA)
    hinge_beta = params.get("hinge_beta", 0.0)
    hinge_margin = params.get("hinge_margin", 1.2)
    lr = params.get("lr", P.LR)
    retain_weight = params.get("retain_weight", 1.0)
    grad_clip = params.get("grad_clip", 1.0)
    retain_n = stab.get("n", 4) if stype == "retain_mix" else 4
    ema_decay = stab.get("decay", 0.9)
    tail_k = stab.get("k", 10)

    named = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    if stype == "ema":
        ema = {n: p.detach().clone() for n, p in named}
    else:
        ema = None
    if stype == "tail_avg":
        tail_sum = {n: torch.zeros_like(p) for n, p in named}
    else:
        tail_sum = None
    tail_count = 0

    def ev(records):
        return eval_nll_chunked(student, tok, records)

    f0, r0 = ev(forget[:64]), ev(retain[:64])
    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    for step in range(P.STEPS):
        fb = C.make_batch(tok, [forget[(step * 4 + i) % len(forget)] for i in range(4)], "cuda")
        rb = C.make_batch(tok, [retain[(step * retain_n + i) % len(retain)] for i in range(retain_n)], "cuda")
        with torch.no_grad(), C.TemperatureContext(ref, P.TAU):
            ref_logits = ref(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits
        stu_logits = student(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits
        kl = kl_on_answer(stu_logits, ref_logits, fb["labels"])
        rlogits = student(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"]).logits
        loss = alpha * kl + retain_weight * nll_on_answer(rlogits, rb["labels"])
        if hinge_beta > 0:
            loss = loss + hinge_beta * F.relu(
                torch.tensor(hinge_margin, device="cuda") - nll_on_answer(stu_logits, fb["labels"])
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        opt.step()

        if stype == "ema":
            with torch.no_grad():
                for n, p in named:
                    ema[n].mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)
        elif stype == "tail_avg" and step >= P.STEPS - tail_k:
            with torch.no_grad():
                for n, p in named:
                    tail_sum[n].add_(p.detach())
            tail_count += 1

    # Apply the stabilizer BEFORE evaluation (this is the actual model we assess).
    if stype == "ema":
        with torch.no_grad():
            for n, p in named:
                p.data.copy_(ema[n])
    elif stype == "tail_avg" and tail_count > 0:
        with torch.no_grad():
            for n, p in named:
                p.data.copy_(tail_sum[n] / tail_count)

    f1, r1 = ev(forget[:64]), ev(retain[:64])
    dF, dR = f1 - f0, r1 - r0
    gen = generation_gate(student, tok, forget)

    p3opt = torch.optim.SGD(student.parameters(), lr=P.P3_LR)
    student.train()
    for k in range(P.P3_STEPS):
        b = C.make_batch(tok, [forget[(k * 4 + i) % 32] for i in range(4)], "cuda")
        out = student(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        p3opt.zero_grad(set_to_none=True)
        nll_on_answer(out.logits, b["labels"]).backward()
        p3opt.step()
    student.eval()
    f3 = ev(forget[:64])

    rho = P.rho_p3(f1, f3, P.base_forget_nll())
    refusal = gen["refusal_rate"]
    val = P.aggregate([{"rho": rho, "dF": dF, "dR": dR,
                        "refusal_rate": refusal, "mean_len": gen["mean_len"]}], 0)["val_bpb"]
    result = {"data_order": args.data_order, "idea_id": spec["id"], "lever": spec["lever"],
              "params": params, "init_seed": P.INIT_SEED,
              "before": {"forget": f0, "retain": r0},
              "after": {"forget": f1, "retain": r1},
              "p3_f3": f3, "rho": rho, "dF": dF, "dR": dR,
              "generation": gen, "refusal_rate": refusal, "mean_len": gen["mean_len"],
              "val_bpb": val,
              "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 ** 2)}
    P.dump_json(result, args.out)

    print(f"val_bpb: {val:.6f}")
    print(f"rho_p3: {rho:.6f}  dF: {dF:.4f}  dR: {dR:.4f}")
    print(f"refusal_rate: {refusal:.4f}  mean_len: {gen['mean_len']:.1f}")
    print(f"peak_vram_mb: {result['peak_vram_mb']:.1f}")


if __name__ == "__main__":
    main()

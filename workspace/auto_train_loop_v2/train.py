"""Universal idea interpreter v2 (only editable code file).

Reads current_idea.json -> fixed-budget training -> evaluation -> generation
quality gate -> P3 recovery attack -> single metric. Unknown levers are REJECTED.
--seed/--out/--skip-gen support the 3-seed confirmation rounds.
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
import common as C  # noqa: E402 (upstream, read-only)

IDEA = P.LOOP / "current_idea.json"


def kl_on_answer(student_logits, teacher_logits, labels):
    s = F.log_softmax(student_logits[:, :-1, :].float(), dim=-1)
    t = F.log_softmax(teacher_logits[:, :-1, :].float(), dim=-1)
    m = labels[:, 1:] != -100
    loss = F.kl_div(s, t, reduction="none", log_target=True).sum(-1)
    return (loss * m).sum() / m.sum().clamp_min(1)


def nll_on_answer(logits, labels):
    tgt = labels[:, 1:]
    m = tgt != -100
    loss = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
        tgt.reshape(-1), reduction="none",
    ).view(tgt.shape)
    return (loss * m).sum() / m.sum().clamp_min(1)


def generation_gate(model, tok, forget):
    """Measure degenerate / refusal-like outputs on the unlearned model."""
    model.eval()
    flags, lens, eos_hits, examples = [], [], 0, []
    with torch.no_grad():
        for rec in forget[:P.GEN_SAMPLES]:
            prompt = C.build_prompt(rec["question"], None)
            ids = tok.encode(prompt, add_special_tokens=False)
            inp = torch.tensor([ids], device="cuda")
            out = model.generate(inp, attention_mask=torch.ones_like(inp),
                                 max_new_tokens=P.GEN_MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            new = out[0][len(ids):]
            text = tok.decode(new, skip_special_tokens=True)
            n = int(new.numel())
            flags.append(P.degenerate(text, n))
            lens.append(n)
            eos_hits += int((new == tok.eos_token_id).any().item())
            if len(examples) < 3:
                examples.append(text[:120])
    model.train()
    return {
        "refusal_rate": sum(flags) / max(len(flags), 1),
        "mean_len": sum(lens) / max(len(lens), 1),
        "eos_rate": eos_hits / max(len(flags), 1),
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--refusal", type=float, default=0.0)
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
    kind = params["kind"]
    if kind not in ("baseline", "hinge", "alpha", "retain_weight", "alpha_hinge"):
        raise ValueError(f"unknown lever kind: {kind}")

    seed = args.seed if args.seed is not None else spec["budget"]["seed"]
    C.set_seed(seed)
    random.seed(seed)
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
    random.Random(seed).shuffle(forget)
    random.Random(seed + 1).shuffle(retain)

    alpha = params.get("alpha", P.ALPHA)
    retain_weight = params.get("retain_weight", 1.0)
    hinge_beta = params.get("hinge_beta", 0.0)
    hinge_margin = params.get("hinge_margin", 1.2)

    def eval_nll_chunked(records, chunk=8):
        """Token-weighted answer NLL in small chunks (GPU-sharing friendly)."""
        student.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(records), chunk):
                b = C.make_batch(tok, records[i:i + chunk], "cuda")
                out = student(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
                tgt = b["labels"][:, 1:]
                m = tgt != -100
                loss = F.cross_entropy(
                    out.logits[:, :-1, :].float().reshape(-1, out.logits.size(-1)),
                    tgt.reshape(-1), reduction="none",
                ).view(tgt.shape)
                total += (loss * m).sum().item()
                count += m.sum().item()
        return total / max(count, 1)

    def eval_all():
        f = eval_nll_chunked(forget[:64])
        r = eval_nll_chunked(retain[:64])
        student.train()
        return f, r

    f0, r0 = eval_all()
    opt = torch.optim.AdamW(student.parameters(), lr=P.LR, weight_decay=0.0)
    for step in range(P.STEPS):
        fb = C.make_batch(tok, [forget[(step * 4 + i) % len(forget)] for i in range(4)], "cuda")
        rb = C.make_batch(tok, [retain[(step * 4 + i) % len(retain)] for i in range(4)], "cuda")
        with torch.no_grad(), C.TemperatureContext(ref, P.TAU):
            ref_logits = ref(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits
        stu_logits = student(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits
        kl = kl_on_answer(stu_logits, ref_logits, fb["labels"])
        rlogits = student(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"]).logits
        rloss = nll_on_answer(rlogits, rb["labels"])
        loss = alpha * kl + retain_weight * rloss
        if hinge_beta > 0:
            loss = loss + hinge_beta * F.relu(
                torch.tensor(hinge_margin, device="cuda") - nll_on_answer(stu_logits, fb["labels"])
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

    f1, r1 = eval_all()
    dF, dR = f1 - f0, r1 - r0

    if args.skip_gen:
        gen = {"refusal_rate": args.refusal, "mean_len": 999.0,
               "eos_rate": None, "examples": [], "skipped": True}
    else:
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
    f3 = eval_nll_chunked(forget[:64])

    f_base = P.base_forget_nll()
    rho = P.rho_p3(f1, f3, f_base)
    refusal = gen["refusal_rate"]
    metric = P.val_bpb(rho, dF, dR, spec["complexity"], refusal)
    checks = P.initial_checks({"rho": rho, "dF": dF, "dR": dR,
                               "refusal_rate": refusal, "mean_len": gen["mean_len"]})
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    result = {
        "round": spec["round"], "idea_id": spec["id"], "lever": spec["lever"],
        "params": params, "complexity": spec["complexity"], "seed": seed,
        "source": spec["source"], "llm_backend": spec.get("llm_backend"),
        "before": {"forget": f0, "retain": r0},
        "after": {"forget": f1, "retain": r1},
        "p3": {"f3": f3, "f_base": f_base, "rho": rho},
        "delta": {"forget": dF, "retain": dR},
        "generation": gen, "refusal_rate": refusal, "mean_len": gen["mean_len"],
        "initial_checks": checks, "rho": rho, "dF": dF, "dR": dR,
        "val_bpb": metric, "seed_confirmed": False, "peak_vram_mb": peak_mb,
    }
    out_path = args.out or str(P.LOOP / "rounds" / f"R{spec['round']}.json")
    P.dump_json(result, out_path)

    print(f"val_bpb: {metric:.6f}")
    print(f"peak_vram_mb: {peak_mb:.1f}")
    print(f"rho_p3: {rho:.6f}  dF: {dF:.4f}  dR: {dR:.4f}")
    print(f"refusal_rate: {refusal:.4f}  mean_len: {gen['mean_len']:.1f}  screen: {checks}")


if __name__ == "__main__":
    main()

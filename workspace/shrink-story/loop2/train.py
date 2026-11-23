"""Single editable file of shrink-story Round 2 (depth / anti-recovery loop).

Each round changes ONLY the CONFIG dict. Metric, budget, P3 protocol and data
are fixed by prepare.py / program.md / upstream common.py.

All rounds use the SAME global tau=2.25 (the Round-1 equal-budget anchor), so any
difference in recovery depth cannot be attributed to a stronger teacher smoothing.
"""

from __future__ import annotations

import random
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import prepare as P

sys.path.insert(0, str(P.AUTO))
import common as C  # noqa: E402  (upstream, read-only)

CONFIG = {
    "round": "B0",
    "kind": "anchor",
    "projection": "none",
    "complexity": 0,
    "description": "anchor: equal-budget depth baseline global tau=2.25"
}


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


def grad_of(module, model, tok, records, device="cuda", batches=2):
    """Accumulated gradient of answer NLL w.r.t. one module's weight."""
    acc = torch.zeros_like(module.weight)
    for i in range(batches):
        b = C.make_batch(tok, records[i * 4: i * 4 + 4], device)
        model.zero_grad(set_to_none=True)
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        nll_on_answer(out.logits, b["labels"]).backward()
        if module.weight.grad is not None:
            acc += module.weight.grad.detach().float()
    model.zero_grad(set_to_none=True)
    return acc


def build_operator(cfg, module, model, tok, forget, retain):
    """Precompute the fixed projection operator / sign mask from the base model.

    This is an approximation of per-step RepSelect / SAGO operators: computed once
    before training so every round keeps the same 60-step budget. Disclosed in program.md.
    """
    if cfg["projection"] in ("none", "hinge_keep"):
        return None, None
    gf = grad_of(module, model, tok, forget)
    gr = grad_of(module, model, tok, retain)
    if cfg["projection"] == "sign_gate":
        return None, (gf * gr >= 0).float()
    if cfg["projection"] == "random_ortho":
        g = torch.randn(gf.shape[1], P.RANK, device=gf.device, generator=torch.Generator(device=gf.device).manual_seed(P.SEED))
        q, _ = torch.linalg.qr(g)
        return q, None
    _, _, Vh = torch.linalg.svd(gf.float(), full_matrices=False)
    V = Vh.T.contiguous()  # [d_in, min(d_out,d_in)]
    if cfg["projection"] == "top_pc":
        return V[:, :P.RANK], None
    if cfg["projection"] == "bottom_pc":
        return V[:, -P.RANK:], None
    raise ValueError(cfg["projection"])


def apply_operator(cfg, module, basis, sign_mask):
    """Modify .grad in place after backward(), before optimizer.step()."""
    if cfg["projection"] in ("none", "hinge_keep"):
        return
    g = module.weight.grad
    if g is None:
        return
    if cfg["projection"] == "sign_gate":
        module.weight.grad = g * sign_mask.to(g.dtype)
    else:
        B = basis.to(g.dtype)
        module.weight.grad = (g @ B) @ B.T


def main():
    issues = P.preflight()
    if issues:
        print("PREFLIGHT FAILED:", issues, flush=True)
        sys.exit(2)

    cfg = CONFIG
    C.set_seed(P.SEED)
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
    random.Random(P.SEED).shuffle(forget)
    random.Random(P.SEED + 1).shuffle(retain)

    module = student.model.layers[P.TARGET_LAYER].self_attn.v_proj
    student.train()
    basis, sign_mask = build_operator(cfg, module, student, tok, forget, retain)

    def eval_all():
        student.eval()
        f = C.answer_nll(student, C.make_batch(tok, forget[:64], "cuda"))[0]
        r = C.answer_nll(student, C.make_batch(tok, retain[:64], "cuda"))[0]
        student.train()
        return f, r

    f0, r0 = eval_all()

    opt = torch.optim.AdamW(student.parameters(), lr=P.LR, weight_decay=0.0)
    for step in range(P.STEPS):
        fb = C.make_batch(tok, [forget[(step * 4 + i) % len(forget)] for i in range(4)], "cuda")
        rb = C.make_batch(tok, [retain[(step * 4 + i) % len(retain)] for i in range(4)], "cuda")
        with torch.no_grad(), C.TemperatureContext(ref, cfg.get("tau", P.TAU)):
            ref_logits = ref(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits
        stu_logits = student(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits
        kl = kl_on_answer(stu_logits, ref_logits, fb["labels"])
        rlogits = student(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"]).logits
        rloss = nll_on_answer(rlogits, rb["labels"])
        loss = P.ALPHA * kl + rloss
        if cfg["projection"] == "hinge_keep":
            loss = loss + 0.1 * F.relu(torch.tensor(1.2, device="cuda") - nll_on_answer(stu_logits, fb["labels"]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        apply_operator(cfg, module, basis, sign_mask)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

    f1, r1 = eval_all()
    dF, dR = f1 - f0, r1 - r0

    # Fixed P3 recovery attack: 20 steps of SGD on a forget subset, then re-eval.
    p3opt = torch.optim.SGD(student.parameters(), lr=P.P3_LR)
    student.train()
    for k in range(P.P3_STEPS):
        b = C.make_batch(tok, [forget[(k * 4 + i) % 32] for i in range(4)], "cuda")
        out = student(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        p3opt.zero_grad(set_to_none=True)
        nll_on_answer(out.logits, b["labels"]).backward()
        p3opt.step()
    student.eval()
    f3 = C.answer_nll(student, C.make_batch(tok, forget[:64], "cuda"))[0]

    f_base = P.base_forget_nll()
    rho = P.rho_p3(f1, f3, f_base)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    if cfg["kind"] == "anchor":
        P.dump_json(
            {"round": cfg["round"], "rho": rho, "dF": dF, "dR": dR,
             "f1": f1, "f3": f3, "f_base": f_base},
            P.ANCHOR,
        )
        anchor = {"rho": rho, "dF": dF, "dR": dR}
    else:
        if not P.ANCHOR.exists():
            print("ERROR: depth anchor missing; B0 must run first", flush=True)
            sys.exit(3)
        anchor = P.load_json(P.ANCHOR)

    val_bpb = P.compute_regret(rho, dF, dR, anchor, cfg["complexity"], cfg["kind"])
    P.dump_json(
        {"round": cfg["round"], "kind": cfg["kind"], "config": cfg,
         "before": {"forget": f0, "retain": r0},
         "after": {"forget": f1, "retain": r1},
         "p3": {"f3": f3, "f_base": f_base, "rho": rho},
         "delta": {"forget": dF, "retain": dR},
         "anchor": anchor, "val_bpb": val_bpb, "peak_vram_mb": peak_mb},
        P.ROUNDS / f"{cfg['round']}.json",
    )

    print(f"val_bpb: {val_bpb:.6f}")
    print(f"peak_vram_mb: {peak_mb:.1f}")
    print(f"rho_p3: {rho:.6f}  dF: {dF:.4f}  dR: {dR:.4f}  f3: {f3:.4f}")


if __name__ == "__main__":
    main()

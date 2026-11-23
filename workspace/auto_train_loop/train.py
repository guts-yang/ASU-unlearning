"""Universal idea interpreter (the only editable code file).

Reads current_idea.json -> configures one fixed-budget training run -> evaluates
-> runs the frozen P3 recovery attack -> emits the single metric. Unknown levers
are REJECTED with ValueError (never silently ignored).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

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


def main():
    issues = P.preflight()
    if issues:
        print("PREFLIGHT FAILED:", issues, flush=True)
        sys.exit(2)
    if not IDEA.exists():
        print("ERROR: current_idea.json missing", flush=True)
        sys.exit(3)

    spec = json.loads(IDEA.read_text(encoding="utf-8"))
    kind = spec["params"]["kind"]
    if kind not in ("baseline", "hinge", "alpha", "retain_weight"):
        raise ValueError(f"unknown lever kind: {kind}")

    seed = spec["budget"]["seed"]
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

    alpha = spec["params"].get("alpha", P.ALPHA)
    retain_weight = spec["params"].get("retain_weight", 1.0)
    hinge_beta = spec["params"].get("hinge_beta", 0.0)
    hinge_margin = spec["params"].get("hinge_margin", 1.2)

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
    metric = P.val_bpb(rho, dF, dR, spec["complexity"])
    ok, checks = P.is_good({"rho": rho, "dF": dF, "dR": dR, "seed_confirmed": False})
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    result = {
        "round": spec["round"], "idea_id": spec["id"], "lever": spec["lever"],
        "params": spec["params"], "complexity": spec["complexity"],
        "source": spec["source"], "llm_backend": spec.get("llm_backend"),
        "before": {"forget": f0, "retain": r0},
        "after": {"forget": f1, "retain": r1},
        "p3": {"f3": f3, "f_base": f_base, "rho": rho},
        "delta": {"forget": dF, "retain": dR},
        "rho": rho, "dF": dF, "dR": dR,
        "val_bpb": metric, "target_met": ok, "target_checks": checks,
        "peak_vram_mb": peak_mb,
    }
    P.dump_json(result, P.LOOP / "rounds" / f"R{spec['round']}.json")

    print(f"val_bpb: {metric:.6f}")
    print(f"peak_vram_mb: {peak_mb:.1f}")
    print(f"rho_p3: {rho:.6f}  dF: {dF:.4f}  dR: {dR:.4f}  target_met: {ok}")


if __name__ == "__main__":
    main()

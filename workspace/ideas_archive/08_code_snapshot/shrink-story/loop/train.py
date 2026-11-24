"""Single editable file of the shrink-story autoresearch loop.

Each round changes ONLY the CONFIG dict below. Everything else (metric definition,
data, model, budget) is fixed by prepare.py / program.md / upstream common.py.
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
import common as C  # noqa: E402  (upstream, read-only)

CONFIG = {
    "round": "A1",
    "kind": "anchor",
    "schedule": "global",
    "tau": 2.25,
    "tau_start": 1.5,
    "tau_end": 3.0,
    "mean_tau": 2.25,
    "complexity": 0,
    "description": "anchor: global tau=2.25 equal-budget control"
}


def kl_on_answer(student_logits, teacher_logits, labels):
    s = F.log_softmax(student_logits[:, :-1, :].float(), dim=-1)
    t = F.log_softmax(teacher_logits[:, :-1, :].float(), dim=-1)
    m = labels[:, 1:] != -100
    loss = F.kl_div(s, t, reduction="none", log_target=True).sum(-1)
    return (loss * m).sum() / m.sum().clamp_min(1)


def build_field(cfg, n_layer, n_head):
    """Spatial temperature field (SensiTemp / shuffled), identical construction to upstream."""
    gate = P.load_json(P.SENS_FILE)
    ls, hs_map = gate["layer_sensitivity"], gate["head_sensitivity"]
    lscale = max(1e-6, max(abs(b) for b in ls.values()))
    all_hs = [b for hs in hs_map.values() for b in hs]
    hscale = max(1e-6, max(abs(b) for b in all_hs))
    tau_map = {}
    for l in range(n_layer):
        layer_mod = 2.0 * max(0.0, ls.get(str(l), 0.0)) / lscale
        hs = hs_map.get(str(l))
        if hs is None:
            tau_map[l] = [2.0 + layer_mod] * n_head
        else:
            tau_map[l] = [2.0 + layer_mod + max(0.0, b) / hscale for b in hs]
    if cfg["schedule"] == "shuffled_field":
        rng = random.Random(P.SEED)
        keys = list(tau_map)
        shuffled_keys = keys[:]
        rng.shuffle(shuffled_keys)
        out = {}
        for k, sk in zip(keys, shuffled_keys):
            vals = tau_map[k][:]
            rng.shuffle(vals)
            out[sk] = vals
        return out
    return tau_map


def tau_for_step(cfg, k, K, rng, field):
    s = cfg["schedule"]
    if s == "global":
        return cfg["tau"]
    if s == "linear_up":
        return cfg["tau_start"] + (cfg["tau_end"] - cfg["tau_start"]) * k / max(K - 1, 1)
    if s == "linear_down":
        return cfg["tau_end"] - (cfg["tau_end"] - cfg["tau_start"]) * k / max(K - 1, 1)
    if s == "random":
        return rng.uniform(cfg["tau_start"], cfg["tau_end"])
    if s == "step":
        return cfg["tau_start"] if k < K // 2 else cfg["tau_end"]
    if s in ("field", "shuffled_field"):
        return field
    raise ValueError(s)


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
    n_layer, n_head = student.config.num_hidden_layers, student.config.num_attention_heads
    field = build_field(cfg, n_layer, n_head) if cfg["schedule"] in ("field", "shuffled_field") else None

    forget = C.load_jsonl(P.FORGET)
    retain = C.load_jsonl(P.RETAIN)
    random.Random(P.SEED).shuffle(forget)
    random.Random(P.SEED + 1).shuffle(retain)
    rng = random.Random(P.SEED)

    def eval_all():
        student.eval()
        f = C.answer_nll(student, C.make_batch(tok, forget[:64], "cuda"))[0]
        r = C.answer_nll(student, C.make_batch(tok, retain[:64], "cuda"))[0]
        student.train()
        return f, r

    f0, r0 = eval_all()

    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=P.LR, weight_decay=0.0)
    K = P.STEPS
    for step in range(K):
        fb = C.make_batch(tok, [forget[(step * 4 + i) % len(forget)] for i in range(4)], "cuda")
        rb = C.make_batch(tok, [retain[(step * 4 + i) % len(retain)] for i in range(4)], "cuda")
        t = tau_for_step(cfg, step, K, rng, field)
        with torch.no_grad(), C.TemperatureContext(ref, t):
            ref_logits = ref(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits
        stu_logits = student(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits
        kl = kl_on_answer(stu_logits, ref_logits, fb["labels"])
        rlogits = student(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"]).logits
        rlabels = rb["labels"][:, 1:]
        rm = rlabels != -100
        rloss = F.cross_entropy(
            rlogits[:, :-1, :].float().reshape(-1, rlogits.size(-1)),
            rlabels.reshape(-1), reduction="none",
        ).view(rlabels.shape)
        rloss = (rloss * rm).sum() / rm.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True)
        (P.ALPHA * kl + rloss).backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

    f1, r1 = eval_all()
    dF, dR = f1 - f0, r1 - r0
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    mu = cfg["mean_tau"]
    if cfg["kind"] == "anchor":
        P.write_anchor(mu, dF, dR, {"round": cfg["round"], "before": [f0, r0], "after": [f1, r1]})
        anchor = {"dF": dF, "dR": dR}
    else:
        ap = P.anchor_path(mu)
        if not ap.exists():
            print(f"ERROR: missing anchor for mean_tau={mu}", flush=True)
            sys.exit(3)
        anchor = P.load_json(ap)

    val_bpb = P.compute_regret(dF, dR, anchor, cfg["complexity"], cfg["kind"])
    P.dump_json(
        {
            "round": cfg["round"], "kind": cfg["kind"], "config": cfg,
            "before": {"forget": f0, "retain": r0},
            "after": {"forget": f1, "retain": r1},
            "delta": {"forget": dF, "retain": dR},
            "anchor": anchor, "val_bpb": val_bpb,
            "peak_vram_mb": peak_mb,
        },
        P.ROUNDS / f"{cfg['round']}.json",
    )

    print(f"val_bpb: {val_bpb:.6f}")
    print(f"peak_vram_mb: {peak_mb:.1f}")
    print(f"delta_forget: {dF:.4f}  delta_retain: {dR:.4f}  mean_tau: {mu}")


if __name__ == "__main__":
    main()

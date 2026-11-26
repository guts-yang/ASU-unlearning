"""Training variants D1 / D7 / D10 on the proxy v0.6.1 engine.

D1  : donor direction becomes a *forbidden* direction (hinge on cos(h-A+, dA)<=0)
D7  : progressive layer schedule driven by per-layer dA purity (answer alignment)
D10 : orthogonalize the L_F update against the retain-gradient top direction

All variants keep the non-donor target (T = A+), stop-gradient teacher, and a
donor-support hinge. Proxy caveats are the same as engine.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import (
    RETAIN,
    encode,
    gen_stats,
    load_jsonl,
    load_model,
    nll_ids,
    span_ids,
)
from pairs import build_pairs, extract_entity

LAYERS = [3, 7, 11, 15]
MAIN_LAYERS = [7, 11]
STEPS = 60
LR = 5e-6
R_F_MAX = 0.35
DONOR_MAX = 0.30


def _hidden(model, ids, layers, device="cuda"):
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(
        input_ids=inp,
        attention_mask=torch.ones_like(inp),
        output_hidden_states=True,
        use_cache=False,
    )
    return out


def layer_purity(teacher, tok, pairs, device="cuda"):
    """cos(dA_l, mean answer-token embedding) per layer -> purity ordering."""
    emb = teacher.get_input_embeddings().weight
    scores = {l: [] for l in LAYERS}
    with torch.no_grad():
        for pr in pairs:
            ip, sp = encode(tok, pr["plus"]["question"], pr["plus"]["answer"])
            im, sm = encode(tok, pr["minus"]["question"], pr["plus"]["answer"])
            op = _hidden(teacher, ip, LAYERS, device)
            om = _hidden(teacher, im, LAYERS, device)
            n = min(len(sp), len(sm))
            a_ids = span_ids(tok, pr["plus"]["answer"])[:n]
            ans_vec = emb[torch.tensor(a_ids, device=device)].mean(0)
            for l in LAYERS:
                dA = (op.hidden_states[l + 1][0][sp[:n]] - om.hidden_states[l + 1][0][sm[:n]]).mean(0)
                scores[l].append(float(F.cosine_similarity(dA, ans_vec, dim=0)))
    mean = {l: sum(v) / max(1, len(v)) for l, v in scores.items()}
    order = sorted(mean, key=mean.get, reverse=True)
    return mean, order


def layer_schedule(step, steps, order):
    """D7: add layers progressively in purity order."""
    frac = step / max(1, steps - 1)
    if frac < 0.34:
        return [order[0]]
    if frac < 0.67:
        return sorted(order[:2])
    return sorted(order[:3])


def train_variant(variant, seed=0, steps=STEPS, samples=16, target_mode="align"):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    teacher, tok = load_model()
    student, _ = load_model()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.train()

    pairs, pstats = build_pairs(n=samples, seed=seed)
    retain_pool = load_jsonl(RETAIN)
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.0)

    purity, order = layer_purity(teacher, tok, pairs)
    retain_dir = None
    if variant == "D10":
        # retain-gradient top direction proxy: mean (base - retain-donor) at layer 8
        with torch.no_grad():
            acc = []
            for r in retain_pool[:8]:
                ip, sp = encode(tok, r["question"], r["answer"])
                op = _hidden(teacher, ip, LAYERS)
                acc.append(op.hidden_states[9][0][sp].mean(0))
            retain_dir = torch.stack(acc).mean(0)
            retain_dir = F.normalize(retain_dir, dim=0)

    history = []
    for step in range(steps):
        pr = pairs[step % len(pairs)]
        f_rec, d_rec = pr["plus"], pr["minus"]
        ip, sp = encode(tok, f_rec["question"], f_rec["answer"])
        im, sm = encode(tok, d_rec["question"], f_rec["answer"])

        with torch.no_grad():
            tp = _hidden(teacher, ip, LAYERS)
            tm = _hidden(teacher, im, LAYERS)

        layers = layer_schedule(step, steps, order) if variant == "D7" else MAIN_LAYERS
        out = _hidden(student, ip, layers)
        loss_f = torch.zeros((), device="cuda")
        forbid = torch.zeros((), device="cuda")
        n = min(len(sp), len(sm))
        for l in layers:
            sh = out.hidden_states[l + 1][0][sp]
            th = tp.hidden_states[l + 1][0][sp].detach()
            loss_f = loss_f + F.mse_loss(sh, th)
            if variant == "D1":
                dA = (tp.hidden_states[l + 1][0][sp[:n]] - tm.hidden_states[l + 1][0][sm[:n]]).detach()
                dA = F.normalize(dA, dim=-1)
                cos = F.cosine_similarity(sh[:n] - th[:n], dA, dim=-1)
                forbid = forbid + F.relu(cos).pow(2).mean()
            if variant == "D10" and retain_dir is not None:
                cos = F.cosine_similarity(sh, retain_dir.expand_as(sh), dim=-1)
                forbid = forbid + F.relu(cos).pow(2).mean()

        logits = out.logits[0]
        r_f_t = torch.zeros((), device="cuda")
        cnt = 0
        for p in sp:
            if p == 0:
                continue
            r_f_t = r_f_t + (-torch.log_softmax(logits[p - 1], -1)[ip[p]])
            cnt += 1
        r_f_t = r_f_t / max(1, cnt)
        r_f = float(r_f_t)

        d_ids = span_ids(tok, d_rec["answer"])
        s_d = 0.0
        for i, p in enumerate(sp[: len(d_ids)]):
            if p == 0:
                continue
            s_d += float(torch.log_softmax(logits[p - 1], -1)[d_ids[i]])
        s_d = s_d / max(1, min(len(sp), len(d_ids)))

        r_rec = retain_pool[step % len(retain_pool)]
        r_ids, r_span = encode(tok, r_rec["question"], r_rec["answer"])
        r_out = _hidden(student, r_ids, MAIN_LAYERS)
        l_r = F.cross_entropy(
            r_out.logits[0][:-1].float(), torch.tensor(r_ids[1:], device="cuda")
        )

        extra = -1.0 * r_f_t if target_mode == "neutral" else 0.0
        loss = 0.1 * (loss_f + extra + 0.5 * forbid) + l_r
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

        if step % 15 == 0 or step == steps - 1:
            history.append(
                {
                    "step": step,
                    "layers": layers,
                    "loss": round(float(loss), 4),
                    "align": round(float(loss_f), 4),
                    "forbid": round(float(forbid), 4),
                    "r_f": round(r_f, 4),
                    "s_d": round(s_d, 4),
                }
            )

    # post-training evaluation on held-out-style samples
    eval_pairs, _ = build_pairs(n=16, seed=100 + seed)
    def dF_dR(model):
        fs, rs = [], []
        for pr in eval_pairs:
            ids, span = encode(tok, pr["plus"]["question"], pr["plus"]["answer"])
            fs.append(nll_ids(model, ids, span))
        for rec in retain_pool[:16]:
            ids, span = encode(tok, rec["question"], rec["answer"])
            rs.append(nll_ids(model, ids, span))
        return sum(fs) / len(fs), sum(rs) / len(rs)

    f1, r1 = dF_dR(student)
    teacher.eval()
    f0, r0 = dF_dR(teacher)
    behavior = gen_stats(student, tok, [p["plus"] for p in eval_pairs], samples=12, max_new_tokens=64)
    return {
        "variant": variant,
        "seed": seed,
        "target_mode": target_mode,
        "steps": steps,
        "layers": MAIN_LAYERS,
        "before": {"forget": round(f0, 4), "retain": round(r0, 4)},
        "after": {"forget": round(f1, 4), "retain": round(r1, 4)},
        "delta": {"forget": round(f1 - f0, 4), "retain": round(r1 - r0, 4)},
        "layer_purity": {str(k): round(v, 4) for k, v in purity.items()},
        "purity_order": order,
        "history": history,
        "behavior": behavior,
        "pair_stats": pstats,
    }

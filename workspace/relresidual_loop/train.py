"""Universal idea interpreter v6 (only editable code file).

One invocation = one INDEPENDENT data-order replication of a RelResidual variant.
Targets are precomputed from a FIXED counterfactual pairing (seed 0) and cached;
only the training-sample order varies with --data-order.
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
import common as C  # noqa: E402 (upstream, read-only)

IDEA = P.LOOP / "current_idea.json"
CACHE = P.LOOP / "targets"


def log_soft(x):
    return F.log_softmax(x.float(), dim=-1)


def kl_on_mask(s_logits, t_logits, mask):
    s = log_soft(s_logits[:, :-1, :])
    t = log_soft(t_logits[:, :-1, :])
    m = mask[:, 1:]
    loss = F.kl_div(s, t, reduction="none", log_target=True).sum(-1)
    return (loss * m).sum() / m.sum().clamp_min(1)


def nll_on_mask(logits, mask, labels):
    ls = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
                         labels[:, 1:].reshape(-1), reduction="none").view(labels[:, 1:].shape)
    m = mask[:, 1:] & (labels[:, 1:] != -100)
    return (ls * m).sum() / m.sum().clamp_min(1)


def build_cache(tok, model_teacher, quads):
    """Precompute h(s_f,r), B_l and counterfactual teacher logits (fixed pairing)."""
    CACHE.mkdir(exist_ok=True)
    P.dump_json(quads, CACHE / "quads.json")
    q_fr = [q["q_fr"] for q in quads]
    q_rr = [q["q_rr"] for q in quads]
    q_fr0 = [q["q_fr0"] for q in quads]
    q_rr0 = [q["q_rr0"] for q in quads]

    for l in P.TARGET_LAYERS:
        hp = P.hidden_last(model_teacher, tok, q_fr, [l])[l]
        hm = P.hidden_last(model_teacher, tok, q_rr, [l])[l]
        hp0 = P.hidden_last(model_teacher, tok, q_fr0, [l])[l]
        hm0 = P.hidden_last(model_teacher, tok, q_rr0, [l])[l]
        torch.save({"h_fr": hp.cpu(), "B": (hp - hm - hp0 + hm0).cpu()},
                   CACHE / f"layer{l}.pt")

    # shared output prefix: prompt(X-) + answer(X+); token-equal to prompt(X+)+answer
    seqs, masks, labels = [], [], []
    for q in quads:
        p_ids = tok.encode(P.prompt_of(q["q_rr"]), add_special_tokens=False)
        a_ids = tok.encode(" " + q["answer_plus"].strip(), add_special_tokens=False) + [tok.eos_token_id]
        ids = p_ids + a_ids
        seqs.append(ids)
        masks.append([0] * len(p_ids) + [1] * len(a_ids))
        labels.append([-100] * len(p_ids) + a_ids)
    W = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), W), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), W), dtype=torch.long)
    ans_mask = torch.zeros((len(seqs), W), dtype=torch.bool)
    lab = torch.full((len(seqs), W), -100, dtype=torch.long)
    for i, (s, m, y) in enumerate(zip(seqs, masks, labels)):
        input_ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
        ans_mask[i, :len(m)] = torch.tensor(m, dtype=torch.bool)
        lab[i, :len(y)] = torch.tensor(y)

    chunks = []
    for i in range(0, len(seqs), 8):
        b = {"input_ids": input_ids[i:i+8].to("cuda"),
             "attention_mask": attn[i:i+8].to("cuda")}
        with torch.no_grad():
            out = model_teacher(**b).logits
        chunks.append(out.half().cpu())
    torch.save({"logits": torch.cat(chunks, 0), "input_ids": input_ids,
                "attention_mask": attn, "ans_mask": ans_mask, "labels": lab},
               CACHE / "teacher.pt")
    # student-side ids (prompt X+ + same answer); assert token equality
    plus_ids = []
    for q in quads:
        p_ids = tok.encode(P.prompt_of(q["q_fr"]), add_special_tokens=False)
        a_ids = tok.encode(" " + q["answer_plus"].strip(), add_special_tokens=False) + [tok.eos_token_id]
        plus_ids.append(p_ids + a_ids)
    Wp = max(len(s) for s in plus_ids)
    pi = torch.full((len(plus_ids), Wp), tok.pad_token_id, dtype=torch.long)
    pm = torch.zeros((len(plus_ids), Wp), dtype=torch.long)
    pl = torch.full((len(plus_ids), Wp), -100, dtype=torch.long)
    for i, s in enumerate(plus_ids):
        p_len = len(tok.encode(P.prompt_of(quads[i]["q_fr"]), add_special_tokens=False))
        pi[i, :len(s)] = torch.tensor(s)
        pm[i, :len(s)] = 1
        pl[i, p_len:len(s)] = torch.tensor(s[p_len:len(s)])
    torch.save({"input_ids": pi, "attention_mask": pm, "labels": pl,
                "prompt_lens": [len(tok.encode(P.prompt_of(q["q_fr"]), add_special_tokens=False)) for q in quads]},
               CACHE / "student.pt")
    assert all(len(a) == len(b) for a, b in zip(plus_ids, seqs)), "counterfactual token length mismatch"
    print(f"[cache] built: {len(quads)} quads, layers {P.TARGET_LAYERS}", flush=True)


def eval_nll(model, tok, records, chunk=8):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(records), chunk):
            b = C.make_batch(tok, records[i:i+chunk], "cuda")
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            labels = b["labels"][:, 1:]
            m = labels != -100
            loss = F.cross_entropy(out.logits[:, :-1, :].float().reshape(-1, out.logits.size(-1)),
                                   labels.reshape(-1), reduction="none").view(labels.shape)
            total += (loss * m).sum().item()
            count += m.sum().item()
    return total / max(count, 1)


def generate_texts(model, tok, quads, n):
    model.eval()
    texts = []
    with torch.no_grad():
        for q in quads[:n]:
            ids = tok.encode(P.prompt_of(q["q_fr"]), add_special_tokens=False)
            inp = torch.tensor([ids], device="cuda")
            out = model.generate(inp, attention_mask=torch.ones_like(inp),
                                 max_new_tokens=P.GEN_MAX_NEW_TOKENS, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            texts.append(tok.decode(out[0][len(ids):], skip_special_tokens=True))
    model.train()
    return texts


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
    prm = spec["params"]
    for k in ("layer_select", "alpha_B", "kl_weight", "lf_form", "lf_weight"):
        if k not in prm:
            raise ValueError(f"missing lever param {k}")
    if prm["layer_select"] not in ("cos_high", "all", "cos_weighted"):
        raise ValueError(prm["layer_select"])
    if prm["lf_form"] not in ("mse", "cosine"):
        raise ValueError(prm["lf_form"])

    torch.manual_seed(P.INIT_SEED)
    random.seed(args.data_order)
    torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(P.MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    student = AutoModelForCausalLM.from_pretrained(P.MODEL, torch_dtype=torch.float32).to("cuda")
    teacher = AutoModelForCausalLM.from_pretrained(P.MODEL, torch_dtype=torch.float32).to("cuda").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    quads = P.build_quadruples(tok, n_pairs=P.PAIRS, seed=P.INIT_SEED)
    if not (CACHE / "teacher.pt").exists() or not (CACHE / "quads.json").exists():
        build_cache(tok, teacher, quads)
    cached = torch.load(CACHE / "teacher.pt", map_location="cpu", weights_only=False)
    stu = torch.load(CACHE / "student.pt", map_location="cpu", weights_only=False)
    quads = P.load_json(CACHE / "quads.json")

    g0 = P.load_json(P.LOOP / "gate0.json") if (P.LOOP / "gate0.json").exists() else {}
    per = ((g0.get("layers") or {}).get("query_last") or {}).get("per_layer", {})
    cos_map = {int(k): float(v.get("cos_r_r0", 0.0)) for k, v in per.items()}
    if prm["layer_select"] == "cos_high":
        layers = [l for l in P.TARGET_LAYERS if cos_map.get(l, 0.0) >= 0.30] or [15]
    else:
        layers = list(P.TARGET_LAYERS)
    weights = {l: (max(0.0, cos_map.get(l, 0.0)) if prm["layer_select"] == "cos_weighted" else 1.0)
               for l in layers}
    if not any(weights.values()):
        weights = {l: 1.0 for l in layers}

    layer_data = {}
    for l in layers:
        d = torch.load(CACHE / f"layer{l}.pt", map_location="cpu", weights_only=False)
        layer_data[l] = {"T": (d["h_fr"] - prm["alpha_B"] * d["B"]).to("cuda")}

    forget = C.load_jsonl(P.FORGET)
    retain = C.load_jsonl(P.RETAIN)
    random.Random(args.data_order).shuffle(forget)
    random.Random(args.data_order + 1).shuffle(retain)
    f0, r0 = eval_nll(student, tok, forget[:64]), eval_nll(student, tok, retain[:64])

    student.train()
    opt = torch.optim.AdamW(student.parameters(), lr=P.LR, weight_decay=0.0)
    rng = random.Random(args.data_order + 100)
    nq = len(quads)
    for step in range(P.STEPS):
        idx = [rng.randrange(nq) for _ in range(4)]
        b = {"input_ids": stu["input_ids"][idx].to("cuda"),
             "attention_mask": stu["attention_mask"][idx].to("cuda"),
             "labels": stu["labels"][idx].to("cuda")}
        plens = [stu["prompt_lens"][i] for i in idx]
        out = student(**{k: v for k, v in b.items() if k != "labels"}, output_hidden_states=True)
        logits = out.logits

        # L_F: query-last activation vs T_l
        lf = 0.0
        for l in layers:
            h = out.hidden_states[l + 1]
            pos = torch.tensor([p - 1 for p in plens], device=h.device)
            hs = h[torch.arange(len(idx), device=h.device), pos].float()
            tgt = layer_data[l]["T"][idx].float()
            if prm["lf_form"] == "mse":
                term = ((hs - tgt) ** 2).mean(-1).mean()
            else:
                term = (1.0 - F.cosine_similarity(hs, tgt, dim=-1)).mean()
            lf = lf + weights[l] * term
        lf = lf / max(sum(weights.values()), 1e-9)

        # L_KL: counterfactual teacher distribution, shared output prefix
        am = cached["ans_mask"][idx].to("cuda")
        tl = cached["logits"][idx].to("cuda")
        T = min(tl.shape[1], logits.shape[1])
        kl = kl_on_mask(logits[:, :T, :], tl[:, :T, :].float(), am[:, :T])

        # L_R: retain NLL
        rb = C.make_batch(tok, [retain[(step * 4 + i) % len(retain)] for i in range(4)], "cuda")
        rlogits = student(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"]).logits
        rlabels = rb["labels"][:, 1:]
        rm = rlabels != -100
        rloss = F.cross_entropy(rlogits[:, :-1, :].float().reshape(-1, rlogits.size(-1)),
                                rlabels.reshape(-1), reduction="none").view(rlabels.shape)
        rloss = (rloss * rm).sum() / rm.sum().clamp_min(1)

        loss = prm["lf_weight"] * lf + prm["kl_weight"] * kl + rloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()

    student.eval()
    f1, r1 = eval_nll(student, tok, forget[:64]), eval_nll(student, tok, retain[:64])
    dF, dR = f1 - f0, r1 - r0
    texts = generate_texts(student, tok, quads, min(P.GEN_SAMPLES, len(quads)))
    gen = P.evaluate_generation(texts, quads)

    # P3 recovery (diagnostic, not part of the frozen gate)
    student.train()
    p3opt = torch.optim.SGD(student.parameters(), lr=1e-5)
    for k in range(10):
        b = C.make_batch(tok, [forget[(k * 4 + i) % 32] for i in range(4)], "cuda")
        lg = student(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        y = b["labels"][:, 1:]
        m = y != -100
        ls = F.cross_entropy(lg[:, :-1, :].float().reshape(-1, lg.size(-1)),
                             y.reshape(-1), reduction="none").view(y.shape)
        p3opt.zero_grad(set_to_none=True)
        (ls * m).sum().div(m.sum().clamp_min(1)).backward()
        p3opt.step()
    student.eval()
    f3 = eval_nll(student, tok, forget[:64])
    base_f = f0
    # rho is undefined when the model barely forgot anything (denominator ~0)
    rho = max(0.0, (f1 - f3) / (f1 - base_f)) if (f1 - base_f) >= 0.05 else None

    single = {"dF": dF, "dR": dR, "donor_leak": gen["donor_leak"],
              "premature": gen["premature"], "truth_leak": gen["truth_leak"]}
    agg = P.aggregate([single], spec["complexity"])
    result = {"data_order": args.data_order, "idea_id": spec["id"], "lever": spec["lever"],
              "params": prm, "layers": layers, "layer_weights": weights,
              "init_seed": P.INIT_SEED, "before": {"forget": f0, "retain": r0},
              "after": {"forget": f1, "retain": r1},
              "dF": dF, "dR": dR, "generation": gen,
              "donor_leak": gen["donor_leak"], "premature": gen["premature"],
              "truth_leak": gen["truth_leak"], "mean_words": gen["mean_words"],
              "rho_p3": rho, "sample_outputs": texts[:3],
              "val_bpb": agg["val_bpb"],
              "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 ** 2)}
    P.dump_json(result, args.out)

    print(f"val_bpb: {agg['val_bpb']:.6f}")
    print(f"dF: {dF:.4f}  dR: {dR:.4f}")
    print(f"donor_leak: {gen['donor_leak']:.4f}  premature: {gen['premature']:.4f}  truth_leak: {gen['truth_leak']:.4f}")
    print(f"rho_p3: {rho if rho is None else round(rho,4)}  peak_vram_mb: {result['peak_vram_mb']:.1f}")


if __name__ == "__main__":
    main()

"""Self-contained data/eval helpers for the CF-04 depth audit.

Ported verbatim (read-only reference) from ``workspace/auto_experiments/common.py``.
No algorithm or evaluation change: the token-weighted ``eval_nll`` and the
``answer_nll`` batch semantics are kept exactly as in the historical audit.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

DATA = Path("/jizhicfs/chenyliao/data/datasets/locuslab-TOFU")
MODEL = "/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
FORGET = str(DATA / "forget10.json")
RETAIN = str(DATA / "retain90.json")
HOLDOUT = str(DATA / "holdout10.json")
CHECKPOINTS = Path("/jizhicfs/chenyliao/llm-unlearning/checkpoints")
CF04_V2_FORMAL = str(CHECKPOINTS / "full_cf04_v2_formal")
GRAD_ASCENT = "/jizhicfs/chenyliao/runs/open-unlearning/20261122_forget10_1b/GradAscent"
ASU_FORGET05 = str(CHECKPOINTS / "asu-forget05-asu-gd")


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {p}", flush=True)


def set_seed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(question, answer=None, fewshot_prefix=None):
    prefix = fewshot_prefix or ""
    s = f"{prefix}Question: {question}\nAnswer:"
    if answer is not None:
        s += f" {answer}"
    return s


def make_batch(tok, records, device="cuda", max_len=512, fewshot_prefix=None, with_labels=True):
    seqs = []
    for rec in records:
        q, a = rec["question"], rec["answer"]
        prompt = build_prompt(q, None, fewshot_prefix)
        p_ids = tok.encode(prompt, add_special_tokens=False)
        a_ids = tok.encode(" " + a.strip(), add_special_tokens=False) + [tok.eos_token_id]
        ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len] if with_labels else None
        seqs.append((ids, labels))
    width = max(len(x[0]) for x in seqs)
    input_ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(seqs), width), -100, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, (ids, lab) in enumerate(seqs):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        mask[i, : len(ids)] = 1
        if lab is not None:
            labels[i, : len(lab)] = torch.tensor(lab)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": mask.to(device),
        "labels": labels.to(device),
    }


def answer_nll(model, batch, reduction="mean", temp_field=None):
    """Compute NLL on answer tokens. ``temp_field`` is not used by this audit."""
    if temp_field is not None:
        raise ValueError("temperature fields are outside this audit's scope")
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    m = labels != -100
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view(labels.shape)
    nll = (loss * m).sum(-1)
    if reduction == "mean":
        denom = m.sum(-1).clamp_min(1)
        return (nll / denom).mean().item(), nll.detach(), m
    return nll, m


def eval_nll(model, tok, records, device="cuda", batch_size=8, temp_field=None, fewshot_prefix=None):
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            chunk = records[i : i + batch_size]
            batch = make_batch(tok, chunk, device, fewshot_prefix=fewshot_prefix)
            val, _, _ = answer_nll(model, batch, temp_field=temp_field)
            n = sum(1 for r in chunk for _ in tok.encode(" " + r["answer"].strip(), add_special_tokens=False)) + len(chunk)
            total += val * n
            count += n
    return total / max(count, 1)

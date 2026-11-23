"""Shared utilities for automated idea validation (all code lives under workspace/)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA = Path("/jizhicfs/chenyliao/data/datasets/locuslab-TOFU")
MODEL = "/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
FORGET = str(DATA / "forget10.json")
RETAIN = str(DATA / "retain90.json")
HOLDOUT = str(DATA / "holdout10.json")
RESULTS = ROOT / "results"
CKPT = ROOT / "checkpoints"

FUNCTION_WORDS = set(
    """a an the and or but if then than that this these those of in on at to for from with by as is are was were be been being it its he she they them his her their we you i my your our do does did done have has had will would can could should may might must not no nor so such very more most some any all each every which who whom whose what when where why how there here about into over under again further once only just also""".split()
)
PUNCT = "".join(set(".,;:!?\"'()[]{}<>/-–—…\n\t "))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {p}", flush=True)


def load_model(path=MODEL, device="cuda", dtype=torch.bfloat16, attn="sdpa"):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation=attn)
    model.to(device).eval()
    return model, tok


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
    """Compute NLL on answer tokens. temp_field: (tau_per_layer[H] or dict layer->[H])."""
    ctx = None
    if temp_field is not None:
        ctx = TemperatureContext(model, temp_field)
        ctx.__enter__()
    try:
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
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)


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


def factual_function_mask(tok, answer):
    """Heuristic token-level factual/function split using offset mapping.

    ASU Appendix N uses GPT-4o annotation; we use a deterministic stopword/punctuation
    proxy and disclose this limitation in the report.
    """
    enc = tok(" " + answer.strip(), add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    text = " " + answer.strip()
    flags = []
    for (a, b) in offsets:
        word = text[a:b].strip().lower().strip(PUNCT)
        flags.append(0 if (not word or word in FUNCTION_WORDS or word.isdigit()) else 1)
    return torch.tensor(flags, dtype=torch.bool), ids


class TemperatureContext:
    """Apply attention temperature by scaling q_proj outputs (equivalent to logits/tau)."""

    def __init__(self, model, field):
        self.model = model
        self.field = field
        self.handles = []

    def __enter__(self):
        n_layer = self.model.config.num_hidden_layers
        n_head = self.model.config.num_attention_heads
        head_dim = getattr(self.model.config, "head_dim", self.model.config.hidden_size // n_head)
        for l in range(n_layer):
            if isinstance(self.field, dict):
                if l not in self.field:
                    continue
                vals = self.field[l]
            else:
                vals = self.field
            if isinstance(vals, (int, float)):
                vals = [float(vals)] * n_head
            tau = torch.tensor([float(v) for v in vals], device=self.model.device, dtype=torch.float32)
            if (tau <= 1.0001).all():
                continue
            inv = (1.0 / tau).to(torch.bfloat16 if self.model.dtype == torch.bfloat16 else torch.float32)

            def hook(module, inp, out, inv=inv, H=n_head, D=head_dim):
                shape = out.shape
                return out.view(shape[0], shape[1], H, D) * inv.view(1, 1, H, 1).to(out.dtype)

            attn = self.model.model.layers[l].self_attn
            self.handles.append(attn.q_proj.register_forward_hook(hook))
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles = []


def get_attn_modules(model):
    return [layer.self_attn for layer in model.model.layers]


def token_span_factual(tok, rec, prefix=""):
    """Return token indices of factual answer tokens inside a full prompt."""
    prompt = build_prompt(rec["question"], None, prefix)
    p_len = len(tok.encode(prompt, add_special_tokens=False))
    flags, _ = factual_function_mask(tok, rec["answer"])
    return [p_len + i for i, f in enumerate(flags.tolist()) if f]


def counterfactual_pairs(tok, forget_path=FORGET, retain_path=RETAIN, max_pairs=64, seed=0):
    import sys

    sys.path.insert(0, str(ROOT.parent / "experiments" / "direction_i"))
    from counterfactual import build_pairs

    return build_pairs(tok, forget_path, retain_path, max_pairs=max_pairs, seed=seed)


def set_seed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

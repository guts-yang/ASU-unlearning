"""Self-contained model I/O for one CF2 method. No cross-method imports."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self._offset = 3

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self._offset + (sum((i + 1) * ord(c) for i, c in enumerate(w)) % (self.vocab_size - self._offset)) for w in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(f"tok{int(i)}" for i in ids if int(i) > self.eos_token_id)

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tiny_tokenizer.json").write_text(json.dumps({"vocab_size": self.vocab_size}))


class TinyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, width * 2)
        self.down = nn.Linear(width * 2, width)

    def forward(self, x):
        return x + self.down(torch.tanh(self.up(self.norm(x))))


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=256, width=32, layers=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width, num_hidden_layers=layers, vocab_size=vocab_size)
        self.embed = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList([TinyBlock(width) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.model = SimpleNamespace(layers=self.blocks)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False):
        del use_cache
        h = self.embed(input_ids)
        positions = torch.arange(1, h.shape[1] + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
        causal_mean = torch.cumsum(h, dim=1) / positions
        h = h + 0.1 * (causal_mean - h)
        states = [h]
        for block in self.blocks:
            h = block(h)
            states.append(h)
        logits = self.lm_head(self.norm(h))
        if attention_mask is not None:
            logits = logits * attention_mask.unsqueeze(-1).to(logits.dtype)
        return SimpleNamespace(logits=logits, hidden_states=tuple(states) if output_hidden_states else None)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=16, do_sample=False, pad_token_id=0, eos_token_id=2, **kwargs):
        del do_sample, kwargs
        ids = input_ids
        mask = attention_mask
        for _ in range(max_new_tokens):
            out = self.forward(ids, mask)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if mask is not None:
                mask = torch.cat([mask, torch.ones_like(nxt)], dim=1)
            if bool((nxt == eos_token_id).all()):
                break
        return ids


def create_model_and_tokenizer(cfg, device, student=False):
    backend = cfg["model"].get("backend", "hf")
    if backend == "tiny":
        mcfg = cfg["model"]
        model = TinyCausalLM(int(mcfg.get("tiny_vocab", 256)), int(mcfg.get("tiny_width", 32)), int(mcfg.get("tiny_layers", 4)))
        tok = TinyTokenizer(int(mcfg.get("tiny_vocab", 256)))
        model.to(device)
        return model, tok
    if backend != "hf":
        raise ValueError(f"unknown model backend: {backend}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = cfg["model"]["path"]
    dtype_name = cfg["model"].get("dtype", "bfloat16")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    tok = AutoTokenizer.from_pretrained(path, local_files_only=cfg["model"].get("local_files_only", True))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        attn_implementation=cfg["model"].get("attn_implementation", "sdpa"),
        local_files_only=cfg["model"].get("local_files_only", True),
    )
    if student and cfg["model"].get("peft", False):
        from peft import LoraConfig, get_peft_model

        pcfg = cfg["model"]
        model = get_peft_model(model, LoraConfig(
            r=int(pcfg.get("lora_rank", 16)), lora_alpha=int(pcfg.get("lora_alpha", 32)),
            lora_dropout=float(pcfg.get("lora_dropout", 0.0)), bias="none",
            task_type="CAUSAL_LM", target_modules=pcfg.get("lora_targets", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ))
    if cfg["model"].get("gradient_checkpointing", False) and student:
        # Non-reentrant checkpointing keeps LoRA gradients alive without a
        # requires-grad hook on the embedding output (which breaks mixed-dtype
        # backward and inference-time generation).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    model.to(device)
    return model, tok


def encode_qa(tok, question, answer, max_len=512):
    prompt = f"Question: {question}\nAnswer:"
    p_ids = tok.encode(prompt, add_special_tokens=False)
    a_ids = tok.encode(" " + answer.strip(), add_special_tokens=False) + [tok.eos_token_id]
    ids = (p_ids + a_ids)[:max_len]
    if len(ids) <= len(p_ids):
        raise ValueError("max_len truncated every answer token")
    labels = ([-100] * len(p_ids) + a_ids)[:max_len]
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(ids), dtype=torch.long),
        "answer_start": len(p_ids), "answer_end": len(ids), "prompt_length": len(p_ids),
    }


def move_encoded(enc, device):
    out = dict(enc)
    for key in ("input_ids", "labels", "attention_mask"):
        out[key] = enc[key].to(device)
    return out


def answer_nll(logits, labels, answer_start=None):
    shift_logits = logits[..., :-1, :].float()
    shift_labels = labels[..., 1:]
    mask = shift_labels != -100
    losses = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), reduction="none", ignore_index=-100).view(shift_labels.shape)
    if answer_start is not None:
        local = torch.zeros_like(mask)
        begin = max(0, answer_start - 1)
        local[..., begin:] = mask[..., begin:]
        mask = local
    return (losses * mask).sum() / mask.sum().clamp_min(1)


def forward_model(model, enc, device, layers=None, grad=False):
    enc = move_encoded(enc, device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        out = model(input_ids=enc["input_ids"].unsqueeze(0), attention_mask=enc["attention_mask"].unsqueeze(0), output_hidden_states=True, use_cache=False)
        hidden = {i: h[0] for i, h in enumerate(out.hidden_states) if layers is None or i in layers}
        nll = answer_nll(out.logits[0], enc["labels"])
    return {"hidden": hidden, "logits": out.logits[0], "nll": nll, "enc": enc}


def span_hidden(forward, layer, start=None, end=None):
    h = forward["hidden"][layer]
    if start is None:
        start, end = forward["enc"]["answer_start"], forward["enc"]["answer_end"]
    # Teacher targets are fp32; keep student activations fp32 so mixed-dtype
    # mse_loss backward does not fail with "Found dtype Float but expected BFloat16".
    return h[start:end].float()


@torch.no_grad()
def evaluate_ground_truth(model, tok, records, device, max_len=512, max_samples=32):
    rows = []
    for rec in list(records)[:max_samples]:
        enc = encode_qa(tok, rec["question"], rec["answer"], max_len)
        out = forward_model(model, enc, device)
        rows.append(float(out["nll"]))
    return {"nll": sum(rows) / max(1, len(rows)), "count": len(rows), "ground_truth": "dataset_answer"}


@torch.no_grad()
def generation_behaviour(model, tok, records, device, max_new_tokens=24, max_samples=8):
    stats = []
    for rec in list(records)[:max_samples]:
        prompt = f"Question: {rec['question']}\nAnswer:"
        ids = tok.encode(prompt, add_special_tokens=False)
        # HF generate() forces num_logits_to_keep=1; that bf16 lm_head GEMM aborts
        # with SIGFPE on this torch/CUDA build. Explicit greedy decoding over the
        # full-sequence logits keeps the same decoding and metrics.
        gen_ids = []
        for _ in range(max_new_tokens):
            inp = torch.tensor(ids + gen_ids, dtype=torch.long, device=device).unsqueeze(0)
            mask = torch.ones_like(inp)
            out = model(input_ids=inp, attention_mask=mask, use_cache=False)
            nxt = int(out.logits[0, -1].float().argmax().item())
            gen_ids.append(nxt)
            if nxt == tok.eos_token_id:
                break
        gen = torch.tensor(gen_ids, dtype=torch.long)
        text = tok.decode(gen.tolist(), skip_special_tokens=True)
        target = set(re.findall(r"[a-z0-9']+", rec["answer"].lower()))
        generated = set(re.findall(r"[a-z0-9']+", text.lower()))
        stats.append({
            "tokens": int(gen.numel()), "eos": bool((gen == tok.eos_token_id).any()),
            "overlap": len(target & generated) / max(1, len(target)), "text": text[:160],
        })
    n = max(1, len(stats))
    return {"samples": len(stats), "mean_tokens": sum(x["tokens"] for x in stats) / n, "eos_rate": sum(x["eos"] for x in stats) / n, "target_overlap": sum(x["overlap"] for x in stats) / n, "examples": [x["text"] for x in stats[:2]]}


@torch.no_grad()
def patch_forward_nll_multi(model, enc, device, replacements):
    modules = model.model.layers
    enc = move_encoded(enc, device)
    handles = []
    for layer, replacement in replacements.items():
        if layer < 0 or layer >= len(modules):
            raise IndexError(f"decoder layer {layer} out of range")
        replacement = replacement.to(device)

        def hook(_module, _args, output, new_hidden=replacement):
            old = output[0] if isinstance(output, tuple) else output
            new = new_hidden.unsqueeze(0).to(dtype=old.dtype, device=old.device)
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

        handles.append(modules[layer].register_forward_hook(hook))
    try:
        out = model(input_ids=enc["input_ids"].unsqueeze(0), attention_mask=enc["attention_mask"].unsqueeze(0), use_cache=False)
        return answer_nll(out.logits[0], enc["labels"])
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def patch_forward_nll(model, enc, device, layer, replacement):
    return patch_forward_nll_multi(model, enc, device, {layer: replacement})


def save_trained(model, tok, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path, safe_serialization=True)
    else:
        torch.save(model.state_dict(), path / "tiny_model.pt")
    if hasattr(tok, "save_pretrained"):
        tok.save_pretrained(path)

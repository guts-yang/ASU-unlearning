"""Self-contained model I/O for one CF3 method. No cross-method imports."""
from __future__ import annotations

import json
import random
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


class TinySelfAttention(nn.Module):
    def __init__(self, width, heads=4):
        super().__init__()
        self.num_heads = min(heads, width)
        while width % self.num_heads:
            self.num_heads -= 1
        self.head_dim = width // self.num_heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, x, output_attentions=False):
        bsz, length, width = x.shape
        shape = (bsz, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / (self.head_dim ** 0.5)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal, -torch.inf)
        probs = F.softmax(scores.float(), dim=-1).to(v.dtype)
        attended = (probs @ v).transpose(1, 2).reshape(bsz, length, width)
        return self.o_proj(attended), probs if output_attentions else None


class TinyMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate_proj = nn.Linear(width, width * 2, bias=False)
        self.up_proj = nn.Linear(width, width * 2, bias=False)
        self.down_proj = nn.Linear(width * 2, width, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(width)
        self.post_attention_layernorm = nn.LayerNorm(width)
        self.self_attn = TinySelfAttention(width)
        self.mlp = TinyMLP(width)

    def forward(self, x, output_attentions=False):
        attention, probs = self.self_attn(self.input_layernorm(x), output_attentions)
        x = x + attention
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, probs


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=256, width=32, layers=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width, num_hidden_layers=layers, vocab_size=vocab_size, model_type="tiny")
        self.embed = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList([TinyBlock(width) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.model = SimpleNamespace(layers=self.blocks)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, output_hidden_states=False, output_attentions=False, use_cache=False):
        del use_cache
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds required")
            h = self.embed(input_ids)
        else:
            h = inputs_embeds
        states = [h] if output_hidden_states else None
        attentions = [] if output_attentions else None
        for block in self.blocks:
            h, probs = block(h, output_attentions)
            if output_hidden_states:
                states.append(h)
            if output_attentions:
                attentions.append(probs)
        logits = self.lm_head(self.norm(h))
        if attention_mask is not None:
            logits = logits * attention_mask.unsqueeze(-1).to(logits.dtype)
        return SimpleNamespace(logits=logits, hidden_states=tuple(states) if output_hidden_states else None, attentions=tuple(attentions) if output_attentions else None)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=16, do_sample=False, pad_token_id=0, eos_token_id=2, **kwargs):
        del do_sample, kwargs
        ids = input_ids
        mask = attention_mask
        for _ in range(max_new_tokens):
            out = self.forward(input_ids=ids, attention_mask=mask)
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if mask is not None:
                mask = torch.cat([mask, torch.ones_like(nxt)], dim=1)
            if bool((nxt == eos_token_id).all()):
                break
        return ids


def decoder_layers(model):
    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(core, "model") and hasattr(core.model, "layers"):
        return core.model.layers
    if hasattr(core, "layers"):
        return core.layers
    raise TypeError("model does not expose decoder layers")


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
    if cfg["model"].get("model_type") == "llama" and model.config.model_type != "llama":
        raise TypeError(f"CF3 low-level capture expects Llama, got {model.config.model_type}")
    if student and cfg["model"].get("peft", False):
        from peft import LoraConfig, get_peft_model

        pcfg = cfg["model"]
        model = get_peft_model(model, LoraConfig(
            r=int(pcfg.get("lora_rank", 16)), lora_alpha=int(pcfg.get("lora_alpha", 32)),
            lora_dropout=float(pcfg.get("lora_dropout", 0.0)), bias="none",
            task_type="CAUSAL_LM", target_modules=pcfg.get("lora_targets", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        ))
    if cfg["model"].get("gradient_checkpointing", False) and student:
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


def encode_prefix(tok, question, prefix_ids, max_len=512):
    prompt = f"Question: {question}\nAnswer:"
    p_ids = tok.encode(prompt, add_special_tokens=False)
    ids = (p_ids + list(prefix_ids))[:max_len]
    if len(ids) <= len(p_ids):
        raise ValueError("prefix state needs at least one generated token")
    labels = [-100] * len(ids)
    return {"input_ids": torch.tensor(ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long), "answer_start": len(p_ids),
            "answer_end": len(ids), "prompt_length": len(p_ids)}


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
        local[..., max(0, answer_start - 1):] = mask[..., max(0, answer_start - 1):]
        mask = local
    return (losses * mask).sum() / mask.sum().clamp_min(1)


def forward_model(model, enc, device, layers=None, grad=False, capture_bilinear=False, capture_glu=False):
    enc = move_encoded(enc, device)
    modules = decoder_layers(model)
    values, glu, block_hidden = {}, {}, {}
    handles = []
    # HF's hidden_states[-1] is the FINAL-NORMED state, not decoder layer L's
    # output.  Patch-based Stage A must use the real block output for every
    # layer (especially layer 16), so capture decoder-layer outputs directly.
    for index, layer in enumerate(modules):
        def block_hook(_module, _args, output, i=index):
            block_hidden[i + 1] = output[0] if isinstance(output, tuple) else output
        handles.append(layer.register_forward_hook(block_hook))
    if capture_bilinear:
        for index, layer in enumerate(modules):
            def value_hook(_module, _args, output, i=index):
                values[i] = output
            handles.append(layer.self_attn.v_proj.register_forward_hook(value_hook))
    if capture_glu:
        for index, layer in enumerate(modules):
            for name in ("gate_proj", "up_proj"):
                def glu_hook(_module, _args, output, i=index, key=name):
                    glu[(i, key)] = output
                handles.append(getattr(layer.mlp, name).register_forward_hook(glu_hook))
    ctx = torch.enable_grad() if grad else torch.no_grad()
    try:
        with ctx:
            out = model(input_ids=enc["input_ids"].unsqueeze(0), attention_mask=enc["attention_mask"].unsqueeze(0),
                        output_hidden_states=True, output_attentions=capture_bilinear, use_cache=False)
            # Student activations are bf16 while teacher targets are fp32; keep the
            # shared interface fp32 so mixed-dtype mse_loss backward does not fail.
            hidden = {}
            for i, h in enumerate(out.hidden_states):
                if layers is not None and i not in layers:
                    continue
                if i in block_hidden:
                    hidden[i] = block_hidden[i][0].float()
                else:
                    hidden[i] = h[0].float()
            nll = answer_nll(out.logits[0], enc["labels"])
            attention = {}
            start, end = enc["answer_start"], enc["answer_end"]
            if capture_bilinear:
                for index in range(len(modules)):
                    if layers is not None and index + 1 not in layers:
                        continue
                    probs = out.attentions[index][0]
                    value = values[index][0]
                    heads = probs.shape[0]
                    kv_heads = int(getattr(model.config, "num_key_value_heads", heads))
                    value = value.view(value.shape[0], kv_heads, value.shape[-1] // kv_heads)
                    if kv_heads != heads:
                        if heads % kv_heads:
                            raise ValueError("attention heads must be divisible by key-value heads")
                        value = value.repeat_interleave(heads // kv_heads, dim=1)
                    attention[index + 1] = {"pi": probs[:, start:end, :], "v": value}
            gates = {}
            if capture_glu:
                for index in range(len(modules)):
                    if layers is not None and index + 1 not in layers:
                        continue
                    gates[index + 1] = {"g": glu[(index, "gate_proj")][0, start:end],
                                        "u": glu[(index, "up_proj")][0, start:end]}
    finally:
        for handle in handles:
            handle.remove()
    return {"hidden": hidden, "logits": out.logits[0], "nll": nll, "enc": enc, "attention": attention, "glu": gates}


def span_hidden(forward, layer, start=None, end=None):
    h = forward["hidden"][layer]
    if start is None:
        start, end = forward["enc"]["answer_start"], forward["enc"]["answer_end"]
    return h[start:end]


def eos_margin(logits, answer_start, answer_end, eos_id):
    """Mean (EOS logit - best non-EOS logit) on the answer span.

    Positive means the model prefers EOS; used only as a relative guard
    (Student vs frozen Teacher), never as an absolute forgetting signal.
    """
    span = logits[answer_start:max(answer_start, answer_end - 1)].float()
    if span.numel() == 0 or int(eos_id) >= span.size(-1):
        return logits.new_zeros(())
    without = span.clone()
    without[:, int(eos_id)] = -torch.inf
    return (span[:, int(eos_id)] - without.max(dim=-1).values).mean()


def stratified_sample(records, max_samples, group_key=None, seed=0):
    """Deterministic sample spread across groups (e.g. authors), not a prefix."""
    records = list(records)
    if max_samples is None or max_samples >= len(records):
        return records
    rng = random.Random(seed)
    if group_key is None:
        return rng.sample(records, max_samples)
    groups = {}
    for rec in records:
        groups.setdefault(group_key(rec), []).append(rec)
    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])
    picked = []
    while len(picked) < max_samples and any(groups[key] for key in keys):
        for key in keys:
            if groups[key] and len(picked) < max_samples:
                picked.append(groups[key].pop())
    return picked


@torch.no_grad()
def evaluate_ground_truth(model, tok, records, device, max_len=512, max_samples=32,
                          group_key=None, seed=0):
    sample = stratified_sample(records, max_samples, group_key, seed)
    rows = []
    for rec in sample:
        out = forward_model(model, encode_qa(tok, rec["question"], rec["answer"], max_len), device)
        rows.append(float(out["nll"]))
    coverage = len({group_key(rec) for rec in sample}) if group_key is not None else None
    return {"nll": sum(rows) / max(1, len(rows)), "count": len(rows),
            "ground_truth": "dataset_answer", "author_coverage": coverage}


@torch.no_grad()
def generation_behaviour(model, tok, records, device, max_new_tokens=24, max_samples=8,
                         group_key=None, seed=0):
    sample = stratified_sample(records, max_samples, group_key, seed)
    stats = []
    for rec in sample:
        ids = tok.encode(f"Question: {rec['question']}\nAnswer:", add_special_tokens=False)
        inp = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        # HF generate() forces num_logits_to_keep=1; that bf16 lm_head GEMM aborts
        # with SIGFPE on this torch/CUDA build. Explicit greedy decoding over the
        # full-sequence logits keeps identical decoding, EOS stop and metrics.
        gen_ids = []
        for _ in range(max_new_tokens):
            cur = torch.tensor(ids + gen_ids, dtype=torch.long, device=device).unsqueeze(0)
            cur_out = model(input_ids=cur, attention_mask=torch.ones_like(cur), use_cache=False)
            nxt = int(cur_out.logits[0, -1].float().argmax().item())
            gen_ids.append(nxt)
            if nxt == tok.eos_token_id:
                break
        gen = torch.tensor(gen_ids, dtype=torch.long)
        text = tok.decode(gen.tolist(), skip_special_tokens=True)
        target = set(re.findall(r"[a-z0-9']+", rec["answer"].lower()))
        generated = set(re.findall(r"[a-z0-9']+", text.lower()))
        stats.append({"tokens": int(gen.numel()), "eos": bool((gen == tok.eos_token_id).any()),
                      "overlap": len(target & generated) / max(1, len(target)), "text": text[:160]})
    n = max(1, len(stats))
    coverage = len({group_key(rec) for rec in sample}) if group_key is not None else None
    return {"samples": len(stats), "mean_tokens": sum(x["tokens"] for x in stats) / n,
            "eos_rate": sum(x["eos"] for x in stats) / n,
            "target_overlap": sum(x["overlap"] for x in stats) / n,
            "author_coverage": coverage,
            "examples": [x["text"] for x in stats[:2]]}


def patch_forward_output_multi(model, enc, device, replacements, layers=None, inputs_embeds=None):
    enc = move_encoded(enc, device)
    handles = []
    for layer, replacement in replacements.items():
        modules = decoder_layers(model)
        if layer < 0 or layer >= len(modules):
            raise IndexError(f"decoder layer {layer} out of range")
        replacement = replacement.to(device)

        def hook(_module, _args, output, new_hidden=replacement):
            old = output[0] if isinstance(output, tuple) else output
            new = new_hidden.unsqueeze(0).to(dtype=old.dtype, device=old.device)
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        handles.append(modules[layer].register_forward_hook(hook))
    try:
        kwargs = {"attention_mask": enc["attention_mask"].unsqueeze(0), "output_hidden_states": True, "use_cache": False}
        if inputs_embeds is None:
            kwargs["input_ids"] = enc["input_ids"].unsqueeze(0)
        else:
            kwargs["inputs_embeds"] = inputs_embeds.unsqueeze(0) if inputs_embeds.ndim == 2 else inputs_embeds
        out = model(**kwargs)
        hidden = {i: h[0] for i, h in enumerate(out.hidden_states) if layers is None or i in layers}
        return {"logits": out.logits[0], "hidden": hidden, "nll": answer_nll(out.logits[0], enc["labels"]), "enc": enc}
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def patch_forward_nll_multi(model, enc, device, replacements):
    return patch_forward_output_multi(model, enc, device, replacements)["nll"]


@torch.no_grad()
def patch_forward_nll(model, enc, device, layer, replacement):
    return patch_forward_nll_multi(model, enc, device, {layer: replacement})


@torch.no_grad()
def forward_with_attention_patch(model, enc, device, layer, replacement, answer_start=None):
    """Swap ONLY the self_attn write-back rows on the answer span, then run the
    remaining frozen layers and return the answer NLL.

    Used by the CF3-01 v1.1 §5.4 native QK/V 2x2 factorial, which is deliberately
    DIFFERENT from block-output patching (patch_forward_nll): here we replace the
    attention sublayer output (o_proj(pi @ v)) rather than the whole DecoderLayer
    hidden state. `replacement` is [n_ans, width] already in self_attn output space.

    Handles both HF (self_attn returns a tuple) and the tiny backend deterministically.
    """
    enc = move_encoded(enc, device)
    modules = decoder_layers(model)
    if layer < 1 or layer > len(modules):
        raise IndexError(f"decoder layer {layer} out of range for {len(modules)} layers")
    replacement = replacement.to(device)
    start = int(enc["answer_start"]) if answer_start is None else int(answer_start)
    attn_mod = modules[layer - 1].self_attn

    def hook(_module, _args, output, rep=replacement, s=start):
        old = output[0] if isinstance(output, tuple) else output
        new = old.clone()
        n = max(0, min(rep.shape[0], new.shape[1] - s))
        if n > 0:
            new[0, s:s + n] = rep[:n].to(dtype=old.dtype, device=old.device)
        return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new

    handle = attn_mod.register_forward_hook(hook)
    try:
        out = model(input_ids=enc["input_ids"].unsqueeze(0),
                    attention_mask=enc["attention_mask"].unsqueeze(0),
                    output_hidden_states=False, use_cache=False)
        return float(answer_nll(out.logits[0], enc["labels"]))
    finally:
        handle.remove()


def save_trained(model, tok, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path, safe_serialization=True)
    else:
        torch.save(model.state_dict(), path / "tiny_model.pt")
    if hasattr(tok, "save_pretrained"):
        tok.save_pretrained(path)

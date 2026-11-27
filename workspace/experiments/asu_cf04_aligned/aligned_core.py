"""Shared primitives for the CF-04 vs official-ASU aligned reproduction.

All functions are verbatim ports (read-only reference) of existing implementations:
  - data/eval helpers: auto_experiments/common.py, cf04_depth_audit/common_lite.py
  - ASU loss:         auto_experiments/train_tau.py (kl_on_answer, TemperatureContext)
  - CF-04 v2 loss:    diagnostic_methods/cf04_dual_probes.py, run_one_method.py
  - counterfactual:   experiments/direction_i/counterfactual.py
No algorithm or evaluation change; the two arms share this module.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

DATA = Path("/jizhicfs/chenyliao/data/datasets/locuslab-TOFU")
MODEL = "/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
FORGET = str(DATA / "forget10.json")
RETAIN = str(DATA / "retain90.json")
HOLDOUT = str(DATA / "holdout10.json")

# official tofu.yaml / asu.sh values, applied identically to both arms
EPOCHS = 5
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 1e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
EVAL_BATCH_SIZE = 8
MAX_LEN = 512


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(obj, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(question, answer=None, fewshot_prefix=None):
    prefix = fewshot_prefix or ""
    s = f"{prefix}Question: {question}\nAnswer:"
    if answer is not None:
        s += f" {answer}"
    return s


def make_batch(tok, records, device="cuda", max_len=MAX_LEN, fewshot_prefix=None):
    seqs = []
    for rec in records:
        prompt = build_prompt(rec["question"], None, fewshot_prefix)
        p_ids = tok.encode(prompt, add_special_tokens=False)
        a_ids = tok.encode(" " + rec["answer"].strip(), add_special_tokens=False) + [tok.eos_token_id]
        ids = (p_ids + a_ids)[:max_len]
        labels = ([-100] * len(p_ids) + a_ids)[:max_len]
        seqs.append((ids, labels))
    width = max(len(x[0]) for x in seqs)
    input_ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
    labels = torch.full((len(seqs), width), -100, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, (ids, lab) in enumerate(seqs):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        mask[i, : len(ids)] = 1
        labels[i, : len(lab)] = torch.tensor(lab)
    return {"input_ids": input_ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}


def answer_nll(model, batch, reduction="mean"):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    m = labels != -100
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none").view(labels.shape)
    nll = (loss * m).sum(-1)
    if reduction == "mean":
        return (nll / m.sum(-1).clamp_min(1)).mean().item(), nll.detach(), m
    return nll, m


def eval_nll(model, tok, records, device="cuda", batch_size=EVAL_BATCH_SIZE, fewshot_prefix=None):
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            chunk = records[i : i + batch_size]
            batch = make_batch(tok, chunk, device, fewshot_prefix=fewshot_prefix)
            val, _, _ = answer_nll(model, batch)
            n = sum(1 for r in chunk for _ in tok.encode(" " + r["answer"].strip(), add_special_tokens=False)) + len(chunk)
            total += val * n
            count += n
    return total / max(count, 1)


def retain_nll(model, batch):
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    return F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
    )


def kl_on_answer(student_logits, teacher_logits, labels):
    s = F.log_softmax(student_logits[:, :-1, :].float(), dim=-1)
    t = F.log_softmax(teacher_logits[:, :-1, :].float(), dim=-1)
    m = labels[:, 1:] != -100
    loss = F.kl_div(s, t, reduction="none", log_target=True).sum(-1)
    return (loss * m).sum() / m.sum().clamp_min(1)


class TemperatureContext:
    """Apply attention temperature by scaling q_proj outputs (verbatim from auto_experiments/common.py)."""

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


# ---- counterfactual pairs (verbatim from experiments/direction_i/counterfactual.py) ----
NAME_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,2}\b")


def subject_spans(question):
    return [re.sub(r"'s$", "", name) for name in NAME_RE.findall(question)]


def diff_is_single_span(ids_plus, ids_minus):
    if len(ids_plus) != len(ids_minus):
        return None
    starts = [i for i, (a, b) in enumerate(zip(ids_plus, ids_minus)) if a != b]
    if not starts:
        return None
    start, end = starts[0], starts[-1] + 1
    if any(ids_plus[i] != ids_minus[i] for i in range(start, end)) is False:
        return None
    if all(ids_plus[i] == ids_minus[i] for i in list(range(0, start)) + list(range(end, len(ids_plus)))):
        return (start, end)
    return None


def build_pairs(tokenizer, forget_records, retain_records, max_pairs=36, seed=0):
    rng = random.Random(seed)
    donor_names = sorted({name for rec in retain_records for name in subject_spans(rec["question"])})
    pairs = []
    for rec in forget_records:
        q_plus = rec["question"]
        spans = subject_spans(q_plus)
        if not spans:
            continue
        subject = spans[0]
        s_len = len(tokenizer.encode(subject, add_special_tokens=False))
        candidates = [n for n in donor_names if n != subject and len(tokenizer.encode(n, add_special_tokens=False)) == s_len]
        rng.shuffle(candidates)
        ids_plus = tokenizer.encode(q_plus, add_special_tokens=False)
        for donor in candidates:
            q_minus = q_plus.replace(subject, donor, 1)
            ids_minus = tokenizer.encode(q_minus, add_special_tokens=False)
            span = diff_is_single_span(ids_plus, ids_minus)
            if span is None:
                continue
            pairs.append({"x_plus": q_plus, "x_minus": q_minus, "forget_subject": subject,
                          "donor_subject": donor, "token_span": list(span), "answer_plus": rec["answer"]})
            break
        if len(pairs) >= max_pairs:
            break
    return pairs


# ---- dual probes (verbatim ports from diagnostic_methods/cf04_dual_probes.py + base.py) ----
def as_tensor(value, dtype=torch.float32):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def safe_cosine(a, b, eps=1e-8):
    a, b = as_tensor(a).flatten(), as_tensor(b).flatten()
    denom = a.norm().clamp_min(eps) * b.norm().clamp_min(eps)
    return (a @ b) / denom


def auc_score(scores, labels):
    s = as_tensor(scores).flatten()
    y = as_tensor(labels).flatten().bool().to(s.device)
    pos, neg = int(y.sum().item()), int((~y).sum().item())
    if pos == 0 or neg == 0:
        return 0.5
    order = torch.argsort(s)
    ranks = torch.empty(s.numel(), dtype=torch.float32, device=s.device)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=torch.float32, device=s.device)
    sorted_s = s[order]
    i = 0
    while i < s.numel():
        j = i
        while j + 1 < s.numel() and torch.isclose(sorted_s[j + 1], sorted_s[i]):
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float(((ranks[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg)).item())


@dataclass
class LinearProbe:
    weight: torch.Tensor
    bias: torch.Tensor
    history: list

    def score(self, x):
        return as_tensor(x) @ self.weight + self.bias


def fit_linear_probe(x, y, steps=300, lr=0.05, l2=1e-3):
    x = as_tensor(x)
    y = as_tensor(y).flatten().float().to(x.device)
    w = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device, requires_grad=True)
    b = torch.zeros((), dtype=x.dtype, device=x.device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    history = []
    for step in range(steps):
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y) + l2 * (w * w).sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.item())})
    return LinearProbe(w.detach(), b.detach(), history)


def _split(x, y, train_fraction=0.6):
    x, y = as_tensor(x), as_tensor(y).to(as_tensor(x).device)
    n = x.shape[0]
    order = torch.randperm(n, device=x.device)
    n_train = max(2, min(n - 2, int(n * train_fraction)))
    tr, te = order[:n_train], order[n_train:]
    return x[tr], y[tr], x[te], y[te]


def fit_dual_probes(fact_x, fact_y, identity_x, identity_y, steps=300, lr=0.05):
    fx, fy, ftx, fty = _split(fact_x, fact_y)
    ix, iy, itx, ity = _split(identity_x, identity_y)
    fact_probe = fit_linear_probe(fx, fy, steps=steps, lr=lr)
    id_probe = fit_linear_probe(ix, iy, steps=steps, lr=lr)
    auc_f = auc_score(fact_probe.score(ftx), fty)
    auc_id = auc_score(id_probe.score(itx), ity)
    cos = float(safe_cosine(fact_probe.weight, id_probe.weight).item())
    report = {"auc_fact": auc_f, "auc_identity": auc_id, "auc_margin": auc_f - auc_id, "probe_cosine": cos,
              "fact_probe_norm": float(fact_probe.weight.norm().item()), "identity_probe_norm": float(id_probe.weight.norm().item()),
              "train_size": int(fx.shape[0]), "test_size": int(ftx.shape[0]),
              "pass": auc_f - auc_id >= 0.10 and abs(cos) <= 0.8}
    return fact_probe, id_probe, report


def fit_layerwise_dual_probes(layer_fact_x, fact_y, layer_identity_x, identity_y, steps=300, lr=0.05):
    probes, report, weights = {}, {}, {}
    for layer in sorted(layer_fact_x):
        fp, ip, rep = fit_dual_probes(layer_fact_x[layer], fact_y, layer_identity_x[layer], identity_y, steps=steps, lr=lr)
        probes[layer] = (fp, ip)
        report[layer] = rep
        weights[layer] = max(0.0, float(rep["auc_margin"]))
    if sum(weights.values()) <= 1e-8:
        weights = {k: 1.0 for k in weights}
    total = sum(weights.values())
    return probes, {k: v / total for k, v in weights.items()}, report


def pooled_hidden(hidden, mask=None):
    h = as_tensor(hidden)
    if mask is None:
        return h.mean(dim=1)
    m = as_tensor(mask).to(h.dtype).unsqueeze(-1)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def probe_guided_loss_v2(forget_hiddens, retain_hiddens, probes, layer_weights=None,
                         margin=0.5, temperature=1.0, identity_weight=1.0, retain_weight=1.0):
    layers = sorted(forget_hiddens)
    weights = layer_weights or {k: 1.0 / len(layers) for k in layers}
    total = 0.0
    parts = {"fact": [], "identity": [], "retain_fact": []}
    for layer in layers:
        fp, ip = probes[layer]
        hf = pooled_hidden(forget_hiddens[layer]) / temperature
        hr = pooled_hidden(retain_hiddens[layer]) / temperature
        l_fact = F.softplus(fp.score(hf) + margin).mean()
        l_id = F.softplus(-ip.score(hf) + 0.5 * margin).mean()
        l_ret = F.softplus(-fp.score(hr) + margin).mean()
        w = float(weights.get(layer, 0.0))
        total = total + w * (l_fact + identity_weight * l_id + retain_weight * l_ret)
        parts["fact"].append(l_fact.detach())
        parts["identity"].append(l_id.detach())
        parts["retain_fact"].append(l_ret.detach())
    return {"loss": total, "fact_forget": torch.stack(parts["fact"]).mean(),
            "identity_preserve": torch.stack(parts["identity"]).mean(),
            "retain_fact_preserve": torch.stack(parts["retain_fact"]).mean()}


def hidden_layers(model, batch, layers):
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
    return {layer: out.hidden_states[layer].float() for layer in layers}


def hidden_for_records(model, tok, records, layer, device="cuda", micro_batch=8):
    chunks = []
    for start in range(0, len(records), micro_batch):
        batch = make_batch(tok, records[start : start + micro_batch], device)
        with torch.no_grad():
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
        h = out.hidden_states[layer].float()
        if h.size(1) < MAX_LEN:
            h = F.pad(h, (0, 0, 0, MAX_LEN - h.size(1)))
        chunks.append(h)
    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, MAX_LEN, 2048))


def make_order(seed, forget, retain, epochs=EPOCHS, batch_size=BATCH_SIZE, grad_accum=GRAD_ACCUM):
    """Identical, deterministic micro-step sample order for BOTH arms of a paired seed."""
    rng = random.Random(seed)
    f_order = list(range(len(forget)))
    r_order = list(range(len(retain)))
    rng.shuffle(f_order)
    rng.shuffle(r_order)
    steps_per_epoch = -(-len(forget) // (batch_size * grad_accum))  # ceil
    total_steps = epochs * steps_per_epoch
    order = []
    for step in range(total_steps):
        micro = []
        for a in range(grad_accum):
            base = (step * grad_accum + a) * batch_size
            fb = [forget[f_order[(base + i) % len(f_order)]] for i in range(batch_size)]
            rb = [retain[r_order[(base + i) % len(r_order)]] for i in range(batch_size)]
            micro.append((fb, rb))
        order.append(micro)
    return order, steps_per_epoch, total_steps

#!/usr/bin/env python3
"""RIU post-training evaluation: NLL + linear fact decodability.

Compares the original model (theta_p) with the unlearned checkpoint (theta_u)
on TOFU forget05 / holdout05.  Reports:

  * mean answer NLL (higher on forget = more forgetting; lower on holdout = utility)
  * fact-identity linear probe accuracy trained on theta_p or theta_u and
    evaluated on theta_u (lower = less linearly recoverable information)

This is a recoverability proxy, not the full LeakRU FQ/RR protocol.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_DIR = Path("/jizhicfs/chenyliao/data/datasets/locuslab-TOFU")
IGNORE = -100


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode(tokenizer, question: str, answer: str, max_length: int = 384):
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = f"Question: {question}\nAnswer:"
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(" " + answer.strip(), add_special_tokens=False).input_ids
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    input_ids = (prompt_ids + answer_ids + eos)[-max_length:]
    labels = ([IGNORE] * len(prompt_ids) + answer_ids + eos)[-max_length:]
    return input_ids, labels


def collate(batch):
    width = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), width), 0, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (input_ids, lab) in enumerate(batch):
        ids[i, : len(input_ids)] = torch.tensor(input_ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        mask[i, : len(input_ids)] = 1
    return ids, labels, mask


@torch.no_grad()
def extract_nll(model, tokenizer, rows, batch_size, device):
    nll_sum, nll_count = 0.0, 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        encoded = [encode(tokenizer, r["question"], r["answer"]) for r in chunk]
        ids, labels, mask = collate(encoded)
        ids, labels, mask = ids.to(device), labels.to(device), mask.to(device)
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        shift_logits = out.logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        valid = shift_labels.ne(IGNORE)
        loss = F.cross_entropy(
            shift_logits.transpose(1, 2), shift_labels.clamp_min(0), reduction="none"
        )
        nll_sum += float((loss * valid).sum())
        nll_count += int(valid.sum())
    return nll_sum / max(nll_count, 1)


def sanitize_question(question: str, names: list[str]) -> str:
    import re
    for name in names:
        question = re.sub(re.escape(name), "the author", question, flags=re.I)
    return question


def encode_prompt(tokenizer, question: str, max_length: int = 384):
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = f"Question: {question}\nAnswer:"
    return tokenizer(prompt, add_special_tokens=False).input_ids[-max_length:]


@torch.no_grad()
def extract_features(model, tokenizer, rows, layer, batch_size, device, sanitize=False, names=None):
    """Probe features from PROMPT-only hidden states.

    Teacher-forced answer tokens would leak the target into the representation,
    making any probe trivially 100% accurate.
    """
    feats = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        questions = [
            sanitize_question(r["question"], names) if sanitize else r["question"]
            for r in chunk
        ]
        encoded = [encode_prompt(tokenizer, q) for q in questions]
        width = max(len(x) for x in encoded)
        ids = torch.full((len(encoded), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for i, seq in enumerate(encoded):
            ids[i, : len(seq)] = torch.tensor(seq)
            mask[i, : len(seq)] = 1
        out = model(
            input_ids=ids.to(device), attention_mask=mask.to(device),
            output_hidden_states=True, use_cache=False,
        )
        hidden = out.hidden_states[layer + 1]
        w = mask.unsqueeze(-1).to(hidden.device).to(hidden.dtype)
        pooled = (hidden * w).sum(1) / w.sum(1).clamp_min(1.0)
        feats.append(F.normalize(pooled, dim=-1).float().cpu())
    return torch.cat(feats, dim=0)


def train_probe(features, labels, num_classes, steps=300, lr=0.05, seed=0):
    """Train on half of the facts, evaluate on the disjoint held-out half."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(features), generator=generator)
    split = len(order) // 2
    train_idx, test_idx = order[:split], order[split:]
    probe = torch.nn.Linear(features.shape[1], num_classes)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(probe(features[train_idx]), labels[train_idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float(
            (probe(features[test_idx]).argmax(-1) == labels[test_idx]).float().mean()
        )
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta-p", default="/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-2-7b-chat-hf_full")
    parser.add_argument("--theta-u", required=True)
    parser.add_argument("--layer", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default="/usr/local/ASU-unlearning-RIU/workspace/results/riu_eval_report.json")
    parser.add_argument("--sanitize-names", action="store_true", help="mask author names in probe prompts")
    args = parser.parse_args()

    forget = load_rows(DATA_DIR / "forget05.json")
    holdout = load_rows(DATA_DIR / "holdout05.json")
    facts_path = Path("/usr/local/ASU-unlearning-RIU/workspace/data/leakru/tofu_forget05_facts.json")
    names = [f["name"] for f in json.loads(facts_path.read_text(encoding="utf-8"))] if args.sanitize_names else []
    labels = torch.arange(len(forget)) // 20
    num_classes = int(labels.max().item()) + 1

    def load(path):
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).to(args.device)
        model.eval()
        return tok, model

    tok, model = load(args.theta_p)
    nll_p_forget = extract_nll(model, tok, forget, args.batch_size, args.device)
    nll_p_hold = extract_nll(model, tok, holdout, args.batch_size, args.device)
    feat_p = extract_features(
        model, tok, forget, args.layer, args.batch_size, args.device,
        args.sanitize_names, names,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    tok, model = load(args.theta_u)
    nll_u_forget = extract_nll(model, tok, forget, args.batch_size, args.device)
    nll_u_hold = extract_nll(model, tok, holdout, args.batch_size, args.device)
    feat_u = extract_features(
        model, tok, forget, args.layer, args.batch_size, args.device,
        args.sanitize_names, names,
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    report = {
        "theta_p": args.theta_p,
        "theta_u": args.theta_u,
        "layer": args.layer,
        "n_forget": len(forget),
        "n_holdout": len(holdout),
        "nll_forget_theta_p": nll_p_forget,
        "nll_forget_theta_u": nll_u_forget,
        "nll_holdout_theta_p": nll_p_hold,
        "nll_holdout_theta_u": nll_u_hold,
        "fact_probe_train_on_theta_p_eval_theta_u": train_probe(feat_p, labels, num_classes),
        "fact_probe_train_on_theta_u_eval_theta_u": train_probe(feat_u, labels, num_classes),
        "sanitize_names": args.sanitize_names,
        "feature_source": "prompt-only" if args.sanitize_names else "prompt-only (names visible)",
        "note": "Recoverability proxy; not LeakRU FQ/RR.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

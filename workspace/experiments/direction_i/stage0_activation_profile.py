"""Stage-0 (zero-training) measurement for Direction I.

Measures the layer-wise structure of the counterfactual activation difference
Delta A_l = A_l(X+) - A_l(X-) on a frozen teacher, where X+ and X- differ only
in the subject slot (token-equal donor replacement).

Purpose: decide whether layer-wise progressive alignment has a viable signal
(deep-layer peak) before spending any training budget.

Usage:
    python stage0_activation_profile.py --model gpt2 --num-pairs 100 --device cuda
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FIRST_NAMES = [
    "Elena", "Marcus", "Sofia", "Adrian", "Priya", "Julian", "Nora", "Darius",
    "Iris", "Theo", "Camila", "Owen", "Leila", "Victor", "Amina", "Felix",
    "Rosa", "Hugo", "Maya", "Silas", "Zara", "Emil", "Lena", "Kai",
]
LAST_NAMES = [
    "Vasquez", "Chen", "Okafor", "Lindqvist", "Haddad", "Moreau", "Kowalski",
    "Nakamura", "Rossi", "Petrov", "Silva", "Kim", "Novak", "Fischer", "Duarte",
    "Berg", "Costa", "Ivanov", "Larsen", "Mensah", "Park", "Reyes",
]
TEMPLATE = "The fictional character {subject} works as a"


def build_token_equal_pairs(tokenizer, num_pairs):
    names = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    by_len = defaultdict(list)
    for name in names:
        # Group by the length of the FULL templated prompt: subword tokenization
        # is context-dependent, so counting the name alone is not sufficient.
        n = len(tokenizer.encode(TEMPLATE.format(subject=name), add_special_tokens=False))
        by_len[n].append(name)

    pairs = []
    for n, group in sorted(by_len.items()):
        if len(group) < 2:
            continue
        for i in range(0, len(group) - 1, 2):
            pairs.append((group[i], group[i + 1], n))
            if len(pairs) >= num_pairs:
                return pairs
    if not pairs:
        raise RuntimeError("No token-equal name pairs found; enlarge name lists.")
    return pairs


def encode_prompt(tokenizer, subject):
    text = TEMPLATE.format(subject=subject)
    ids = tokenizer.encode(text, return_tensors="pt")
    subj_ids = tokenizer.encode(subject, add_special_tokens=False)
    return ids, len(subj_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--num-pairs", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="stage0_activation_profile.json")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device).eval()
    n_layer = model.config.n_layer if hasattr(model.config, "n_layer") else model.config.num_hidden_layers

    pairs = build_token_equal_pairs(tok, args.num_pairs)
    print(f"model={args.model} layers={n_layer} pairs={len(pairs)} device={args.device}")

    # Accumulators: per layer, per position mode ("all" and "last").
    stats = {
        "all": defaultdict(list),
        "last": defaultdict(list),
    }
    with torch.no_grad():
        for start in range(0, len(pairs), args.batch_size):
            chunk = pairs[start : start + args.batch_size]
            ids_plus, ids_minus = [], []
            for subj_f, subj_r, _ in chunk:
                a, _ = encode_prompt(tok, subj_f)
                b, _ = encode_prompt(tok, subj_r)
                assert a.shape == b.shape, "Prompt token lengths must match."
                ids_plus.append(a.squeeze(0))
                ids_minus.append(b.squeeze(0))

            def pad(seqs):
                m = max(s.numel() for s in seqs)
                out = torch.full((len(seqs), m), tok.pad_token_id or 0, dtype=torch.long)
                mask = torch.zeros((len(seqs), m), dtype=torch.bool)
                for i, s in enumerate(seqs):
                    out[i, : s.numel()] = s
                    mask[i, : s.numel()] = True
                return out.to(args.device), mask.to(args.device)

            x_plus, mask = pad(ids_plus)
            x_minus, _ = pad(ids_minus)
            out_plus = model(input_ids=x_plus, attention_mask=mask, output_hidden_states=True)
            out_minus = model(input_ids=x_minus, attention_mask=mask, output_hidden_states=True)

            for l in range(1, n_layer + 1):
                h_plus = out_plus.hidden_states[l]
                h_minus = out_minus.hidden_states[l]
                delta = (h_plus - h_minus).pow(2).sum(-1).sqrt()  # [B,T]
                denom_all = h_plus.pow(2).sum(-1).sqrt().clamp_min(1e-6)
                ratio_all = (delta / denom_all)[mask]
                stats["all"][l].extend(ratio_all.cpu().tolist())

                # last-token position of each sequence (final prompt token)
                last_idx = mask.sum(1) - 1
                b_idx = torch.arange(x_plus.size(0), device=args.device)
                ratio_last = (delta / denom_all)[b_idx, last_idx]
                stats["last"][l].extend(ratio_last.cpu().tolist())

    result = {
        "model": args.model,
        "num_pairs": len(pairs),
        "layers": n_layer,
        "per_layer": {},
    }
    for mode in ("all", "last"):
        for l in range(1, n_layer + 1):
            vals = stats[mode][l]
            result["per_layer"].setdefault(str(l), {})[mode] = {
                "mean": sum(vals) / max(1, len(vals)),
                "max": max(vals) if vals else 0.0,
            }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\nlayer   delta/act(all)   delta/act(last)")
    for l in range(1, n_layer + 1):
        a = result["per_layer"][str(l)]["all"]["mean"]
        b = result["per_layer"][str(l)]["last"]["mean"]
        print(f"{l:>5}   {a:>15.4f}   {b:>16.4f}")

    vals = [result["per_layer"][str(l)]["last"]["mean"] for l in range(1, n_layer + 1)]
    peak = max(range(len(vals)), key=lambda i: vals[i]) + 1
    print(f"\nlast-position peak at layer {peak}/{n_layer}  (value={vals[peak-1]:.4f})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

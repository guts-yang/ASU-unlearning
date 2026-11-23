"""Refined Delta-A probe: subject-level pooled DeltaA (optimization after stage7).

Stage7 token-level DeltaA had top5 explained variance 0.17-0.23 (KILL gate).
This variant pools DeltaA over prompt positions per pair before PCA, testing
whether the low-rank structure exists at the subject/event level rather than
at the individual-token level. Same kill gates as stage7.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from counterfactual import build_pairs, load_jsonl, token_ids
from stage7_da_probe import pca_basis, subspace_overlap

NAME_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,2}\b")


def pooled_delta(model, tok, xps, xms, layer, device):
    out = []
    with torch.no_grad():
        for xp, xm in zip(xps, xms):
            hp = model(input_ids=tok.encode(xp, return_tensors="pt").to(device), output_hidden_states=True).hidden_states[layer][0].float().cpu().numpy()
            hm = model(input_ids=tok.encode(xm, return_tensors="pt").to(device), output_hidden_states=True).hidden_states[layer][0].float().cpu().numpy()
            L = min(len(hp), len(hm))
            out.append((hp[:L] - hm[:L]).mean(axis=0))
    return np.stack(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    ap.add_argument("--forget", default="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json")
    ap.add_argument("--retain", default="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json")
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--layers", default="8,12,16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="stage7b_da_probe_pooled.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    pairs = build_pairs(tok, args.forget, args.retain, max_pairs=args.pairs, seed=0)
    retain = load_jsonl(args.retain)

    # wrong donor: donor name from an unrelated retain question, matched token length
    tmpls = {}
    for r in retain:
        tmpls.setdefault(re.sub(NAME_RE, "<NAME>", r["question"]), []).append(r)
    keys = list(tmpls)
    wrong_xp, wrong_xm = [], []
    for i, p in enumerate(pairs):
        t = re.sub(NAME_RE, "<NAME>", p["x_plus"])
        other = keys[(keys.index(t) + 11 + i) % len(keys)] if t in keys else keys[i % len(keys)]
        pool = set()
        for r in tmpls[other]:
            for n in NAME_RE.findall(r["question"]):
                pool.add(re.sub(r"'s$", "", n))
        pool = [n for n in pool if len(token_ids(tok, n)) == len(token_ids(tok, p["forget_subject"]))]
        if not pool:
            continue
        donor = pool[(i * 5) % len(pool)]
        wrong_xp.append(p["x_plus"])
        wrong_xm.append(p["x_plus"].replace(p["forget_subject"], donor, 1))

    ret_xp, ret_xm = [], []
    for i in range(0, min(64, len(retain) - 2), 2):
        ret_xp.append(retain[i]["question"])
        ret_xm.append(retain[i + 1]["question"])

    result = {"pairs": len(pairs), "rank": args.rank, "layers": {}}
    for layer in [int(x) for x in args.layers.split(",")]:
        D = pooled_delta(model, tok, [p["x_plus"] for p in pairs], [p["x_minus"] for p in pairs], layer, device)
        B, ck, c5, c10 = pca_basis(D, args.rank)
        ent = {"topk_explained": float(ck), "top5_explained": float(c5), "top10_explained": float(c10)}
        if wrong_xp:
            Dw = pooled_delta(model, tok, wrong_xp, wrong_xm, layer, device)
            Bw, *_ = pca_basis(Dw, args.rank)
            ent["overlap_true_wrong"] = subspace_overlap(B, Bw)
        if ret_xp:
            Dr = pooled_delta(model, tok, ret_xp, ret_xm, layer, device)
            Br, *_ = pca_basis(Dr, args.rank)
            ent["overlap_true_retain"] = subspace_overlap(B, Br)
        result["layers"][str(layer)] = ent
        print(f"layer {layer}: top5={c5:.3f} top8={ck:.3f} "
              f"ov_wrong={ent.get('overlap_true_wrong', float('nan')):.3f} "
              f"ov_retain={ent.get('overlap_true_retain', float('nan')):.3f}", flush=True)
        np.save(Path(args.output).with_name(f"da_basis_pooled_layer{layer}.npy"), B)

    primary = result["layers"][str(max(int(x) for x in args.layers.split(",")))]
    result["gate"] = {
        "low_rank_pass": bool(primary["top5_explained"] >= 0.40),
        "retain_orthogonal_pass": bool(primary.get("overlap_true_retain", 1.0) <= 0.50),
        "verdict": "PASS" if primary["top5_explained"] >= 0.40 and primary.get("overlap_true_retain", 1.0) <= 0.50 else "KILL",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("GATE:", result["gate"], flush=True)
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()

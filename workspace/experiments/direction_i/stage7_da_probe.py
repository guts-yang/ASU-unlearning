"""Block A gate (idea-discovery): Delta-A spectral probe, zero training.

Kill gates from remote-agent review:
  1) top-5 explained variance of DeltaA subspace < 0.40 -> kill (no low-rank structure)
  2) overlap(P_fact, P_retain) > 0.50 -> kill (same damage direction as Idea 4)
  3) wrong-donor subspace should NOT be indistinguishable from true-donor subspace

Also writes the cached subspaces (rank k) for the later Idea1 training runs.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from counterfactual import build_pairs, load_jsonl, token_ids

NAME_RE = re.compile(r"\b[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){1,2}\b")


def norm_template(q):
    return NAME_RE.sub("<NAME>", q)


def hidden_all(model, ids, layer, device):
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    return out.hidden_states[layer][0].float().cpu().numpy()  # [T,d]


def collect_delta(model, tok, x_plus_list, x_minus_list, layer, device, pool="none"):
    mats = []
    for xp, xm in zip(x_plus_list, x_minus_list):
        hp = hidden_all(model, tok.encode(xp, return_tensors="pt").to(device), layer, device)
        hm = hidden_all(model, tok.encode(xm, return_tensors="pt").to(device), layer, device)
        L = min(hp.shape[0], hm.shape[0])
        d = hp[:L] - hm[:L]
        if pool == "mean":
            d = d.mean(axis=0, keepdims=True)
        mats.append(d)
    return np.concatenate(mats, axis=0)  # [N*T, d] or [N, d]


def pca_basis(X, k):
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    ratio = var / max(var.sum(), 1e-12)
    return Vt[:k].T, ratio[:k].sum(), ratio[:5].sum(), ratio[:10].sum()


def subspace_overlap(A, B):
    # orthonormal bases A,B: [d,k]; Frobenius-normalized principal-angle overlap
    M = A.T @ B
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.linalg.norm(s) / np.sqrt(min(A.shape[1], B.shape[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    ap.add_argument("--forget", default="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget10.json")
    ap.add_argument("--retain", default="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json")
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--layers", default="8,12,16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="stage7_da_probe.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    pairs = build_pairs(tok, args.forget, args.retain, max_pairs=args.pairs, seed=0)
    retain = load_jsonl(args.retain)

    # Wrong-donor: donor name drawn from a DIFFERENT normalized question template.
    by_tmpl = {}
    for r in retain:
        by_tmpl.setdefault(norm_template(r["question"]), []).append(r)
    all_tmpl = list(by_tmpl)
    wrong_plus, wrong_minus = [], []
    for i, p in enumerate(pairs):
        tmpl = norm_template(p["x_plus"])
        other = all_tmpl[(all_tmpl.index(tmpl) + 7 + i) % len(all_tmpl)] if tmpl in all_tmpl else all_tmpl[i % len(all_tmpl)]
        cand = by_tmpl[other]
        donor_pool = set()
        for r in cand:
            for n in NAME_RE.findall(r["question"]):
                donor_pool.add(re.sub(r"'s$", "", n))
        donor_pool = [n for n in donor_pool if n and len(token_ids(tok, n)) == len(token_ids(tok, p["forget_subject"]))]
        if not donor_pool:
            continue
        donor = donor_pool[(i * 3) % len(donor_pool)]
        wrong_plus.append(p["x_plus"])
        wrong_minus.append(p["x_plus"].replace(p["forget_subject"], donor, 1))

    # Retain-difference subspace: pairs of retain questions with different subjects.
    retain_plus, retain_minus = [], []
    for i in range(0, min(64, len(retain) - 2), 2):
        retain_plus.append(retain[i]["question"])
        retain_minus.append(retain[i + 1]["question"])

    result = {"model": args.model, "pairs": len(pairs), "rank": args.rank, "layers": {}}
    for layer in [int(x) for x in args.layers.split(",")]:
        D_fact = collect_delta(model, tok, [p["x_plus"] for p in pairs], [p["x_minus"] for p in pairs], layer, device)
        B_fact, cumk, cum5, cum10 = pca_basis(D_fact, args.rank)
        entry = {
            "topk_explained": float(cumk),
            "top5_explained": float(cum5),
            "top10_explained": float(cum10),
        }
        if wrong_plus:
            D_wrong = collect_delta(model, tok, wrong_plus, wrong_minus, layer, device)
            B_wrong, *_ = pca_basis(D_wrong, args.rank)
            entry["overlap_true_wrong"] = subspace_overlap(B_fact, B_wrong)
        if retain_plus:
            D_ret = collect_delta(model, tok, retain_plus, retain_minus, layer, device)
            B_ret, *_ = pca_basis(D_ret, args.rank)
            entry["overlap_true_retain"] = subspace_overlap(B_fact, B_ret)
        result["layers"][str(layer)] = entry
        print(f"layer {layer}: top5={cum5:.3f} top{args.rank}={cumk:.3f} "
              f"ov_wrong={entry.get('overlap_true_wrong', float('nan')):.3f} "
              f"ov_retain={entry.get('overlap_true_retain', float('nan')):.3f}", flush=True)

        # cache basis for training
        np.save(Path(args.output).with_name(f"da_basis_layer{layer}.npy"), B_fact)

    # Gates: use the deepest probed layer (closest to output) as primary.
    primary = result["layers"][str(max(int(x) for x in args.layers.split(",")))]
    result["gate"] = {
        "low_rank_pass": bool(primary["top5_explained"] >= 0.40),
        "retain_orthogonal_pass": bool(primary.get("overlap_true_retain", 1.0) <= 0.50),
        "verdict": "PASS" if (primary["top5_explained"] >= 0.40 and primary.get("overlap_true_retain", 1.0) <= 0.50) else "KILL",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("GATE:", result["gate"], flush=True)
    print("wrote", args.output, flush=True)


if __name__ == "__main__":
    main()

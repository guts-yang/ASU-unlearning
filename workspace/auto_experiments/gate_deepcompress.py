"""V3 Idea 3 gate: can deep Delta-A directions be suppressed/aligned without hurting retain?

V2 showed plain suppression fails. This gate additionally tests V3's proposed
"align toward retain component" variant at deep layers, with equal-hook evaluation.
"""

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import FORGET, MODEL, RETAIN, RESULTS, answer_nll, counterfactual_pairs, load_jsonl, load_model, make_batch, save_json, set_seed


def pooled_delta(model, tok, xs, xms, layer, device):
    out = []
    with torch.no_grad():
        for xp, xm in zip(xs, xms):
            hp = model(input_ids=tok.encode(xp, return_tensors="pt").to(device), output_hidden_states=True).hidden_states[layer][0].float().cpu().numpy()
            hm = model(input_ids=tok.encode(xm, return_tensors="pt").to(device), output_hidden_states=True).hidden_states[layer][0].float().cpu().numpy()
            L = min(len(hp), len(hm))
            out.append((hp[:L] - hm[:L]).mean(axis=0))
    return np.stack(out)


def pca_basis(D, rank):
    D = D - D.mean(axis=0, keepdims=True)
    U, S, Vh = np.linalg.svd(D, full_matrices=False)
    return Vh[:rank].T, float(S[:5].sum() / S.sum())


def retain_component(model, tok, records, layer, B, device):
    hs = []
    with torch.no_grad():
        for r in records[:24]:
            ids = tok.encode(r["question"], return_tensors="pt").to(device)
            out = model(input_ids=ids, output_hidden_states=True)
            h = out.hidden_states[layer][0].float().cpu().numpy()
            hs.append((h @ B).mean(axis=0))
    return np.stack(hs).mean(axis=0)


class DirectionHook:
    def __init__(self, B, target, mode):
        self.B = torch.tensor(B, dtype=torch.float32)
        self.target = None if target is None else torch.tensor(target, dtype=torch.float32)
        self.mode = mode
        self.handle = None

    def __enter__(self):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            orig_dtype = h.dtype
            hf = h.float()
            B = self.B.to(hf.device)
            comp = hf @ B
            if self.mode == "suppress":
                hf = hf - comp @ B.T
            elif self.mode == "align":
                target = self.target.to(hf.device)
                hf = hf - comp @ B.T + target.unsqueeze(0).unsqueeze(0) @ B.T
            new = hf.to(orig_dtype)
            return (new,) + out[1:] if isinstance(out, tuple) else new

        self.handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *args):
        self.handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--forget", default=FORGET)
    ap.add_argument("--retain", default=RETAIN)
    ap.add_argument("--pairs", type=int, default=48)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--layers", default="10,12,14,15")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=str(RESULTS / "gate_deepcompress.json"))
    args = ap.parse_args()
    set_seed(0)

    model, tok = load_model(args.model, args.device)
    pairs = counterfactual_pairs(tok, args.forget, args.retain, max_pairs=args.pairs, seed=0)
    forget = load_jsonl(args.forget)[:64]
    retain = load_jsonl(args.retain)[:64]
    f_base = answer_nll(model, make_batch(tok, forget, args.device))[0]
    r_base = answer_nll(model, make_batch(tok, retain, args.device))[0]
    print(f"baseline forget={f_base:.4f} retain={r_base:.4f}", flush=True)

    results = {}
    for l in [int(x) for x in args.layers.split(",")]:
        D = pooled_delta(model, tok, [p["x_plus"] for p in pairs], [p["x_minus"] for p in pairs], l, args.device)
        B, evr = pca_basis(D, args.rank)
        target = retain_component(model, tok, retain, l, B, args.device)
        results[str(l)] = {"top5_explained": evr, "modes": {}}
        for mode in ["suppress", "align"]:
            hook = DirectionHook(B, target, mode)
            hook.module = model.model.layers[min(l, model.config.num_hidden_layers - 1)]
            with hook:
                f = answer_nll(model, make_batch(tok, forget, args.device))[0]
                r = answer_nll(model, make_batch(tok, retain, args.device))[0]
            results[str(l)]["modes"][mode] = {
                "forget_nll": f,
                "retain_nll": r,
                "forget_delta": f - f_base,
                "retain_damage": r - r_base,
            }
            print(f"L{l} {mode}: dF={f-f_base:+.4f} dR={r-r_base:+.4f}", flush=True)

    any_pass = any(
        m["forget_delta"] >= 0.10 and m["retain_damage"] <= 0.10
        for v in results.values()
        for m in v["modes"].values()
    )
    result = {
        "idea": "V3-Idea3-shallow-deep",
        "baseline": {"forget": f_base, "retain": r_base},
        "layers": results,
        "gate": {"deep_direction_usable": any_pass, "verdict": "PASS" if any_pass else "KILL"},
        "threshold": {"forget_delta": 0.10, "retain_damage": 0.10},
    }
    save_json(result, args.output)
    print("GATE", result["gate"], flush=True)


if __name__ == "__main__":
    main()

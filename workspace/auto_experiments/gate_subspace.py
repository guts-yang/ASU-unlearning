"""V2 Idea A gate: RepSelect-style forget-gradient subspace selectivity.

Tests whether bottom-PC directions of the forget gradient are more forget-specific
(lower overlap with retain gradient) than top-PC directions, on an ASU-unlearned model.
Attention v_proj and MLP down_proj are tested because RepSelect only studied MLP.
"""

import argparse
import math

import torch
import torch.nn.functional as F

from common import CKPT, FORGET, MODEL, RETAIN, RESULTS, load_jsonl, make_batch, save_json, set_seed


def grad_sums(model, tok, records, modules, device="cuda", batch_size=4, max_batches=8):
    sums = {name: torch.zeros_like(mod.weight) for name, mod in modules.items()}
    model.zero_grad(set_to_none=True)
    for bi in range(max_batches):
        chunk = records[bi * batch_size : (bi + 1) * batch_size]
        if not chunk:
            break
        batch = make_batch(tok, chunk, device)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = out.logits[:, :-1, :].float()
        labels = batch["labels"][:, 1:]
        m = labels != -100
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none").view(labels.shape)
        ((loss * m).sum() / m.sum().clamp_min(1)).backward()
        for name, mod in modules.items():
            if mod.weight.grad is not None:
                sums[name] += mod.weight.grad.detach().float()
                mod.weight.grad = None
    return sums


def overlap(g, basis, k):
    """Fraction of ||g||^2 explained by top-k right singular directions."""
    if g.ndim != 2:
        g = g.reshape(g.shape[0], -1)
    proj = g @ basis[:, :k]
    return float((proj**2).sum() / g.pow(2).sum().clamp_min(1e-12))


def singular_basis(g, rank):
    U, S, Vh = torch.linalg.svd(g.float(), full_matrices=False)
    return Vh.T.contiguous(), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CKPT / "tau_global_s0"))
    ap.add_argument("--fallback", default=MODEL)
    ap.add_argument("--forget", default=FORGET)
    ap.add_argument("--retain", default=RETAIN)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=str(RESULTS / "gate_subspace.json"))
    args = ap.parse_args()
    set_seed(0)

    path = args.model
    try:
        model = torch.load  # noqa: F841 - keep torch imported
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(args.device)
    except Exception as e:
        print(f"checkpoint {path} unavailable ({e}); using fallback {args.fallback}", flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.fallback)
        model = AutoModelForCausalLM.from_pretrained(args.fallback, torch_dtype=torch.float32).to(args.device)
        path = args.fallback
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.train()

    modules = {}
    layers = sorted(set([2, model.config.num_hidden_layers // 2, model.config.num_hidden_layers - 2]))
    for l in layers:
        modules[f"L{l}_v"] = model.model.layers[l].self_attn.v_proj
        modules[f"L{l}_down"] = model.model.layers[l].mlp.down_proj

    forget = load_jsonl(args.forget)[: args.max_batches * 4]
    retain = load_jsonl(args.retain)[: args.max_batches * 4]
    gf = grad_sums(model, tok, forget, modules, args.device, batch_size=4, max_batches=args.max_batches)
    gr = grad_sums(model, tok, retain, modules, args.device, batch_size=4, max_batches=args.max_batches)

    per_module = {}
    for name in modules:
        Bf, S = singular_basis(gf[name], args.rank)
        g = gr[name].reshape(gr[name].shape[0], -1)
        top = overlap(g, Bf, args.rank)
        bottom = overlap(g, Bf[:, -args.rank :], args.rank)
        per_module[name] = {
            "top_overlap": top,
            "bottom_overlap": bottom,
            "top_singular": [float(x) for x in S[:3]],
            "selectivity": top - bottom,
        }
        print(f"{name}: top={top:.4f} bottom={bottom:.4f} sel={top-bottom:+.4f}", flush=True)

    mean_sel = sum(v["selectivity"] for v in per_module.values()) / max(len(per_module), 1)
    verdict = "PASS" if mean_sel >= 0.05 else "KILL"
    result = {
        "idea": "V2-IdeaA-ASU-Selective",
        "model": path,
        "rank": args.rank,
        "modules": per_module,
        "mean_selectivity": mean_sel,
        "gate": {"bottom_more_specific": mean_sel >= 0.05, "verdict": verdict},
        "interpretation": "Positive means forget-gradient top-PC aligns more with retain gradient than bottom-PC, i.e. bottom-PC is forget-specific (RepSelect condition).",
    }
    save_json(result, args.output)
    print("GATE", result["gate"], flush=True)


if __name__ == "__main__":
    main()

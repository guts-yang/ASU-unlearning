"""V3 Idea 1 gate: factual/function temperature sensitivity across layers and heads.

Intervention: scale q_proj per head by 1/tau, which is exactly attention-logit
temperature tau. Sensitivity = (dNLL_factual - dNLL_function) at tau=2.
If the field is near-uniform, SensiTemp degenerates to global ASU tau and is killed.
"""

import argparse
import math

import torch
import torch.nn.functional as F

from common import (
    FORGET,
    MODEL,
    RESULTS,
    TemperatureContext,
    factual_function_mask,
    load_jsonl,
    load_model,
    make_batch,
    save_json,
    set_seed,
)


def token_nll(model, batch, temp_field=None):
    ctx = TemperatureContext(model, temp_field) if temp_field is not None else None
    if ctx:
        ctx.__enter__()
    try:
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = out.logits[:, :-1, :].float()
        labels = batch["labels"][:, 1:]
        m = labels != -100
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
        ).view(labels.shape)
        return loss.detach(), m
    finally:
        if ctx:
            ctx.__exit__(None, None, None)


def fact_func_stats(model, tok, records, temp_field=None, batch_size=8):
    fact_tot, fact_n, func_tot, func_n = 0.0, 0, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            chunk = records[i : i + batch_size]
            batch = make_batch(tok, chunk)
            loss, m = token_nll(model, batch, temp_field)
            for j, rec in enumerate(chunk):
                flags, _ = factual_function_mask(tok, rec["answer"])
                flags = torch.cat([flags, torch.tensor([False])])  # EOS treated as function
                valid = m[j].cpu()
                length = min(len(flags), valid.sum().item())
                f = flags[:length]
                v = valid.nonzero(as_tuple=True)[0][:length]
                lv = loss[j].cpu()[v]
                fact_tot += lv[f].sum().item()
                fact_n += int(f.sum().item())
                func_tot += lv[~f].sum().item()
                func_n += int((~f).sum().item())
    return fact_tot / max(fact_n, 1), func_tot / max(func_n, 1)


def cv(vals):
    mean = sum(vals) / max(len(vals), 1)
    if mean <= 1e-9:
        return 0.0, mean
    var = sum((v - mean) ** 2 for v in vals) / max(len(vals), 1)
    return math.sqrt(var) / abs(mean), mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--forget", default=FORGET)
    ap.add_argument("--num-records", type=int, default=48)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default=str(RESULTS / "gate_sensitemp.json"))
    args = ap.parse_args()
    set_seed(0)

    model, tok = load_model(args.model, args.device)
    n_layer = model.config.num_hidden_layers
    n_head = model.config.num_attention_heads
    records = load_jsonl(args.forget)[: args.num_records]
    fact0, func0 = fact_func_stats(model, tok, records)
    print(f"baseline fact={fact0:.4f} func={func0:.4f}", flush=True)

    layer_sens = {}
    for l in range(n_layer):
        f, u = fact_func_stats(model, tok, records, temp_field={l: [args.tau] * n_head})
        layer_sens[l] = (f - fact0) - (u - func0)
        print(f"layer {l:2d} sens={layer_sens[l]:+.4f}", flush=True)

    probe_layers = sorted(set([0, n_layer // 4, n_layer // 2, 3 * n_layer // 4, n_layer - 1]))
    head_sens = {}
    for l in probe_layers:
        hs = []
        for h in range(n_head):
            field = [1.0] * n_head
            field[h] = args.tau
            f, u = fact_func_stats(model, tok, records, temp_field={l: field})
            hs.append((f - fact0) - (u - func0))
        head_sens[l] = hs
        hcv, hmean = cv(hs)
        print(f"layer {l:2d} head mean={hmean:+.4f} cv={hcv:.3f}", flush=True)

    lcv, lmean = cv(list(layer_sens.values()))
    hcvs = [cv(head_sens[l])[0] for l in probe_layers]
    hcv_mean = sum(hcvs) / len(hcvs)
    top_layers = sorted(layer_sens, key=layer_sens.get, reverse=True)[:5]
    top_heads = []
    for l in probe_layers:
        for h in sorted(range(n_head), key=lambda x: head_sens[l][x], reverse=True)[:3]:
            top_heads.append([l, h, round(head_sens[l][h], 4)])

    verdict = "PASS" if (lcv >= 0.25 or hcv_mean >= 0.35) else "KILL"
    result = {
        "idea": "V3-Idea1-SensiTemp",
        "model": args.model,
        "records": len(records),
        "tau": args.tau,
        "baseline": {"fact_nll": fact0, "func_nll": func0},
        "layer_sensitivity": {str(k): round(v, 5) for k, v in layer_sens.items()},
        "head_sensitivity": {str(k): [round(x, 5) for x in v] for k, v in head_sens.items()},
        "layer_cv": lcv,
        "head_cv_mean": hcv_mean,
        "top_layers": top_layers,
        "top_heads": top_heads,
        "gate": {
            "nonuniform_layer": lcv >= 0.25,
            "nonuniform_head": hcv_mean >= 0.35,
            "verdict": verdict,
        },
        "limitations": "factual/function tokens use a stopword heuristic, not GPT-4o labels; sensitivity measured with q scaling at tau=2 only.",
    }
    save_json(result, args.output)
    print("GATE", result["gate"], flush=True)


if __name__ == "__main__":
    main()

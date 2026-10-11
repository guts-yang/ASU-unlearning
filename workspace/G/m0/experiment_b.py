"""Pivot localization: integrated gradients vs activation patching, then column temperature.

Site is the residual entering Llama layer 16 (0-based, L/2). Integrated
gradients use a zero baseline and a 32-step Riemann sum. Captum is not
installed, so the sum is written out here. B2 applies eager attention
QK^T / (sqrt(head_dim) * tau) on the chosen key columns at that same layer.
The downgrade rerun moves only that temperature to layer 24.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from common import (
    DOWNGRADE_LAYER,
    EXP_B,
    IG_STEPS,
    LAYER_INDEX,
    TAU,
    TOKENIZER_ID,
    bootstrap_mean_ci,
    read_jsonl,
    write_json,
    write_jsonl,
)
from judge import gold_spans, is_hit

TOP_FRACTION = 0.10


def continuation_ids(tokenizer, prefix_ids: list[int], text: str) -> list[int]:
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    full = tokenizer.encode(prefix_text + text, add_special_tokens=False)
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    if full[: len(prefix)] == prefix and len(full) > len(prefix):
        return full[len(prefix):]
    spaced = tokenizer.encode(prefix_text + " " + text, add_special_tokens=False)
    if spaced[: len(prefix)] == prefix and len(spaced) > len(prefix):
        return spaced[len(prefix):]
    return tokenizer.encode(text, add_special_tokens=False)


def top_fraction(scores: np.ndarray) -> list[int]:
    count = len(scores)
    k = int(round(count * TOP_FRACTION))
    k = min(count, max(1, k))
    order = sorted(range(count), key=lambda index: (-float(scores[index]), index))
    return order[:k]


def jaccard(left: list[int], right: list[int]) -> float:
    a, b = set(left), set(right)
    union = a | b
    if not union:
        return float("nan")
    return len(a & b) / len(union)


class ResidualSite:
    def __init__(self, layer: torch.nn.Module):
        self.clean = None
        self.mode = "off"
        self.cot_index = None
        self.alpha = 1.0
        self.mean_vec = None
        self.patch_pos = None
        self.leaf = None
        self.handle = layer.register_forward_pre_hook(self._pre)

    def close(self) -> None:
        self.handle.remove()

    def _pre(self, module, args):
        hidden = args[0]
        if self.mode == "capture":
            self.clean = hidden.detach()
            return None
        if self.mode == "patch":
            patched = hidden.clone()
            patched[:, self.patch_pos] = self.mean_vec.to(dtype=hidden.dtype, device=hidden.device)
            return (patched,) + args[1:]
        if self.mode == "ig":
            base = hidden.detach()
            piece = (self.alpha * self.clean[:, self.cot_index]).detach()
            piece = piece.to(dtype=hidden.dtype, device=hidden.device).requires_grad_(True)
            edited = base.index_copy(1, self.cot_index, piece)
            self.leaf = piece
            return (edited,) + args[1:]
        return None


def gold_logprob(model, input_ids: torch.Tensor, gold_start: int) -> torch.Tensor:
    attention = torch.ones_like(input_ids).unsqueeze(0)
    outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention, use_cache=False)
    logits = outputs.logits[0, gold_start - 1:-1].float()
    targets = input_ids[gold_start:]
    return F.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).sum()


def signed_ig(model, site: ResidualSite, input_ids: torch.Tensor, gold_start: int, cot_index: torch.Tensor) -> np.ndarray:
    site.cot_index = cot_index
    grads = []
    for step in range(1, IG_STEPS + 1):
        site.mode = "ig"
        site.alpha = step / IG_STEPS
        with torch.enable_grad():
            score = gold_logprob(model, input_ids, gold_start)
            score.backward()
        grads.append(site.leaf.grad.detach().float()[0])
        site.leaf = None
    average = torch.stack(grads).mean(dim=0)
    residual = site.clean[0, cot_index].float()
    return (residual * average).sum(dim=-1).cpu().numpy()


def column_tau_forward(attn, key_tau: torch.Tensor):
    def forward(
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        del kwargs
        query_states = attn.q_proj(hidden_states)
        key_states = attn.k_proj(hidden_states)
        value_states = attn.v_proj(hidden_states)
        bsz, q_len, _ = hidden_states.size()
        query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, attn.num_key_value_groups)
        value_states = repeat_kv(value_states, attn.num_key_value_groups)
        tau = key_tau.view(1, 1, 1, -1).to(dtype=query_states.dtype, device=query_states.device)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (math.sqrt(attn.head_dim) * tau)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = attn.o_proj(attn_output)
        return attn_output, None, past_key_value

    return forward


def logprob_with_tau(model, layer_index: int, input_ids: torch.Tensor, gold_start: int, columns: list[int], tau: float) -> float:
    attn = model.model.layers[layer_index].self_attn
    original = attn.forward
    key_tau = torch.ones(input_ids.shape[0], device=input_ids.device)
    if columns:
        key_tau[columns] = tau
    attn.forward = column_tau_forward(attn, key_tau)
    try:
        with torch.inference_mode():
            value = gold_logprob(model, input_ids, gold_start)
        return float(value.item())
    finally:
        attn.forward = original


def prepare_examples(rows: list[dict], tokenizer) -> list[dict]:
    prepared = []
    for row in rows:
        spans = gold_spans(row["answer"], row.get("question"))
        if not is_hit(row["output"], spans):
            continue
        prefix = list(row["prompt_token_ids"]) + list(row["output_token_ids"])
        gold = continuation_ids(tokenizer, prefix, row["answer"])
        if not gold or not row["output_token_ids"]:
            continue
        prepared.append({
            "index": row["index"],
            "prefix_len": len(row["prompt_token_ids"]),
            "cot_len": len(row["output_token_ids"]),
            "ids": prefix + gold,
            "gold_start": len(prefix),
        })
    return prepared


def locate(model, prepared: list[dict]) -> tuple[list[dict], torch.Tensor]:
    layer = model.model.layers[LAYER_INDEX]
    site = ResidualSite(layer)
    located = []
    residual_sum = None
    residual_count = 0
    for example in prepared:
        input_ids = torch.tensor(example["ids"], device=model.device)
        cot = torch.arange(
            example["prefix_len"],
            example["prefix_len"] + example["cot_len"],
            device=model.device,
        )
        site.mode = "capture"
        with torch.no_grad():
            clean_logp = float(gold_logprob(model, input_ids, example["gold_start"]).item())
        site.clean = site.clean.clone()
        cot_residual = site.clean[0, cot].float()
        residual_sum = cot_residual.sum(dim=0) if residual_sum is None else residual_sum + cot_residual.sum(dim=0)
        residual_count += cot_residual.shape[0]
        attribution = signed_ig(model, site, input_ids, example["gold_start"], cot)
        located.append({
            "index": example["index"],
            "prefix_len": example["prefix_len"],
            "cot_len": example["cot_len"],
            "gold_start": example["gold_start"],
            "ids": example["ids"],
            "clean_logp": clean_logp,
            "signed_ig": attribution.tolist(),
        })
        print(f"ig {len(located)}/{len(prepared)}", flush=True)
    site.close()
    mean_residual = residual_sum / residual_count
    return located, mean_residual


def mediate(model, located: list[dict], mean_residual: torch.Tensor) -> None:
    layer = model.model.layers[LAYER_INDEX]
    site = ResidualSite(layer)
    site.mean_vec = mean_residual.to(model.device)
    for example in located:
        input_ids = torch.tensor(example["ids"], device=model.device)
        drops = []
        for offset in range(example["cot_len"]):
            site.mode = "patch"
            site.patch_pos = example["prefix_len"] + offset
            with torch.inference_mode():
                patched = float(gold_logprob(model, input_ids, example["gold_start"]).item())
            drops.append(example["clean_logp"] - patched)
        example["mediation_drop"] = drops
        ig_set = top_fraction(np.array(example["signed_ig"]))
        med_set = top_fraction(np.array(drops))
        example["ig_top"] = ig_set
        example["mediation_top"] = med_set
        example["jaccard"] = jaccard(ig_set, med_set)
        print(f"mediate {example['index']} jaccard {example['jaccard']:.3f}", flush=True)
    site.mode = "off"
    site.close()


def temperature(model, located: list[dict], layer_index: int) -> list[dict]:
    rows = []
    for example in located:
        pivot_local = example["ig_top"]
        non_pivot = [offset for offset in range(example["cot_len"]) if offset not in set(pivot_local)]
        if len(non_pivot) < len(pivot_local):
            rows.append({
                "index": example["index"],
                "valid": False,
                "reason": "not enough non-pivot CoT tokens",
            })
            continue
        rng = np.random.default_rng(1001 + int(example["index"]))
        random_local = rng.choice(non_pivot, size=len(pivot_local), replace=False).tolist()
        input_ids = torch.tensor(example["ids"], device=model.device)
        base = logprob_with_tau(model, layer_index, input_ids, example["gold_start"], [], 1.0)
        pivot_columns = [example["prefix_len"] + offset for offset in pivot_local]
        random_columns = [example["prefix_len"] + offset for offset in random_local]
        pivot_logp = logprob_with_tau(model, layer_index, input_ids, example["gold_start"], pivot_columns, TAU)
        random_logp = logprob_with_tau(model, layer_index, input_ids, example["gold_start"], random_columns, TAU)
        rows.append({
            "index": example["index"],
            "valid": True,
            "layer": layer_index,
            "D_pivot": base - pivot_logp,
            "D_random": base - random_logp,
            "n_columns": len(pivot_local),
        })
        print(f"tau L{layer_index} {example['index']} Dp {rows[-1]['D_pivot']:.4f} Dr {rows[-1]['D_random']:.4f}", flush=True)
    return rows


def paired_d(diff: np.ndarray) -> float:
    if diff.size < 2:
        return float("nan")
    sd = float(diff.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(diff.mean() / sd)


def b2_report(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("valid")]
    n = len(valid)
    if n == 0:
        return {"n": 0, "pass": False, "mean_D_pivot": None, "mean_D_random": None, "ratio": None, "wilcoxon_p": None, "paired_d": None}
    d_pivot = np.array([row["D_pivot"] for row in valid], dtype=np.float64)
    d_random = np.array([row["D_random"] for row in valid], dtype=np.float64)
    mean_p = float(d_pivot.mean())
    mean_r = float(d_random.mean())
    if mean_r > 0:
        ratio = mean_p / mean_r
        ratio_pass = mean_p >= 3 * mean_r
    else:
        ratio = None
        ratio_pass = False
    diff = d_pivot - d_random
    try:
        stat = wilcoxon(d_pivot, d_random, zero_method="wilcox", alternative="two-sided")
        p_value = float(stat.pvalue)
    except ValueError:
        p_value = None
    effect = paired_d(diff)
    passed = bool(
        n >= 40
        and ratio_pass
        and p_value is not None
        and p_value < 0.05
        and effect == effect
        and effect >= 0.8
    )
    return {
        "n": n,
        "n_invalid": len(rows) - n,
        "pass": passed,
        "mean_D_pivot": mean_p,
        "mean_D_random": mean_r,
        "ratio": ratio,
        "wilcoxon_p": p_value,
        "wilcoxon_alternative": "two-sided",
        "paired_d": effect,
    }


def b1_report(located: list[dict]) -> dict:
    values = np.array([row["jaccard"] for row in located], dtype=np.float64)
    point, low, high = bootstrap_mean_ci(values)
    return {
        "n": int(values.size),
        "mean_jaccard": point,
        "ci95": [low, high],
        "pass": bool(values.size >= 40 and point >= 0.4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--downgrade-layer", action="store_true")
    args = parser.parse_args()
    out_dir = EXP_B / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    rows = read_jsonl(args.generations)
    prepared = prepare_examples(rows, tokenizer)
    write_json(out_dir / "eligible.json", {"n_generations": len(rows), "n_recoveries": len(prepared)})
    if len(prepared) < 40:
        write_json(out_dir / "report.json", {
            "n": len(prepared),
            "B1": {"pass": False, "n": len(prepared), "mean_jaccard": None, "ci95": None},
            "B2": {"pass": False, "n": len(prepared), "ratio": None, "wilcoxon_p": None, "paired_d": None},
            "reason": "fewer than 40 DefaultThink recoveries",
        })
        print(f"recoveries {len(prepared)} < 40")
        return
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": 0},
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    located_path = out_dir / "located.jsonl"
    if located_path.exists() and not args.downgrade_layer:
        located = read_jsonl(located_path)
    else:
        located, mean_residual = locate(model, prepared)
        mediate(model, located, mean_residual)
        torch.save(mean_residual.cpu(), out_dir / "mean_residual.pt")
        write_jsonl(located_path, located)
    if args.downgrade_layer:
        tau_rows = temperature(model, located, DOWNGRADE_LAYER)
        write_jsonl(out_dir / "temperature_layer24.jsonl", tau_rows)
        write_json(out_dir / "b2_layer24.json", b2_report(tau_rows))
        return
    tau_rows = temperature(model, located, LAYER_INDEX)
    write_jsonl(out_dir / "temperature_layer16.jsonl", tau_rows)
    report = {"n": len(located), "B1": b1_report(located), "B2": b2_report(tau_rows), "layer": LAYER_INDEX, "ig_steps": IG_STEPS, "ig_baseline": 0}
    write_json(out_dir / "report.json", report)
    print(report)


if __name__ == "__main__":
    main()

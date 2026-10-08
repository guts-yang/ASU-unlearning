"""Base-model NLL against attention temperature, grouped by token class."""

import torch
import torch.nn.functional as F

from .kill_switch import decide, problem_slopes
from .labels import encode_labeled
from .protocol import TAU_GRID, normalize_layers_id


def shifted_nll(logits, labels):
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    batch, length, vocab = shift_logits.shape
    nll = F.cross_entropy(
        shift_logits.reshape(-1, vocab),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )
    return nll.view(batch, length)


def class_means_from_nll(nll, class_masks):
    """Mean NLL at query positions. ``class_masks`` values are ``[seq]`` bool."""
    means = {}
    for name, mask in class_masks.items():
        selected = mask[: nll.shape[0]].bool()
        if int(selected.sum()) == 0:
            means[name] = None
        else:
            means[name] = float(nll[selected].mean().item())
    return means


@torch.no_grad()
def nll_at_temperature(model, input_ids, labels, attention_mask, attention_temp, layers_id):
    single = input_ids.dim() == 1
    if single:
        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=None,
        attention_temp=attention_temp,
        layers_id=layers_id,
        use_cache=False,
        return_dict=True,
    )
    nll = shifted_nll(outputs.logits, labels)
    return nll[0] if single else nll


def _prepare(tokenizer, records, model_configs, max_length):
    prepared = []
    for problem_id, record in enumerate(records):
        input_ids, labels, attention_mask, classes = encode_labeled(
            tokenizer,
            record["question"],
            record["answer"],
            model_configs,
            max_length,
            record.get("factual_spans") or [],
            record.get("function_spans") or [],
        )
        prepared.append(
            {
                "id": record.get("id", problem_id),
                "problem_id": problem_id,
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "classes": classes,
            }
        )
    return prepared


def scan_records(
    model,
    tokenizer,
    records,
    model_configs,
    max_length,
    taus=TAU_GRID,
    layers_id=None,
    batch_size=4,
):
    """Return per-problem mean NLL curves and the kill-switch verdict.

    Each record needs ``question`` and ``answer``. Optional ``factual_spans``
    and ``function_spans`` are answer-character intervals. Derivation spans are
    read from ``<<expr=result>>``. Missing factual or function labels leave
    that class empty and the verdict stays ``insufficient_data``.
    """
    layers_id = normalize_layers_id(layers_id)
    prepared = _prepare(tokenizer, records, model_configs, max_length)
    device = next(model.parameters()).device
    curves = []
    bucket = {name: {} for name in ("derivation", "factual", "function")}
    for start in range(0, len(prepared), batch_size):
        print(f"scan {start}/{len(prepared)}", flush=True)
        chunk = prepared[start : start + batch_size]
        input_ids = torch.stack([item["input_ids"] for item in chunk]).to(device)
        labels = torch.stack([item["labels"] for item in chunk]).to(device)
        attention_mask = torch.stack([item["attention_mask"] for item in chunk]).to(device)
        per_tau = [[] for _ in chunk]
        for tau in taus:
            nll = nll_at_temperature(
                model, input_ids, labels, attention_mask, float(tau), layers_id
            )
            for row, item in enumerate(chunk):
                aligned = {name: mask[:-1].to(device) for name, mask in item["classes"].items()}
                means = class_means_from_nll(nll[row], aligned)
                per_tau[row].append({"tau": float(tau), **means})
        for row, item in enumerate(chunk):
            curves.append({"id": item["id"], "curve": per_tau[row]})
            for name in bucket:
                series = [point[name] for point in per_tau[row]]
                if all(value is not None for value in series):
                    bucket[name][item["problem_id"]] = series
    slopes = {name: problem_slopes(means, taus) for name, means in bucket.items()}
    verdict = decide(slopes)
    verdict["taus"] = [float(tau) for tau in taus]
    verdict["n_records"] = len(records)
    verdict["layers_id"] = layers_id
    return {"curves": curves, "verdict": verdict}

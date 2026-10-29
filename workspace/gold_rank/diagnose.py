"""Calibration-only teacher diagnostic. It does not train or download weights."""

import json
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

import torch

from gold_rank.demotion import (
    apply_gold_rank_logits,
    fit_logit_penalty,
    gold_components,
    select_operating_point,
)
from gold_rank.preflight import enable_offline, local_snapshot_issue

REPO = Path(__file__).resolve().parents[2]


def load_json_records(path):
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def calibration_indices(count, fraction, seed):
    if count <= 0:
        return [], []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(count, generator=generator).tolist()
    calibration_count = max(1, int(round(count * float(fraction))))
    if count > 1 and calibration_count >= count:
        calibration_count = count - 1
    return order[:calibration_count], order[calibration_count:]


def empty_bucket():
    return {
        "n_valid": 0,
        "n_edited": 0,
        "n_skipped": 0,
        "gold_prob_before_sum": 0.0,
        "gold_prob_after_sum": 0.0,
        "rank_before_sum": 0.0,
        "rank_after_sum": 0.0,
    }


def add_stats(bucket, stats):
    for key in bucket:
        bucket[key] += stats[key]
    return bucket


def evaluate_grid(logits, gold_ids, mask, k_grid, delta_grid):
    cells = {}
    parts_gold = []
    parts_rest = []
    valid = mask.bool() & (gold_ids >= 0)
    if bool(valid.any()):
        gold_logit, log_rest = gold_components(logits.float()[valid], gold_ids[valid])
        parts_gold.append(gold_logit.detach().cpu())
        parts_rest.append(log_rest.detach().cpu())
    for k in k_grid:
        for delta in delta_grid:
            _, stats = apply_gold_rank_logits(
                logits,
                gold_ids,
                mask,
                {"mode": "demote", "k": int(k), "delta": float(delta), "collect_stats": True},
            )
            key = (int(k), float(delta))
            cells[key] = add_stats(cells.get(key, empty_bucket()), stats or empty_bucket())
            cells[key]["k"] = int(k)
            cells[key]["delta"] = float(delta)
    return cells, parts_gold, parts_rest


def merge_cells(parts):
    merged = {}
    for cells in parts:
        for key, bucket in cells.items():
            target = merged.setdefault(key, empty_bucket())
            add_stats(target, bucket)
            target["k"] = bucket["k"]
            target["delta"] = bucket["delta"]
    return list(merged.values())


def penalty_rank_increase(logits, gold_ids, mask, penalty):
    _, stats = apply_gold_rank_logits(
        logits,
        gold_ids,
        mask,
        {"mode": "fixed_penalty", "penalty": float(penalty), "collect_stats": True},
    )
    if stats is None or stats["n_valid"] == 0:
        return 0.0, 0
    increase = stats["rank_after_sum"] - stats["rank_before_sum"]
    return increase, stats["n_valid"]


def write_lock(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_lock(protocol, cells, components, forget_sha, attention_temp, calibration_count, heldout_count):
    chosen = select_operating_point(cells, protocol["min_rank_separation"])
    chosen.update(
        {
            "k_grid": list(protocol["k_grid"]),
            "delta_grid": list(protocol["delta_grid"]),
            "selection_rule": protocol["selection_rule"],
            "min_rank_separation": protocol["min_rank_separation"],
            "forget_sha256": forget_sha,
            "attention_temp": attention_temp,
            "calibration_seed": protocol["calibration_seed"],
            "calibration_fraction": protocol["calibration_fraction"],
            "n_calibration_examples": calibration_count,
            "n_heldout_examples": heldout_count,
            "cells": cells,
            "test_metrics_read": False,
            "matched_penalty": None,
        }
    )
    if chosen.get("status") != "frozen_pending_penalty_check":
        return chosen
    if not components:
        chosen["status"] = "stop_not_distinct"
        return chosen
    valid = next(cell for cell in cells if cell["k"] == chosen["k"] and cell["delta"] == chosen["delta"])
    target_drop = (valid["gold_prob_before_sum"] - valid["gold_prob_after_sum"]) / valid["n_valid"]
    gold_logit = torch.cat([item[0] for item in components])
    log_rest = torch.cat([item[1] for item in components])
    penalty, achieved, saturated = fit_logit_penalty(
        gold_logit,
        log_rest,
        target_drop,
        tolerance=float(protocol["penalty_match_tolerance"]),
    )
    chosen["matched_penalty"] = penalty
    chosen["matched_probability_drop"] = achieved
    chosen["target_probability_drop"] = target_drop
    chosen["penalty_saturated"] = saturated
    if saturated:
        chosen["status"] = "stop_unmatched_penalty"
    return chosen


def finalize_penalty_separation(lock, matched_increase, matched_count, min_separation):
    """Veto a frozen cell without replacing it. Held-out test metrics are not an input."""
    if lock.get("status") != "frozen_pending_penalty_check":
        return lock
    valid = lock["n_valid"] if "n_valid" in lock else None
    cell = next(item for item in lock["cells"] if item["k"] == lock["k"] and item["delta"] == lock["delta"])
    valid = cell["n_valid"]
    demotion_increase = (cell["rank_after_sum"] - cell["rank_before_sum"]) / valid
    matched_mean = matched_increase / matched_count if matched_count else 0.0
    lock["matched_rank_increase"] = matched_mean
    lock["mean_rank_increase"] = demotion_increase
    if demotion_increase < matched_mean + float(min_separation):
        lock["status"] = "stop_not_distinct"
    else:
        lock["status"] = "frozen"
    return lock


def _load_local_teacher(model_path):
    enable_offline()
    issue = local_snapshot_issue(model_path)
    if issue:
        raise SystemExit(issue)
    import sys

    sys.path.insert(0, str(REPO / "Right-to-be-forgotten"))
    from my_models.my_llama import LlamaForCausalLM
    from transformers import AutoTokenizer

    last_error = None
    model = None
    for implementation in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation=implementation,
                local_files_only=True,
            )
            break
        except (ImportError, ValueError, OSError) as error:
            last_error = error
    if model is None:
        raise SystemExit(f"could not load local snapshot {model_path}: {last_error}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def _model_configs():
    import yaml

    path = REPO / "Right-to-be-forgotten" / "config" / "model_config.yaml"
    return yaml.safe_load(path.read_text())["llama3-8b"]


def _walk(model, tokenizer, records, indices, max_length, attention_temp, batch_size, k_grid, delta_grid, collect_parts):
    from gold_rank.masking import encode_forget_example

    configs = _model_configs()
    device = next(model.parameters()).device
    cell_parts = []
    components = []
    for start in range(0, len(indices), batch_size):
        batch_ids = indices[start:start + batch_size]
        encoded = [
            encode_forget_example(tokenizer, max_length, records[index]["question"], records[index]["answer"], configs, mask=True)
            for index in batch_ids
        ]
        input_ids = torch.stack([item[0] for item in encoded]).to(device)
        labels = torch.stack([item[1] for item in encoded]).to(device)
        attention_mask = torch.stack([item[2] for item in encoded]).to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                attention_temp=attention_temp,
                layers_id=None,
                use_cache=False,
            )
        logits = outputs.logits[:, :-1, :].detach()
        gold_ids = labels[:, 1:]
        answer_mask = gold_ids != -100
        cells, gold_parts, rest_parts = evaluate_grid(logits, gold_ids, answer_mask, k_grid, delta_grid)
        cell_parts.append(cells)
        if collect_parts:
            components.extend(zip(gold_parts, rest_parts))
    return merge_cells(cell_parts), components


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Freeze gold-rank K and delta on calibration teacher logits")
    parser.add_argument("--model", required=True, help="Local snapshot directory. Hub ids are refused.")
    parser.add_argument("--forget-json", required=True)
    parser.add_argument("--protocol", default=str(Path(__file__).with_name("protocol.json")))
    parser.add_argument("--lock-out", required=True)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    enable_offline()
    issue = local_snapshot_issue(args.model)
    if issue or not Path(args.forget_json).is_file():
        missing = [item for item in (issue, None if Path(args.forget_json).is_file() else f"forget split missing: {args.forget_json}") if item]
        raise SystemExit("Refusing to download.\n" + "\n".join(missing))

    protocol = json.loads(Path(args.protocol).read_text())
    records = load_json_records(args.forget_json)
    calibration, heldout = calibration_indices(len(records), protocol["calibration_fraction"], protocol["calibration_seed"])
    from gold_rank.lock import sha256_file

    model, tokenizer = _load_local_teacher(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    cells, components = _walk(
        model,
        tokenizer,
        records,
        calibration,
        args.max_length,
        float(protocol["attention_temp"]),
        args.batch_size,
        protocol["k_grid"],
        protocol["delta_grid"],
        collect_parts=True,
    )
    lock = build_lock(
        protocol,
        cells,
        components,
        sha256_file(args.forget_json),
        float(protocol["attention_temp"]),
        len(calibration),
        len(heldout),
    )
    if lock.get("status") == "frozen_pending_penalty_check":
        increase_sum = 0.0
        count = 0
        from gold_rank.masking import encode_forget_example

        configs = _model_configs()
        for index in calibration:
            encoded = encode_forget_example(
                tokenizer, args.max_length, records[index]["question"], records[index]["answer"], configs, True
            )
            input_ids = encoded[0].unsqueeze(0).to(device)
            labels = encoded[1].unsqueeze(0).to(device)
            attention_mask = encoded[2].unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    attention_temp=float(protocol["attention_temp"]),
                    layers_id=None,
                    use_cache=False,
                )
            increase, n_valid = penalty_rank_increase(
                outputs.logits[:, :-1, :],
                labels[:, 1:],
                labels[:, 1:] != -100,
                lock["matched_penalty"],
            )
            increase_sum += increase
            count += n_valid
        lock = finalize_penalty_separation(lock, increase_sum, count, protocol["min_rank_separation"])
    write_lock(args.lock_out, lock)
    print(json.dumps({"status": lock["status"], "k": lock.get("k"), "delta": lock.get("delta"), "lock": args.lock_out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

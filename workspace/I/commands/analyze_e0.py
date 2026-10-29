#!/usr/bin/env python3
"""Calibration temperature choice and holdout gate for MCSD E0. No threshold retuning."""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import yaml


def load_forward(path):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    for row in rows:
        if row["n_tokens"] <= 0 or row["gold_logp_mean"] is None:
            raise RuntimeError(f"missing gold logp in {path} sample {row['sample_id']}")
    return rows


def author_means(rows, split, subset, field):
    grouped = {}
    for row in rows:
        if row["split"] == split and row["subset"] == subset:
            grouped.setdefault(int(row["author_index"]), []).append(float(row[field]))
    if not grouped:
        raise RuntimeError(f"no rows for {split} {subset} {field}")
    return {author: float(np.mean(values)) for author, values in grouped.items()}


def aligned(left, right, sign):
    authors = sorted(set(left) & set(right))
    if authors != sorted(left) or authors != sorted(right):
        raise RuntimeError("author ids differ")
    return np.asarray([sign * (left[author] - right[author]) for author in authors], dtype=np.float64)


def bootstrap_mean_ci(values, seed, samples, quantiles):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, quantiles)
    return {
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / np.sqrt(len(values))),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_authors": int(len(values)),
    }


def condition_record(rows, baseline_rows, config):
    result = {}
    for split in ("calibration", "holdout"):
        forget_s = bootstrap_mean_ci(
            aligned(
                author_means(baseline_rows, split, "forget", "gold_logp_mean"),
                author_means(rows, split, "forget", "gold_logp_mean"),
                1.0,
            ),
            int(config["bootstrap_seed"]),
            int(config["bootstrap_samples"]),
            config["bootstrap_ci"],
        )
        retain_nll = bootstrap_mean_ci(
            aligned(
                author_means(rows, split, "retain", "nll_mean"),
                author_means(baseline_rows, split, "retain", "nll_mean"),
                1.0,
            ),
            int(config["bootstrap_seed"]) + 1,
            int(config["bootstrap_samples"]),
            config["bootstrap_ci"],
        )
        result[split] = {"S_forget": forget_s, "delta_nll_retain": retain_nll}
    return result


def retain_cap(baseline_rows, config):
    means = author_means(baseline_rows, "calibration", "retain", "nll_mean")
    values = np.asarray([means[author] for author in sorted(means)], dtype=np.float64)
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return {"cap": float(2.0 * se), "baseline_mean": float(values.mean()), "baseline_se": se}


def eligible(record, cap):
    forget = record["calibration"]["S_forget"]
    retain = record["calibration"]["delta_nll_retain"]
    return bool(forget["ci_low"] > 0.0 and retain["mean"] <= cap)


def job_path(output_dir, name):
    return Path(output_dir) / "forwards" / f"{name}.jsonl"


def temp_name(kind, temperature):
    label = str(temperature).replace(".", "p")
    return f"{kind}_tau{label}"


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def select(config):
    output_dir = Path(config["output_dir"])
    baseline_path = job_path(output_dir, "T00")
    if not baseline_path.exists():
        raise RuntimeError(f"missing baseline forward: {baseline_path}")
    baseline = load_forward(baseline_path)
    cap_info = retain_cap(baseline, config)
    cap = cap_info["cap"]
    attention = []
    for temperature in config["attention_temperature_grid"]:
        path = job_path(output_dir, temp_name("T10", temperature))
        if not path.exists():
            raise RuntimeError(f"missing {path}")
        record = condition_record(load_forward(path), baseline, config)
        record["temperature"] = float(temperature)
        record["eligible_on_calibration"] = eligible(record, cap)
        attention.append(record)
    mlp = []
    for temperature in config["mlp_temperature_grid"]:
        path = job_path(output_dir, temp_name("T01", temperature))
        if not path.exists():
            raise RuntimeError(f"missing {path}")
        record = condition_record(load_forward(path), baseline, config)
        record["temperature"] = float(temperature)
        record["eligible_on_calibration"] = eligible(record, cap)
        mlp.append(record)
    chosen_attention = [row for row in attention if row["eligible_on_calibration"]]
    tau_a = None if not chosen_attention else min(row["temperature"] for row in chosen_attention)
    tau_m = None
    intensity_matched = False
    if tau_a is not None:
        reference = next(row for row in attention if row["temperature"] == tau_a)
        target = reference["calibration"]["S_forget"]["mean"]
        floor = float(config["intensity_match_floor"])
        tolerance = float(config["intensity_match_relative"])
        ranked = []
        for row in mlp:
            if not row["eligible_on_calibration"]:
                continue
            gap = abs(row["calibration"]["S_forget"]["mean"] - target)
            relative = gap / max(abs(target), floor)
            ranked.append((relative, gap, row["temperature"]))
        if ranked:
            relative, _, tau_m = min(ranked)
            intensity_matched = bool(relative <= tolerance)
            if not intensity_matched:
                tau_m = None
    payload = {
        "assessed_at": now(),
        "retain_damage_cap": cap_info,
        "attention_grid": attention,
        "mlp_grid": mlp,
        "tau_A": tau_a,
        "tau_M": tau_m,
        "intensity_matched": intensity_matched,
        "rule": (
            "smallest calibration-eligible attention temperature; "
            "mlp temperature within 10% relative gold-logp drop and under the same retain cap"
        ),
    }
    write_json(output_dir / "e0_selection.json", payload)
    write_json(Path("/usr/local/ASU-unlearning/workspace/I/derived/e0_selection.json"), payload)
    return payload


def interaction(s11, s10, s01, s00):
    return s11 - s10 - s01 + s00


def gate(config, selection):
    output_dir = Path(config["output_dir"])
    decision = "do_not_start_e1"
    reasons = []
    holdout = None
    if selection["tau_A"] is None:
        reasons.append("no calibration attention temperature met forget and retain rules")
    elif not selection["intensity_matched"]:
        reasons.append("no calibration mlp temperature matched T10 within 10% at the retain cap")
        decision = "ambiguous_stop"
    else:
        baseline = load_forward(job_path(output_dir, "T00"))
        t10 = load_forward(job_path(output_dir, temp_name("T10", selection["tau_A"])))
        t01 = load_forward(job_path(output_dir, temp_name("T01", selection["tau_M"])))
        t11_path = job_path(output_dir, "T11")
        if not t11_path.exists():
            reasons.append("T11 forward is missing")
        else:
            t11 = load_forward(t11_path)
            cap = selection["retain_damage_cap"]["cap"]
            rec10 = condition_record(t10, baseline, config)
            rec01 = condition_record(t01, baseline, config)
            rec11 = condition_record(t11, baseline, config)
            holdout = {"T10": rec10["holdout"], "T01": rec01["holdout"], "T11": rec11["holdout"]}
            s00 = 0.0
            s10 = rec10["holdout"]["S_forget"]["mean"]
            s01 = rec01["holdout"]["S_forget"]["mean"]
            s11 = rec11["holdout"]["S_forget"]["mean"]
            noise = rec10["calibration"]["S_forget"]["se"]
            i_am = interaction(s11, s10, s01, s00)
            t01_ok = rec01["holdout"]["S_forget"]["ci_low"] > 0.0 and rec01["holdout"]["delta_nll_retain"]["mean"] <= cap
            t11_ok = s11 + 1e-12 >= s10 - noise and rec11["holdout"]["delta_nll_retain"]["mean"] <= cap
            gain = (s01 - s10) > noise or (s11 - s10) > noise
            if rec11["holdout"]["delta_nll_retain"]["mean"] > cap:
                reasons.append("T11 holdout retain delta exceeds cap")
            elif (not t01_ok) and s11 <= s10 + noise:
                reasons.append("matched T01 has no stable holdout forget effect and T11 does not beat T10")
            elif (t01_ok or t11_ok) and gain:
                decision = "eligible_for_e1_pending_approval"
                reasons.append("holdout matched-intensity rule passed; E1 still needs a separate approval")
            else:
                decision = "ambiguous_stop"
                reasons.append("holdout effect is inside calibration noise or disagrees with the pass rule")
            holdout["I_AM"] = i_am
            holdout["calibration_noise_se_T10"] = noise
    payload = {
        "stage": "e0_teacher_2x2",
        "assessed_at": now(),
        "decision": decision,
        "reasons": reasons,
        "tau_A": selection["tau_A"],
        "tau_M": selection["tau_M"],
        "intensity_matched": selection["intensity_matched"],
        "retain_damage_cap": selection["retain_damage_cap"],
        "holdout": holdout,
        "substitution_rate": "未运行",
        "generation_coherence": "未运行",
        "note": "I_AM is reported only as an interaction contrast. A positive value is not a success call.",
    }
    write_json(Path("/usr/local/ASU-unlearning/workspace/I/gates/e0.json"), payload)
    write_json(output_dir / "e0_gate.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["select", "gate"], required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.phase == "select":
        print(json.dumps(select(config), ensure_ascii=False, indent=2))
    else:
        selection_path = Path(config["output_dir"]) / "e0_selection.json"
        selection = json.loads(selection_path.read_text())
        print(json.dumps(gate(config, selection), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

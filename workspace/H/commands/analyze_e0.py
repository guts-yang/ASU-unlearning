#!/usr/bin/env python3
"""Apply the frozen E0 selection rule, then the holdout gate."""

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/usr/local/ASU-unlearning")


def load_forward(path):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    for row in rows:
        if row["n_tokens"] <= 0 or row["nll_mean"] is None:
            raise RuntimeError(f"missing answer NLL in {path} sample {row['sample_id']}")
    return rows


def author_means(rows, split, subset):
    grouped = {}
    for row in rows:
        if row["split"] == split and row["subset"] == subset:
            grouped.setdefault(int(row["author_index"]), []).append(float(row["nll_mean"]))
    if not grouped:
        raise RuntimeError(f"no rows for {split} {subset}")
    return {author: float(np.mean(values)) for author, values in grouped.items()}


def aligned_delta(intervention, baseline):
    authors = sorted(set(intervention) & set(baseline))
    if authors != sorted(baseline) or authors != sorted(intervention):
        raise RuntimeError("author ids differ between intervention and baseline")
    return np.asarray([intervention[author] - baseline[author] for author in authors], dtype=np.float64)


def bootstrap_mean_ci(values, seed, samples, quantiles):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.quantile(means, quantiles)
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_authors": int(len(values)),
    }


def retain_cap(baseline_rows, config):
    means = author_means(baseline_rows, "calibration", "retain")
    values = np.asarray([means[author] for author in sorted(means)], dtype=np.float64)
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return {"cap": float(2.0 * se), "baseline_mean": float(values.mean()), "baseline_se": se}


def layer_record(rows, baseline_rows, config, cap):
    result = {}
    for split in ("calibration", "holdout"):
        forget = bootstrap_mean_ci(
            aligned_delta(author_means(rows, split, "forget"), author_means(baseline_rows, split, "forget")),
            int(config["bootstrap_seed"]),
            int(config["bootstrap_samples"]),
            config["bootstrap_ci"],
        )
        retain = bootstrap_mean_ci(
            aligned_delta(author_means(rows, split, "retain"), author_means(baseline_rows, split, "retain")),
            int(config["bootstrap_seed"]) + 1,
            int(config["bootstrap_samples"]),
            config["bootstrap_ci"],
        )
        result[split] = {"delta_nll_forget": forget, "delta_nll_retain": retain}
    forget_cal = result["calibration"]["delta_nll_forget"]
    retain_cal = result["calibration"]["delta_nll_retain"]
    result["eligible_on_calibration"] = bool(forget_cal["ci_low"] > 0.0 and retain_cal["mean"] <= cap)
    return result


def choose_temperature(per_tau, config):
    counts = {tau: sum(1 for row in rows if row["eligible_on_calibration"]) for tau, rows in per_tau.items()}
    best_count = max(counts.values())
    winners = [tau for tau, count in counts.items() if count == best_count]
    tie = float(config["tie_temperature"])
    chosen = tie if tie in winners else sorted(winners)[0]
    return float(chosen), counts


def sample_random_subsets(selected, config):
    rng = np.random.default_rng(int(config["random_subset_seed"]))
    selected = sorted(int(layer) for layer in selected)
    width = len(selected)
    target = float(np.mean(selected))
    tolerance = float(config["random_mean_index_tolerance"])
    wanted = int(config["random_subset_count"])
    pool = [layer for layer in range(int(config["layer_count"])) if layer not in set(selected)]
    chosen = []
    seen = set()

    def take(draw, require_band):
        key = tuple(draw)
        if key in seen:
            return False
        if require_band and abs(float(np.mean(draw)) - target) > tolerance:
            return False
        seen.add(key)
        chosen.append(draw)
        return True

    for _ in range(400):
        if len(chosen) >= wanted or len(pool) < width:
            break
        take(sorted(int(value) for value in rng.choice(pool, size=width, replace=False)), True)
    for _ in range(400):
        if len(chosen) >= wanted or len(pool) < width:
            break
        take(sorted(int(value) for value in rng.choice(pool, size=width, replace=False)), False)
    universe = list(range(int(config["layer_count"])))
    banned = tuple(selected)
    for _ in range(400):
        if len(chosen) >= wanted:
            break
        draw = sorted(int(value) for value in rng.choice(universe, size=width, replace=False))
        if tuple(draw) == banned:
            continue
        take(draw, False)
    return chosen


def job_name(prefix, temperature):
    return f"{prefix}_tau{str(temperature).replace('.', 'p')}"


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["select", "gate"], required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    forward_dir = Path(config["output_dir"]) / "forwards"
    baseline = load_forward(forward_dir / "baseline_tau1.jsonl")
    cap_info = retain_cap(baseline, config)
    cap = cap_info["cap"]

    single = {}
    for layer in range(int(config["layer_count"])):
        single[layer] = {}
        for temperature in config["temperatures"]:
            name = job_name(f"single_L{layer:02d}", temperature)
            single[layer][float(temperature)] = layer_record(load_forward(forward_dir / f"{name}.jsonl"), baseline, config, cap)

    per_tau = {}
    for temperature in config["temperatures"]:
        temperature = float(temperature)
        per_tau[temperature] = []
        for layer in range(int(config["layer_count"])):
            record = single[layer][temperature]
            per_tau[temperature].append({
                "layer": layer,
                "eligible_on_calibration": record["eligible_on_calibration"],
                "calibration_delta_nll_forget": record["calibration"]["delta_nll_forget"]["mean"],
            })
    shared_tau, counts = choose_temperature(per_tau, config)
    selected = sorted(layer["layer"] for layer in per_tau[shared_tau] if layer["eligible_on_calibration"])
    selection = {
        "assessed_at": now(),
        "retain_damage_cap": cap_info,
        "eligible_counts": {str(tau): count for tau, count in counts.items()},
        "shared_temperature": shared_tau,
        "selected_layers": selected,
        "rule": "calibration forget CI lower bound > 0 and calibration retain mean delta <= cap; shared tau is the grid value with more eligible layers, tie 2.3",
    }

    if args.phase == "select":
        jobs = []
        if selected:
            jobs.append({"name": job_name("joint_Lstar", shared_tau), "attention_temp": shared_tau, "layers_id": selected})
            for index, layers in enumerate(sample_random_subsets(selected, config)):
                jobs.append({
                    "name": job_name(f"random_{index:02d}", shared_tau),
                    "attention_temp": shared_tau,
                    "layers_id": layers,
                })
        selection["phase2_jobs"] = jobs
        selection["phase2_complete"] = len(jobs) == 0 or len(jobs) == 1 + int(config["random_subset_count"])
        write_json(ROOT / "workspace/H/derived/e0_selection.json", selection)
        write_json(Path(config["output_dir"]) / "phase2_jobs.json", jobs)
        print(json.dumps({"shared_temperature": shared_tau, "selected_layers": selected, "phase2_jobs": len(jobs), "cap": cap}, indent=2))
        return

    saved = json.loads((ROOT / "workspace/H/derived/e0_selection.json").read_text())
    if saved["selected_layers"] != selected or float(saved["shared_temperature"]) != shared_tau:
        raise RuntimeError("selection changed after it was frozen")

    controls = {}
    for temperature in config["temperatures"]:
        for prefix, layers in (
            ("control_L6_8", [int(layer) for layer in config["fixed_control_layers"]]),
            ("control_all", None),
        ):
            name = job_name(prefix, temperature)
            controls[name] = {
                "layers_id": layers,
                "attention_temp": float(temperature),
                **layer_record(load_forward(forward_dir / f"{name}.jsonl"), baseline, config, cap),
            }

    reasons = []
    joint = None
    random_rows = []
    if not selected:
        reasons.append("no calibration layer met the forget and retain rule")
    else:
        joint_name = job_name("joint_Lstar", shared_tau)
        joint = layer_record(load_forward(forward_dir / f"{joint_name}.jsonl"), baseline, config, cap)
        for layer in selected:
            holdout = single[layer][shared_tau]["holdout"]
            if holdout["delta_nll_forget"]["ci_low"] <= 0.0:
                reasons.append(f"layer {layer} forget response did not replicate on holdout")
            if holdout["delta_nll_retain"]["mean"] > cap:
                reasons.append(f"layer {layer} retain damage exceeded the calibration cap on holdout")
        if joint["holdout"]["delta_nll_forget"]["ci_low"] <= 0.0:
            reasons.append("joint forget response did not replicate on holdout")
        if joint["holdout"]["delta_nll_retain"]["mean"] > cap:
            reasons.append("joint retain damage exceeded the calibration cap on holdout")
        for index in range(int(config["random_subset_count"])):
            name = job_name(f"random_{index:02d}", shared_tau)
            record = layer_record(load_forward(forward_dir / f"{name}.jsonl"), baseline, config, cap)
            random_rows.append({"name": name, **record})
            if joint["holdout"]["delta_nll_forget"]["mean"] <= record["holdout"]["delta_nll_forget"]["mean"]:
                reasons.append(f"joint forget response did not beat {name}")

    decision = "start_e1" if not reasons else "do_not_start_e1"
    single_summary = []
    for layer in range(int(config["layer_count"])):
        for temperature in config["temperatures"]:
            record = single[layer][float(temperature)]
            single_summary.append({
                "layer": layer,
                "temperature": float(temperature),
                "eligible_on_calibration": record["eligible_on_calibration"],
                "calibration_delta_nll_forget": record["calibration"]["delta_nll_forget"],
                "calibration_delta_nll_retain": record["calibration"]["delta_nll_retain"],
                "holdout_delta_nll_forget": record["holdout"]["delta_nll_forget"],
                "holdout_delta_nll_retain": record["holdout"]["delta_nll_retain"],
            })
    gate = {
        "stage": "e0_teacher_probe",
        "assessed_at": now(),
        "decision": decision,
        "reasons": reasons,
        "model_id": config["model_id"],
        "model_path": config["model_path"],
        "shared_temperature": shared_tau,
        "selected_layers": selected,
        "retain_damage_cap": cap_info,
        "joint": joint,
        "random_controls": random_rows,
        "fixed_controls": controls,
    }
    write_json(ROOT / "workspace/H/gates/e0.json", {key: gate[key] for key in ("stage", "assessed_at", "decision", "reasons", "shared_temperature", "selected_layers", "retain_damage_cap")})
    write_json(ROOT / "workspace/H/derived/e0_summary.json", {"gate": gate, "single_layers": single_summary})
    print(json.dumps({"decision": decision, "reasons": reasons, "selected_layers": selected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Wire alignment, patch effects, sparse selection, rho, and serialization."""

from __future__ import annotations

import subprocess
from pathlib import Path

from I2.mcsd.alignment import check_alignment, questions_present
from I2.mcsd.bootstrap import bootstrap_from_rows
from I2.mcsd.config import config_sha256
from I2.mcsd.effects import control_suite, interaction_diagnostic
from I2.mcsd.gates import localization_gate, retain_gate
from I2.mcsd.hooks import layer_count, parse_path
from I2.mcsd.patch import capture_clean, freeze_model
from I2.mcsd.serialize import write_run
from I2.mcsd.sparse import select_binary
from I2.mcsd.teacher import run_rho_sweep

_REPO = Path(__file__).resolve().parents[3]


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_metadata(model, config, tokenizer_name, seed):
    return {
        "model_id": getattr(model, "mcsd_model_id", type(model).__name__),
        "layer_count": layer_count(model),
        "tokenizer": tokenizer_name,
        "attention_temperature": None,
        "parameters_frozen": True,
        "seed": seed,
        "code_version": git_head(),
        "config_version": config.get("version"),
        "config_sha256": config_sha256(config),
        "path_unit": "module,layer,token,head",
        "real_model_executed": False,
        "flash_attention_boundary": "not_run",
        "e0_decision_preserved": "do_not_start_e1",
        "method_effectiveness_claimed": False,
        "shapes": [],
    }


def _teacher_verdict(config, sweep):
    gates = config.get("gates") or {}
    judge_rho = gates.get("retain_rho")
    if gates.get("retain_nll_cap") is None or judge_rho is None:
        return {"pass": None, "status": "thresholds_missing"}
    if not sweep:
        return {"pass": None, "status": "metrics_missing"}
    match = next((row for row in sweep if row["rho"] == float(judge_rho)), None)
    if match is None:
        return {"pass": None, "status": "rho_not_in_grid"}
    verdict = retain_gate(match.get("delta_retain_nll"), config)
    verdict["judged_rho"] = float(judge_rho)
    return verdict


def _public_rho(rho_result):
    if not rho_result or not rho_result.get("sweep"):
        return rho_result
    public = dict(rho_result)
    public["sweep"] = []
    for row in rho_result["sweep"]:
        copied = dict(row)
        copied.pop("teacher_logits", None)
        public["sweep"].append(copied)
    return public


def run_examples(model, examples, config, loaded_rows, tokenizer_name="fake", output_dir=None, wrong_donors=None):
    """Run the candidate measurement path. This does not launch Student E1."""
    freeze_model(model)
    seed = config.get("control_seed")
    metadata = build_metadata(model, config, tokenizer_name, seed)
    excluded = []
    measured = []
    donors = wrong_donors or {}
    for example in examples:
        aligned, reasons = check_alignment(
            example["prompt_plus_ids"],
            example["prompt_minus_ids"],
            example.get("fact_span"),
            example["score_plus_ids"],
            example["score_minus_ids"],
        )
        missing = questions_present(
            [example["question_plus"], example["question_minus"]],
            loaded_rows,
        )
        if missing:
            reasons = list(reasons) + ["row_not_in_loaded_records"]
            aligned = False
        if not aligned:
            excluded.append(
                {
                    "sample_id": example["sample_id"],
                    "author_id": example["author_id"],
                    "split": example["split"],
                    "status": "excluded",
                    "reasons": reasons,
                    "N": None,
                    "R": None,
                    "controls": None,
                    "rho": None,
                    "score_scope": None,
                }
            )
            continue
        example["cache_plus"] = capture_clean(model, example["prompt_plus"])
        example["cache_minus"] = capture_clean(model, example["prompt_minus"])
        if not metadata["shapes"]:
            metadata["shapes"] = list(example["cache_plus"].shapes)
        selection = select_binary(model, example, config, example["split"])
        paths = [parse_path(item) for item in selection["binary_paths"]]
        retain_example = example.get("retain_example")
        retain_reason = example.get("retain_reason")
        if retain_example is not None:
            retain_ok, retain_reasons = check_alignment(
                retain_example["prompt_plus_ids"],
                retain_example["prompt_minus_ids"],
                retain_example.get("fact_span"),
                retain_example["score_plus_ids"],
                retain_example["score_minus_ids"],
            )
            if not retain_ok:
                retain_example = None
                retain_reason = "retain pair excluded: " + ",".join(retain_reasons)
            elif retain_example.get("cache_plus") is None:
                retain_example["cache_plus"] = capture_clean(model, retain_example["prompt_plus"])
                retain_example["cache_minus"] = capture_clean(model, retain_example["prompt_minus"])
        controls = None
        if paths and seed is not None:
            controls = control_suite(
                model,
                example,
                paths,
                [parse_path(item) for item in config["selection"]["candidate_pool"]],
                seed,
                wrong_donor=donors.get(example["sample_id"]),
                retain_example=retain_example,
                retain_reason=retain_reason,
            )
        elif seed is None:
            controls = {
                "no_swap": {"status": "skipped", "reason": "control_seed is missing", "N": None, "R": None},
                "random": {"status": "skipped", "reason": "control_seed is missing", "N": None, "R": None},
                "wrong_donor": {"status": "skipped", "reason": "control_seed is missing", "N": None, "R": None},
                "retain": {"status": "skipped", "reason": "control_seed is missing", "N": None, "R": None},
            }
        else:
            controls = {
                "no_swap": {"status": "skipped", "reason": "empty path set", "N": None, "R": None},
                "random": {"status": "skipped", "reason": "empty path set", "N": None, "R": None},
                "wrong_donor": {"status": "skipped", "reason": "empty path set", "N": None, "R": None},
                "retain": {"status": "skipped", "reason": retain_reason or "empty path set", "N": None, "R": None},
            }
        if config.get("compute_interaction"):
            interaction = interaction_diagnostic(model, example, paths)
        else:
            interaction = {"status": "unavailable", "value": None}
        rho = run_rho_sweep(
            model,
            example,
            paths,
            config,
            retain_example=retain_example,
            retain_reason=retain_reason,
        )
        effects = selection["binary_effects"]
        measured.append(
            {
                "sample_id": example["sample_id"],
                "author_id": example["author_id"],
                "split": example["split"],
                "status": "measured",
                "reasons": [],
                "score_scope": effects.get("score_scope"),
                "selection_method": selection["method"],
                "paths": selection["binary_paths"],
                "N": effects.get("N"),
                "R": effects.get("R"),
                "computed_by": effects.get("computed_by"),
                "controls": controls,
                "interaction": interaction,
                "rho": _public_rho(rho),
                "localization_gate": localization_gate(effects.get("N"), effects.get("R"), config),
                "teacher_retain_gate": _teacher_verdict(config, rho.get("sweep")),
                "shapes": list(example["cache_plus"].shapes),
            }
        )
    summary = {
        "metadata": metadata,
        "excluded": excluded,
        "samples": measured,
        "author_bootstrap": bootstrap_from_rows(
            measured,
            "N",
            min_authors=config.get("min_authors_for_interval"),
            samples=config.get("bootstrap_samples") or 0,
            seed=config.get("bootstrap_seed") or 0,
        ),
        "real_model_executed": False,
        "student_e1_started": False,
        "method_effectiveness_claimed": False,
        "e0_decision_preserved": "do_not_start_e1",
    }
    if output_dir is not None:
        write_run(output_dir, summary, excluded + measured)
    return summary

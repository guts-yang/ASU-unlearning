"""Eval-only Llama hook smoke test. Does not localize, select rho, or train."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
_REPO = _WORKSPACE.parent
_RBF = _REPO / "Right-to-be-forgotten"
for _path in (str(_WORKSPACE), str(_RBF)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from I2.mcsd.config import load_config  # noqa: E402
from I2.mcsd.data import read_records  # noqa: E402
from I2.mcsd.errors import HookBoundaryError  # noqa: E402
from I2.mcsd.gates import student_block_reason  # noqa: E402
from I2.mcsd.hooks import ActivationCache, module_layer_index  # noqa: E402
from I2.mcsd.patch import capture_clean, freeze_model, patched_logits  # noqa: E402
from I2.mcsd.storage import measurement_dir  # noqa: E402

MODEL_DIR = Path("/root/autodl-tmp/models/tofu_ft_llama2-7b")
FORGET_PATH = _RBF / "data" / "tofu" / "forget05.json"
CANDIDATE = Path(__file__).resolve().parents[1] / "configs" / "mcsd_candidate.yaml"
BACKENDS = ("eager", "sdpa", "flash_attention_2")
SAMPLE_ID = "forget05_task1_0000"


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    protocol_dir = _mkdir(measurement_dir("protocol", stamp))
    smoke_dir = _mkdir(measurement_dir("smoke", stamp))
    localization_dir = _mkdir(measurement_dir("localization", stamp))
    teacher_dir = _mkdir(measurement_dir("teacher", stamp))

    config = load_config(CANDIDATE)
    protocol = protocol_status(config)
    _write(protocol_dir / "summary.json", protocol)
    smoke = run_smoke(config)
    smoke["protocol_dir"] = str(protocol_dir)
    smoke["student_e1_started"] = False
    smoke["optimizer_steps"] = 0
    _write(smoke_dir / "summary.json", smoke)

    localization = blocked_localization(protocol)
    _write(localization_dir / "summary.json", localization)
    teacher = blocked_teacher(config, protocol, localization, smoke)
    _write(teacher_dir / "summary.json", teacher)
    sys.stdout.write(
        json.dumps(
            {
                "protocol": str(protocol_dir),
                "smoke": str(smoke_dir),
                "localization": str(localization_dir),
                "teacher": str(teacher_dir),
                "smoke_passed": smoke["smoke_passed"],
                "student_e1_started": False,
            }
        )
        + "\n"
    )
    return 0 if smoke["smoke_passed"] else 1


def protocol_status(config):
    configs = sorted(path.name for path in CANDIDATE.parent.glob("*.yaml"))
    gates = config.get("gates") or {}
    missing = []
    for name in (
        "pairing.mode",
        "selection.method",
        "selection.candidate_pool",
        "alpha",
        "gates.localization_n_min",
        "gates.localization_r_min",
        "gates.retain_nll_cap",
        "gates.retain_rho",
        "gates.localization_result_id",
        "gates.teacher_gate_id",
    ):
        section, _, key = name.partition(".")
        value = config.get(section, {}).get(key) if key else config.get(name)
        if value is None:
            missing.append(name)
    if gates.get("preregistered") is not True:
        missing.append("gates.preregistered")
    if (config.get("rho") or {}).get("grid_status") != "frozen":
        missing.append("rho.grid_status")
    return {
        "approved_config_present": not missing,
        "candidate_config": str(CANDIDATE),
        "config_sha256": config.get("_config_sha256"),
        "yaml_files_in_i2_configs": configs,
        "allow_student_e1": config.get("allow_student_e1"),
        "e0_decision_preserved": config.get("e0_decision_preserved"),
        "missing_fields": missing,
        "localization_started": False,
        "reason": "no separately approved protocol config; localization stays stopped",
        "e0_retain_cap_reused": False,
        "code_version": git_head(),
    }


def run_smoke(config):
    import torch

    report = {
        "real_model_executed": False,
        "eval_only": True,
        "optimizer_steps": 0,
        "causal_localization_claimed": False,
        "method_effectiveness_claimed": False,
        "model_dir": str(MODEL_DIR),
        "config_sha256": config.get("_config_sha256"),
        "code_version": git_head(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [],
        "backends": [],
        "smoke_passed": False,
    }
    if not torch.cuda.is_available():
        report["status"] = "cuda_unavailable"
        return report
    report["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    donor = donor_batch()
    report["sample_id"] = SAMPLE_ID
    report["donor_length"] = int(donor["input_ids"].shape[1])
    report["answer_tokens_in_donor"] = False
    passed = []
    for backend in BACKENDS:
        passed.append(run_backend(backend, donor, report))
    report["real_model_executed"] = any(row.get("forward_executed") for row in report["backends"])
    report["smoke_passed"] = all(passed)
    report["status"] = "passed" if report["smoke_passed"] else "backend_failed"
    return report


def donor_batch():
    import torch
    import yaml
    from transformers import AutoTokenizer

    tags = yaml.load(
        (_RBF / "config" / "model_config.yaml").read_text(),
        Loader=yaml.FullLoader,
    )["llama2-7b"]
    row = next(record for record in read_records(FORGET_PATH) if str(record.get("task_id")) == "1")
    question = tags["question_start_tag"] + row["question"] + tags["question_end_tag"]
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    encoded = tokenizer(question, add_special_tokens=True, return_tensors="pt")
    if tags["answer_tag"] + row["answer"] in question:
        raise RuntimeError("answer text was included in the donor question")
    return {
        "input_ids": encoded["input_ids"].to("cuda:0"),
        "attention_mask": encoded["attention_mask"].to("cuda:0"),
        "sample_ids": [SAMPLE_ID],
        "attention_temp": None,
        "question": row["question"],
    }


def run_backend(backend, donor, report):
    import gc
    import torch

    row = {"backend": backend, "forward_executed": False, "optimizer_steps": 0}
    report["backends"].append(row)
    model = None
    try:
        from my_models.my_llama import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            str(MODEL_DIR),
            torch_dtype=torch.bfloat16,
            attn_implementation=backend,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        model.to("cuda:0")
        freeze_model(model)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("smoke model still has trainable parameters")
        clean = capture_clean(model, donor)
        clean_logits = _logits(model, donor)
        zero_logits, zero_editor = patched_logits(
            model, donor, [("attention", 0, 0, 0)], clean, clean, strength=0.0
        )
        donor_cache = _clone_cache(clean)
        donor_cache.attention[(SAMPLE_ID, 0)][0, 0, :] += 1
        captured = {}
        one_logits, one_editor = _patched_with_capture(
            model, donor, clean, donor_cache, captured
        )
        shapes = _shape_summary(clean.shapes)
        attention_layers = {item["layer"] for item in clean.shapes if item["kind"] == "attention"}
        live = captured["live"]
        written = captured["written"]
        site_delta = _max_abs(written[:, 0, 0, :], live[:, 0, 0, :])
        other_heads = _max_abs(written[:, 0, 1:, :], live[:, 0, 1:, :])
        other_tokens = _max_abs(written[:, 1:, :, :], live[:, 1:, :, :])
        zero_delta = (zero_logits.float() - clean_logits.float()).abs().max().item()
        one_delta = (one_logits.float() - clean_logits.float()).abs().max().item()
        attention_shape = next(item for item in shapes if item["kind"] == "attention" and item["layer"] == 0)
        mlp_shape = next(item for item in shapes if item["kind"] == "mlp" and item["layer"] == 0)
        expected_layout = "flash" if backend == "flash_attention_2" else "bhsd"
        checks = {
            "attention_layout": attention_shape["layout"] == expected_layout,
            "attention_raw_rank": len(attention_shape["shape"]) == 4,
            "mlp_layout": mlp_shape["layout"] == "bth",
            "mlp_rank": len(mlp_shape["shape"]) == 3,
            "strength_zero_identical": zero_delta == 0.0 and zero_editor.last_written is None,
            "patched_site_changed": site_delta > 0.0,
            "other_heads_unchanged": other_heads == 0.0,
            "other_tokens_unchanged": other_tokens == 0.0,
            "downstream_logits_changed": one_delta > 0.0,
            "layer_count": len(attention_layers) == 32,
        }
        row.update(
            {
                "forward_executed": True,
                "attn_implementation": model.config._attn_implementation,
                "dtype": str(next(model.parameters()).dtype),
                "shapes_layer0": [attention_shape, mlp_shape],
                "strength_zero_max_abs": zero_delta,
                "strength_one_logit_max_abs": one_delta,
                "patched_site_max_abs": site_delta,
                "other_heads_max_abs": other_heads,
                "other_tokens_max_abs": other_tokens,
                "checks": checks,
                "passed": all(checks.values()),
                "hook_boundary_error": None,
            }
        )
        return row["passed"]
    except HookBoundaryError as exc:
        row["hook_boundary_error"] = str(exc)
        row["passed"] = False
        row["relabelled_backend"] = False
        return False
    except Exception as exc:
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
        row["passed"] = False
        row["relabelled_backend"] = False
        return False
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _logits(model, batch):
    import torch

    with torch.no_grad():
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            attention_temp=None,
            use_cache=False,
        )
    return output.logits


def _clone_cache(cache):
    cloned = ActivationCache()
    cloned.attention = {key: value.clone() for key, value in cache.attention.items()}
    cloned.mlp = {key: value.clone() for key, value in cache.mlp.items()}
    cloned.shapes = list(cache.shapes)
    cloned.sample_ids = list(cache.sample_ids)
    return cloned


def _patched_with_capture(model, batch, source, donor, captured):
    from I2.mcsd.hooks import ActivationEditor, bind_editor, restore_editor
    from I2.mcsd.patch import module_logits

    editor = ActivationEditor(
        mode="patch",
        sample_ids=list(batch["sample_ids"]),
        paths=[("attention", 0, 0, 0)],
        source=source,
        donor=donor,
        strength=1.0,
    )
    original = editor.__call__

    def wrapped(kind, tensor, module, layout):
        result = original(kind, tensor, module, layout)
        if (
            kind == "attention"
            and module_layer_index(module) == 0
            and editor.last_written is not None
            and "written" not in captured
        ):
            captured["live"] = editor.snapshots["attention"]["live"].detach().clone()
            captured["written"] = editor.last_written.detach().clone()
        return result

    editor.__call__ = wrapped
    saved = bind_editor(model, editor)
    try:
        import torch

        with torch.no_grad():
            logits = module_logits(model, batch)
    finally:
        restore_editor(saved)
    if "written" not in captured:
        raise RuntimeError("layer 0 attention patch was not observed")
    return logits, editor


def _max_abs(left, right):
    if left.numel() == 0 or right.numel() == 0:
        return 0.0
    return (left.float() - right.float()).abs().max().item()


def _shape_summary(shapes):
    kept = []
    seen = set()
    for item in shapes:
        if item["layer"] not in (0, 31):
            continue
        key = (item["kind"], item["layer"], item["layout"], tuple(item["shape"]), tuple(item["canonical_shape"]))
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def blocked_localization(protocol):
    return {
        "real_model_executed": False,
        "localization_started": False,
        "reason": protocol["reason"],
        "approved_q_minus_present": False,
        "pairs_constructed": 0,
        "N": None,
        "R": None,
        "controls": None,
        "binary_S_frozen": False,
        "author_interval_unstable": True,
        "calibration_author_count": 5,
        "min_authors_for_interval": 8,
        "matched_retain_constructable": False,
        "matched_retain_reason": "retain50 has no approved q+/q- fact span",
        "e0_retain_cap_reused": False,
    }


def blocked_teacher(config, protocol, localization, smoke):
    return {
        "student_e1_started": False,
        "forget_py_called": False,
        "smoke_passed": smoke.get("smoke_passed"),
        "rho_sweep_executed": False,
        "holdout_used": False,
        "frozen_S_present": localization["binary_S_frozen"],
        "rho_grid_status": (config.get("rho") or {}).get("grid_status"),
        "retain_rho": (config.get("gates") or {}).get("retain_rho"),
        "reason": "no frozen S and no approved retain rho; holdout was not used",
        "protocol_missing": protocol["missing_fields"],
        "training_conditions": [
            {"condition": "frozen approved gate", "satisfied": False},
            {"condition": "hook smoke, alignment, and patch", "satisfied": False},
            {"condition": "calibration localization and controls", "satisfied": False},
            {"condition": "rho passes retain delta-NLL gate", "satisfied": False},
            {"condition": "student config points at the new gate", "satisfied": False},
            {"condition": "trainer wiring verified", "satisfied": False},
        ],
        "student_block_reason": student_block_reason(config),
    }


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO)).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _mkdir(path):
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())

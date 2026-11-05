"""CPU preflight for MCSD-ASU. No Llama weights, no patch search, no training.

Writes an environment record, a task-1 token manifest, an alignment exclusion
audit, the unset protocol fields, and the Student E1 refusal. Measurements that
require a forward stay null.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
_REPO = _WORKSPACE.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from I2.commands import student_entry  # noqa: E402
from I2.mcsd.alignment import check_alignment, questions_present  # noqa: E402
from I2.mcsd.config import load_config  # noqa: E402
from I2.mcsd.data import (  # noqa: E402
    author_index_for_row,
    read_records,
    split_for_author,
)
from I2.mcsd.errors import StorageError, StudentE1Blocked  # noqa: E402
from I2.mcsd.gates import (  # noqa: E402
    E0_DECISION,
    E0_RETAIN_CAP,
    localization_gate,
    retain_gate,
    student_block_reason,
)
from I2.mcsd.sparse import GREEDY_KEYS, PGB_KEYS  # noqa: E402
from I2.mcsd.storage import measurement_dir, reject_forbidden  # noqa: E402

MODEL_DIR = Path("/root/autodl-tmp/models/tofu_ft_llama2-7b")
FORGET_PATH = _REPO / "Right-to-be-forgotten" / "data" / "tofu" / "forget05.json"
RETAIN_PATH = _REPO / "Right-to-be-forgotten" / "data" / "tofu" / "retain50.json"
PERTURBED_PATH = _REPO / "Right-to-be-forgotten" / "data" / "tofu" / "forget05_perturbed.json"
MODEL_CONFIG_PATH = _REPO / "Right-to-be-forgotten" / "config" / "model_config.yaml"
CANDIDATE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "mcsd_candidate.yaml"
# forget.py builds TextForgetDatasetQA with this length. It is a record of the
# existing formatter, not a new training hyperparameter.
SCORE_MAX_LENGTH = 500
MODEL_FAMILY = "llama2-7b"


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    output_dir = _output_dir(arguments)
    output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(CANDIDATE_CONFIG)
    environment = collect_environment()
    _write_json(output_dir / "environment.json", environment)
    (output_dir / "config_snapshot.yaml").write_text(CANDIDATE_CONFIG.read_text())

    manifest = build_task_manifest(config)
    _write_jsonl(output_dir / "task1_samples.jsonl", manifest["rows"])
    _write_json(output_dir / "alignment_audit.json", manifest["audit"])
    _write_json(output_dir / "retain_inventory.json", manifest["retain"])
    gaps = protocol_gaps(config)
    _write_json(output_dir / "protocol_gaps.json", gaps)
    blocked = student_preflight(config)
    _write_json(output_dir / "student_block.json", blocked)

    summary = {
        "real_model_executed": False,
        "llama_for_causal_lm_instantiated": False,
        "optimizer_step": False,
        "student_e1_started": False,
        "method_effectiveness_claimed": False,
        "causal_localization_claimed": False,
        "config_version": config.get("version"),
        "config_sha256": config.get("_config_sha256"),
        "code_version": environment["git"]["head"],
        "cuda_available": environment["torch"]["cuda_available"],
        "task1_rows": manifest["audit"]["task1_rows"],
        "task1_excluded": manifest["audit"]["excluded"],
        "task1_aligned": manifest["audit"]["aligned"],
        "protocol_ready_for_experiment": gaps["ready_for_real_experiment"],
        "student_e1_block_reason": blocked["student_entry"]["stderr"],
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "summary.json", summary)
    sys.stdout.write(str(output_dir) + "\n")
    return 0


def _output_dir(arguments):
    if arguments:
        path = Path(arguments[0])
        reject_forbidden(path)
        if path.exists():
            raise StorageError("refusing to overwrite %s" % path)
        return path
    return measurement_dir("pre_experiment")


def collect_environment():
    import torch
    import transformers

    packages = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    try:
        import flash_attn

        packages["flash_attn"] = flash_attn.__version__
    except Exception as exc:  # pragma: no cover - environment dependent
        packages["flash_attn"] = "IMPORT_FAIL %s: %s" % (type(exc).__name__, exc)
    nvidia_smi = Path("/usr/bin/nvidia-smi")
    return {
        "collected_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "packages": packages,
        "torch": {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "version_cuda": torch.version.cuda,
        },
        "nvidia_smi": {
            "path": str(nvidia_smi),
            "exists": nvidia_smi.exists(),
            "size_bytes": nvidia_smi.stat().st_size if nvidia_smi.exists() else None,
            "executed": False,
        },
        "device_nodes_created": False,
        "packages_installed": False,
        "git": git_record(),
        "files": {
            "candidate_config": file_record(CANDIDATE_CONFIG),
            "forget05": file_record(FORGET_PATH),
            "retain50": file_record(RETAIN_PATH),
            "forget05_perturbed": file_record(PERTURBED_PATH),
            "model_config_json": file_record(MODEL_DIR / "config.json"),
            "generation_config": file_record(MODEL_DIR / "generation_config.json"),
            "tokenizer_json": file_record(MODEL_DIR / "tokenizer.json"),
            "tokenizer_config": file_record(MODEL_DIR / "tokenizer_config.json"),
            "special_tokens_map": file_record(MODEL_DIR / "special_tokens_map.json"),
        },
        "weight_files_hashed": False,
        "model_dir": str(MODEL_DIR),
    }


def git_record():
    def run(args):
        return subprocess.check_output(args, cwd=str(_REPO), stderr=subprocess.DEVNULL).decode().strip()

    status = subprocess.check_output(["git", "status", "--short"], cwd=str(_REPO)).decode()
    branch = subprocess.check_output(["git", "status", "-sb"], cwd=str(_REPO)).decode().splitlines()
    return {
        "head": run(["git", "rev-parse", "HEAD"]),
        "branch_status": branch[0] if branch else "",
        "status_short_before_results": status,
        "ahead_behind_origin_fuxian": run(["git", "rev-list", "--left-right", "--count", "origin/fuxian...HEAD"]),
    }


def file_record(path):
    data = Path(path).read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def build_task_manifest(config):
    tokenizer, tags = load_tokenizer_and_tags()
    forget_rows = read_records(FORGET_PATH)
    task_rows = [(index, row) for index, row in enumerate(forget_rows) if str(row.get("task_id")) == "1"]
    per_author = int(config["questions_per_author"])
    if len(task_rows) % per_author != 0:
        raise RuntimeError("task 1 row count is not divisible by questions_per_author")
    loaded = [row for _index, row in task_rows]
    records = []
    reason_counts = Counter()
    author_counts = Counter()
    split_counts = Counter()
    for task_index, (source_index, row) in enumerate(task_rows):
        author_id = author_index_for_row(task_index, per_author)
        split = split_for_author(
            author_id,
            config["calibration_author_indices"],
            config["holdout_author_indices"],
        )
        encoded = encode_row(tokenizer, row["question"], row["answer"], tags)
        alignment = exclude_unpaired(encoded)
        if not encoded["donor_ids_equal_score_prefix"]:
            alignment["reasons"].append("score_prefix_boundary_mismatch")
            alignment["answer_span_reliable"] = False
        missing = questions_present([row["question"]], loaded)
        if missing:
            alignment["reasons"].append("row_not_in_loaded_records")
            alignment["aligned"] = False
        for reason in alignment["reasons"]:
            reason_counts[reason] += 1
        author_counts[str(author_id)] += 1
        split_counts[split] += 1
        records.append(
            {
                "sample_id": "forget05_task1_%04d" % task_index,
                "source_path": str(FORGET_PATH),
                "source_row_index": source_index,
                "task_id": str(row["task_id"]),
                "author_id": author_id,
                "split": split,
                "question": row["question"],
                "question_in_loaded_records": not missing,
                "donor_input_ids": encoded["donor_input_ids"],
                "donor_attention_mask": encoded["donor_attention_mask"],
                "donor_contains_answer_text": False,
                "score_input_ids": encoded["score_input_ids"],
                "score_attention_mask": encoded["score_attention_mask"],
                "score_labels": encoded["score_labels"],
                "prompt_token_span": encoded["prompt_token_span"],
                "answer_token_span": encoded["answer_token_span"],
                "pad_count": encoded["pad_count"],
                "truncated": encoded["truncated"],
                "num_question_tokens": encoded["num_question_tokens"],
                "donor_ids_equal_score_prefix": encoded["donor_ids_equal_score_prefix"],
                "answer_tokens_in_donor": encoded["answer_tokens_in_donor"],
                "answer_span_reliable": encoded["donor_ids_equal_score_prefix"],
                "label_ignore_index": -100,
                "alignment": alignment,
                "N": None,
                "R": None,
                "controls": None,
                "rho": None,
            }
        )
    calibration_authors = sorted(set(config["calibration_author_indices"]))
    audit = {
        "task1_rows": len(records),
        "forget05_rows": len(forget_rows),
        "aligned": sum(1 for row in records if row["alignment"]["aligned"]),
        "excluded": sum(1 for row in records if not row["alignment"]["aligned"]),
        "exclusion_reason_counts": dict(reason_counts),
        "author_row_counts": dict(author_counts),
        "split_row_counts": dict(split_counts),
        "questions_per_author": per_author,
        "author_count": len(task_rows) // per_author,
        "calibration_author_indices": calibration_authors,
        "holdout_author_indices": list(config["holdout_author_indices"]),
        "calibration_author_count": len(calibration_authors),
        "min_authors_for_interval": config.get("min_authors_for_interval"),
        "author_interval_unstable": len(calibration_authors) < int(config["min_authors_for_interval"]),
        "independent_sample_unit": "author",
        "qa_rows_are_not_independent_samples": True,
        "pairs_constructed": 0,
        "perturbed_answers_used_as_donors": False,
        "shared_token_index_alignment": False,
        "score_prefix_boundary_mismatches": reason_counts.get("score_prefix_boundary_mismatch", 0),
        "score_prefix_boundary_note": (
            "On a mismatch, joint question+answer tokenization fused the template boundary "
            "with the first answer character. The donor sequence stays question-only. "
            "Its token index is not reused as the answer span."
        ),
        "donor_forwards_executed": False,
        "score_formatter": {
            "source": "Right-to-be-forgotten/dataset/data_module.py convert_raw_data_to_model_format",
            "max_length": SCORE_MAX_LENGTH,
            "max_length_source": "Right-to-be-forgotten/forget.py TextForgetDatasetQA max_length=500",
            "model_family": MODEL_FAMILY,
            "question_start_tag": tags["question_start_tag"],
            "question_end_tag": tags["question_end_tag"],
            "answer_tag": tags["answer_tag"],
            "pad_id_source": "tokenizer.eos_token_id",
        },
        "check_alignment_contract": alignment_contract(),
    }
    retain = retain_inventory()
    return {"rows": records, "audit": audit, "retain": retain}


def load_tokenizer_and_tags():
    from transformers import AutoTokenizer

    if "LlamaForCausalLM" in sys.modules:
        raise RuntimeError("refusing to continue after LlamaForCausalLM was imported")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tags = llama2_tags()
    return tokenizer, tags


def llama2_tags():
    import yaml

    document = yaml.load(MODEL_CONFIG_PATH.read_text(), Loader=yaml.FullLoader)
    family = document[MODEL_FAMILY]
    return {
        "question_start_tag": family["question_start_tag"],
        "question_end_tag": family["question_end_tag"],
        "answer_tag": family["answer_tag"],
    }


def encode_row(tokenizer, question, answer, tags):
    """Match the ASU score formatter. The donor sequence is the question only."""
    new_question = tags["question_start_tag"] + question + tags["question_end_tag"]
    new_answer = tags["answer_tag"] + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=SCORE_MAX_LENGTH,
        truncation=True,
    )
    unpadded = list(encoded["input_ids"])
    unpadded_len = len(unpadded)
    pad_length = SCORE_MAX_LENGTH - unpadded_len
    pad_id = tokenizer.eos_token_id
    score_ids = unpadded + [pad_id] * pad_length
    attention_mask = list(encoded["attention_mask"]) + [0] * pad_length
    if unpadded_len == SCORE_MAX_LENGTH:
        labels = list(unpadded)
    else:
        labels = unpadded + [pad_id] + [-100] * (pad_length - 1)
    for index in range(min(num_question_tokens, len(labels))):
        labels[index] = -100
    donor = tokenizer(new_question, add_special_tokens=True, truncation=True, max_length=SCORE_MAX_LENGTH)
    donor_ids = list(donor["input_ids"])
    answer_span = [num_question_tokens, unpadded_len]
    answer_in_donor = donor_ids[answer_span[0] : answer_span[1]] == unpadded[answer_span[0] : answer_span[1]]
    if answer_span[0] >= answer_span[1]:
        answer_in_donor = False
    return {
        "donor_input_ids": donor_ids,
        "donor_attention_mask": list(donor["attention_mask"]),
        "score_input_ids": score_ids,
        "score_attention_mask": attention_mask,
        "score_labels": labels,
        "prompt_token_span": [0, num_question_tokens],
        "answer_token_span": answer_span,
        "pad_count": pad_length,
        "truncated": unpadded_len == SCORE_MAX_LENGTH,
        "num_question_tokens": num_question_tokens,
        "donor_ids_equal_score_prefix": score_ids[: len(donor_ids)] == donor_ids,
        "answer_tokens_in_donor": bool(answer_in_donor and answer_span[1] > answer_span[0]),
    }


def exclude_unpaired(encoded):
    """Ask check_alignment to reject the row. No counterfactual question is stored."""
    donor_ids = encoded["donor_input_ids"]
    score_ids = encoded["score_input_ids"]
    # Same-length ids reach the fact-span check. They are not saved as q-.
    aligned, function_reasons = check_alignment(donor_ids, donor_ids, None, score_ids, score_ids)
    if aligned:
        raise RuntimeError("unpaired row was accepted by check_alignment")
    reasons = ["pairing_mode_unset", "counterfactual_question_missing"]
    for reason in function_reasons:
        if reason not in reasons:
            reasons.append(reason)
    return {
        "aligned": False,
        "pair_constructed": False,
        "answer_span_reliable": True,
        "prompt_minus_stored": False,
        "check_alignment_called": True,
        "check_alignment_reasons": function_reasons,
        "reasons": reasons,
    }


def alignment_contract():
    aligned, reasons = check_alignment([11, 12], [11, 12], None, [11, 12, 13], [11, 12, 13])
    return {
        "inputs": "synthetic ids, not a TOFU row",
        "aligned": aligned,
        "reasons": reasons,
    }


def retain_inventory():
    rows = read_records(RETAIN_PATH)
    keys = sorted(rows[0].keys()) if rows else []
    task_ids = sorted({row.get("task_id") for row in rows})
    perturbed = read_records(PERTURBED_PATH)
    perturbed_keys = sorted(perturbed[0].keys()) if perturbed else []
    return {
        "retain_rows": len(rows),
        "retain_keys": keys,
        "retain_task_ids": [None if item is None else str(item) for item in task_ids],
        "matched_retain_control": {
            "constructable": False,
            "reason": (
                "retain50 has no task_id and no q+/q- fact span aligned to forget05 task 1; "
                "perturbed_answer values are scoring foils, not donor questions"
            ),
        },
        "forget05_perturbed_rows": len(perturbed),
        "forget05_perturbed_keys": perturbed_keys,
        "perturbed_answer_used_as_donor": False,
    }


def protocol_gaps(config):
    selection = config.get("selection") or {}
    gates = config.get("gates") or {}
    rho = config.get("rho") or {}
    fields = {
        "pairing.mode": (config.get("pairing") or {}).get("mode"),
        "fact_span_source": None,
        "counterfactual_question_source": None,
        "selection.method": selection.get("method"),
        "selection.candidate_pool": selection.get("candidate_pool"),
        "alpha": config.get("alpha"),
        "gates.preregistered": gates.get("preregistered"),
        "gates.localization_passed": gates.get("localization_passed"),
        "gates.teacher_retain_gate_passed": gates.get("teacher_retain_gate_passed"),
        "gates.localization_n_min": gates.get("localization_n_min"),
        "gates.localization_r_min": gates.get("localization_r_min"),
        "gates.retain_nll_cap": gates.get("retain_nll_cap"),
        "gates.retain_rho": gates.get("retain_rho"),
        "gates.localization_result_id": gates.get("localization_result_id"),
        "gates.teacher_gate_id": gates.get("teacher_gate_id"),
        "rho.grid_status": rho.get("grid_status"),
        "rho.select_best": rho.get("select_best"),
        "run_real_model": config.get("run_real_model"),
        "allow_student_e1": config.get("allow_student_e1"),
    }
    missing = []
    for name, value in fields.items():
        if name == "rho.grid_status" and value != "frozen":
            missing.append({"field": name, "value": value, "status": "not_frozen"})
        elif name == "rho.select_best" and value is not False:
            missing.append({"field": name, "value": value, "status": "unexpected"})
        elif value is None or value is False:
            missing.append({"field": name, "value": value, "status": "unset"})
    localization = localization_gate(None, None, config)
    retain = retain_gate(None, config)
    return {
        "ready_for_real_experiment": False,
        "thresholds_filled": False,
        "e0_retain_cap_reused": False,
        "e0_decision_preserved": config.get("e0_decision_preserved"),
        "e0_decision_constant": E0_DECISION,
        "e0_retain_cap_constant_not_copied": E0_RETAIN_CAP,
        "selection_method_chosen": None,
        "pgb_ct_keys_not_set": list(PGB_KEYS),
        "discrete_greedy_keys_not_set": list(GREEDY_KEYS),
        "rho_grid_exploratory": rho.get("grid"),
        "fields": missing,
        "localization_gate_without_metrics": localization,
        "retain_gate_without_metrics": retain,
        "holdout_used_to_choose_thresholds": False,
    }


def student_preflight(config):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        return_code = student_entry.main([str(CANDIDATE_CONFIG)])
    reason = student_block_reason(config)
    loss_check = loss_refuses_before_forward()
    blockers = [
        {
            "condition": "frozen approved mcsd_v2 gate",
            "satisfied": False,
            "detail": "gates.preregistered is false and thresholds are null",
        },
        {
            "condition": "real hook, token alignment, and patch smoke passed",
            "satisfied": False,
            "detail": "no CUDA run; alignment excluded every task-1 row; patch smoke was not run",
        },
        {
            "condition": "author-isolated calibration localization and controls meet the new gate",
            "satisfied": False,
            "detail": "N, R, and controls are null",
        },
        {
            "condition": "one rho passes an independent retain delta-NLL gate",
            "satisfied": False,
            "detail": "rho grid is exploratory_unregistered and retain_nll_cap is null",
        },
        {
            "condition": "student config points at that new version rather than E0",
            "satisfied": False,
            "detail": "allow_student_e1 is false; e0_decision_preserved remains do_not_start_e1",
        },
        {
            "condition": "training entry verified: student-only updates, frozen teacher and donor, paths and resume",
            "satisfied": False,
            "detail": "student_entry.py does not call forget.py, so those checks were not performed",
        },
    ]
    return {
        "student_e1_started": False,
        "student_entry_calls_forget_py": False,
        "runnable_training_command_written": False,
        "block_reason": reason,
        "student_entry": {"return_code": return_code, "stderr": stderr.getvalue().strip()},
        "loss_refusal_before_forward": loss_check,
        "blockers": blockers,
        "all_conditions_satisfied": all(item["satisfied"] for item in blockers),
        "existing_trainer_sites": {
            "train_entry": "Right-to-be-forgotten/forget.py",
            "save_dir_field": "cfg.save_dir",
            "save_dir_template": (
                "results/${model_family}/${split}/${forget_loss}/seed_${seed}/"
                "epoch${num_epochs}_${lr}_FixRef${fix_ref_model}_mask${mask}_"
                "Fcoeff${forget_coeff}_Rcoeff${regularization_coeff}_Temp${attention_temp}_"
                "Layers${layers_id}_beta${beta}"
            ),
            "seed_field": "cfg.seed",
            "max_steps": "computed in forget.py from epochs, dataset length, batch size, accumulation, and device count",
            "deepspeed_config": "Right-to-be-forgotten/config/ds_config/llama2.json",
            "reference_model_when_fix_ref_model": "cfg.model_path",
            "reference_frozen_in": "Right-to-be-forgotten/trainer/trainer.py sets param.requires_grad = False",
        },
        "stop_lines_for_a_future_run": [
            "NaN loss or non-finite scores",
            "out-of-memory or data-shape errors",
            "retain delta NLL above the retain cap frozen at that time",
            "gate no longer valid",
            "keep failed artifacts; current retain cap is null and the E0 cap is not a stop line",
        ],
    }


def loss_refuses_before_forward():
    path = _REPO / "Right-to-be-forgotten" / "trainer" / "losses.py"
    spec = importlib.util.spec_from_file_location("asu_losses_pre_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Boom(object):
        def __init__(self):
            self.calls = 0

        def __call__(self, *args, **kwargs):
            del args, kwargs
            self.calls += 1
            raise AssertionError("forward was called")

    model = Boom()
    reference = Boom()
    try:
        module.get_loss(model, reference, [], "MCSD_ASU")
    except StudentE1Blocked as exc:
        message = str(exc)
    else:
        raise RuntimeError("MCSD_ASU loss did not refuse")
    return {
        "exception": "StudentE1Blocked",
        "message": message,
        "model_calls": model.calls,
        "reference_calls": reference.calls,
        "forward_reached": False,
    }


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())

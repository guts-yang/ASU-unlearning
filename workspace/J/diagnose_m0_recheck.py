#!/usr/bin/env python3
"""Versioned TOFU M0 recheck that preserves the original preregistered gate."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

import diagnose_m0 as original


RECHECK_NAME = "m0_recheck_rel1e-4_v1"
RECHECK_DIR = original.WORKSPACE / "J" / "derived" / RECHECK_NAME
RECHECK_VERDICT = RECHECK_DIR / "verdict.md"


def _tensor_metadata(tensor: torch.Tensor) -> dict:
    return {
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "shape": list(tensor.shape),
    }


def kl_api_error_with_location(student_logits, teacher_logits, input_ids, labels, attention_mask):
    """Run the frozen API check and add diagnostics without changing its decision."""
    from trainer.losses import asu_loss

    student = student_logits.detach().float().requires_grad_(True)
    teacher = teacher_logits.detach().float()
    loss = asu_loss(
        original.FrozenLogits(student),
        original.FrozenLogits(teacher),
        [(input_ids, labels, attention_mask)],
        attention_temp=original.ATTENTION_TEMP,
        layers_id=None,
    )
    loss.backward()

    gold = labels[:, 1:]
    mask = gold != -100
    if int(mask.sum()) == 0:
        raise RuntimeError("API check batch has no answer tokens")
    positions = mask.nonzero(as_tuple=False)
    gold_ids = gold[positions[:, 0], positions[:, 1]]
    gathered_s = student[:, :-1, :][positions[:, 0], positions[:, 1]]
    gathered_t = teacher[:, :-1, :][positions[:, 0], positions[:, 1]]
    prob_s = torch.softmax(gathered_s, dim=-1).gather(1, gold_ids[:, None]).squeeze(1)
    prob_t = torch.softmax(gathered_t, dim=-1).gather(1, gold_ids[:, None]).squeeze(1)
    n_valid = mask.sum(dim=-1).float()
    batch = float(labels.shape[0])
    scale = n_valid[positions[:, 0]] * batch
    analytic = (prob_s - prob_t) / scale
    observed = student.grad[:, :-1, :][positions[:, 0], positions[:, 1], gold_ids]
    abs_err = (observed - analytic).abs()

    # This is intentionally identical to the frozen preregistered definition.
    rel_denominator = analytic.abs().clamp_min(1e-6)
    rel = abs_err / rel_denominator
    max_index = int(rel.argmax().item())
    max_position = positions[max_index]
    sign_ok = bool(torch.all((observed < 0) == (prob_s < prob_t)))
    max_abs = float(abs_err.max().detach())
    max_rel = float(rel.max().detach())
    report = {
        "loss": float(loss.detach()),
        "n_positions": int(positions.shape[0]),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "sign_ok": sign_ok,
        "ok": bool(sign_ok and max_rel < original.REL_TOL and max_abs < 1e-5),
        "frozen_gate": {
            "relative_tolerance": original.REL_TOL,
            "absolute_tolerance": 1e-5,
            "relative_denominator": "abs(analytic).clamp_min(1e-6)",
            "comparison": "strictly less than",
        },
        "max_relative_error_position": {
            "flat_valid_index": max_index,
            "batch_index": int(max_position[0].item()),
            "query_index": int(max_position[1].item()),
            "label_index": int(max_position[1].item() + 1),
            "gold_token_id": int(gold_ids[max_index].item()),
            "n_valid_in_sequence": int(n_valid[max_position[0]].item()),
            "batch_size": int(labels.shape[0]),
            "scale": float(scale[max_index].item()),
            "p_student_gold": float(prob_s[max_index].detach()),
            "p_teacher_gold": float(prob_t[max_index].detach()),
            "analytic_gradient": float(analytic[max_index].detach()),
            "autograd_gradient": float(observed[max_index].detach()),
            "absolute_error": float(abs_err[max_index].detach()),
            "relative_denominator": float(rel_denominator[max_index].detach()),
            "relative_error": float(rel[max_index].detach()),
            "denominator_clamped": bool(analytic[max_index].abs() < 1e-6),
        },
        "numeric_path": {
            "model_student_logits": _tensor_metadata(student_logits),
            "model_teacher_logits": _tensor_metadata(teacher_logits),
            "checked_student_logits": _tensor_metadata(student),
            "checked_teacher_logits": _tensor_metadata(teacher),
            "analytic": _tensor_metadata(analytic),
            "autograd": _tensor_metadata(observed),
        },
    }
    return report


def write_recheck_verdict(summary):
    api = summary["kl_api"]
    point = api["max_relative_error_position"]
    decision = summary["decision"]
    lines = [
        "# 方向 J M0 数值复核",
        "",
        f"复核版本：`{RECHECK_NAME}`。旧判决与旧 derived 文件未覆盖。",
        "",
        f"决策：**{decision}**。",
        "",
        "旧预登记口径保持不变：最大相对误差必须严格小于 `1e-4`；"
        "相对误差分母仍为 `abs(analytic).clamp_min(1e-6)`，并同时要求最大绝对误差小于 `1e-5`、符号一致。",
        "",
        f"- 最大相对误差：`{api['max_rel_error']:.9e}`",
        f"- 最大绝对误差：`{api['max_abs_error']:.9e}`",
        f"- 最大相对误差位置：batch `{point['batch_index']}`，query `{point['query_index']}`，"
        f"label `{point['label_index']}`，gold token `{point['gold_token_id']}`",
        f"- 解析梯度 / autograd：`{point['analytic_gradient']:.12e}` / "
        f"`{point['autograd_gradient']:.12e}`",
        f"- 相对分母是否被 clamp：`{point['denominator_clamped']}`",
        "",
    ]
    if decision != "GO":
        lines.extend(
            [
                "结论：旧门未通过，保持训练关闭。绝对误差较小不能替代预登记的相对误差门。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "数值门通过仅解除 M0 数值阻塞；事实 span 尚未审核，因此候选训练仍保持关闭。",
                "",
            ]
        )
    RECHECK_VERDICT.write_text("\n".join(lines))


def main():
    original.local_dir(original.INIT_MODEL, "init")
    original.local_dir(original.ASU_MODEL, "asu")
    original.local_dir(original.TOKENIZER_DIR, "tokenizer")
    if not original.FORGET_PATH.is_file():
        raise SystemExit(f"forget data missing: {original.FORGET_PATH}")
    if RECHECK_DIR.exists():
        raise SystemExit(f"refusing to overwrite existing recheck evidence: {RECHECK_DIR}")
    RECHECK_DIR.mkdir(parents=True)

    original.OUT_DIR = RECHECK_DIR
    original.VERDICT_PATH = RECHECK_VERDICT
    original.kl_api_error = kl_api_error_with_location
    original.write_verdict = write_recheck_verdict
    original.main()

    summary_path = RECHECK_DIR / "m0_summary.json"
    summary = json.loads(summary_path.read_text())
    try:
        commit = subprocess.run(
            ["git", "-C", str(original.ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    run_manifest = {
        "recheck_name": RECHECK_NAME,
        "preserves_original_gate": True,
        "relative_tolerance": original.REL_TOL,
        "original_summary": str(original.WORKSPACE / "J/derived/m0_summary.json"),
        "decision": summary.get("decision"),
        "command": f"{sys.executable} {Path(__file__).resolve()}",
        "commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }
    (RECHECK_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))


if __name__ == "__main__":
    main()

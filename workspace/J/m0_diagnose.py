#!/usr/bin/env python3
"""Read-only DARD-ASU M0 diagnostic for the real-world ASU+KL protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "workspace"))
sys.path.insert(0, str(REPO / "Right-to-be-forgotten"))

MODEL_CONFIG = {
    "question_start_tag": "<|start_header_id|>user<|end_header_id|>\n\n",
    "question_end_tag": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "answer_tag": "",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_snapshot(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise SystemExit(
            f"{label} must be an existing local snapshot directory with config.json; "
            f"refusing download or substitution: {value}"
        )
    return path


def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise SystemExit("forget JSON must be a JSON array or JSONL of object records")
    missing = [index for index, row in enumerate(payload) if not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str)]
    if missing:
        raise SystemExit(f"forget JSON records require string question and answer; invalid rows: {missing[:10]}")
    return payload


def load_teacher(path: Path, dtype: torch.dtype):
    from my_models.my_llama import LlamaForCausalLM

    last_error = None
    for implementation in ("flash_attention_2", "sdpa", "eager"):
        try:
            return LlamaForCausalLM.from_pretrained(
                str(path), local_files_only=True, torch_dtype=dtype, attn_implementation=implementation
            )
        except (ImportError, OSError, ValueError) as error:
            last_error = error
    raise RuntimeError(f"unable to load local Teacher snapshot {path}") from last_error


def load_student(path: Path, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, torch_dtype=dtype)


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "n": int(tensor.numel()),
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "p05": float(torch.quantile(tensor, 0.05)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def rank_summary(ranks: list[int]) -> dict[str, float | int | None]:
    if not ranks:
        return {"n": 0, "top1": None, "top5": None, "top10": None, "median_rank": None}
    values = torch.tensor(ranks)
    return {
        "n": int(values.numel()),
        "top1": float((values <= 1).float().mean()),
        "top5": float((values <= 5).float().mean()),
        "top10": float((values <= 10).float().mean()),
        "median_rank": float(values.float().median()),
    }


def selected_group(record: dict[str, Any], requested: str | None) -> str | None:
    keys = (requested,) if requested else ("author", "entity", "author_id", "entity_id")
    for key in keys:
        if key and record.get(key) not in (None, ""):
            return str(record[key])
    return None


def flattened(entries: list[dict[str, Any]], student_name: str | None = None) -> dict[str, list[float] | list[int]]:
    data: dict[str, list[float] | list[int]] = {
        "teacher_probs": [],
        "teacher_ranks": [],
        "student_probs": [],
        "student_ranks": [],
        "student_lt_teacher": [],
        "analytic_negative_gradient": [],
    }
    for entry in entries:
        data["teacher_probs"].extend(entry["teacher"]["gold_probabilities"])
        data["teacher_ranks"].extend(entry["teacher"]["gold_ranks"])
        if student_name:
            student = entry["students"][student_name]
            data["student_probs"].extend(student["gold_probabilities"])
            data["student_ranks"].extend(student["gold_ranks"])
            data["student_lt_teacher"].extend(student["student_lt_teacher"])
            data["analytic_negative_gradient"].extend(student["analytic_negative_gradient"])
    return data


def bootstrap_ratio(entries: list[dict[str, Any]], student_name: str, draws: int, seed: int) -> dict[str, Any]:
    """Question-cluster bootstrap for the primary token-level proportions."""
    generator = torch.Generator().manual_seed(seed)
    metrics = {"student_lt_teacher": [], "analytic_negative_gradient": [], "student_top1": [], "student_top5": [], "student_top10": []}
    usable = [entry for entry in entries if entry["valid_answer_tokens"]]
    if len(usable) < 2 or draws <= 0:
        return {"method": "question-cluster bootstrap", "draws": 0, "reason": "fewer than two usable questions", "metrics": {}}
    for _ in range(draws):
        sample = torch.randint(len(usable), (len(usable),), generator=generator).tolist()
        rows = [usable[index]["students"][student_name] for index in sample]
        comparisons = [value for row in rows for value in row["student_lt_teacher"]]
        gradients = [value for row in rows for value in row["analytic_negative_gradient"]]
        ranks = [value for row in rows for value in row["gold_ranks"]]
        metrics["student_lt_teacher"].append(sum(comparisons) / len(comparisons))
        metrics["analytic_negative_gradient"].append(sum(gradients) / len(gradients))
        for cutoff in (1, 5, 10):
            metrics[f"student_top{cutoff}"].append(sum(rank <= cutoff for rank in ranks) / len(ranks))
    return {
        "method": "nonparametric bootstrap resampling questions; tokens remain within question clusters",
        "draws": draws,
        "metrics": {
            name: {"estimate": float(torch.tensor(values).mean()), "ci95": [float(torch.quantile(torch.tensor(values), 0.025)), float(torch.quantile(torch.tensor(values), 0.975))]}
            for name, values in metrics.items()
        },
    }


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# DARD-ASU M0 结果",
        "",
        f"结论：**{result['decision']}**。{result['decision_reason']}",
        "",
        f"有效答案 token：{result['counts']['valid_answer_tokens']}；题目：{result['counts']['usable_questions']}/{result['counts']['input_questions']}；"
        f"可用 author/entity group：{result['counts']['available_groups'] or '未提供'}。",
        "",
        "| 对象 | token数 | Top-1 | Top-5 | Top-10 | rank中位数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    teacher = result["teacher"]["rank"]
    lines.append(f"| Teacher | {teacher['n']} | {teacher['top1']:.4f} | {teacher['top5']:.4f} | {teacher['top10']:.4f} | {teacher['median_rank']:.1f} |")
    for name, values in result["students"].items():
        rank = values["rank"]
        lines.append(f"| Student ({name}) | {rank['n']} | {rank['top1']:.4f} | {rank['top5']:.4f} | {rank['top10']:.4f} | {rank['median_rank']:.1f} |")
    lines.extend(["", "| Student | pS(g)<pT(g) | 解析 dL/dzS(g)<0 | 小批量 autograd 检查 |", "|---|---:|---:|---|"])
    for name, values in result["students"].items():
        lines.append(
            f"| {name} | {values['student_lt_teacher_fraction']:.4f} | {values['analytic_negative_gradient_fraction']:.4f} | "
            f"{values['autograd_check']['passed']}（n={values['autograd_check']['checked_tokens']}，max误差={values['autograd_check']['max_abs_error']:.3g}） |"
        )
    lines.extend(
        [
            "",
            "区间为题目 cluster bootstrap；token 未被视作独立样本。所有结果都是指定资产和 teacher-forcing 前缀下的描述统计，不构成显著性检验、硬下限、零泄漏或 DARD 有效性声明。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    global torch
    global encode_forget_example, analytic_gold_gradient, autograd_gradient_check
    global gold_probabilities, gold_ranks, shifted_answer_mask

    parser = argparse.ArgumentParser(description="Offline, read-only DARD-ASU M0 teacher-forcing diagnostic")
    parser.add_argument("--forget-json", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--student-initial", required=True)
    parser.add_argument("--student-asu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--cluster-key", help="Optional author/entity field for inventory only")
    parser.add_argument("--limit", type=int, help="Read-only debugging limit; do not use for final M0")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 2 or args.bootstrap_draws < 0:
        raise SystemExit("batch-size >= 1, max-length >= 2, and bootstrap-draws >= 0 are required")

    forget_path = Path(args.forget_json).expanduser().resolve()
    if not forget_path.is_file():
        raise SystemExit(f"forget JSON is missing: {forget_path}; refusing download or substitute data")
    teacher_path = require_snapshot(args.teacher_model, "teacher-model")
    initial_path = require_snapshot(args.student_initial, "student-initial")
    asu_path = require_snapshot(args.student_asu, "student-asu") if args.student_asu else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists; refusing to overwrite evidence: {output_dir}")

    records = load_records(forget_path)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("forget JSON has no records")

    import torch
    from transformers import AutoTokenizer, __version__ as transformers_version
    from gold_rank.masking import encode_forget_example
    from m0_core import (
        analytic_gold_gradient,
        autograd_gradient_check,
        gold_probabilities,
        gold_ranks,
        shifted_answer_mask,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(teacher_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher = load_teacher(teacher_path, dtype).to(device).eval()
    students = {"initial": load_student(initial_path, dtype).to(device).eval()}
    if asu_path:
        students["asu"] = load_student(asu_path, dtype).to(device).eval()

    entries: list[dict[str, Any]] = []
    gradient_checks: dict[str, Any] = {}
    for start in range(0, len(records), args.batch_size):
        batch_records = records[start : start + args.batch_size]
        encoded = [encode_forget_example(tokenizer, args.max_length, row["question"], row["answer"], MODEL_CONFIG, mask=True) for row in batch_records]
        input_ids = torch.stack([row[0] for row in encoded]).to(device)
        labels = torch.stack([row[1] for row in encoded]).to(device)
        attention_mask = torch.stack([row[2] for row in encoded]).to(device)
        valid = shifted_answer_mask(labels)
        with torch.no_grad():
            teacher_logits = teacher(input_ids, attention_mask=attention_mask, attention_temp=2.0, layers_id=None).logits
            student_logits = {name: model(input_ids, attention_mask=attention_mask).logits for name, model in students.items()}

        student_metrics = {}
        for name, logits in student_logits.items():
            valid_rows = valid.sum(dim=1).bool()
            if name not in gradient_checks and valid_rows.any():
                gradient_checks[name] = autograd_gradient_check(
                    logits[valid_rows], teacher_logits[valid_rows], labels[valid_rows]
                ).__dict__
            probabilities, _ = gold_probabilities(logits, labels)
            student_metrics[name] = {
                "probabilities": probabilities,
                "ranks": gold_ranks(logits, labels),
                "analytic_gradient": analytic_gold_gradient(logits, teacher_logits, labels),
            }
        teacher_probs, _ = gold_probabilities(teacher_logits, labels)
        teacher_rank = gold_ranks(teacher_logits, labels)
        cursor = 0
        for position, record in enumerate(batch_records):
            count = int(valid[position].sum().item())
            entry = {
                "index": start + position,
                "group": selected_group(record, args.cluster_key),
                "valid_answer_tokens": count,
                "teacher": {"gold_probabilities": teacher_probs[cursor : cursor + count].tolist(), "gold_ranks": teacher_rank[cursor : cursor + count].tolist()},
                "students": {},
            }
            for name, metrics in student_metrics.items():
                student_probs = metrics["probabilities"][cursor : cursor + count]
                teacher_slice = teacher_probs[cursor : cursor + count]
                analytic_slice = metrics["analytic_gradient"][cursor : cursor + count]
                entry["students"][name] = {
                    "gold_probabilities": student_probs.tolist(),
                    "gold_ranks": metrics["ranks"][cursor : cursor + count].tolist(),
                    "student_lt_teacher": (student_probs < teacher_slice).tolist(),
                    "analytic_negative_gradient": (analytic_slice < 0).tolist(),
                }
            entries.append(entry)
            cursor += count

    flat = flattened(entries)
    if not flat["teacher_probs"]:
        raise SystemExit("zero valid answer positions after real training encoding; refusing to report denominators of zero")
    result: dict[str, Any] = {
        "decision": "INCONCLUSIVE",
        "decision_reason": "M0 is descriptive only; this run does not establish a DARD-ASU objective or a hard forgetting bound.",
        "protocol": {
            "forget_loss": "KL(p_T || p_S)",
            "attention_temp_teacher": 2.0,
            "layers_id_teacher": None,
            "student_attention_temp": None,
            "causal_alignment": "logits[:, :-1, :] predicts labels[:, 1:]",
            "valid_mask": "labels[:, 1:] != -100",
        },
        "assets": {
            "forget_json": {"path": str(forget_path), "sha256": sha256_file(forget_path)},
            "teacher_model": {"path": str(teacher_path), "config_sha256": sha256_file(teacher_path / "config.json")},
            "student_initial": {"path": str(initial_path), "config_sha256": sha256_file(initial_path / "config.json")},
            "student_asu": None if not asu_path else {"path": str(asu_path), "config_sha256": sha256_file(asu_path / "config.json")},
        },
        "runtime": {"created_at_utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__, "transformers": transformers_version, "device": str(device), "dtype": str(dtype)},
        "counts": {
            "input_questions": len(records),
            "usable_questions": sum(entry["valid_answer_tokens"] > 0 for entry in entries),
            "skipped_no_answer_tokens": sum(entry["valid_answer_tokens"] == 0 for entry in entries),
            "valid_answer_tokens": len(flat["teacher_probs"]),
            "available_groups": len({entry["group"] for entry in entries if entry["group"] is not None}),
        },
        "teacher": {"gold_probability": summary(flat["teacher_probs"]), "rank": rank_summary(flat["teacher_ranks"])},
        "students": {},
        "statistics_note": "Question-cluster bootstrap; tokens remain nested within questions. These are descriptive statistics, not significance tests.",
    }
    for name in students:
        values = flattened(entries, name)
        autograd = gradient_checks.get(name, {"checked_tokens": 0, "passed": False, "max_abs_error": None, "negative_autograd_tokens": 0, "negative_analytic_tokens": 0})
        result["students"][name] = {
            "gold_probability": summary(values["student_probs"]),
            "rank": rank_summary(values["student_ranks"]),
            "student_lt_teacher_fraction": sum(values["student_lt_teacher"]) / len(values["student_lt_teacher"]) if values["student_lt_teacher"] else None,
            "analytic_negative_gradient_fraction": sum(values["analytic_negative_gradient"]) / len(values["analytic_negative_gradient"]) if values["analytic_negative_gradient"] else None,
            "autograd_check": autograd,
            "question_cluster_bootstrap": bootstrap_ratio(entries, name, args.bootstrap_draws, args.seed),
        }

    output_dir.mkdir(parents=True)
    with (output_dir / "per_question.jsonl").open("w") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    (output_dir / "m0_summary.json").write_text(json.dumps(result, indent=2))
    (output_dir / "m0_summary.md").write_text(markdown_report(result))
    print(json.dumps({"decision": result["decision"], "output_dir": str(output_dir), "valid_answer_tokens": result["counts"]["valid_answer_tokens"]}, indent=2))


if __name__ == "__main__":
    main()

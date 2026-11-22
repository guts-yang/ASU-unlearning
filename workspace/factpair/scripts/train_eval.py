#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

BASE = Path(__file__).resolve().parents[1]
CORE = BASE / "core"
sys.path.insert(0, str(CORE))

from factpair import FactPairDataset, read_jsonl, write_json, write_jsonl  # noqa: E402
from factpair_collator import FactPairCollator  # noqa: E402
from factpair_activation import (  # noqa: E402
    ActivationReader,
    LayerSchedule,
    activation_alignment_loss,
)
from factpair_seqkl import compute_seqkl, generate_pair_output_traces  # noqa: E402
from factpair_behavior import classify_generation  # noqa: E402
from factpair_recovery import aggregate_recovery  # noqa: E402


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(config: Dict[str, Any]):
    dtype = torch.bfloat16 if config["dtype"] == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        torch_dtype=dtype,
        attn_implementation=config["attn_implementation"],
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.to("cuda:0")
    return tokenizer, model


def make_dataloader(tokenizer, config: Dict[str, Any]) -> DataLoader:
    dataset = FactPairDataset(
        pairs_path=Path(config["data_dir"]) / "paired_prompts.jsonl",
        retain_path=Path(config["data_dir"]) / "retain_train.jsonl",
        tokenizer=tokenizer,
        template_args={"apply_chat_template": True, "system_prompt": None},
        max_length=512,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=FactPairCollator(padding_side="right"),
        num_workers=0,
        drop_last=False,
    )


def prompt_inputs(batch: Dict[str, Any], key: str) -> Dict[str, torch.Tensor]:
    item = batch[key]
    return {
        "input_ids": item["input_ids"].to("cuda:0", non_blocking=True),
        "attention_mask": item["attention_mask"].to("cuda:0", non_blocking=True),
    }


def compute_generation_metrics(tokenizer, model, pairs: List[Dict[str, Any]], max_samples: int) -> Dict[str, Any]:
    rows = pairs[:max_samples]
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device
    records: List[Dict[str, Any]] = []
    was_training = model.training
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(rows), 4):
                batch = rows[start : start + 4]
                encoded = tokenizer(
                    [row["prompt_plus"] for row in batch],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                sequences = model.generate(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=64,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    use_cache=True,
                )
                prompt_length = encoded["input_ids"].shape[1]
                for row, sequence in zip(batch, sequences):
                    generated = sequence[prompt_length:]
                    eos_present = bool((generated == eos_id).any().item())
                    text = tokenizer.decode([int(token) for token in generated], skip_special_tokens=True)
                    record = classify_generation(
                        generation=text,
                        target_answer=row["answer_plus"],
                        donor_answer=row["answer_minus"],
                        eos_present=eos_present,
                        truncated=(not eos_present and len(generated) >= 64),
                    )
                    record["generated_tokens"] = int(generated.ne(pad_id).sum().item())
                    records.append(record)
    finally:
        model.train(was_training)
        tokenizer.padding_side = original_padding_side
    total = max(1, len(records))
    categories = [record["category"] for record in records]
    return {
        "samples": len(records),
        "target_correct_rate": categories.count("target_correct") / total,
        "donor_answer_rate": categories.count("donor_answer") / total,
        "refusal_rate": categories.count("refusal") / total,
        "other_wrong_rate": categories.count("other_wrong") / total,
        "premature_eos_rate": sum(1 for r in records if r["truncated"]) / total,
        "truncated_rate": sum(1 for r in records if r["truncated"]) / total,
        "eos_rate": sum(1 for r in records if r["eos_present"]) / total,
        "mean_generated_tokens": sum(int(r["generated_tokens"]) for r in records) / total,
        "complete_answer_rate": sum(1 for r in records if r["complete_answer"]) / total,
    }


def compute_retain_nll(tokenizer, model, rows: List[Dict[str, Any]], max_samples: int) -> Dict[str, float]:
    """Answer-position NLL on retain rows; prompt and padding are masked."""
    rows = rows[:max_samples]
    device = next(model.parameters()).device
    eos_token = tokenizer.eos_token or ""
    prompt_ids_list: List[List[int]] = []
    texts: List[str] = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_ids_list.append([int(token) for token in prompt_ids])
        answer = row["answer"]
        if eos_token and not answer.rstrip().endswith(eos_token):
            answer = answer + eos_token
        texts.append(prompt + answer)

    loss_sum = 0.0
    count = 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            ids = encoded["input_ids"].to(device)
            mask = encoded["attention_mask"].to(device)
            labels = ids.clone()
            for row_index, prompt_ids in enumerate(prompt_ids_list):
                prefix = torch.tensor(prompt_ids, dtype=torch.long, device=device)
                if not torch.equal(labels[row_index, : len(prompt_ids)], prefix):
                    raise RuntimeError("Retain prompt tokenization prefix mismatch")
                labels[row_index, : len(prompt_ids)] = -100
            labels[mask == 0] = -100
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            if outputs.loss is not None:
                loss_sum = float(outputs.loss.item()) * len(rows)
                count = len(rows)
    finally:
        model.train(was_training)
    mean_loss = loss_sum / max(1, count)
    return {"retain_nll": mean_loss, "retain_ppl": math.exp(min(20.0, mean_loss)), "samples": count}


def evaluate_model(tokenizer, model, config: Dict[str, Any], pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    data_dir = Path(config["data_dir"])
    retain_eval_path = data_dir / "retain_eval.jsonl"
    retain_path = retain_eval_path if retain_eval_path.exists() else data_dir / "retain_train.jsonl"
    retain_rows = read_jsonl(retain_path)
    generation = compute_generation_metrics(tokenizer, model, pairs, int(config["eval_samples"]))
    retain = compute_retain_nll(tokenizer, model, retain_rows, min(100, len(retain_rows)))
    return {"generation": generation, "retain": retain, "retain_source": str(retain_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=BASE / "configs" / "experiment.json")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_path = output_dir / "train_metrics.jsonl"
    checkpoint_path = output_dir / "checkpoint-last.pt"
    state_path = output_dir / "trainer_state.json"
    final_path = output_dir / "final_results.json"
    if final_path.exists():
        print("EXPERIMENT_ALREADY_COMPLETE")
        return

    tokenizer, student = build_model(config)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    dataloader = make_dataloader(tokenizer, config)
    dataiter = iter(dataloader)

    schedule = LayerSchedule(
        num_layers=int(student.config.num_hidden_layers),
        schedule_type=config["schedule_type"],
        initial_fraction=config["initial_fraction"],
        expansion_fraction=config["expansion_fraction"],
        stage_steps=int(config["stage_steps"]),
        seed=int(config["seed"]),
    )
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    total_steps = int(config["max_steps"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 20),
        num_training_steps=total_steps,
    )

    global_step = 0
    if checkpoint_path.exists() and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cuda:0")
        student.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(state.get("global_step", 0))
        print(f"RESUMED_FROM={global_step}")

    pairs = read_jsonl(Path(config["data_dir"]) / "paired_prompts.jsonl")
    base_eval = None
    base_eval_path = output_dir / "base_eval.json"
    if base_eval_path.exists():
        base_eval = json.loads(base_eval_path.read_text(encoding="utf-8"))
    else:
        print("EVALUATING_BASE")
        base_eval = evaluate_model(tokenizer, teacher, config, pairs)
        write_json(base_eval_path, base_eval)

    seq_cache = {"step": -10**9, "targets": None}
    history: List[Dict[str, Any]] = []
    student.train()
    start_time = time.time()
    while global_step < total_steps:
        try:
            batch = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            batch = next(dataiter)
        forget = prompt_inputs(batch, "forget")
        teacher_minus = prompt_inputs(batch, "teacher_minus")
        retain = prompt_inputs(batch, "retain")
        retain["labels"] = batch["retain"]["labels"].to("cuda:0")

        with torch.no_grad():
            teacher_plus_act = ActivationReader(teacher, point=config["activation_point"]).read(
                forget["input_ids"], forget["attention_mask"], detach=True
            )
            teacher_minus_act = ActivationReader(teacher, point=config["activation_point"]).read(
                teacher_minus["input_ids"], teacher_minus["attention_mask"], detach=True
            )
            activation_delta_l2 = float(
                F.mse_loss(
                    F.normalize(teacher_plus_act, p=2, dim=-1),
                    F.normalize(teacher_minus_act, p=2, dim=-1),
                ).item()
            )
        student_act = ActivationReader(student, point=config["activation_point"]).read(
            forget["input_ids"], forget["attention_mask"], detach=False
        )
        active_layers = schedule.active_layers(global_step)
        layer_weights = schedule.layer_weights(global_step)
        act_loss = activation_alignment_loss(
            student_act,
            teacher_minus_act,
            active_layers,
            layer_weights,
            norm=config["activation_norm"],
            distance=config["activation_distance"],
        )

        if global_step - seq_cache["step"] >= int(config["seq_refresh_steps"]) or seq_cache["targets"] is None:
            seq_cache["targets"] = generate_pair_output_traces(
                teacher_model=teacher,
                student_model=student,
                teacher_prompt_ids=teacher_minus["input_ids"],
                teacher_prompt_mask=teacher_minus["attention_mask"],
                student_prompt_ids=forget["input_ids"],
                student_prompt_mask=forget["attention_mask"],
                max_new_tokens=int(config["max_new_tokens"]),
                eos_token_id=int(tokenizer.eos_token_id),
                pad_token_id=int(tokenizer.pad_token_id),
                seed=int(config["seed"]),
            )
            seq_cache["step"] = global_step
        targets = {key: value.detach() for key, value in seq_cache["targets"].items()}
        seq_metrics = compute_seqkl(
            student_model=student,
            teacher_model=teacher,
            teacher_prompt_ids=teacher_minus["input_ids"],
            teacher_prompt_attention_mask=teacher_minus["attention_mask"],
            student_prompt_ids=forget["input_ids"],
            student_prompt_attention_mask=forget["attention_mask"],
            teacher_output_ids=targets["teacher_output_ids"],
            student_output_ids=targets["student_output_ids"],
            teacher_output_mask=targets["teacher_output_mask"],
            student_output_mask=targets["student_output_mask"],
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            eta=float(config["eta"]),
            start_index=int(config["sequence_start_index"]),
        )
        retain_loss = student(**retain).loss
        loss = (
            float(config["lambda_act"]) * act_loss
            + float(config["lambda_seq"]) * seq_metrics["loss"]
            + float(config["beta_retain"]) * retain_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), float(config["grad_clip"]))
        optimizer.step()
        scheduler.step()

        global_step += 1
        record = {
            "step": global_step,
            "loss": float(loss.detach().item()),
            "act_loss": float(act_loss.detach().item()),
            "seq_loss": float(seq_metrics["loss"].detach().item()),
            "seq_kl_teacher_prefix": float(seq_metrics["loss_teacher_prefix"].detach().item()),
            "seq_kl_student_prefix": float(seq_metrics["loss_student_prefix"].detach().item()),
            "seq_valid_positions_teacher": int(seq_metrics["valid_positions_teacher"].detach().item()),
            "seq_valid_positions_student": int(seq_metrics["valid_positions_student"].detach().item()),
            "eta": float(config["eta"]),
            "activation_delta_l2": activation_delta_l2,
            "retain_loss": float(retain_loss.detach().item()),
            "active_layers": len(active_layers),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "elapsed_sec": time.time() - start_time,
        }
        history.append(record)
        with logs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if global_step % int(config["save_every"]) == 0:
            torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_path)
            write_json(state_path, {"global_step": global_step, "config": config, "timestamp": time.time()})
            print(f"CHECKPOINT step={global_step} loss={record['loss']:.6f}", flush=True)
        elif global_step % int(config["log_every"]) == 0:
            print(f"STEP {global_step}/{total_steps} loss={record['loss']:.6f}", flush=True)

    torch.save({"model": student.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_path)
    write_json(state_path, {"global_step": global_step, "config": config, "timestamp": time.time()})
    print("EVALUATING_FINAL", flush=True)
    final_eval = evaluate_model(tokenizer, student, config, pairs)

    g0_path = Path(config["data_dir"]) / "g0_gate_report.json"
    g0_report = json.loads(g0_path.read_text(encoding="utf-8")) if g0_path.exists() else {"overall_status": "UNKNOWN"}
    recovery_rows = []
    recovery_path = output_dir / "recovery_results.jsonl"
    if recovery_path.exists():
        recovery_rows = read_jsonl(recovery_path)
    recovery_summary = aggregate_recovery(recovery_rows) if recovery_rows else {"status": "BLOCKED_NO_RESULTS"}

    base_gen = base_eval["generation"]
    final_gen = final_eval["generation"]
    target_drop = base_gen["target_correct_rate"] - final_gen["target_correct_rate"]
    retain_delta = final_eval["retain"]["retain_nll"] - base_eval["retain"]["retain_nll"]
    if (
        target_drop >= 0.2
        and final_gen["donor_answer_rate"] <= 0.15
        and final_gen["refusal_rate"] <= 0.25
        and retain_delta <= 0.5
    ):
        conclusion = "观测到目标事实生成率显著下降，且 donor 替换、拒答与 retain NLL 退化未越过预设解释门槛；仍需 G2-G6 抗恢复证据才能称为深度遗忘。"
    else:
        conclusion = "当前结果未通过选择性遗忘组合门槛：目标事实下降不足，或主要由 donor 替换/拒答/retain 退化解释；不能主张该机制已成立。"

    results = {
        "protocol": config["protocol"],
        "gpu": config["gpu"],
        "model": config["model"],
        "eval_mode": "student_without_runtime_patch",
        "train_steps": global_step,
        "train_metrics": {
            "final_loss": history[-1] if history else None,
            "mean_loss_last_20": sum(r["loss"] for r in history[-20:]) / max(1, len(history[-20:])),
            "mean_act_loss_last_20": sum(r["act_loss"] for r in history[-20:]) / max(1, len(history[-20:])),
            "mean_seq_loss_last_20": sum(r["seq_loss"] for r in history[-20:]) / max(1, len(history[-20:])),
            "mean_seq_teacher_prefix_last_20": sum(r["seq_kl_teacher_prefix"] for r in history[-20:]) / max(1, len(history[-20:])),
            "mean_seq_student_prefix_last_20": sum(r["seq_kl_student_prefix"] for r in history[-20:]) / max(1, len(history[-20:])),
            "mean_retain_loss_last_20": sum(r["retain_loss"] for r in history[-20:]) / max(1, len(history[-20:])),
        },
        "base_eval": base_eval,
        "final_eval": final_eval,
        "deltas": {
            "target_correct_drop": target_drop,
            "donor_answer_rate_change": final_gen["donor_answer_rate"] - base_gen["donor_answer_rate"],
            "refusal_rate_change": final_gen["refusal_rate"] - base_gen["refusal_rate"],
            "retain_nll_delta": retain_delta,
        },
        "gates": {"g0_g1": g0_report.get("overall_status", "UNKNOWN"), "g5_g6_recovery": recovery_summary},
        "conclusion": conclusion,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "train_log": str(logs_path),
            "base_eval": str(base_eval_path),
        },
    }
    write_json(final_path, results)
    write_jsonl(output_dir / "final_summary.jsonl", [results])
    print(f"FINAL_RESULTS={final_path}", flush=True)


if __name__ == "__main__":
    main()

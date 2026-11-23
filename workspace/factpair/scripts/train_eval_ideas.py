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
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

BASE = Path(__file__).resolve().parents[1]
CORE = BASE / "core"
SCRIPTS = BASE / "scripts"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(SCRIPTS))

from factpair import read_jsonl, write_json, write_jsonl  # noqa: E402
from factpair_activation import LayerSchedule  # noqa: E402
from factpair_seqkl import compute_seqkl, generate_pair_output_traces  # noqa: E402
from factpair_ideas import (  # noqa: E402
    FactPairIdeaCollator,
    FactPairIdeaDataset,
    apply_gradient_surgery,
    compute_gradient_causal_gate,
    delta_subspace_target,
    gather_subject,
    layer_weight_tensor,
    layered_mse,
    read_donor_subject_activations,
    read_layer_activations,
    to_device,
    tricon_loss,
)
from train_eval import evaluate_model  # noqa: E402


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


def make_eval_pairs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "prompt_plus": row["prompt_plus"],
            "answer_plus": row["answer_plus"],
            "answer_minus": row["answer_minus_list"][0],
            "relation_family": row["relation_family"],
        }
        for row in rows
    ]


def cosine_mean(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(left.float(), right.float(), dim=-1).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    idea = config["idea"]
    condition = config.get("condition", idea)
    set_seed(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final_results.json"
    if final_path.exists():
        print("EXPERIMENT_ALREADY_COMPLETE", flush=True)
        return

    tokenizer, student = build_model(config)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    device = "cuda:0"

    pairs = read_jsonl(Path(config["data_dir"]) / "paired_prompts_multidonor.jsonl")
    dataset = FactPairIdeaDataset(
        pairs_path=str(Path(config["data_dir"]) / "paired_prompts_multidonor.jsonl"),
        retain_path=str(Path(config["data_dir"]) / "retain_train.jsonl"),
        tokenizer=tokenizer,
        unknown_word=config.get("unknown_word", "unknown"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=FactPairIdeaCollator(int(tokenizer.pad_token_id)),
        num_workers=0,
        drop_last=False,
    )
    dataiter = iter(dataloader)

    schedule = LayerSchedule(
        num_layers=int(student.config.num_hidden_layers),
        schedule_type=config["schedule_type"],
        initial_fraction=config["initial_fraction"],
        expansion_fraction=config["expansion_fraction"],
        stage_steps=int(config["stage_steps"]),
        seed=int(config["seed"]),
    )
    named_parameters = list(student.named_parameters())
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

    eval_pairs = make_eval_pairs(pairs)
    base_eval_path = output_dir / "base_eval.json"
    base_eval = (
        json.loads(base_eval_path.read_text(encoding="utf-8"))
        if base_eval_path.exists()
        else evaluate_model(tokenizer, teacher, config, eval_pairs)
    )
    if not base_eval_path.exists():
        write_json(base_eval_path, base_eval)

    logs_path = output_dir / "train_metrics.jsonl"
    seq_cache = {"step": -10**9, "targets": None}
    gate_cache = {"step": -10**9, "gate": None}
    history: List[Dict[str, Any]] = []
    student.train()
    start_time = time.time()
    global_step = 0

    while global_step < total_steps:
        try:
            batch = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            batch = next(dataiter)
        batch = to_device(batch, device)
        forget = batch["forget"]
        supervision = batch["supervision"]
        donors = batch["donors"]
        null = batch["null"]
        retain = batch["retain"]
        entity_last = batch["entity_last"]
        donor_ids0 = donors["input_ids"][:, 0]
        donor_mask0 = donors["attention_mask"][:, 0]

        layer_indices = schedule.active_layers(global_step)
        layer_weights = layer_weight_tensor(
            layer_indices, schedule.layer_weights(global_step), torch.device(device)
        )
        point = config["activation_point"]

        with torch.no_grad():
            teacher_plus_full = read_layer_activations(
                teacher, point, layer_indices, forget["input_ids"], forget["attention_mask"], True
            )
            teacher_donor_subject = read_donor_subject_activations(
                teacher, point, layer_indices, donors, entity_last, True
            )
            teacher_null_full = read_layer_activations(
                teacher, point, layer_indices, null["input_ids"], null["attention_mask"], True
            )
            a_plus = gather_subject(teacher_plus_full, entity_last)
            a_minus = teacher_donor_subject
            unknown = gather_subject(teacher_null_full, entity_last)
            unknown_target_margin = cosine_mean(unknown, a_plus).item()
            unknown_donor_margin = cosine_mean(unknown, a_minus[:, 0]).item()

        student_full = read_layer_activations(
            student, point, layer_indices, forget["input_ids"], forget["attention_mask"], False
        )
        h_student = gather_subject(student_full, entity_last)

        delta_info = None
        if idea in {"delta_subspace", "causal_delta", "delta_guide"}:
            delta_info = delta_subspace_target(
                target=a_plus,
                donors=a_minus,
                donor_valid=donors["valid"],
                alpha=float(config["alpha"]),
                rank=int(config["svd_rank"]),
            )

        if idea == "baseline_direct":
            act_loss = layered_mse(h_student, a_minus[:, 0], layer_weights)
            aux_metrics: Dict[str, float] = {}
        elif idea == "tricon":
            tri = tricon_loss(
                student=h_student,
                unknown=unknown,
                target=a_plus,
                donors=a_minus,
                donor_valid=donors["valid"],
                weights=layer_weights,
                tau_c=float(config["tau_c"]),
            )
            act_loss = tri["loss"]
            aux_metrics = {
                "unknown_target_margin": float(tri["unknown_target_margin"].detach().item()),
                "unknown_donor_margin": float(tri["unknown_donor_margin"].detach().item()),
            }
        elif idea == "delta_subspace":
            act_loss = layered_mse(h_student, delta_info["target"], layer_weights)
            aux_metrics = {
                "projection_energy": float(delta_info["projection_energy"].mean().item())
            }
        elif idea == "causal_delta":
            if (
                global_step - gate_cache["step"] >= int(config["gate_refresh_steps"])
                or gate_cache["gate"] is None
                or gate_cache["gate"].shape[1] != len(layer_indices)
            ):
                gate_cache["gate"] = compute_gradient_causal_gate(
                    teacher=teacher,
                    forget=supervision,
                    entity_last=entity_last,
                    target=a_plus,
                    mean_delta=delta_info["mean_delta"],
                    layer_indices=layer_indices,
                    point=point,
                )
                gate_cache["step"] = global_step
            gate = gate_cache["gate"]
            gated_delta = gate.unsqueeze(-1) * delta_info["mean_delta"]
            basis = delta_info["basis"]
            projection = (gated_delta @ basis) @ basis.T
            causal_target = (a_plus - float(config["alpha"]) * projection).detach()
            act_loss = layered_mse(h_student, causal_target, layer_weights)
            aux_metrics = {
                "gate_mean": float(gate.mean().item()),
                "projection_energy": float(delta_info["projection_energy"].mean().item()),
            }
        elif idea == "delta_guide":
            act_loss = torch.zeros((), device=device)
            aux_metrics = {
                "projection_energy": float(delta_info["projection_energy"].mean().item())
            }
        else:
            raise ValueError(f"Unsupported idea: {idea}")

        if global_step - seq_cache["step"] >= int(config["seq_refresh_steps"]) or seq_cache["targets"] is None:
            seq_cache["targets"] = generate_pair_output_traces(
                teacher_model=teacher,
                student_model=student,
                teacher_prompt_ids=donor_ids0,
                teacher_prompt_mask=donor_mask0,
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
            teacher_prompt_ids=donor_ids0,
            teacher_prompt_attention_mask=donor_mask0,
            student_prompt_ids=forget["input_ids"],
            student_prompt_attention_mask=forget["attention_mask"],
            teacher_output_ids=targets["teacher_output_ids"],
            student_output_ids=targets["student_output_ids"],
            teacher_output_mask=targets["teacher_output_mask"],
            student_output_mask=targets["student_output_mask"],
            pad_token_id=int(tokenizer.pad_token_id),
            eos_token_id=int(tokenizer.eos_token_id),
            eta=float(config["eta"]),
            start_index=1,
        )
        seq_loss = seq_metrics["loss"]

        retain_out = student(
            input_ids=retain["input_ids"],
            attention_mask=retain["attention_mask"],
            labels=retain["labels"],
        )
        retain_loss = retain_out.loss

        if idea == "delta_guide":
            forget_out = student(
                input_ids=supervision["input_ids"],
                attention_mask=supervision["attention_mask"],
                labels=supervision["labels"],
            )
            npo_loss = -forget_out.loss
            forget_objective = npo_loss + float(config["lambda_seq"]) * seq_loss
            optimizer.zero_grad(set_to_none=True)
            forget_objective.backward()
            forget_grads = [
                parameter.grad.detach().clone() if parameter.grad is not None else None
                for _, parameter in named_parameters
            ]
            optimizer.zero_grad(set_to_none=True)
            retain_loss.backward()
            retain_grads = [
                parameter.grad.detach().clone() if parameter.grad is not None else None
                for _, parameter in named_parameters
            ]
            energy = delta_info["projection_energy"].mean(dim=0)
            layer_scales = {
                layer: float(energy[index]) for index, layer in enumerate(layer_indices)
            }
            apply_gradient_surgery(
                named_parameters,
                forget_grads,
                retain_grads,
                layer_scales,
                retain_weight=float(config["beta_retain"]),
            )
            logged_loss = float(forget_objective.detach().item()) + float(
                config["beta_retain"]
            ) * float(retain_loss.detach().item())
            aux_metrics["npo_loss"] = float(npo_loss.detach().item())
        else:
            logged_loss = (
                float(config["lambda_act"]) * float(act_loss.detach().item())
                + float(config["lambda_seq"]) * float(seq_loss.detach().item())
                + float(config["beta_retain"]) * float(retain_loss.detach().item())
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss = (
                float(config["lambda_act"]) * act_loss
                + float(config["lambda_seq"]) * seq_loss
                + float(config["beta_retain"]) * retain_loss
            )
            total_loss.backward()
            logged_loss = float(total_loss.detach().item())

        torch.nn.utils.clip_grad_norm_(student.parameters(), float(config["grad_clip"]))
        optimizer.step()
        scheduler.step()
        global_step += 1

        record: Dict[str, Any] = {
            "step": global_step,
            "idea": idea,
            "condition": condition,
            "loss": logged_loss,
            "act_loss": float(act_loss.detach().item()),
            "seq_loss": float(seq_loss.detach().item()),
            "retain_loss": float(retain_loss.detach().item()),
            "active_layers": len(layer_indices),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "elapsed_sec": time.time() - start_time,
            "unknown_cos_target": unknown_target_margin,
            "unknown_cos_donor": unknown_donor_margin,
            **aux_metrics,
        }
        history.append(record)
        with logs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if global_step % int(config["log_every"]) == 0:
            print(
                f"{condition} STEP {global_step}/{total_steps} loss={logged_loss:.6f} "
                f"act={float(act_loss.detach().item()):.6f} seq={float(seq_loss.detach().item()):.6f}",
                flush=True,
            )

    print(f"EVALUATING_FINAL {condition}", flush=True)
    final_eval = evaluate_model(tokenizer, student, config, eval_pairs)
    base_gen = base_eval["generation"]
    final_gen = final_eval["generation"]
    target_drop = base_gen["target_correct_rate"] - final_gen["target_correct_rate"]
    donor_rate = final_gen["donor_answer_rate"]
    refusal_rate = final_gen["refusal_rate"]
    retain_delta = final_eval["retain"]["retain_nll"] - base_eval["retain"]["retain_nll"]
    passed = (
        target_drop >= 0.2
        and donor_rate <= 0.15
        and refusal_rate <= 0.25
        and retain_delta <= 0.5
    )
    candidate_score = target_drop - 0.5 * donor_rate - 0.5 * refusal_rate - max(0.0, retain_delta)

    results = {
        "protocol": config["protocol"],
        "idea": idea,
        "condition": condition,
        "model": config["model"],
        "eval_mode": "student_without_runtime_patch",
        "train_steps": global_step,
        "config": config,
        "train_metrics_last": history[-1] if history else None,
        "train_metrics_mean_last_20": {
            key: sum(record.get(key, 0.0) for record in history[-20:]) / max(1, len(history[-20:]))
            for key in ("loss", "act_loss", "seq_loss", "retain_loss", "projection_energy", "gate_mean")
        },
        "base_eval": base_eval,
        "final_eval": final_eval,
        "deltas": {
            "target_correct_drop": target_drop,
            "donor_answer_rate": donor_rate,
            "refusal_rate": refusal_rate,
            "retain_nll_delta": retain_delta,
        },
        "selection": {
            "threshold_pass": passed,
            "candidate_score": candidate_score,
        },
        "conclusion": (
            "通过选择性遗忘组合门槛，仍需 G5/G6 抗恢复证据才能称深度遗忘。"
            if passed
            else "未通过选择性遗忘组合门槛，不能主张该 idea 已验证成立。"
        ),
        "artifacts": {
            "checkpoint": None,
            "train_log": str(logs_path),
            "base_eval": str(base_eval_path),
        },
    }
    checkpoint_path = output_dir / "checkpoint-last.pt"
    torch.save({"model": student.state_dict()}, checkpoint_path)
    results["artifacts"]["checkpoint"] = str(checkpoint_path)
    write_json(final_path, results)
    write_jsonl(output_dir / "final_summary.jsonl", [results])
    print(f"FINAL_RESULTS={final_path}", flush=True)
    print(
        f"SUMMARY {condition} target_drop={target_drop:.4f} donor={donor_rate:.4f} "
        f"refusal={refusal_rate:.4f} retain_delta={retain_delta:.4f} score={candidate_score:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

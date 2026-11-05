#!/usr/bin/env python3
"""TOFU eval for one local checkpoint. Does not delete weights or touch NO-GO files."""

import os

os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface"
os.environ["HF_HUB_CACHE"] = "/root/autodl-tmp/huggingface/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/autodl-tmp/huggingface/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/usr/local/ASU-unlearning")
REPO = ROOT / "Right-to-be-forgotten"
sys.path.insert(0, str(REPO))

from eval import model_eval  # noqa: E402

TOKENIZER_DIR = Path(
    "/root/autodl-tmp/huggingface/hub/models--NousResearch--Llama-2-7b-chat-hf/"
    "snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee"
)
CHAT_CONFIG = TOKENIZER_DIR
TOKENMAP = ROOT / "workspace/J/spans/base/tokenmap_v1/forget05_pilot200_tokenmap.jsonl"
PERTURBED = REPO / "data/tofu/forget05_perturbed.json"
TASK1_FORGET = Path(
    "/root/autodl-tmp/ASU-unlearning/checkpoints/tofu_replay/llama2-7b/forget05/ASU+GD/"
    "seed_1001/epoch5_1e-05_FixRefTrue_maskTrue_Fcoeff0.1_Rcoeff1.0_Temp2.3_LayersNone_beta0.1/"
    "1/unlearn_times_1/task_data"
)
PREPARED = ROOT / "workspace/J/eval_sets"


def load_cfg(forget_loss: str, save_dir: Path):
    cfg = OmegaConf.load(REPO / "config/tofu.yaml")
    cfg.model_family = "llama2-7b"
    cfg.forget_loss = forget_loss
    cfg.split = "forget05"
    cfg.task_id = 1
    cfg.num_epochs = 5
    cfg.lr = 1e-5
    cfg.forget_coeff = 0.1
    cfg.regularization_coeff = 1.0
    cfg.fix_ref_model = True
    cfg.mask = True
    cfg.save_checkpoint = True
    cfg.save_dir = str(save_dir)
    cfg.eval.model_family = "llama2-7b"
    cfg.eval.ds_size = 300
    cfg.eval.batch_size = 30
    cfg.eval.generation.max_length = 200
    cfg.eval.generation.max_new_tokens = None
    cfg.eval.data_path = ["data/tofu", "data/tofu", "data/tofu", "data/tofu"]
    cfg.eval.split_list = [
        "retain_perturbed",
        "real_authors_perturbed",
        "world_facts_perturbed",
        "forget05_perturbed",
    ]
    cfg.eval.eval_task = [
        "eval_log",
        "eval_real_author_wo_options",
        "eval_real_world_wo_options",
        "eval_log_forget",
    ]
    cfg.eval.question_key = ["question", "question", "question", "question"]
    cfg.eval.answer_key = ["answer", "answer", "answer", "answer"]
    cfg.eval.base_answer_key = ["paraphrased_answer", "answer", "answer", "paraphrased_answer"]
    cfg.eval.perturbed_answer_key = [
        "perturbed_answer",
        "perturbed_answer",
        "perturbed_answer",
        "perturbed_answer",
    ]
    return cfg


def prepare_subsets():
    PREPARED.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in TOKENMAP.read_text().splitlines() if line.strip()]
    wanted = {row["question"]: row for row in rows}
    if len(wanted) != 200:
        raise SystemExit(f"tokenmap questions are not unique: {len(wanted)}")
    pilot = []
    overlap = []
    seen = set()
    with PERTURBED.open() as handle:
        for line in handle:
            item = json.loads(line)
            if item["question"] not in wanted:
                continue
            if item["question"] in seen:
                raise SystemExit(f"duplicate perturbed question: {item['question']}")
            seen.add(item["question"])
            pilot.append(item)
            if str(item["task_id"]) == "1":
                overlap.append(item)
    missing = set(wanted) - seen
    if missing:
        raise SystemExit(f"pilot questions missing from forget05_perturbed: {len(missing)}")
    write_jsonl(PREPARED / "pilot200" / "forget_perturbed.json", pilot)
    write_jsonl(PREPARED / "overlap_task1" / "forget_perturbed.json", overlap)
    meta = {"pilot": len(pilot), "overlap_task1": len(overlap)}
    (PREPARED / "counts.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("prepared", meta, flush=True)
    return meta


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_suite(cfg, model, tokenizer, save_dir: Path, forget_path: Path, tasks):
    cfg.eval.data_path = ["data/tofu"] * len(tasks["eval_task"])
    cfg.eval.split_list = tasks["split_list"]
    cfg.eval.eval_task = tasks["eval_task"]
    cfg.eval.question_key = ["question"] * len(tasks["eval_task"])
    cfg.eval.answer_key = ["answer"] * len(tasks["eval_task"])
    cfg.eval.base_answer_key = tasks["base_answer_key"]
    cfg.eval.perturbed_answer_key = ["perturbed_answer"] * len(tasks["eval_task"])
    save_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    return model_eval(cfg, 1, 1, model, tokenizer, str(save_dir), str(forget_path), "last")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--forget-loss", required=True)
    parser.add_argument("--suites", default="task1,pilot,overlap")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
        raise SystemExit(f"checkpoint is not a local HF directory: {checkpoint}")
    if "locuslab" in str(checkpoint):
        raise SystemExit(f"refusing hub id path: {checkpoint}")

    prepare_subsets()
    cfg = load_cfg(args.forget_loss, args.out)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(CHAT_CONFIG, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map={"": int(os.environ["LOCAL_RANK"])},
        local_files_only=True,
    )
    model.eval()

    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    summary = {}
    if "task1" in suites:
        summary["task1"] = run_suite(
            cfg,
            model,
            tokenizer,
            args.out / "task1",
            TASK1_FORGET,
            {
                "split_list": [
                    "retain_perturbed",
                    "real_authors_perturbed",
                    "world_facts_perturbed",
                    "forget05_perturbed",
                ],
                "eval_task": [
                    "eval_log",
                    "eval_real_author_wo_options",
                    "eval_real_world_wo_options",
                    "eval_log_forget",
                ],
                "base_answer_key": ["paraphrased_answer", "answer", "answer", "paraphrased_answer"],
            },
        )
    if "pilot" in suites:
        summary["pilot"] = run_suite(
            cfg,
            model,
            tokenizer,
            args.out / "pilot200",
            PREPARED / "pilot200",
            {
                "split_list": ["forget_perturbed"],
                "eval_task": ["eval_log_forget"],
                "base_answer_key": ["paraphrased_answer"],
            },
        )
    if "overlap" in suites:
        summary["overlap"] = run_suite(
            cfg,
            model,
            tokenizer,
            args.out / "overlap_task1",
            PREPARED / "overlap_task1",
            {
                "split_list": ["forget_perturbed"],
                "eval_task": ["eval_log_forget"],
                "base_answer_key": ["paraphrased_answer"],
            },
        )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print("EVAL_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()

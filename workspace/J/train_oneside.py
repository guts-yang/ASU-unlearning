#!/usr/bin/env python3
"""Minimal ASU+GD+Cap run on the confirmed pilot200 span set.

Ignores the written M0 NO-GO by explicit user decision. Does not rewrite that
verdict and does not overwrite the existing ASU+GD checkpoint.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import sys
from pathlib import Path

import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer

ROOT = Path("/usr/local/ASU-unlearning")
REPO = ROOT / "Right-to-be-forgotten"
sys.path.insert(0, str(ROOT / "workspace"))
sys.path.insert(0, str(REPO))

from dataset import custom_data_collator_forget  # noqa: E402
from gold_rank.masking import encode_forget_example  # noqa: E402
from my_models.my_llama import LlamaForCausalLM  # noqa: E402
from trainer import CustomTrainerForgetting  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

TOKENMAP = ROOT / "workspace/J/spans/base/tokenmap_v1/forget05_pilot200_tokenmap.jsonl"
RETAIN = REPO / "data/tofu/retain50.json"
TOKENIZER_DIR = Path(
    "/root/autodl-tmp/huggingface/hub/models--NousResearch--Llama-2-7b-chat-hf/"
    "snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee"
)
MODEL_DIR = Path("/root/autodl-tmp/models/tofu_ft_llama2-7b")
SAVE_DIR = Path("/root/autodl-tmp/ASU-unlearning/checkpoints/dard_oneside/seed_1001")
MODEL_CONFIGS = {
    "question_start_tag": "[INST] ",
    "question_end_tag": " [/INST]",
    "answer_tag": "",
}
MAX_LENGTH = 500


def local_dir(path: Path, label: str) -> Path:
    if "locuslab" in str(path) or not path.is_dir() or not (path / "config.json").is_file():
        raise SystemExit(f"{label} is not a local snapshot: {path}")
    return path


class CapDataset(Dataset):
    def __init__(self, rows, retain_rows, tokenizer):
        self.rows = rows
        self.retain_rows = retain_rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def _encode(self, question, answer):
        return encode_forget_example(
            self.tokenizer, MAX_LENGTH, question, answer, MODEL_CONFIGS, mask=True
        )

    def __getitem__(self, idx):
        row = self.rows[idx]
        input_ids, labels, attention_mask = self._encode(row["question"], row["answer"])
        m_tar = torch.zeros(MAX_LENGTH, dtype=torch.long)
        for position in row["m_tar_token_positions"]:
            m_tar[int(position)] = 1
        forget = (input_ids, labels, attention_mask, m_tar)

        generator = torch.Generator()
        generator.manual_seed(1001 + idx)
        retain_index = int(torch.randint(0, len(self.retain_rows), (1,), generator=generator))
        retain_row = self.retain_rows[retain_index]
        retain = self._encode(retain_row["question"], retain_row["answer"])
        return [forget, retain, retain, retain, retain]


def load_rows():
    rows = [json.loads(line) for line in TOKENMAP.read_text().splitlines() if line.strip()]
    if len(rows) != 200:
        raise SystemExit(f"expected 200 tokenmap rows, found {len(rows)}")
    retain = [json.loads(line) for line in RETAIN.read_text().splitlines() if line.strip()]
    if not retain:
        raise SystemExit(f"retain set is empty: {RETAIN}")
    return rows, retain


def main():
    if (SAVE_DIR / "checkpoint-last").exists():
        raise SystemExit(f"checkpoint already exists: {SAVE_DIR / 'checkpoint-last'}")
    local_dir(MODEL_DIR, "model")
    local_dir(TOKENIZER_DIR, "tokenizer")
    rows, retain_rows = load_rows()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    dataset = CapDataset(rows, retain_rows, tokenizer)

    batch_size = 4
    grad_accum = 4
    num_epochs = 5
    max_steps = int(num_epochs * len(dataset)) // (batch_size * grad_accum * world)
    steps_per_epoch = len(dataset) // (batch_size * grad_accum * world)
    if max_steps < 1:
        raise SystemExit(f"max_steps computed as {max_steps}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if local_rank == 0:
        print(f"Saving to: {SAVE_DIR}", flush=True)
        print(f"max_steps={max_steps} world={world}", flush=True)

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=steps_per_epoch if steps_per_epoch > 1 else 0,
        max_steps=max_steps,
        learning_rate=1e-5,
        bf16=True,
        bf16_full_eval=True,
        output_dir=str(SAVE_DIR),
        optim="paged_adamw_32bit",
        deepspeed=str(REPO / "config/ds_config/llama2.json"),
        save_steps=max_steps,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        weight_decay=0.01,
        seed=1001,
        eval_strategy="no",
        report_to="none",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        config=config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    reference = LlamaForCausalLM.from_pretrained(
        MODEL_DIR,
        config=config,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    reference.eval()

    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=custom_data_collator_forget,
        loss_type="ASU+GD+Cap",
        ref_model=reference,
        attention_temp=2.3,
        layers_id=None,
        beta=0.1,
        forget_coeff=0.1,
        regularization_coeff=1.0,
    )
    if local_rank == 0:
        print("Start Training ...", flush=True)
    trainer.train()
    if local_rank == 0:
        finished = SAVE_DIR / f"checkpoint-{max_steps}"
        target = SAVE_DIR / "checkpoint-last"
        if finished.exists() and not target.exists():
            finished.rename(target)
            print(f"checkpoint-last: {target}", flush=True)


if __name__ == "__main__":
    main()

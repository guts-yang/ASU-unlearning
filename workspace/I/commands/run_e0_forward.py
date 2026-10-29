#!/usr/bin/env python3
"""Teacher-only MCSD forwards. No Student update and no edits to upstream sources."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

ROOT = Path("/usr/local/ASU-unlearning")
RTBF = ROOT / "Right-to-be-forgotten"
sys.path.insert(0, str(RTBF))
sys.path.insert(0, str(ROOT / "workspace/I"))
os.chdir(RTBF)

from dataset.data_module import convert_raw_forget_data_to_model_format  # noqa: E402
from mcsd.teacher import GateTemperatureTeacher  # noqa: E402
from my_models.my_llama import LlamaForCausalLM  # noqa: E402


def load_jobs(path):
    return json.loads(Path(path).read_text())


def tokenize_samples(samples, tokenizer, config):
    model_configs = {
        "question_start_tag": config["question_start_tag"],
        "question_end_tag": config["question_end_tag"],
        "answer_tag": config["answer_tag"],
    }
    encoded = []
    for sample in samples:
        input_ids, labels, attention_mask = convert_raw_forget_data_to_model_format(
            tokenizer,
            int(config["max_length"]),
            sample["question"],
            sample["answer"],
            model_configs,
            mask=bool(config["mask_question"]),
        )
        encoded.append((input_ids, labels, attention_mask))
    return encoded


def answer_stats(logits, labels):
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size(0), -1)
    mask = shift_labels != -100
    n_tokens = mask.sum(dim=-1)
    nll_sum = token_nll.sum(dim=-1)
    safe_labels = shift_labels.clamp(min=0)
    gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    gold_sum = (gathered * mask).sum(dim=-1)
    gold_logit = shift_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    rank = (shift_logits > gold_logit.unsqueeze(-1)).sum(dim=-1) + 1
    rank_sum = (rank.float() * mask).sum(dim=-1)
    return nll_sum, n_tokens, gold_sum, rank_sum


def forward_job(teacher, encoded, samples, job, batch_size, device):
    rows = []
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start:start + batch_size]
        input_ids = torch.stack([item[0] for item in batch]).to(device)
        labels = torch.stack([item[1] for item in batch]).to(device)
        attention_mask = torch.stack([item[2] for item in batch]).to(device)
        outputs = teacher.forward(
            input_ids,
            attention_mask,
            attention_temp=float(job["attention_temp"]),
            mlp_gate_temp=float(job["mlp_gate_temp"]),
            layers_id=None,
            mlp_layers_id=None,
        )
        nll_sum, n_tokens, gold_sum, rank_sum = answer_stats(outputs.logits, labels)
        for offset, sample in enumerate(samples[start:start + batch_size]):
            tokens = int(n_tokens[offset].item())
            rows.append({
                "sample_id": sample["sample_id"],
                "subset": sample["subset"],
                "split": sample["split"],
                "author_index": sample["author_index"],
                "n_tokens": tokens,
                "nll_sum": float(nll_sum[offset].item()),
                "nll_mean": None if tokens == 0 else float(nll_sum[offset].item()) / tokens,
                "gold_logp_sum": float(gold_sum[offset].item()),
                "gold_logp_mean": None if tokens == 0 else float(gold_sum[offset].item()) / tokens,
                "gold_rank_mean": None if tokens == 0 else float(rank_sum[offset].item()) / tokens,
            })
        del outputs, input_ids, labels, attention_mask
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--gpu", default="cpu")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    manifest = json.loads(Path(config["manifest_path"]).read_text())
    samples = manifest["samples"]
    jobs = load_jobs(args.jobs)
    output_dir = Path(config["output_dir"]) / "forwards"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gpu == "cpu" or not torch.cuda.is_available():
        if args.gpu != "cpu":
            raise RuntimeError("CUDA is not available; pass --gpu cpu only for an explicit CPU run")
        device = torch.device("cpu")
        dtype = torch.float32
        attn = "eager"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda:0")
        dtype = torch.bfloat16
        attn = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenize_samples(samples, tokenizer, config)
    empty = sum(int((labels[1:] != -100).sum()) == 0 for _, labels, _ in encoded)
    if empty:
        raise RuntimeError(f"{empty} samples have no answer tokens at max_length")

    model = LlamaForCausalLM.from_pretrained(
        config["model_path"],
        attn_implementation=attn,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    teacher = GateTemperatureTeacher(model)
    if int(model.config.num_hidden_layers) != int(config["layer_count"]):
        raise RuntimeError(
            f"config layer_count={config['layer_count']} but checkpoint has {model.config.num_hidden_layers}"
        )

    expected = len(samples)
    batch_size = int(config["batch_size"])
    with torch.inference_mode():
        for job in jobs:
            destination = output_dir / f"{job['name']}.jsonl"
            if destination.exists() and sum(1 for _ in destination.open()) == expected:
                print(f"skip {job['name']}", flush=True)
                continue
            started = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            rows = forward_job(teacher, encoded, samples, job, batch_size, device)
            if len(rows) != expected:
                raise RuntimeError(f"{job['name']} wrote {len(rows)} rows, expected {expected}")
            with destination.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            peak = None
            if device.type == "cuda":
                peak = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
            meta = {
                "name": job["name"],
                "attention_temp": job["attention_temp"],
                "mlp_gate_temp": job["mlp_gate_temp"],
                "seconds": round(time.time() - started, 3),
                "peak_mem_mb": peak,
                "device": str(device),
                "attn_implementation": attn,
                "rows": len(rows),
            }
            (output_dir / f"{job['name']}.meta.json").write_text(json.dumps(meta) + "\n")
            print(f"done {job['name']} {meta['seconds']}s peak={peak}", flush=True)


if __name__ == "__main__":
    main()

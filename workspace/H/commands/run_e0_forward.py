#!/usr/bin/env python3
"""Teacher-only answer-token NLL for one GPU's E0 job list. No backward and no checkpoint."""

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

RTBF = Path("/usr/local/ASU-unlearning/Right-to-be-forgotten")
sys.path.insert(0, str(RTBF))
os.chdir(RTBF)

from dataset.data_module import convert_raw_forget_data_to_model_format  # noqa: E402
from my_models.my_llama import LlamaForCausalLM  # noqa: E402


def load_jobs(path):
    jobs = json.loads(Path(path).read_text())
    for job in jobs:
        if job.get("layers_id") is not None:
            job["layers_id"] = [int(layer) for layer in job["layers_id"]]
    return jobs


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


def answer_nll(logits, labels):
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size(0), -1)
    n_tokens = (shift_labels != -100).sum(dim=-1)
    nll_sum = token_loss.sum(dim=-1)
    return nll_sum, n_tokens


def forward_job(model, encoded, samples, job, batch_size, device):
    rows = []
    layers_id = job["layers_id"]
    attention_temp = float(job["attention_temp"])
    for start in range(0, len(encoded), batch_size):
        batch = encoded[start:start + batch_size]
        input_ids = torch.stack([item[0] for item in batch]).to(device)
        labels = torch.stack([item[1] for item in batch]).to(device)
        attention_mask = torch.stack([item[2] for item in batch]).to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_temp=attention_temp,
            layers_id=layers_id,
            use_cache=False,
        )
        nll_sum, n_tokens = answer_nll(outputs.logits, labels)
        for offset, sample in enumerate(samples[start:start + batch_size]):
            tokens = int(n_tokens[offset].item())
            total = float(nll_sum[offset].item())
            row = {
                "sample_id": sample["sample_id"],
                "subset": sample["subset"],
                "split": sample["split"],
                "author_index": sample["author_index"],
                "n_tokens": tokens,
                "nll_sum": total,
                "nll_mean": None if tokens == 0 else total / tokens,
            }
            rows.append(row)
        del outputs, input_ids, labels, attention_mask
    return rows


def smoke_check(model, encoded, device):
    batch = encoded[:2]
    input_ids = torch.stack([item[0] for item in batch]).to(device)
    attention_mask = torch.stack([item[2] for item in batch]).to(device)
    base = model(input_ids=input_ids, attention_mask=attention_mask, attention_temp=1.0, layers_id=None, use_cache=False)
    alt = model(input_ids=input_ids, attention_mask=attention_mask, attention_temp=2.3, layers_id=[0], use_cache=False)
    delta = (base.logits.float() - alt.logits.float()).abs().max().item()
    if delta == 0.0:
        raise RuntimeError("layer-0 temperature did not change logits")
    return delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    manifest = json.loads(Path(config["manifest_path"]).read_text())
    samples = manifest["samples"]
    jobs = load_jobs(args.jobs)
    output_dir = Path(config["output_dir"]) / "forwards"
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenize_samples(samples, tokenizer, config)
    empty = sum(int((labels[1:] != -100).sum()) == 0 for _, labels, _ in encoded)
    if empty:
        raise RuntimeError(f"{empty} samples have no answer tokens at max_length")

    model = LlamaForCausalLM.from_pretrained(
        config["model_path"],
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False

    if args.smoke:
        delta = smoke_check(model, encoded, device)
        smoke_path = Path(config["output_dir"]) / "smoke.json"
        smoke_path.write_text(json.dumps({"max_abs_logit_delta": delta, "gpu": args.gpu}) + "\n")
        print(f"smoke max_abs_logit_delta={delta}", flush=True)

    expected = len(samples)
    batch_size = int(config["batch_size"])
    with torch.inference_mode():
        for job in jobs:
            destination = output_dir / f"{job['name']}.jsonl"
            meta_path = output_dir / f"{job['name']}.meta.json"
            if destination.exists() and sum(1 for _ in destination.open()) == expected:
                print(f"skip {job['name']}", flush=True)
                continue
            started = time.time()
            torch.cuda.reset_peak_memory_stats()
            try:
                rows = forward_job(model, encoded, samples, job, batch_size, device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows = forward_job(model, encoded, samples, job, max(1, batch_size // 2), device)
            if len(rows) != expected:
                raise RuntimeError(f"{job['name']} wrote {len(rows)} rows, expected {expected}")
            with destination.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            meta = {
                "name": job["name"],
                "attention_temp": job["attention_temp"],
                "layers_id": job["layers_id"],
                "seconds": round(time.time() - started, 3),
                "peak_mem_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1),
                "gpu": args.gpu,
                "rows": len(rows),
            }
            meta_path.write_text(json.dumps(meta) + "\n")
            print(f"done {job['name']} {meta['seconds']}s {meta['peak_mem_mb']}MB", flush=True)


if __name__ == "__main__":
    main()

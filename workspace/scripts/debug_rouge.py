"""Isolate the SIGFPE: generate from student, then score with rouge_scorer."""
import sys, os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "open-unlearning" / "src"), str(REPO / "workspace" / "src")]

import torch
from omegaconf import OmegaConf
from datasets import load_dataset
from transformers import AutoTokenizer
from model import get_model

STUDENT = "/jizhicfs/chenyliao/runs/qkspec/unlearn/qkspec_Llama-3.2-1B-Instruct_forget10_20261120_232713"
TOFU = "/jizhicfs/chenyliao/data/datasets/locuslab-TOFU"

model_cfg = OmegaConf.create({
    "model_args": {"pretrained_model_name_or_path": STUDENT, "attn_implementation": "eager", "torch_dtype": "bfloat16"},
    "tokenizer_args": {"pretrained_model_name_or_path": STUDENT},
})
model, tok = get_model(model_cfg)
model.eval()

ds = load_dataset("json", data_files=f"{TOFU}/forget10_perturbed.json", split="train").select(range(16))
print(f"[1] dataset ok ({len(ds)})", flush=True)

questions = [ds[i]["question"] for i in range(len(ds))]
prompts = [f"Question: {q}\nAnswer:" for q in questions]
enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
print("[2] tokenized", enc["input_ids"].shape, flush=True)

with torch.inference_mode():
    out = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
print("[3] generate ok", out.shape, flush=True)

texts = tok.batch_decode(out, skip_special_tokens=True)
print("[4] decode ok, sample:", repr(texts[0][:80]), flush=True)

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
for i, t in enumerate(texts):
    s = scorer.score(ds[i]["answer"] if isinstance(ds[i]["answer"], str) else ds[i]["answer"][0], t)
    print(f"[5] rouge {i}: rougeL_recall={s['rougeL'].recall:.4f}", flush=True)
print("DEBUG OK", flush=True)

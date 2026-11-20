"""Reproduce SIGFPE with retain90 left-padded generation using model's own gen config."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "open-unlearning" / "src"), str(REPO / "workspace" / "src")]
import torch
from omegaconf import OmegaConf
from datasets import load_dataset
from model import get_model

STUDENT = "/jizhicfs/chenyliao/runs/qkspec/unlearn/qkspec_Llama-3.2-1B-Instruct_forget10_20261120_233401"
TOFU = "/jizhicfs/chenyliao/data/datasets/locuslab-TOFU"
model_cfg = OmegaConf.create({
    "model_args": {"pretrained_model_name_or_path": STUDENT, "attn_implementation": "eager", "torch_dtype": "bfloat16"},
    "tokenizer_args": {"pretrained_model_name_or_path": STUDENT},
})
model, tok = get_model(model_cfg)
model.eval()
if tok.padding_side != "left":
    tok.padding_side = "left"

ds = load_dataset("json", data_files=f"{TOFU}/retain90.json", split="train")
print(f"[1] retain90 {len(ds)}", flush=True)
questions = [f"Question: {ds[i]['question']}\nAnswer:" for i in range(32)]
enc = tok(questions, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
print("[2] tokenized", enc["input_ids"].shape, "gen_cfg:", model.generation_config.do_sample,
      model.generation_config.temperature, model.generation_config.top_p, flush=True)
# Use model's own generation config (as eval.py does) -- do NOT force do_sample
with torch.inference_mode():
    out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                         max_new_tokens=200, pad_token_id=tok.eos_token_id)
print("[3] generate ok", out.shape, flush=True)
print("DEBUG2 OK", flush=True)

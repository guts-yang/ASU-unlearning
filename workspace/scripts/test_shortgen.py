"""Test whether short generation avoids the bf16 cuBLAS SIGFPE in ROUGE eval."""
import sys
from pathlib import Path
REPO = Path("/usr/local/workspace/ASU-unlearning-K4")
sys.path[:0] = [str(REPO / "open-unlearning" / "src"), str(REPO / "workspace" / "src")]
import torch
from omegaconf import OmegaConf
from datasets import load_dataset
from model import get_model
from rouge_score import rouge_scorer

S = "/jizhicfs/chenyliao/runs/qkspec/unlearn/qkspec_Llama-3.2-1B-Instruct_forget10_20261120_233401"
cfg = OmegaConf.create({
    "model_args": {"pretrained_model_name_or_path": S, "attn_implementation": "eager", "torch_dtype": "bfloat16"},
    "tokenizer_args": {"pretrained_model_name_or_path": S},
})
model, tok = get_model(cfg)
model.eval()
tok.padding_side = "left"
print("[1] model loaded on", next(model.parameters()).device, flush=True)

ds = load_dataset("json", data_files="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain90.json", split="train")
prompts = [f"Question: {ds[i]['question']}\nAnswer:" for i in range(64)]
enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
print("[2] tokenized", tuple(enc["input_ids"].shape), flush=True)

for max_new in (32,):
    print(f"[3] generate max_new_tokens={max_new} ...", flush=True)
    with torch.inference_mode():
        out = model.generate(
            input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
            max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id,
            temperature=None, top_p=None, top_k=None,
        )
    print(f"[4] generate OK {tuple(out.shape)}", flush=True)
    texts = tok.batch_decode(out, skip_special_tokens=True)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = [scorer.score(str(ds[i]["answer"]), texts[i])["rougeL"].recall for i in range(len(texts))]
    print(f"[5] rouge OK mean={sum(scores)/len(scores):.4f}", flush=True)
print("SHORTGEN TEST OK", flush=True)

"""Evaluate a QKSpec student with the Right-to-be-forgotten FE/MU metric.

Faithful to Right-to-be-forgotten/metrics/metrics.py::get_eval_results:
  * 4 splits: Forget / Retain / Real Authors / Real World
  * 6 metrics each: ROUGE-L recall, Probability, Truth Ratio, Token Entropy,
    Cosine Similarity, Entailment Score
  * MU   = hmean(18 non-forget components)
  * MU_R = hmean(6 retain components)
  * FE   = hmean(5 forget components: all except cosine; non-entropy use 1-v,
    entropy used as-is)

Model runs in fp32+TF32 on GPU to avoid the bf16 cuBLAS SIGFPE seen during
generation on this node. NLI / sentence-transformer models load from local
snapshot paths.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

REPO = Path("/usr/local/workspace/ASU-unlearning-K4")
RTF = REPO / "Right-to-be-forgotten"
sys.path[:0] = [str(RTF), str(REPO / "open-unlearning" / "src"), str(REPO / "workspace" / "src")]

from scipy.stats import hmean
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

from metrics import (  # type: ignore
    eval_rouge_recall,
    eval_cosine_similarity,
    get_entailment_results,
    get_entailment_score,
    token_entropy,
    eval_perturbation_ratio,
    run_generation,
    get_eval_results,
)
from dataset.data_module import TextDatasetQA, get_batch_loss, custom_data_collator  # type: ignore
from types import SimpleNamespace

TOFU = Path("/jizhicfs/chenyliao/data/datasets/locuslab-TOFU")
NLI_PATH = "/jizhicfs/chenyliao/models/sileod/deberta-v3-base-tasksource-nli"
ST_PATH = "/jizhicfs/chenyliao/models/sentence-transformers/paraphrase-MiniLM-L6-v2"
if not Path(ST_PATH).exists():
    # resolve from HF cache snapshot
    import glob
    cands = glob.glob("/jizhicfs/chenyliao/cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/*")
    if cands:
        ST_PATH = cands[0]

# Llama-3.2 chat template (same family as llama3-8b in RTF model_config)
LLAMA3_CFG = {
    "hf_key": "meta-llama/Llama-3.2-1B-Instruct",
    "question_start_tag": "<|start_header_id|>user<|end_header_id|>\n\n",
    "question_end_tag": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "answer_tag": "",
    "flash_attention2": "false",
    "gradient_checkpointing": "false",
}


def make_loader(tokenizer, split, answer_key, ds_size, max_length, batch_size):
    ds = TextDatasetQA(str(TOFU), tokenizer=tokenizer, model_family="llama3-8b",
                       max_length=max_length, split=split,
                       question_key="question", answer_key=answer_key)
    if ds_size:
        ds.data = ds.data.select(range(min(ds_size, len(ds.data))))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                       collate_fn=custom_data_collator)


def evaluate_split(model, tokenizer, es_pipe, split, answer_key, base_key, pert_key,
                   orig_file, ds_size, max_length, batch_size, eval_task, device):
    """Compute the raw eval_log dict for one split (RTF schema)."""
    t0 = time.time()
    loader = make_loader(tokenizer, split, answer_key, ds_size, max_length, batch_size)
    base_loader = make_loader(tokenizer, split, base_key, ds_size, max_length, max(1, batch_size // 4))
    pert_loader = make_loader(tokenizer, split, pert_key, ds_size, max_length, max(1, batch_size // 4))

    gen_outputs, ground_truths = [], []
    log = {}
    for batch in loader:
        input_ids, labels, attention_mask = [x.to(device) for x in batch]
        b = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        with torch.inference_mode():
            outputs = model(**b)
            gen_cfg = SimpleNamespace(
                model_family="llama3-8b",
                generation=SimpleNamespace(max_length=max_length, max_new_tokens=None),
            )
            _, gen, gt = run_generation(gen_cfg, b, model, tokenizer)
        gen_outputs.extend(gen)
        ground_truths.extend(gt)
        gt_loss = get_batch_loss(outputs.logits, labels)
        ntok = (labels != -100).sum(-1)
        log["avg_gt_loss"] = log.get("avg_gt_loss", []) + (gt_loss / ntok).cpu().numpy().tolist()

    rouge = eval_rouge_recall(gen_outputs, ground_truths)
    log.update(rouge)
    log.update(eval_perturbation_ratio(base_loader, pert_loader, model))
    log.update(token_entropy(tokenizer, gen_outputs, normalize=True))

    # cosine + entailment use the ORIGINAL answers (RTF behavior)
    orig_rows = [json.loads(l) for l in open(TOFU / orig_file) if l.strip()]
    orig_key = "original_answer" if "original_answer" in orig_rows[0] else "answer"
    org_answers = [str(r[orig_key]) for r in orig_rows[: len(gen_outputs)]]
    log.update(eval_cosine_similarity(gen_outputs, org_answers))
    ent = get_entailment_results(es_pipe, gen_outputs, org_answers, eval_task,
                                 rouge["rougeL_recall"], bs=30, tofu=True)
    log.update(ent)
    print(f"  [{eval_task}] {len(gen_outputs)} samples in {time.time()-t0:.0f}s", flush=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ds-size", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda:0"
    print("[1/3] loading student ...", flush=True)
    cfg = AutoConfig.from_pretrained(args.student)
    model = AutoModelForCausalLM.from_pretrained(
        args.student, config=cfg, torch_dtype=torch.float32, attn_implementation="eager",
    ).to(device).float().eval()
    tok = AutoTokenizer.from_pretrained(args.student)
    tok.pad_token = tok.eos_token

    print("[2/3] loading NLI + sentence-transformer ...", flush=True)
    es_pipe = pipeline("text-classification", model=NLI_PATH, device=device)
    # monkey-patch RTF's cosine model path to local snapshot
    import metrics as rtf_metrics
    _orig_embed = rtf_metrics.SentenceTransformer
    class _LocalST:
        def __init__(self, *a, **k):
            self._m = _orig_embed(ST_PATH, device="cuda")
        def encode(self, *a, **k):
            return self._m.encode(*a, **k)
    rtf_metrics.SentenceTransformer = _LocalST

    print("[3/3] evaluating 4 splits ...", flush=True)
    specs = [
        ("eval_log", "forget10_perturbed", "answer", "paraphrased_answer", "perturbed_answer", "eval_log_forget.json"),
        ("eval_log", "retain_perturbed", "answer", "answer", "perturbed_answer", "eval_log.json"),
        ("eval_real_author_wo_options", "real_authors_perturbed", "answer", "answer", "perturbed_answer", "eval_real_author_wo_options.json"),
        ("eval_real_world_wo_options", "world_facts_perturbed", "answer", "answer", "perturbed_answer", "eval_real_world_wo_options.json"),
    ]
    agg = {}
    for eval_task, split, ak, bk, pk, out_key in specs:
        agg[out_key] = evaluate_split(model, tok, es_pipe, split, ak, bk, pk,
                                      f"{split}.json", args.ds_size, args.max_length,
                                      args.batch_size, eval_task, device)

    results = get_eval_results(agg)
    os.makedirs(args.out, exist_ok=True)
    with open(Path(args.out) / "rtf_fe_mu.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(Path(args.out) / "rtf_eval_logs.json", "w") as f:
        json.dump(agg, f, default=str)
    print("=== RTF FE/MU ===", flush=True)
    print(json.dumps(results, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

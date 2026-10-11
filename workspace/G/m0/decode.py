"""Greedy decoding for the three locked M0 prompt modes."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import TOKENIZER_ID, chat_prompt, max_new_tokens, read_jsonl, write_jsonl


def generate_split(model, tokenizer, rows: list[dict], mode: str, batch_size: int) -> list[dict]:
    prompts = [chat_prompt(row["question"], mode) for row in rows]
    encoded = tokenizer(
        prompts,
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
    )
    outputs = []
    limit = max_new_tokens(mode)
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        input_ids = encoded["input_ids"][start:stop].to(model.device)
        attention_mask = encoded["attention_mask"][start:stop].to(model.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=limit,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for offset, row in enumerate(rows[start:stop]):
            prompt_len = int(prompt_lengths[offset])
            # Left padding: the prompt occupies the last prompt_len tokens.
            new_ids = generated[offset, input_ids.shape[1]:].tolist()
            # Drop a trailing eos / pad if generate appended it.
            while new_ids and new_ids[-1] in {tokenizer.eos_token_id, tokenizer.pad_token_id}:
                new_ids.pop()
            prompt_ids = input_ids[offset, -prompt_len:].tolist()
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            outputs.append({
                "index": row["index"],
                "question": row["question"],
                "answer": row["answer"],
                "mode": mode,
                "prompt": prompts[start + offset],
                "prompt_token_ids": prompt_ids,
                "output": text,
                "output_token_ids": new_ids,
            })
        print(f"{mode} {stop}/{len(rows)}", flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mode", required=True, choices=["DirectQA", "DefaultThink", "ZeroThink"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        done = read_jsonl(args.out)
        rows = read_jsonl(args.data)
        if len(done) == len(rows):
            print(f"exists {args.out} n={len(done)}")
            return
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    model.eval()
    model.generation_config.do_sample = False
    rows = read_jsonl(args.data)
    decoded = generate_split(model, tokenizer, rows, args.mode, args.batch_size)
    write_jsonl(args.out, decoded)


if __name__ == "__main__":
    main()

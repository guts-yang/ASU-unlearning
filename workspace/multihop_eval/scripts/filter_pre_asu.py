#!/usr/bin/env python3
"""Keep probes the model already answers, before any unlearning run.

Uses the frozen primary score: case-insensitive containment of the answer or an alias.
Does not download weights. A hub id that is not a local directory is refused.
Subset files are written only when all six reasoning types still have a kept multi-hop item.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REASONING_TYPES = {
    "chain_implication",
    "contrapositive_reasoning",
    "disjunctive_elimination",
    "identity_substitution",
    "instance_subsumption",
    "multi_element_intersection",
}


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def hit(prediction: str, answer: str, aliases: list[str]) -> bool:
    folded = normalize(prediction)
    candidates = [answer, *aliases]
    return any(normalize(candidate) and normalize(candidate) in folded for candidate in candidates)


def generate_predictions(model_dir: Path, tokenizer_dir: Path, questions: list[str]) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    predictions = []
    for question in questions:
        prompt = f"Question: {question}\nAnswer:"
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        input_ids = input_ids.to(model.device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = output_ids[0, input_ids.shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        for stop in ("\n\n", "\nQuestion", "Question:"):
            text = text.split(stop)[0]
        predictions.append(text)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--singlehop", type=Path, required=True)
    parser.add_argument("--multihop", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.model.is_dir() or not args.tokenizer.is_dir():
        print(
            "pre-ASU filter not run: model and tokenizer must be local directories. "
            "This script does not download MUSE-Books_target or any other weights."
        )
        return

    single = json.loads(args.singlehop.read_text(encoding="utf-8"))
    multi = json.loads(args.multihop.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in single}
    questions = [row["question"] for row in single] + [row["question"] for row in multi]
    predictions = generate_predictions(args.model, args.tokenizer, questions)
    single_pred = predictions[: len(single)]
    multi_pred = predictions[len(single) :]

    correct_single = {}
    kept_single = []
    for row, prediction in zip(single, single_pred):
        ok = hit(prediction, row["answer"], row.get("aliases") or [])
        correct_single[row["id"]] = ok
        if ok:
            kept_single.append(row)

    kept_multi = []
    for row, prediction in zip(multi, multi_pred):
        supports = row.get("support_ids") or []
        if not supports or any(not correct_single.get(support_id, False) for support_id in supports):
            continue
        if support_id_missing(supports, by_id):
            continue
        if not hit(prediction, row["answer"], row.get("aliases") or []):
            continue
        kept_multi.append(row)

    present = {row["reasoning_type"] for row in kept_multi}
    if present != REASONING_TYPES:
        missing = sorted(REASONING_TYPES - present)
        print(
            "subset files not written: kept multi-hop items are missing reasoning types: "
            + ", ".join(missing)
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval_singlehop.json").write_text(
        json.dumps(kept_single, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "eval_multihop.json").write_text(
        json.dumps(kept_multi, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"kept single-hop {len(kept_single)} multi-hop {len(kept_multi)}")


def support_id_missing(support_ids: list[str], by_id: dict) -> bool:
    return any(support_id not in by_id for support_id in support_ids)


if __name__ == "__main__":
    main()

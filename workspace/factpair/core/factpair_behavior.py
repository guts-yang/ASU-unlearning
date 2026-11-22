from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

class unlearning_metric:  # type: ignore[no-redef]
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, metric_fn):
        metric_fn.name = self.name
        return metric_fn


REFUSAL_PATTERNS = (
    "i don't know",
    "i do not know",
    "cannot answer",
    "can't answer",
    "not sure",
    "no information",
    "unable to",
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
}


def _tokens(text: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return [word for word in words if word not in STOPWORDS and len(word) > 2]


def overlap_score(generation: str, answer: str) -> float:
    generated = set(_tokens(generation))
    expected = set(_tokens(answer))
    if not expected:
        return 0.0
    return len(generated & expected) / len(expected)


def classify_generation(
    generation: str,
    target_answer: str,
    donor_answer: str,
    eos_present: bool,
    truncated: bool,
) -> Dict[str, Any]:
    target_score = overlap_score(generation, target_answer)
    donor_score = overlap_score(generation, donor_answer)
    lower = generation.lower()
    if donor_score >= 0.3 and donor_score > target_score + 0.05:
        category = "donor_answer"
    elif target_score >= 0.35:
        category = "target_correct"
    elif any(pattern in lower for pattern in REFUSAL_PATTERNS):
        category = "refusal"
    elif truncated:
        category = "truncated"
    elif not generation.strip():
        category = "empty"
    else:
        category = "other_wrong"
    return {
        "category": category,
        "target_score": target_score,
        "donor_score": donor_score,
        "eos_present": eos_present,
        "truncated": truncated,
        "complete_answer": eos_present and not truncated,
    }


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@unlearning_metric(name="factpair_behavior")
def factpair_behavior(
    model,
    tokenizer=None,
    pairs_path: Optional[str] = None,
    max_samples: int = 50,
    max_new_tokens: int = 64,
    batch_size: int = 8,
    output_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    if not pairs_path:
        return {"agg_value": 0.0, "status": "BLOCKED_NO_PAIRS", "value_by_index": {}}
    rows = _read_jsonl(pairs_path)[:max_samples]
    if not rows or tokenizer is None:
        return {"agg_value": 0.0, "status": "BLOCKED_NO_INPUTS", "value_by_index": {}}

    pad_id = int(getattr(tokenizer, "pad_token_id", None) or 0)
    eos_id = int(getattr(tokenizer, "eos_token_id", 0))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    records: List[Dict[str, Any]] = []
    try:
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                encoded = tokenizer(
                    [row["prompt_plus"] for row in batch],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                sequences = model.generate(
                    input_ids=encoded["input_ids"].to(device),
                    attention_mask=encoded["attention_mask"].to(device),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    use_cache=True,
                )
                prompt_length = encoded["input_ids"].shape[1]
                for row, sequence in zip(batch, sequences):
                    generated = sequence[prompt_length:]
                    eos_positions = (generated == eos_id).nonzero()
                    eos_present = bool(eos_positions.numel())
                    generated_text = tokenizer.decode(
                        [int(token) for token in generated], skip_special_tokens=True
                    )
                    label = classify_generation(
                        generation=generated_text,
                        target_answer=row["answer_plus"],
                        donor_answer=row["answer_minus"],
                        eos_present=eos_present,
                        truncated=(not eos_present) and len(generated) >= max_new_tokens,
                    )
                    label.update({"pair_id": row.get("id")})
                    records.append(label)
    finally:
        model.train(was_training)

    counts = Counter(record["category"] for record in records)
    total = max(1, len(records))
    value_by_index = {
        "target_correct_rate": counts["target_correct"] / total,
        "donor_answer_rate": counts["donor_answer"] / total,
        "refusal_rate": counts["refusal"] / total,
        "premature_eos_rate": sum(1 for r in records if r["truncated"]) / total,
        "complete_answer_rate": sum(1 for r in records if r["complete_answer"]) / total,
        "samples": total,
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps({"records": records, "metrics": value_by_index}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return {
        "agg_value": value_by_index["target_correct_rate"],
        "value_by_index": value_by_index,
        "status": "PASS",
    }

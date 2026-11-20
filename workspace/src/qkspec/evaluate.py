"""Lightweight evaluation of full D_F / D_R after QKSpec editing.

This complements the repository's TOFU evaluator (which is invoked by the
Trainer wrapper) with explicit, reader-specific NLL / token-accuracy numbers
that are stored under separate keys.  No probe / causal-patch / quantization
/ relearning reader is implemented in this first version; those are marked
"not_covered" rather than silently folded into a score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    nll: float = 0.0
    token_accuracy: float = 0.0
    num_samples: int = 0
    num_tokens: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nll": self.nll,
            "token_accuracy": self.token_accuracy,
            "num_samples": self.num_samples,
            "num_tokens": self.num_tokens,
            "extra": self.extra,
        }


def compute_nll_and_accuracy(
    model,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> EvalResult:
    """Compute answer-token NLL and next-token accuracy on a full split."""
    if device is None:
        device = next(model.parameters()).device
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            if max_batches is not None and seen >= max_batches:
                break
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]
            labels = inputs["labels"][:, 1:]
            mask = labels.ne(-100)
            preds = logits.argmax(dim=-1)
            correct = (preds == labels) & mask
            total_correct += int(correct.sum())
            total_tokens += int(mask.sum())
            n_tokens = int(mask.sum())
            if outputs.loss is not None:
                total_loss += float(outputs.loss) * n_tokens
            seen += 1
    nll = total_loss / max(total_tokens, 1)
    acc = total_correct / max(total_tokens, 1)
    return EvalResult(nll=nll, token_accuracy=acc, num_tokens=total_tokens, num_samples=seen)

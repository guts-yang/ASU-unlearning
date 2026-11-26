"""R2-C: relearning-aware minimax objective.

Inner step simulates a bounded relearning attack; the outer loss maximizes the
post-attack forget NLL while keeping retain NLL low.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.func import functional_call


def _as_logits(out):
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def answer_nll_from_logits(logits: torch.Tensor | tuple, labels: torch.Tensor) -> torch.Tensor:
    logits = _as_logits(logits)
    logits = logits[:, :-1, :].float()
    labels = labels[:, 1:]
    mask = labels != -100
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
    ).view(labels.shape)
    return ((loss * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def inner_attack(
    model: torch.nn.Module,
    batch: dict,
    steps: int = 3,
    lr: float = 0.05,
    param_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Return attacked parameters without mutating the original model."""
    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    if param_names is not None:
        keep = set(param_names)
        params = {k: v for k, v in params.items() if k in keep}
    if not params:
        raise ValueError("no trainable parameters selected for inner attack")
    for _ in range(max(1, steps)):
        logits = functional_call(model, params, (batch["input_ids"],), {"attention_mask": batch["attention_mask"]})
        loss = answer_nll_from_logits(_as_logits(logits), batch["labels"])
        grads = torch.autograd.grad(loss, list(params.values()), create_graph=False, allow_unused=True)
        params = {
            k: (p - lr * g if g is not None else p)
            for (k, p), g in zip(params.items(), grads)
        }
    return params


def relearning_aware_loss(
    model: torch.nn.Module,
    forget_batch: dict,
    retain_batch: dict,
    inner_steps: int = 3,
    inner_lr: float = 0.05,
    retain_weight: float = 1.0,
    param_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    attacked = inner_attack(model, forget_batch, steps=inner_steps, lr=inner_lr, param_names=param_names)
    attacked_logits = functional_call(
        model, attacked, (forget_batch["input_ids"],), {"attention_mask": forget_batch["attention_mask"]}
    )
    retain_logits = functional_call(
        model,
        {k: v for k, v in model.named_parameters() if v.requires_grad},
        (retain_batch["input_ids"],),
        {"attention_mask": retain_batch["attention_mask"]},
    )
    after = answer_nll_from_logits(attacked_logits, forget_batch["labels"])
    retain = answer_nll_from_logits(retain_logits, retain_batch["labels"])
    loss = -after + retain_weight * retain
    return {"loss": loss, "post_attack_forget_nll": after, "retain_nll": retain}


def minimax_surrogate(forget_nll: torch.Tensor, recovery_nll: torch.Tensor, retain_nll: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    """Testable scalar form: maximize post-attack recovery NLL, preserve retain."""
    return -recovery_nll + lam * retain_nll + 0.1 * F.relu(forget_nll - recovery_nll)


def self_test() -> dict:
    torch.manual_seed(9)

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(17, 8)
            self.head = torch.nn.Linear(8, 17)

        def forward(self, input_ids, attention_mask=None):
            return self.head(self.emb(input_ids))

    model = TinyLM()
    batch = {
        "input_ids": torch.randint(0, 17, (2, 6)),
        "attention_mask": torch.ones(2, 6, dtype=torch.long),
        "labels": torch.randint(0, 17, (2, 6)),
    }
    params = inner_attack(model, batch, steps=2, lr=0.01, param_names=["head.weight", "head.bias"])
    assert set(params) == {"head.weight", "head.bias"}
    out = minimax_surrogate(torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.2))
    assert out.ndim == 0
    return {"status": "ok", "inner_params": sorted(params)}


__all__ = ["inner_attack", "relearning_aware_loss", "minimax_surrogate", "self_test"]

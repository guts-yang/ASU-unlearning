"""CF-03: causal activation gate.

Core operator: replace a layer/position slice with a counterfactual activation
and measure its causal effect on answer NLL.  Selection uses patch effects, not
correlation with the ΔA norm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .base import as_tensor


def nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits[:, :-1, :].float()
    labels = labels[:, 1:]
    mask = labels != -100
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none"
    ).view(labels.shape)
    return (loss * mask).sum(-1) / mask.sum(-1).clamp_min(1)


class PatchLayer:
    """Context manager replacing a decoder-layer output at selected positions."""

    def __init__(self, module: torch.nn.Module, replacement: torch.Tensor, positions: torch.Tensor | None = None):
        self.module = module
        self.replacement = as_tensor(replacement)
        self.positions = positions
        self.handle = None

    def _hook(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        rep = self.replacement.to(device=h.device, dtype=h.dtype)
        if rep.ndim == 2:
            rep = rep.unsqueeze(0)
        if rep.shape[0] == 1 and h.shape[0] > 1:
            rep = rep.expand(h.shape[0], -1, -1)
        if rep.shape[1] != h.shape[1]:
            raise ValueError(f"replacement length {rep.shape[1]} != hidden length {h.shape[1]}")
        new = h.clone()
        if self.positions is None:
            new[:, : rep.shape[1], :] = rep
        else:
            idx = self.positions.to(device=h.device, dtype=torch.long).flatten()
            new[:, idx, :] = rep[:, idx, :]
        return (new,) + output[1:] if is_tuple else new

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@torch.no_grad()
def patch_effect(
    model: torch.nn.Module,
    batch: dict,
    layer: int,
    replacement: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-sequence NLL change caused by patching ``layer``."""
    base = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    base_nll = nll_from_logits(base, batch["labels"])
    with PatchLayer(model.model.layers[layer], replacement, positions):
        patched = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    patched_nll = nll_from_logits(patched, batch["labels"])
    return patched_nll - base_nll


@dataclass
class CausalGate:
    effects: torch.Tensor
    mask: torch.Tensor
    scores: torch.Tensor
    diagnostics: dict

    def to_dict(self) -> dict:
        return {"diagnostics": self.diagnostics}


def select_causal_gate(effects: torch.Tensor, null_effects: torch.Tensor | None = None, top_k: int = 8) -> CausalGate:
    """Select the strongest causal components and report separation from nulls."""
    e = as_tensor(effects).flatten()
    k = min(max(1, top_k), e.numel())
    order = torch.argsort(e.abs(), descending=True)
    mask = torch.zeros_like(e, dtype=torch.bool)
    mask[order[:k]] = True
    scores = e.abs()
    diag = {
        "top_k": k,
        "selected_mean_abs_effect": float(scores[mask].mean().item()),
        "unselected_mean_abs_effect": float(scores[~mask].mean().item()) if (~mask).any() else 0.0,
        "sparsity": float(mask.float().mean().item()),
    }
    if null_effects is not None:
        n = as_tensor(null_effects).flatten()
        diag["null_mean_abs_effect"] = float(n.abs().mean().item())
        diag["separation_ratio"] = diag["selected_mean_abs_effect"] / max(diag["null_mean_abs_effect"], 1e-8)
    return CausalGate(effects=e, mask=mask, scores=scores, diagnostics=diag)


def self_test() -> dict:
    torch.manual_seed(2)
    e = torch.tensor([0.1, 0.9, -0.2, 0.05, 0.7])
    out = select_causal_gate(e, null_effects=torch.tensor([0.01, 0.02, 0.01, 0.0, 0.03]), top_k=2)
    assert out.mask.sum().item() == 2
    assert out.diagnostics["separation_ratio"] > 1.0
    return {"status": "ok", "diagnostics": out.diagnostics}


__all__ = ["PatchLayer", "patch_effect", "CausalGate", "select_causal_gate", "self_test"]

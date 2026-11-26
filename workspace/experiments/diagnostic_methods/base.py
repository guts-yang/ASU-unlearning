"""Shared, method-agnostic primitives for the non-claim diagnostic modules.

The package is intentionally importable on CPU.  Model-facing helpers are only
constructed when ``ModelContext.load`` is called.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch

WORKSPACE = Path(__file__).resolve().parents[2]
AUTO = WORKSPACE / "auto_experiments"
if str(AUTO) not in sys.path:
    sys.path.insert(0, str(AUTO))


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    title: str
    module: str
    requires_training: bool = True
    requires_lrm: bool = False
    core: tuple[str, ...] = ()


def as_tensor(value, dtype=torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def orthonormalize(basis: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return an orthonormal column basis for ``basis [..., D, K]``."""
    b = as_tensor(basis)
    if b.ndim != 2:
        raise ValueError(f"basis must be [D,K], got {tuple(b.shape)}")
    if b.numel() == 0:
        return b.new_zeros((b.shape[0], 0))
    q, r = torch.linalg.qr(b, mode="reduced")
    diag = torch.abs(torch.diagonal(r))
    keep = diag > eps
    return q[:, keep]


def project_onto(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    b = orthonormalize(basis)
    if b.shape[1] == 0:
        return torch.zeros_like(x)
    return (x @ b) @ b.T


def project_out(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - project_onto(x, basis)


def safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = as_tensor(a).flatten()
    b = as_tensor(b).flatten()
    denom = a.norm().clamp_min(eps) * b.norm().clamp_min(eps)
    return (a @ b) / denom


def ridge_coefficients(targets: torch.Tensor, design: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """Solve ``design @ beta = targets`` with a tiny ridge term."""
    x = as_tensor(design)
    y = as_tensor(targets)
    if x.ndim == 1:
        x = x[:, None]
    d = x.shape[1]
    a = x.T @ x + ridge * torch.eye(d, dtype=x.dtype, device=x.device)
    return torch.linalg.solve(a, x.T @ y)


def auc_score(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based AUC; labels are bool/0-1.  Returns 0.5 for a degenerate set."""
    s = as_tensor(scores).flatten()
    y = as_tensor(labels).flatten().bool().to(s.device)
    pos = int(y.sum().item())
    neg = int((~y).sum().item())
    if pos == 0 or neg == 0:
        return 0.5
    order = torch.argsort(s)
    ranks = torch.empty(s.numel(), dtype=torch.float32, device=s.device)
    ranks[order] = torch.arange(1, s.numel() + 1, dtype=torch.float32, device=s.device)
    # average ties
    sorted_s = s[order]
    i = 0
    while i < s.numel():
        j = i
        while j + 1 < s.numel() and torch.isclose(sorted_s[j + 1], sorted_s[i]):
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return float(((ranks[y].sum() - pos * (pos + 1) / 2.0) / (pos * neg)).item())


def finite_metrics(values: dict) -> dict:
    out = {}
    for k, v in values.items():
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                v = v.item()
            else:
                v = v.detach().float().cpu().tolist()
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        out[k] = v
    return out


@dataclass
class ModelContext:
    model: torch.nn.Module
    tokenizer: object
    forget: list
    retain: list
    holdout: list = field(default_factory=list)
    pairs: list = field(default_factory=list)
    device: str = "cuda"
    ref_model: torch.nn.Module | None = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        model_path: str | None = None,
        device: str = "cuda",
        max_pairs: int = 36,
        seed: int = 0,
        with_ref: bool = False,
        dtype=torch.float32,
        attn: str = "eager",
        forget_path: str | None = None,
        retain_path: str | None = None,
        holdout_path: str | None = None,
    ) -> "ModelContext":
        from common import MODEL, counterfactual_pairs, load_jsonl, load_model, set_seed
        from common import FORGET, HOLDOUT, RETAIN

        set_seed(seed)
        path = model_path or MODEL
        forget_file = str(forget_path or FORGET)
        retain_file = str(retain_path or RETAIN)
        holdout_file = str(holdout_path or HOLDOUT)
        model, tok = load_model(path, device=device, dtype=dtype, attn=attn)
        ref = None
        if with_ref:
            ref, _ = load_model(path, device=device, dtype=dtype, attn=attn)
            for p in ref.parameters():
                p.requires_grad_(False)
            ref.eval()
        return cls(
            model=model,
            tokenizer=tok,
            forget=load_jsonl(forget_file),
            retain=load_jsonl(retain_file),
            holdout=load_jsonl(holdout_file),
            pairs=counterfactual_pairs(tok, forget_path=forget_file, retain_path=retain_file, max_pairs=max_pairs, seed=seed),
            device=device,
            ref_model=ref,
            meta={"model_path": path, "seed": seed, "forget_path": forget_file,
                  "retain_path": retain_file, "holdout_path": holdout_file},
        )

    def records_batch(self, records: Sequence[dict], max_len: int = 512):
        from common import make_batch

        return make_batch(self.tokenizer, list(records), self.device, max_len=max_len)

    def hidden_for_records(self, records: Sequence[dict], layer: int = 16, max_len: int = 512):
        batch = self.records_batch(records, max_len=max_len)
        with torch.no_grad():
            out = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
        return out.hidden_states[layer].float()

    def pooled_delta_for_pairs(self, layer: int = 16, max_pairs: int | None = None) -> torch.Tensor:
        use = self.pairs[: max_pairs or len(self.pairs)]
        plus_records = [{"question": p["x_plus"], "answer": p["answer_plus"]} for p in use]
        minus_records = [{"question": p["x_minus"], "answer": p["answer_plus"]} for p in use]
        plus = self.hidden_for_records(plus_records, layer=layer)
        minus = self.hidden_for_records(minus_records, layer=layer)
        return (plus - minus).mean(dim=(0, 1))

    def answer_nll(self, records: Sequence[dict], model: torch.nn.Module | None = None) -> float:
        from common import answer_nll

        m = model or self.model
        batch = self.records_batch(records)
        with torch.no_grad():
            return float(answer_nll(m, batch)[0])

    def train_with_loss(
        self,
        loss_fn: Callable[[torch.nn.Module, dict, dict, int], torch.Tensor | dict],
        steps: int = 20,
        lr: float = 1e-5,
        batch_size: int = 4,
        seed: int = 0,
    ) -> list[dict]:
        """Minimal generic trainer used by training-bearing diagnostic methods."""
        g = torch.Generator().manual_seed(seed)
        forget = list(self.forget)
        retain = list(self.retain)
        perm_f = torch.randperm(len(forget), generator=g).tolist()
        perm_r = torch.randperm(len(retain), generator=g).tolist()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.0)
        self.model.train()
        history = []
        for step in range(steps):
            step_start = time.perf_counter()
            fb = [forget[perm_f[(step * batch_size + i) % len(forget)]] for i in range(batch_size)]
            rb = [retain[perm_r[(step * batch_size + i) % len(retain)]] for i in range(batch_size)]
            fb_batch = self.records_batch(fb)
            rb_batch = self.records_batch(rb)
            out = loss_fn(self.model, fb_batch, rb_batch, step)
            loss = out["loss"] if isinstance(out, dict) else out
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            row = {
                "step": step,
                "loss": float(loss.detach().item()),
                "step_sec": round(time.perf_counter() - step_start, 4),
            }
            if isinstance(out, dict):
                row.update({k: float(v.detach().item()) for k, v in out.items() if isinstance(v, torch.Tensor) and v.numel() == 1})
            history.append(row)
        return history


__all__ = [
    "MethodSpec",
    "ModelContext",
    "as_tensor",
    "orthonormalize",
    "project_onto",
    "project_out",
    "safe_cosine",
    "ridge_coefficients",
    "auc_score",
    "finite_metrics",
]

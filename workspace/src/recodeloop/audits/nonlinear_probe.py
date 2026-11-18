"""Independent nonlinear reader audit arm."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..config import FrozenConfig
from ..data.never_known import hash_embedding
from ..types import ActivationBatch, AuditResult, Budget
from ..utils.seed import set_global_seed


class MLPFactReader:
    name = "independent_nonlinear_reader"

    def __init__(self, value_ids: Sequence[str], in_dim: int, dim: int = 32, seed: int = 4002) -> None:
        self.value_ids = tuple(value_ids)
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(),
                                 nn.Linear(in_dim, dim, bias=False))
        self.value_embeddings = torch.stack([
            torch.from_numpy(hash_embedding(v, dim, seed)) for v in self.value_ids
        ])

    def parameters(self):
        return self.net.parameters()

    def loss(self, activations: torch.Tensor, labels: Sequence[str]) -> torch.Tensor:
        projected = self.net(activations)
        emb = F.normalize(self.value_embeddings.to(projected.device, dtype=projected.dtype), dim=-1)
        logits = F.normalize(projected, dim=-1) @ emb.T
        targets = torch.tensor([self.value_ids.index(v) for v in labels], device=activations.device)
        return F.cross_entropy(logits, targets)

    def score(self, batch: ActivationBatch) -> dict[str, float]:
        self.net.eval()
        with torch.no_grad():
            projected = self.net(batch.activations)
            emb = F.normalize(self.value_embeddings.to(projected.device, dtype=projected.dtype), dim=-1)
            preds = (F.normalize(projected, dim=-1) @ emb.T).argmax(-1).cpu().tolist()
        values = [self.value_ids[i] for i in preds]
        accuracy = sum(p == l for p, l in zip(values, batch.fact_value_labels)) / len(values)
        return {"fact_value_accuracy": float(accuracy)}


def run_nonlinear_audit(batches: Sequence[ActivationBatch], value_ids: Sequence[str],
                        cfg: FrozenConfig, seed: int | None = None) -> AuditResult:
    arm_cfg = cfg.section("audits")["arms"]["independent_generative_reader"]
    if not batches:
        return AuditResult(arm="independent_nonlinear_reader", independence_statement="not run",
                           budget=Budget(0, 0, 0, 0), raw_metrics={}, control_delta={},
                           coverage="uncovered", note="未覆盖：无数据")
    actual_seed = int(seed if seed is not None else arm_cfg["seed"])
    set_global_seed(actual_seed)
    reader = MLPFactReader(value_ids, int(batches[0].activations.shape[-1]), seed=actual_seed)
    steps = int(arm_cfg["max_steps"])
    optimizer = torch.optim.AdamW(reader.parameters(), lr=0.01)
    for _ in range(steps):
        for batch in batches:
            loss = reader.loss(batch.activations.detach(), batch.fact_value_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    metrics = reader.score(batches[0])
    return AuditResult(
        arm="independent_nonlinear_reader",
        independence_statement=(
            f"独立 MLP 结构、fresh seed={actual_seed}、独立用途；与训练 reader 不同架构"
        ),
        budget=Budget(tokens=sum(b.activations.shape[0] for b in batches) * steps,
                      steps=steps, components=1, rank=0),
        raw_metrics=metrics, control_delta={}, coverage="covered",
    )

"""Frozen-answer-embedding similarity reader.

The reader decodes a concrete fact value by nearest frozen answer embedding.
It never outputs a forget/retain or set-membership class.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..errors import ContractViolation
from ..types import ActivationBatch, RecoveryScore, Split
from ..utils.hashing import sha256_json
from .base import fact_value_metrics
from ..data.never_known import hash_embedding


class FrozenEmbeddingReader:
    name = "frozen_answer_embedding_similarity"

    def __init__(
        self,
        value_ids: Sequence[str],
        in_dim: int,
        dim: int = 32,
        seed: int = 1001,
    ) -> None:
        values = tuple(dict.fromkeys(value_ids))
        if len(values) < 2:
            raise ContractViolation("reader requires a candidate fact-value set of size >= 2")
        self.value_ids = values
        self.dim = dim
        self.seed = seed
        embeddings = torch.stack([
            torch.from_numpy(hash_embedding(v, dim, seed)) for v in values
        ])
        self.value_embeddings = embeddings  # frozen; moved explicitly with inputs
        self.head = nn.Linear(in_dim, dim, bias=False)

    def parameters(self):
        return self.head.parameters()

    def to(self, device: torch.device | str) -> "FrozenEmbeddingReader":
        self.head.to(device)
        self.value_embeddings = self.value_embeddings.to(device)
        return self

    def train(self) -> "FrozenEmbeddingReader":
        self.head.train()
        return self

    def eval(self) -> "FrozenEmbeddingReader":
        self.head.eval()
        return self

    def freeze(self) -> None:
        for p in self.head.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.head.parameters():
            p.requires_grad_(True)

    def correct_mask(self, batch: ActivationBatch) -> list[bool]:
        self.head.eval()
        with torch.no_grad():
            predictions = self._logits(batch.activations).argmax(dim=-1).cpu().tolist()
        values = [self.value_ids[i] for i in predictions]
        return [p == label for p, label in zip(values, batch.fact_value_labels)]

    def _logits(self, activations: torch.Tensor) -> torch.Tensor:
        projected = self.head(activations)
        value_emb = self.value_embeddings.to(projected.device, dtype=projected.dtype)
        return F.normalize(projected, dim=-1) @ F.normalize(value_emb, dim=-1).T

    def differentiable_loss(self, activations: torch.Tensor, targets: Sequence[str]) -> torch.Tensor:
        indices = torch.tensor([self.value_ids.index(t) for t in targets],
                               dtype=torch.long, device=activations.device)
        return F.cross_entropy(self._logits(activations), indices)

    def score(
        self,
        batch: ActivationBatch,
        fact_ids: Sequence[str],
        split: Split,
        nk_ids: Sequence[str],
    ) -> RecoveryScore:
        was_training = self.head.training
        self.head.eval()
        with torch.no_grad():
            logits = self._logits(batch.activations)
            preds = logits.argmax(dim=-1).cpu().tolist()
            predictions = [self.value_ids[i] for i in preds]
            labels = list(batch.fact_value_labels)
            sorted_idx = logits.argsort(dim=-1, descending=True).cpu().tolist()
            ranks, mrrs = [], []
            for row, label in enumerate(labels):
                if label not in self.value_ids:
                    ranks.append(len(self.value_ids) + 1)
                    mrrs.append(1.0 / (len(self.value_ids) + 1))
                    continue
                target = self.value_ids.index(label)
                rank = sorted_idx[row].index(target) + 1
                ranks.append(rank)
                mrrs.append(1.0 / rank)
            confidences = F.softmax(logits, dim=-1).max(dim=-1).values.cpu().tolist()
        if was_training:
            self.head.train()
        metrics = fact_value_metrics(predictions, labels, nk_ids, ranks, mrrs, confidences)
        return RecoveryScore(
            fact_value_accuracy=metrics["fact_value_accuracy"],
            exact_match=metrics["exact_match"],
            candidate_rank=min(ranks) if ranks else None,
            mrr=metrics["mrr"],
            confidence=metrics["confidence"],
            nk_accuracy=metrics["nk_accuracy"],
            cluster_id=batch.entity_ids[0] if batch.entity_ids else "",
            split=split,
            n_samples=len(labels),
        )

    def state_dict_hash(self) -> str:
        return sha256_json({
            "value_ids": self.value_ids,
            "head": [[float(x) for x in p.detach().cpu().flatten()] for p in self.head.parameters()],
            "embeddings": self.value_embeddings.detach().cpu().tolist(),
        })

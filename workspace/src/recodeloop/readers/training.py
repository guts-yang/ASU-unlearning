"""Train the current-round fact-value reader on frozen activations."""

from __future__ import annotations

from typing import Sequence

import torch

from ..types import ActivationBatch, SeedRecord
from ..utils.seed import set_global_seed
from .fact_value import FrozenEmbeddingReader


def train_edit_reader(
    reader: FrozenEmbeddingReader,
    batches: Sequence[ActivationBatch],
    steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[SeedRecord, list[float]]:
    """Train only the reader head; activations are detached, model stays frozen."""
    seed_record = set_global_seed(seed)
    reader.unfreeze()
    reader.train()
    optimizer = torch.optim.AdamW(reader.head.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(steps):
        for batch in batches:
            with torch.no_grad():
                activations = batch.activations.detach().clone()
            loss = reader.differentiable_loss(activations, batch.fact_value_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
    reader.eval()
    reader.freeze()
    return seed_record, losses

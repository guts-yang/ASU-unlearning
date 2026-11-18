"""Shared control helpers: budget-matched low-rank edits and scoring."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from ..activations.extraction import extract_activation_batches
from ..baseline_adapter import BaselineAdapter
from ..config import FrozenConfig
from ..editing.low_rank import LowRankEditor
from ..editing.objectives import editing_objective
from ..localization.causal_effects import InterventionInputs
from ..readers.fact_value import FrozenEmbeddingReader
from ..types import ActivationSpec, Budget, ComponentSelection, ControlResult, SeedRecord
from ..utils.hashing import sha256_json
from ..utils.seed import set_global_seed


def score_inputs(
    adapter: BaselineAdapter,
    reader: FrozenEmbeddingReader,
    specs: Mapping[str, ActivationSpec],
    inputs: InterventionInputs,
) -> dict[str, float]:
    batch = extract_activation_batches(
        adapter, list(specs.values()), inputs.input_ids,
        inputs.entity_ids, inputs.fact_ids, inputs.split,
        inputs.value_labels,
    )
    score = reader.score(
        next(iter(batch.values())), inputs.fact_ids, inputs.split, inputs.nk_ids
    )
    return {
        "fact_value_accuracy": score.fact_value_accuracy,
        "exact_match": score.exact_match,
        "mrr": score.mrr or 0.0,
        "nk_accuracy": score.nk_accuracy,
        "confidence": score.confidence,
    }


def lora_edit(
    adapter: BaselineAdapter,
    reader: FrozenEmbeddingReader,
    specs: Mapping[str, ActivationSpec],
    inputs: InterventionInputs,
    retain_ids: torch.Tensor,
    retain_labels: torch.Tensor,
    components: Sequence[str],
    ranks: Sequence[int],
    steps: int,
    lr: float,
    lambda_ret: float,
    g: float,
    seed: int,
    randomize: bool = False,
) -> Budget:
    """Attach low-rank factors, optionally train, project energy, then remove."""
    seed_record: SeedRecord = set_global_seed(seed)
    selection = ComponentSelection(tuple(components), tuple(ranks),
                                   tuple(g for _ in components), "control_lora",
                                   sha256_json(list(components)))
    initial_hash = adapter.state_hash()
    with adapter.frozen():
        reader.freeze()
        editor = LowRankEditor(adapter)
        editor.attach(selection)
        if randomize:
            with torch.no_grad():
                for _comp, (_A, B) in editor.factors.items():
                    B.normal_(0, 0.1)
        else:
            optimizer = torch.optim.AdamW(editor.low_rank_params(), lr=lr)
            for _ in range(steps):
                batches = extract_activation_batches(
                    adapter, [specs[c] for c in components],
                    inputs.input_ids, inputs.entity_ids, inputs.fact_ids,
                    inputs.split, inputs.value_labels, detach=False,
                )
                rec = sum(
                    reader.differentiable_loss(b.activations, b.fact_value_labels)
                    for b in batches.values()
                ) / len(batches)
                ret = adapter.retain_loss_tensor(retain_ids, retain_labels)
                total = editing_objective(rec, ret, lambda_ret)
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
        editor.project_energy(g)
        editor.assert_ranks()
        editor.assert_base_unchanged(initial_hash)
        editor.remove()
        reader.unfreeze()
    return Budget(
        tokens=inputs.input_ids.numel() * (0 if randomize else steps),
        steps=0 if randomize else steps,
        components=len(components),
        rank=max(ranks),
    )


def make_result(arm: str, budget: Budget, metrics: Mapping[str, float],
                cfg: FrozenConfig, seed: int, note: str = "") -> ControlResult:
    return ControlResult(
        arm=arm, budget=budget, metrics=dict(metrics),
        config_hash=cfg.config_hash,
        seed=set_global_seed(seed), note=note,
    )

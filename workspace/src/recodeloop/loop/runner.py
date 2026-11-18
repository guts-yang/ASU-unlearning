"""RecodeLoop main closed loop. Uses only train/calibration, never confirmation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import torch

from ..activations.extraction import extract_activation_batches
from ..baseline_adapter import BaselineAdapter
from ..config import FrozenConfig
from ..data.access_guard import AccessGuard
from ..errors import ContractViolation
from ..editing.budgets import BudgetTracker
from ..editing.low_rank import LowRankEditor
from ..editing.objectives import editing_objective
from ..localization.causal_effects import (
    InterventionInputs,
    LocalizationRecord,
    compute_localization,
)
from ..localization.selection import select_components
from ..readers.fact_value import FrozenEmbeddingReader
from ..readers.training import train_edit_reader
from ..types import ActivationSpec, Budget, ComponentSelection, LoopState, PatchResult
from ..utils.logging import JsonlLogger
from .stopping import should_stop


@dataclass(frozen=True)
class LoopResult:
    states: tuple[LoopState, ...]
    selections: tuple[ComponentSelection, ...]
    localization: tuple[tuple[LocalizationRecord, ...], ...]
    patches: tuple[PatchResult, ...]
    energies: tuple[Mapping[str, float], ...]


class RecodeLoop:
    def __init__(self, adapter: BaselineAdapter, cfg: FrozenConfig,
                 guard: AccessGuard, logger: JsonlLogger | None = None) -> None:
        self.adapter = adapter
        self.cfg = cfg
        self.guard = guard
        self.logger = logger

    def run(
        self,
        reader: FrozenEmbeddingReader,
        specs: Mapping[str, ActivationSpec],
        forget_inputs: InterventionInputs,
        nk_inputs: InterventionInputs,
        retain_ids: torch.Tensor,
        retain_labels: torch.Tensor,
        nk_ids: Sequence[str],
        reader_seed: int = 1001,
    ) -> LoopResult:
        self.guard.assert_access(forget_inputs.split)
        self.guard.assert_access(nk_inputs.split)
        frozen = self.cfg.section("m0_p08_m2")["frozen_constants"]
        edit_cfg = self.cfg.section("editing")
        reader_cfg = self.cfg.section("readers")["edit_reader"]
        t_max = int(frozen["t_max"])
        edit_steps = int(reader_cfg.get("train_steps", 64))
        lambda_ret = float(edit_cfg["lambda_ret"])
        g = float(edit_cfg["energy_g"])
        seed = int(reader_cfg["seed"])

        states: list[LoopState] = []
        selections: list[ComponentSelection] = []
        localization: list[tuple[LocalizationRecord, ...]] = []
        all_patches: list[PatchResult] = []
        energies: list[Mapping[str, float]] = []

        for step in range(1, t_max + 1):
            initial_hash = self.adapter.state_hash(quick=True)
            with self.adapter.frozen():
                train_batches = extract_activation_batches(
                    self.adapter,
                    [specs[c] for c in specs],
                    forget_inputs.input_ids,
                    forget_inputs.entity_ids,
                    forget_inputs.fact_ids,
                    forget_inputs.split,
                    forget_inputs.value_labels,
                )
                _, _ = train_edit_reader(
                    reader, tuple(train_batches.values()), edit_steps,
                    float(reader_cfg["learning_rate"]), seed + step,
                )
                records, patches = compute_localization(
                    self.adapter, reader, specs, tuple(specs),
                    forget_inputs, nk_inputs, retain_ids, retain_labels,
                    seed=seed + step,
                )
            selection = select_components(records, patches, self.cfg, self.cfg.config_hash)
            localization.append(records)
            all_patches.extend(patches)
            selections.append(selection)

            if not selection.components:
                state = LoopState(
                    step=step, model_hash=self.adapter.state_hash(),
                    reader_hash=reader.state_dict_hash(), components=(),
                    actual_budget=Budget(0, 0, 0, 0), stop_reason="no_components_selected",
                )
                states.append(state)
                break

            with self.adapter.frozen():
                reader.freeze()
                editor = LowRankEditor(self.adapter)
                editor.attach(selection)
                editor.assert_only_low_rank_trainable()
                optimizer = torch.optim.AdamW(
                    editor.low_rank_params(),
                    lr=float(edit_cfg["optimizer"]["learning_rate"]),
                )
                tracker = BudgetTracker(Budget(
                    tokens=forget_inputs.input_ids.numel() * edit_steps,
                    steps=edit_steps,
                    components=len(selection.components),
                    rank=max(selection.ranks),
                ))
                for _ in range(edit_steps):
                    edit_batches = extract_activation_batches(
                        self.adapter,
                        [specs[c] for c in selection.components],
                        forget_inputs.input_ids,
                        forget_inputs.entity_ids,
                        forget_inputs.fact_ids,
                        forget_inputs.split,
                        forget_inputs.value_labels,
                        detach=False,
                    )
                    losses = [
                        reader.differentiable_loss(batch.activations, batch.fact_value_labels)
                        for batch in edit_batches.values()
                    ]
                    rec_loss = sum(losses) / len(losses)
                    ret_loss = self.adapter.retain_loss_tensor(retain_ids, retain_labels)
                    total = editing_objective(rec_loss, ret_loss, lambda_ret)
                    optimizer.zero_grad()
                    total.backward()
                    optimizer.step()

                used = tracker.finish(
                    tokens=forget_inputs.input_ids.numel() * edit_steps,
                    steps=edit_steps,
                )
                projected = editor.project_energy(g)
                rank_report = editor.assert_ranks()
                editor.assert_base_unchanged(initial_hash)
                committed = editor.commit()
                reader.unfreeze()
                if not committed:
                    raise ContractViolation("commit produced no low-rank delta")
            energies.append(projected)
            if self.logger:
                self.logger.log("loop.round", step=step, components=selection.components,
                                ranks=rank_report, energies=projected, budget=used)

            state = LoopState(
                step=step,
                model_hash=self.adapter.state_hash(quick=True),
                reader_hash=reader.state_dict_hash(),
                components=selection.components,
                actual_budget=used,
                stop_reason=None,
            )
            stop_reason = should_stop(
                state, self.cfg, components_selected=True,
                retain_damage=max((r.retain_damage for r in records), default=0.0),
            )
            if stop_reason:
                state = replace(state, stop_reason=stop_reason)
            states.append(state)
            if stop_reason:
                break

        return LoopResult(
            states=tuple(states),
            selections=tuple(selections),
            localization=tuple(localization),
            patches=tuple(all_patches),
            energies=tuple(energies),
        )

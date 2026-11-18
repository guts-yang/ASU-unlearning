"""Causal effects from NK/reverse/random/self patches.

phi is computed as recovery delta from NK patching, never from activation
correlation, attention weights or probe accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from ..activations.extraction import extract_activation_batches, extract_raw_donors
from ..activations.patching import patch_components, random_donors
from ..baseline_adapter import BaselineAdapter
from ..errors import ContractViolation
from ..readers.fact_value import FrozenEmbeddingReader
from ..types import ActivationSpec, PatchKind, PatchResult, RecoveryScore, Split


@dataclass(frozen=True)
class InterventionInputs:
    input_ids: torch.Tensor
    entity_ids: Sequence[str]
    fact_ids: Sequence[str]
    value_labels: Sequence[str]
    split: Split
    nk_ids: Sequence[str]


@dataclass(frozen=True)
class LocalizationRecord:
    component: str
    phi: float
    per_entity_phi: tuple[float, ...]
    retain_damage: float
    self_delta: float
    random_delta: float
    joint_phi: float | None


def _pooled_score(
    adapter: BaselineAdapter,
    reader: FrozenEmbeddingReader,
    spec: ActivationSpec,
    inputs: InterventionInputs,
) -> tuple[RecoveryScore, list[bool]]:
    batch = extract_activation_batches(
        adapter, [spec], inputs.input_ids, inputs.entity_ids, inputs.fact_ids,
        inputs.split, inputs.value_labels,
    )[spec.component]
    return reader.score(batch, inputs.fact_ids, inputs.split, inputs.nk_ids), reader.correct_mask(batch)


def _delta(
    adapter: BaselineAdapter,
    reader: FrozenEmbeddingReader,
    spec: ActivationSpec,
    inputs: InterventionInputs,
    donors: Mapping[str, torch.Tensor],
    kind: PatchKind,
) -> tuple[float, tuple[float, ...]]:
    before, before_mask = _pooled_score(adapter, reader, spec, inputs)
    with patch_components(adapter.model, donors, kind=kind):
        after, after_mask = _pooled_score(adapter, reader, spec, inputs)
    per_entity = tuple(float(b) - float(a) for b, a in zip(before_mask, after_mask))
    return after.fact_value_accuracy - before.fact_value_accuracy, per_entity


def compute_localization(
    adapter: BaselineAdapter,
    reader: FrozenEmbeddingReader,
    specs: Mapping[str, ActivationSpec],
    components: Sequence[str],
    forget_inputs: InterventionInputs,
    nk_inputs: InterventionInputs,
    retain_ids: torch.Tensor,
    retain_labels: torch.Tensor,
    seed: int,
) -> tuple[tuple[LocalizationRecord, ...], tuple[PatchResult, ...]]:
    """One self/NK/reverse/random patch per component plus a joint patch."""
    if forget_inputs.input_ids.shape != nk_inputs.input_ids.shape:
        raise ContractViolation("forget/NK inputs must have matching shapes")

    forget_raw = extract_raw_donors(adapter, components, forget_inputs.input_ids)
    nk_raw = extract_raw_donors(adapter, components, nk_inputs.input_ids)
    retain_before = adapter.retain_nll(retain_ids, retain_labels)

    records: list[LocalizationRecord] = []
    patches: list[PatchResult] = []
    for component in components:
        spec = specs[component]
        before, _ = _pooled_score(adapter, reader, spec, forget_inputs)
        nk_delta, per_entity = _delta(
            adapter, reader, spec, forget_inputs, {component: nk_raw[component]}, "nk"
        )
        self_delta, _ = _delta(
            adapter, reader, spec, forget_inputs, {component: forget_raw[component]}, "self"
        )
        random_delta, _ = _delta(
            adapter, reader, spec, forget_inputs,
            random_donors({component: nk_raw[component]}, seed), "random",
        )
        retain_donor = nk_raw[component]
        target_len = int(retain_ids.shape[1])
        if retain_donor.shape[1] > target_len:
            retain_donor = retain_donor[:, :target_len, :]
        elif retain_donor.shape[1] < target_len:
            repeats = (target_len + retain_donor.shape[1] - 1) // retain_donor.shape[1]
            retain_donor = retain_donor.repeat(1, repeats, 1)[:, :target_len, :]
        with patch_components(adapter.model, {component: retain_donor}, kind="nk"):
            retain_after = adapter.retain_nll(retain_ids, retain_labels)
        phi = -nk_delta
        records.append(LocalizationRecord(
            component=component,
            phi=float(phi),
            per_entity_phi=tuple(per_entity),
            retain_damage=float(retain_after - retain_before),
            self_delta=float(self_delta),
            random_delta=float(random_delta),
            joint_phi=None,
        ))
        for kind, delta in (("self", self_delta), ("nk", nk_delta), ("random", random_delta)):
            patches.append(PatchResult(
                patch_kind=kind,
                scope="single",
                component=component,
                recovery_before=before.fact_value_accuracy,
                recovery_after=before.fact_value_accuracy + delta,
                recovery_delta=float(delta),
                retain_damage=float(retain_after - retain_before),
                ci_low=None,
                ci_high=None,
            ))

    reverse_delta = 0.0
    if components:
        reverse_delta, _ = _delta(
            adapter, reader, specs[components[0]], nk_inputs,
            {components[0]: forget_raw[components[0]]}, "reverse",
        )
        patches.append(PatchResult(
            patch_kind="reverse", scope="single", component=components[0],
            recovery_before=0.0, recovery_after=reverse_delta,
            recovery_delta=float(reverse_delta), retain_damage=0.0,
            ci_low=None, ci_high=None,
        ))

    joint_phi = 0.0
    if components:
        before, _ = _pooled_score(adapter, reader, specs[components[0]], forget_inputs)
        with patch_components(adapter.model, {c: nk_raw[c] for c in components}, kind="nk"):
            after, _ = _pooled_score(adapter, reader, specs[components[0]], forget_inputs)
        joint_phi = before.fact_value_accuracy - after.fact_value_accuracy
        records = [
            LocalizationRecord(
                component=r.component, phi=r.phi, per_entity_phi=r.per_entity_phi,
                retain_damage=r.retain_damage, self_delta=r.self_delta,
                random_delta=r.random_delta, joint_phi=float(joint_phi),
            ) for r in records
        ]
        patches.append(PatchResult(
            patch_kind="nk", scope="joint", component=None,
            recovery_before=before.fact_value_accuracy,
            recovery_after=after.fact_value_accuracy,
            recovery_delta=float(after.fact_value_accuracy - before.fact_value_accuracy),
            retain_damage=0.0, ci_low=None, ci_high=None,
        ))

    return tuple(records), tuple(patches)

"""Budgeted component selection from causal effects only."""

from __future__ import annotations

from typing import Sequence

from ..config import FrozenConfig
from ..types import ComponentSelection
from .causal_effects import LocalizationRecord
from ..types import PatchResult


def select_components(
    records: Sequence[LocalizationRecord],
    patches: Sequence[PatchResult],
    cfg: FrozenConfig,
    config_hash: str,
) -> ComponentSelection:
    loc = cfg.section("localization")
    edit = cfg.section("editing")
    threshold = float(loc["patch_threshold"])
    max_n = int(loc["max_components_per_round"])
    min_n = int(loc["min_components_per_round"])
    ranks_cfg = tuple(int(r) for r in edit["candidate_ranks"])

    eligible = sorted(
        (r for r in records if r.phi > threshold),
        key=lambda r: (-r.phi, r.retain_damage),
    )
    chosen = eligible[:max_n]
    interaction_note = None
    if not chosen:
        joint = [p for p in patches if p.scope == "joint" and p.recovery_delta < 0]
        if joint:
            interaction_note = (
                "单组件 patch 无效但联合 patch 有效：记录冗余/交互，不误报'没有残余'"
            )
            chosen = sorted(records, key=lambda r: -abs(r.phi))[:min(max_n, min_n)]

    energy = float(edit["energy_g"])
    return ComponentSelection(
        components=tuple(r.component for r in chosen),
        ranks=tuple(ranks_cfg[0] for _ in chosen),
        energies=tuple(energy for _ in chosen),
        rule=str(loc["selection_rule"]),
        config_hash=config_hash,
        interaction_note=interaction_note,
    )

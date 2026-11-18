"""Frozen stopping rule; never uses confirmation."""

from __future__ import annotations

from ..config import FrozenConfig
from ..types import LoopState


def should_stop(
    state: LoopState,
    cfg: FrozenConfig,
    components_selected: bool,
    retain_damage: float = 0.0,
) -> str | None:
    frozen = cfg.section("m0_p08_m2")["frozen_constants"]
    gates = cfg.section("m0_p08_m2")["gates"]
    if not components_selected:
        return "no_components_selected"
    if state.step >= int(frozen["t_max"]):
        return "t_max_reached"
    if retain_damage > float(gates["retain_max_abs_drop"]):
        return "retain_gate_failed"
    return None

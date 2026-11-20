"""Baseline controls required by the QKSpec design doc (section 8 / 10).

* ``noop``: rho = 1.0 (or empty J) must leave weights and logits unchanged.
* ``random_isospectral``: same layer/head count, rank, rho, direction count
  and matched spectral energy but random singular directions.
* ``shuffle``: permute directions across heads/layers, keeping ranks/rho.

The controls are important because if a random direction produces the same
forget effect, the observation is generic attention damage, not a
target-specific mechanism.  Each control temporarily folds its own edit,
measures D_F / D_R NLL change with the same loader / decode config as the real
edit, then restores the model to its pre-control state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from .config import QKSpecConfig
from .qk_math import DirectionSpec
from .score import _nll_over_loader
from .edit import apply_directions


@dataclass
class ControlResult:
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    real_edit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "controls": self.results,
            "real_edit": self.real_edit,
            "comparison_note": (
                "A control whose D_F NLL gain matches the real edit falsifies "
                "the target-specific claim."
            ),
        }


def _make_control_directions(
    mode: str,
    real: Sequence[DirectionSpec],
    cfg: QKSpecConfig,
    gen: torch.Generator,
) -> List[DirectionSpec]:
    if mode == "noop":
        return [
            DirectionSpec(
                layer=d.layer, head=d.head, rank=d.rank, sigma=d.sigma,
                rho=cfg.noop_rho, stage_a_score=d.stage_a_score,
                causal_verified=False, meta={"control": "noop"},
            )
            for d in real
        ]
    if mode == "random_isospectral":
        ctrl: List[DirectionSpec] = []
        by_head: Dict[tuple, List[DirectionSpec]] = {}
        for d in real:
            by_head.setdefault((d.layer, d.head), []).append(d)
        for (li, h), dirs in by_head.items():
            sigmas = sorted(d.sigma for d in dirs)
            ranks = torch.randperm(max(8, len(dirs) * 3), generator=gen).tolist()[: len(dirs)]
            for j, d in enumerate(dirs):
                ctrl.append(
                    DirectionSpec(
                        layer=li, head=h, rank=int(ranks[j]),
                        sigma=sigmas[j % len(sigmas)] if sigmas else d.sigma,
                        rho=cfg.rho, stage_a_score=d.stage_a_score,
                        causal_verified=False, meta={"control": "random_isospectral"},
                    )
                )
        return ctrl
    if mode == "shuffle":
        keys = [(d.layer, d.head) for d in real]
        order = torch.randperm(len(keys), generator=gen).tolist()
        ctrl = []
        for d, src in zip(real, order):
            li, h = keys[src]
            ctrl.append(
                DirectionSpec(
                    layer=li, head=h, rank=d.rank, sigma=d.sigma,
                    rho=cfg.rho, stage_a_score=d.stage_a_score,
                    causal_verified=False, meta={"control": "shuffle"},
                )
            )
        return ctrl
    raise ValueError(f"unknown control mode: {mode}")


def _measure_edit_effect(
    model,
    directions: Sequence[DirectionSpec],
    cfg: QKSpecConfig,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    device: torch.device,
    baseline_f: float,
    baseline_r: float,
) -> Dict[str, Any]:
    """Temporarily apply ``directions``, measure NLL delta, restore model."""
    layers = model.model.layers
    touched = sorted({d.layer for d in directions})
    snapshots = {li: layers[li].self_attn.q_proj.weight.detach().clone() for li in touched}
    edit_meta = None
    try:
        edit_meta = apply_directions(
            model, directions, cfg, device, write_back=True
        )
        nll_f = _nll_over_loader(model, forget_loader, device, cfg.max_stage_b_batches)
        nll_r = _nll_over_loader(model, retain_loader, device, cfg.max_stage_b_batches)
    finally:
        with torch.no_grad():
            for li, snap in snapshots.items():
                layers[li].self_attn.q_proj.weight.data.copy_(snap)
    return {
        "update_norm": edit_meta.update_norm if edit_meta else 0.0,
        "forget_nll_delta": float(nll_f - baseline_f) if edit_meta else 0.0,
        "retain_nll_delta": float(nll_r - baseline_r) if edit_meta else 0.0,
        "directions": [d.to_dict() for d in directions],
    }


def run_controls(
    model,
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    real_directions: Sequence[DirectionSpec],
    cfg: QKSpecConfig,
    device: torch.device,
    baseline_f: Optional[float] = None,
    baseline_r: Optional[float] = None,
) -> ControlResult:
    """Run configured controls on the *un-edited* model and compare effects.

    Call this **before** applying the real edit (or after restoring it); the
    function guarantees the model is left in its original state.
    """
    result = ControlResult()
    if not cfg.run_controls:
        result.results["disabled"] = {"reason": "run_controls=False"}
        return result

    if baseline_f is None:
        baseline_f = _nll_over_loader(model, forget_loader, device, cfg.max_stage_b_batches)
    if baseline_r is None:
        baseline_r = _nll_over_loader(model, retain_loader, device, cfg.max_stage_b_batches)

    # Real-edit reference effect (temporary, restored).
    result.real_edit = _measure_edit_effect(
        model, real_directions, cfg, forget_loader, retain_loader,
        device, baseline_f, baseline_r,
    )

    gen = torch.Generator().manual_seed(cfg.control_seed)
    for mode in cfg.control_modes:
        try:
            ctrl_dirs = _make_control_directions(mode, real_directions, cfg, gen)
            result.results[mode] = _measure_edit_effect(
                model, ctrl_dirs, cfg, forget_loader, retain_loader,
                device, baseline_f, baseline_r,
            )
        except Exception as exc:
            result.results[mode] = {"error": f"{type(exc).__name__}: {exc}"}

    return result

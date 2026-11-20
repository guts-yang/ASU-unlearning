"""Two-stage direction selection for QKSpec.

Stage A ranks all shortlisted (layer, head, rank) triples by the screening
score; we take the global Top-K.  Stage B then causally re-checks each of them
via single-direction ablation and applies the retain constraint.  The final
direction set J is the subset of Top-K that (a) increases D_F NLL by at least
``stage_b_min_nll_increase`` and (b) does not increase D_R NLL by more than
``max_retain_impact`` (if configured).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .config import QKSpecConfig
from .qk_math import DirectionSpec
from .score import ScoreResult, score_stage_b


@dataclass
class SelectionResult:
    selected: List[DirectionSpec] = field(default_factory=list)
    top_k: List[DirectionSpec] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    stage_a_all: List[DirectionSpec] = field(default_factory=list)
    stage_b_all: List[DirectionSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": [d.to_dict() for d in self.selected],
            "top_k": [d.to_dict() for d in self.top_k],
            "rejected": list(self.rejected),
            "stage_a_all": [d.to_dict() for d in self.stage_a_all],
            "stage_b_all": [d.to_dict() for d in self.stage_b_all],
        }


def select_directions(
    score_result: ScoreResult,
    model,
    forget_loader,
    retain_loader,
    cfg: QKSpecConfig,
    device,
) -> SelectionResult:
    """Run Top-K screening then stage-B causal re-check and retain filtering."""
    result = SelectionResult(stage_a_all=list(score_result.stage_a_scores))

    ranked = sorted(
        score_result.stage_a_scores,
        key=lambda d: d.stage_a_score,
        reverse=True,
    )
    # Drop non-positive screening scores only if we still have enough; keep at
    # least top_k entries so stage B always has something to check.
    top_k = ranked[: cfg.top_k_directions]
    if not top_k:
        return result
    result.top_k = list(top_k)

    stage_b = score_stage_b(model, forget_loader, retain_loader, top_k, cfg, device)
    result.stage_b_all = list(stage_b)

    for d in stage_b:
        b = d.stage_b_score if d.stage_b_score is not None else 0.0
        reasons = []
        if b < cfg.stage_b_min_nll_increase:
            reasons.append(f"stage_b_nll_increase={b:.6f} < min={cfg.stage_b_min_nll_increase}")
        if cfg.max_retain_impact is not None:
            ri = d.retain_impact if d.retain_impact is not None else 0.0
            if ri > cfg.max_retain_impact:
                reasons.append(f"retain_impact={ri:.6f} > max={cfg.max_retain_impact}")
        if reasons:
            result.rejected.append({"direction": d.to_dict(), "reasons": reasons})
        else:
            result.selected.append(d)

    result.selected.sort(key=lambda d: d.stage_b_score or 0.0, reverse=True)
    return result

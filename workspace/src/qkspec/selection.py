"""Two-stage direction selection for QKSpec.

Stage A ranks all shortlisted (layer, head, rank) triples by the screening
score; we take the global Top-K.  Stage B then causally re-checks each of them
via single-direction ablation.  Final selection uses *causal specificity*:

    specificity_j = stage_b_score_j - w * retain_impact_j

where ``stage_b_score`` is the D_F NLL increase (target effect) and
``retain_impact`` is the D_R NLL increase (collateral damage).  A direction is
kept only if ``specificity > stage_b_min_nll_increase`` and its retain damage
stays under ``max_retain_impact``.  Directions are ranked by specificity, not
by raw target effect, so generic retrieval directions that damage retain are
pushed down.  ``w`` is ``cfg.specificity_weight`` (default 1.0).
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


def _specificity(d: DirectionSpec, w: float) -> float:
    b = d.stage_b_score if d.stage_b_score is not None else 0.0
    r = d.retain_impact if d.retain_impact is not None else 0.0
    return float(b) - w * float(r)


def select_directions(
    score_result: ScoreResult,
    model,
    forget_loader,
    retain_loader,
    cfg: QKSpecConfig,
    device,
) -> SelectionResult:
    """Stage-A Top-K, then stage-B causal re-check, then specificity filter."""
    result = SelectionResult(stage_a_all=list(score_result.stage_a_scores))

    ranked = sorted(
        score_result.stage_a_scores,
        key=lambda d: d.stage_a_score,
        reverse=True,
    )
    top_k = ranked[: cfg.top_k_directions]
    if not top_k:
        return result
    result.top_k = list(top_k)

    stage_b = score_stage_b(model, forget_loader, retain_loader, top_k, cfg, device)
    result.stage_b_all = list(stage_b)

    for d in stage_b:
        spec = _specificity(d, cfg.specificity_weight)
        d.meta["specificity"] = spec
        reasons = []
        if spec < cfg.stage_b_min_nll_increase:
            reasons.append(
                f"specificity={spec:.6f} < min={cfg.stage_b_min_nll_increase} "
                f"(stage_b={d.stage_b_score}, retain_impact={d.retain_impact})"
            )
        if cfg.max_retain_impact is not None:
            ri = d.retain_impact if d.retain_impact is not None else 0.0
            if ri > cfg.max_retain_impact:
                reasons.append(f"retain_impact={ri:.6f} > max={cfg.max_retain_impact}")
        if reasons:
            result.rejected.append({"direction": d.to_dict(), "reasons": reasons})
        else:
            result.selected.append(d)

    result.selected.sort(key=lambda d: _specificity(d, cfg.specificity_weight), reverse=True)
    return result

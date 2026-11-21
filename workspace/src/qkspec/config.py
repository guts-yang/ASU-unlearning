"""Configuration parsing and validation for QKSpec.

The config is intentionally a plain dataclass (no Hydra dependency) so the
algorithm kernel can be used standalone in tests or notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class QKSpecConfig:
    """All knobs for a single QKSpec run.

    Attributes
    ----------
    layer_start / layer_end:
        Inclusive layer index range to consider; ``None`` means all layers.
    shortlist_per_layer:
        Number of heads kept per layer after the cheap pre-filter (stage A
        pre-screening).
    max_candidate_heads:
        Hard cap on candidate heads across all layers after pre-screening.
    top_k_directions:
        Number of (layer, head, singular-index) directions entering stage B.
    ranks_per_head:
        Maximum number of top singular directions kept per head in stage A.
    rho:
        Contraction coefficient applied to selected singular values, 0<=rho<1.
    min_singular_value:
        Singular values below this are never selected.
    max_length:
        Token truncation for discovery and evaluation.
    discovery_batch_size:
        Batch size for the single forward pass used by stage A.
    max_discovery_batches:
        Limit on discovery batches; ``None`` means all of D_F / D_R.
    stage_b_batch_size:
        Batch size for the stage-B ablation re-runs.
    max_stage_b_batches:
        Limit on batches per stage-B direction; ``None`` means all of D_F.
    stage_b_min_nll_increase:
        Minimum NLL increase required to keep a direction after ablation.
        A value <= 0 keeps all Top-K directions.
    specificity_weight:
        Weight on retain damage when ranking stage-B directions. Final score
        is ``stage_b_score - specificity_weight * retain_impact``. A value of
        1.0 ranks by target-effect-minus-retain-damage (causal specificity);
        0.0 ranks by raw target effect only.
    max_retain_impact:
        Maximum allowed relative NLL increase on D_R when a direction is
        ablated; ``None`` disables this constraint.
    edge_top_fraction:
        When selecting target retrieval edges from attention weights, keep the
        top fraction of answer->prompt positions. 1.0 uses all edges.
    run_controls:
        Whether to run no-op / random-isospectral / shuffle controls.
    noop_rho:
        Rho for the no-op control (should be 1.0).
    control_seed:
        Seed for random direction / shuffle controls.
    enable_rope_check:
        Whether to record RoPE-position-wise logit error.
    rope_check_positions:
        Token positions at which to log QK logit error after editing.
    fallback_retain_random:
        If D_R cannot be paired sample-by-sample to D_F, fall back to random
        same-template retain samples and mark ``retain_pairing='random'``.
    empty_direction_action:
        What to do when no direction survives selection: ``'noop'`` returns an
        unmodified model; ``'error'`` raises.
    """

    layer_start: Optional[int] = None
    layer_end: Optional[int] = None
    shortlist_per_layer: int = 4
    max_candidate_heads: int = 128
    top_k_directions: int = 16
    ranks_per_head: int = 4
    rho: float = 0.5
    min_singular_value: float = 1e-6
    max_length: int = 512
    discovery_batch_size: int = 4
    max_discovery_batches: Optional[int] = None
    stage_b_batch_size: int = 4
    max_stage_b_batches: Optional[int] = None
    stage_b_min_nll_increase: float = 0.0
    specificity_weight: float = 1.0
    max_retain_impact: Optional[float] = None
    edge_top_fraction: float = 1.0
    run_controls: bool = True
    noop_rho: float = 1.0
    control_seed: int = 42
    enable_rope_check: bool = True
    rope_check_positions: Sequence[int] = field(default_factory=lambda: [0, 64, 128, 256])
    fallback_retain_random: bool = True
    empty_direction_action: str = "noop"
    control_modes: List[str] = field(
        default_factory=lambda: ["noop", "random_isospectral", "shuffle"]
    )
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {self.rho}")
        if self.top_k_directions < 1:
            raise ValueError("top_k_directions must be >= 1")
        if self.ranks_per_head < 1:
            raise ValueError("ranks_per_head must be >= 1")
        if self.shortlist_per_layer < 1:
            raise ValueError("shortlist_per_layer must be >= 1")
        if not 0.0 < self.edge_top_fraction <= 1.0:
            raise ValueError("edge_top_fraction must be in (0, 1]")
        if self.empty_direction_action not in ("noop", "error"):
            raise ValueError("empty_direction_action must be 'noop' or 'error'")
        if self.layer_start is not None and self.layer_end is not None:
            if self.layer_start > self.layer_end:
                raise ValueError("layer_start must be <= layer_end")
        if self.min_singular_value < 0:
            raise ValueError("min_singular_value must be >= 0")
        if self.specificity_weight < 0:
            raise ValueError("specificity_weight must be >= 0")

    @classmethod
    def from_any(cls, source: Any) -> "QKSpecConfig":
        """Build from dict, OmegaConf DictConfig, or an existing instance."""
        if isinstance(source, cls):
            return source
        if source is None:
            return cls()
        if hasattr(source, "keys"):
            data = {k: source[k] for k in source.keys()}
        else:
            raise TypeError(f"Cannot build QKSpecConfig from {type(source)!r}")
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        unknown = {k: v for k, v in data.items() if k not in known}
        if unknown:
            kwargs["extra"] = dict(unknown)
        # OmegaConf ListConfig -> list, etc.
        for key in ("rope_check_positions", "control_modes"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = list(kwargs[key])
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

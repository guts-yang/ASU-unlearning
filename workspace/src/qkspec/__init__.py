"""QKSpec: targeted QK spectral unlearning (target-only, gradient-free).

Public entry points are re-exported here so callers only need to import from
``qkspec``.
"""

from .config import QKSpecConfig
from .qk_math import (
    DirectionSpec,
    build_qk_matrix,
    compute_delta_wq,
    fold_into_q,
    get_head_slices,
    get_k_head_slices,
    get_kv_head_index,
    svd_qk,
)
from .hooks import ActivationCache, install_qk_hooks
from .score import ScoreResult, score_stage_a, score_stage_b
from .selection import SelectionResult, select_directions
from .controls import ControlResult, run_controls
from .edit import EditResult, run_qkspec_edit
from .evaluate import EvalResult, compute_nll_and_accuracy
from .checks import compute_spectral_residual, attention_distribution_shift
from .report import write_report

__all__ = [
    "QKSpecConfig",
    "DirectionSpec",
    "build_qk_matrix",
    "compute_delta_wq",
    "fold_into_q",
    "get_head_slices",
    "get_k_head_slices",
    "get_kv_head_index",
    "svd_qk",
    "ActivationCache",
    "install_qk_hooks",
    "ScoreResult",
    "score_stage_a",
    "score_stage_b",
    "SelectionResult",
    "select_directions",
    "ControlResult",
    "run_controls",
    "EditResult",
    "run_qkspec_edit",
    "EvalResult",
    "compute_nll_and_accuracy",
    "compute_spectral_residual",
    "attention_distribution_shift",
    "write_report",
]

"""Typed contracts for RecodeLoop M0.

All key state is carried by frozen dataclasses. Mappings, where present, are
restricted to explicitly registered metric-name -> float records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import torch

from .errors import ContractViolation

Split = Literal["forget_train_edit", "forget_train_reader", "calibration", "confirmation"]
Role = Literal["base", "target", "asu_student"]
PatchKind = Literal["self", "nk", "reverse", "random"]
Scope = Literal["single", "joint"]
Coverage = Literal["covered", "uncovered"]

SPLITS: tuple[Split, ...] = (
    "forget_train_edit", "forget_train_reader", "calibration", "confirmation"
)
ROLES: tuple[Role, ...] = ("base", "target", "asu_student")


@dataclass(frozen=True)
class CheckpointRef:
    """Frozen asset reference. Empty sha256 means G0 BLOCKED; never guess."""

    role: Role
    path: str
    revision: str | None
    tokenizer_path: str
    sha256: str | None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ContractViolation(f"invalid checkpoint role: {self.role!r}")
        if not self.path or not self.tokenizer_path:
            raise ContractViolation("CheckpointRef requires path and tokenizer_path")


@dataclass(frozen=True)
class EntityCluster:
    """Entity-level data unit. Split is assigned by canonical author identity."""

    canonical_id: str
    aliases: tuple[str, ...]
    source_rows: tuple[str, ...]
    split: Split
    fact_relation: str
    fact_values: tuple[str, ...]
    provenance: str

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ContractViolation(f"invalid split: {self.split!r}")
        if not self.canonical_id:
            raise ContractViolation("canonical_id must not be empty")
        if not self.fact_values:
            raise ContractViolation("fact_values must not be empty; fact values are required")


@dataclass(frozen=True)
class ActivationSpec:
    """Declared activation view. shape excludes leading batch dimension."""

    layer: int
    component: str
    token_position: int | Literal["all"]
    pooling: Literal["last", "mean", "none"]
    dtype: str
    device: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ContractViolation("layer must be >= 0")
        if self.pooling not in ("last", "mean", "none"):
            raise ContractViolation(f"invalid pooling: {self.pooling!r}")
        if self.pooling == "last" and self.token_position != "all":
            raise ContractViolation("pooling='last' requires token_position='all'")


@dataclass
class ActivationBatch:
    """Activations with explicit labels, split and provenance hash.

    activations: float tensor [N, ...]; dtype/device must match spec.
    """

    activations: torch.Tensor
    entity_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]
    split: Split
    fact_value_labels: tuple[str, ...]
    source_hash: str
    spec: ActivationSpec

    def __post_init__(self) -> None:
        n = self.activations.shape[0]
        if len(self.entity_ids) != n or len(self.fact_ids) != n or len(self.fact_value_labels) != n:
            raise ContractViolation("activation batch length mismatch")
        if self.split not in SPLITS:
            raise ContractViolation(f"invalid split: {self.split!r}")


@dataclass(frozen=True)
class RecoveryScore:
    """Fact-value reader output, including never-known control."""

    fact_value_accuracy: float
    exact_match: float
    candidate_rank: int | None
    mrr: float | None
    confidence: float
    nk_accuracy: float
    cluster_id: str
    split: Split
    n_samples: int

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ContractViolation(f"invalid split: {self.split!r}")
        if self.n_samples <= 0:
            raise ContractViolation("n_samples must be positive")


@dataclass(frozen=True)
class PatchResult:
    """Result of one causal patch intervention on one or several components."""

    patch_kind: PatchKind
    scope: Scope
    component: str | None
    recovery_before: float
    recovery_after: float
    recovery_delta: float
    retain_damage: float
    ci_low: float | None
    ci_high: float | None

    def __post_init__(self) -> None:
        if self.patch_kind not in ("self", "nk", "reverse", "random"):
            raise ContractViolation(f"invalid patch kind: {self.patch_kind!r}")
        if self.scope == "single" and not self.component:
            raise ContractViolation("single-component patch requires component path")


@dataclass(frozen=True)
class ComponentSelection:
    """Selected components with frozen ranks and per-component energies g_{c,t}."""

    components: tuple[str, ...]
    ranks: tuple[int, ...]
    energies: tuple[float, ...]
    rule: str
    config_hash: str
    interaction_note: str | None = None

    def __post_init__(self) -> None:
        if not (len(self.components) == len(self.ranks) == len(self.energies)):
            raise ContractViolation("components/ranks/energies length mismatch")
        if any(r <= 0 for r in self.ranks):
            raise ContractViolation("ranks must be positive")


@dataclass(frozen=True)
class Budget:
    """Matched budget record. Global G2 and per-component g are recorded separately."""

    tokens: int
    steps: int
    components: int
    rank: int
    wall_clock_s: float | None = None

    def __post_init__(self) -> None:
        if min(self.tokens, self.steps, self.components, self.rank) < 0:
            raise ContractViolation("budget values must be non-negative")


@dataclass(frozen=True)
class LoopState:
    step: int
    model_hash: str
    reader_hash: str
    components: tuple[str, ...]
    actual_budget: Budget
    stop_reason: str | None = None


@dataclass(frozen=True)
class SeedRecord:
    seed: int
    python_random: int
    numpy_random: int
    torch_cpu: int
    torch_cuda: tuple[int, ...] = ()


@dataclass(frozen=True)
class EnvironmentRecord:
    python_version: str
    torch_version: str
    transformers_version: str | None
    cuda_available: bool
    gpu_names: tuple[str, ...]
    code_hash: str
    config_hash: str
    seed: SeedRecord


@dataclass(frozen=True)
class AuditResult:
    """One independent audit arm.

    raw_metrics/control_delta are registered metric-name -> float maps only.
    """

    arm: str
    independence_statement: str
    budget: Budget
    raw_metrics: Mapping[str, float]
    control_delta: Mapping[str, float]
    coverage: Coverage
    note: str = ""

    def __post_init__(self) -> None:
        if self.coverage not in ("covered", "uncovered"):
            raise ContractViolation(f"invalid coverage: {self.coverage!r}")


@dataclass(frozen=True)
class ControlResult:
    arm: str
    budget: Budget
    metrics: Mapping[str, float]
    config_hash: str
    seed: SeedRecord
    note: str = ""


@runtime_checkable
class FactValueReader(Protocol):
    """Reader that decodes concrete fact values, not membership labels."""

    name: str

    def score(
        self,
        batch: ActivationBatch,
        fact_ids: Sequence[str],
        split: Split,
        nk_ids: Sequence[str],
    ) -> RecoveryScore: ...

    def state_dict_hash(self) -> str: ...


def uncovered_audit(arm: str, reason: str, budget: Budget | None = None) -> AuditResult:
    """Factory for audits that were not actually executed."""
    return AuditResult(
        arm=arm,
        independence_statement="not run",
        budget=budget or Budget(tokens=0, steps=0, components=0, rank=0),
        raw_metrics={},
        control_delta={},
        coverage="uncovered",
        note=f"未覆盖：{reason}",
    )


def as_plain(value: Any) -> Any:
    """Convert dataclasses/tensors to plain JSON-safe structures for logs."""
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if hasattr(value, "__dataclass_fields__"):
        return {k: as_plain(v) for k, v in vars(value).items()}
    if isinstance(value, (tuple, list)):
        return [as_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): as_plain(v) for k, v in value.items()}
    return value

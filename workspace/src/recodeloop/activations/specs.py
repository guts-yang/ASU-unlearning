"""Build declarative activation views for real linear modules."""

from __future__ import annotations

import re

import torch

from ..baseline_adapter import ALLOWED_COMPONENTS, BaselineAdapter
from ..errors import ContractViolation
from ..types import ActivationSpec


def build_spec(
    adapter: BaselineAdapter,
    component: str,
    token_position: int | str = "all",
    pooling: str = "none",
) -> ActivationSpec:
    """Create a spec for a real o_proj/down_proj output."""
    if not component.endswith(ALLOWED_COMPONENTS):
        raise ContractViolation(f"component not allowed for activation view: {component!r}")
    try:
        module = adapter.model.get_submodule(component)
    except AttributeError as exc:
        raise ContractViolation(f"unknown component path: {component!r}") from exc
    if not isinstance(module, torch.nn.Linear):
        raise ContractViolation(f"{component} is not nn.Linear")
    if pooling not in ("last", "mean", "none"):
        raise ContractViolation(f"invalid pooling: {pooling!r}")

    out_features = int(module.out_features)
    pooled = pooling in ("last", "mean") or isinstance(token_position, int)
    shape = (out_features,) if pooled else (-1, out_features)
    param = next(adapter.model.parameters())
    match = re.search(r"layers\.(\d+)", component)
    return ActivationSpec(
        layer=int(match.group(1)) if match else -1,
        component=component,
        token_position=token_position,
        pooling=pooling,  # type: ignore[arg-type]
        dtype=str(param.dtype).replace("torch.", ""),
        device=str(param.device),
        shape=shape,
    )


def _same_storage(spec: ActivationSpec, tensor: torch.Tensor) -> bool:
    return (
        str(tensor.dtype).replace("torch.", "") == spec.dtype
        and str(tensor.device) == spec.device
    )


def spec_matches_raw(spec: ActivationSpec, tensor: torch.Tensor) -> bool:
    """Validate the pre-pooling module output tensor [N, L, D]."""
    if not _same_storage(spec, tensor) or tensor.ndim < 3:
        return False
    return int(tensor.shape[-1]) == int(spec.shape[-1])


def spec_matches(spec: ActivationSpec, tensor: torch.Tensor) -> bool:
    """Validate the stored (post-pooling, batch-excluded) tensor."""
    if not _same_storage(spec, tensor):
        return False
    actual = tuple(tensor.shape[1:])
    expected = tuple(spec.shape)
    if len(actual) != len(expected):
        return False
    return all(e == -1 or e == a for e, a in zip(expected, actual))

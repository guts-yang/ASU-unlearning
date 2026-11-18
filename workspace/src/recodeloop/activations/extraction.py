"""External forward-hook activation extraction (no baseline topology changes)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import torch

from ..baseline_adapter import BaselineAdapter
from ..errors import ContractViolation
from ..types import ActivationBatch, ActivationSpec, Split
from ..utils.hashing import sha256_json
from .specs import build_spec, spec_matches, spec_matches_raw


def _select_and_pool(tensor: torch.Tensor, spec: ActivationSpec) -> torch.Tensor:
    if tensor.ndim < 3:
        raise ContractViolation(f"expected [N,L,D] activation, got {tuple(tensor.shape)}")
    if isinstance(spec.token_position, int):
        return tensor[:, spec.token_position, :]
    if spec.pooling == "last":
        return tensor[:, -1, :]
    if spec.pooling == "mean":
        return tensor.mean(dim=1)
    return tensor


@contextmanager
def capture_raw_activations(
    model: torch.nn.Module,
    components: Sequence[str],
    detach: bool = True,
) -> Iterator[dict[str, torch.Tensor]]:
    """Capture unpooled [N, L, D] module outputs for donor patching."""
    store: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    for component in components:
        module = model.get_submodule(component)

        def hook(_module: torch.nn.Module, _inputs: Any, output: Any,
                 _component: str = component) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            store[_component] = tensor.detach().clone() if detach else tensor
            return output

        handles.append(module.register_forward_hook(hook))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()
        store.clear()


def extract_raw_donors(
    adapter: BaselineAdapter,
    components: Sequence[str],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    detach: bool = True,
) -> dict[str, torch.Tensor]:
    """Run one forward and return raw donor tensors keyed by component."""
    with capture_raw_activations(adapter.model, components, detach=detach) as store:
        adapter.forward(input_ids, attention_mask=attention_mask)
        return dict(store)


@contextmanager
def capture_activations(
    model: torch.nn.Module,
    specs: Sequence[ActivationSpec],
    detach: bool = True,
) -> Iterator[dict[str, torch.Tensor]]:
    """Capture declared views during one forward pass; always remove hooks."""
    store: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    for spec in specs:
        module = model.get_submodule(spec.component)

        def hook(_module: torch.nn.Module, _inputs: Any, output: Any,
                 _spec: ActivationSpec = spec) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            if not spec_matches_raw(_spec, tensor):
                raise ContractViolation(
                    f"raw activation spec mismatch at {_spec.component}: "
                    f"got shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}"
                )
            pooled = _select_and_pool(tensor, _spec).detach()
            if not spec_matches(_spec, pooled):
                raise ContractViolation(
                    f"pooled activation spec mismatch at {_spec.component}: "
                    f"got shape={tuple(pooled.shape)}, expected {_spec.shape}"
                )
            store[_spec.component] = pooled.detach() if detach else pooled
            return output

        handles.append(module.register_forward_hook(hook))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()
        store.clear()


def extract_activation_batches(
    adapter: BaselineAdapter,
    specs: Sequence[ActivationSpec],
    input_ids: torch.Tensor,
    entity_ids: Sequence[str],
    fact_ids: Sequence[str],
    split: Split,
    fact_value_labels: Sequence[str],
    attention_mask: torch.Tensor | None = None,
    detach: bool = True,
) -> dict[str, ActivationBatch]:
    source_hash = sha256_json({
        "input_ids": input_ids.tolist(),
        "specs": [s.component for s in specs],
        "split": split,
    })
    with capture_activations(adapter.model, specs, detach=detach) as store:
        adapter.forward(input_ids, attention_mask=attention_mask)
        captured = dict(store)
    return {
        spec.component: ActivationBatch(
            activations=captured[spec.component],
            entity_ids=tuple(entity_ids),
            fact_ids=tuple(fact_ids),
            split=split,
            fact_value_labels=tuple(fact_value_labels),
            source_hash=source_hash,
            spec=spec,
        )
        for spec in specs
    }


def extract_activation_batch(
    adapter: BaselineAdapter,
    component: str,
    input_ids: torch.Tensor,
    entity_ids: Sequence[str],
    fact_ids: Sequence[str],
    split: Split,
    fact_value_labels: Sequence[str],
    token_position: int | str = "all",
    pooling: str = "none",
    attention_mask: torch.Tensor | None = None,
) -> ActivationBatch:
    spec = build_spec(adapter, component, token_position=token_position, pooling=pooling)
    return extract_activation_batches(
        adapter, [spec], input_ids, entity_ids, fact_ids, split,
        fact_value_labels, attention_mask,
    )[component]

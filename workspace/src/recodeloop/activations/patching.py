"""Causal patch context: self / NK / reverse / random, single or joint.

Patching replaces module *outputs* via external hooks. It never edits weights
or baseline DecoderLayer math, and restores hooks/state on context exit.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import torch

from ..errors import ContractViolation
from ..types import ActivationBatch, PatchKind

_DONOR_KINDS: tuple[PatchKind, ...] = ("self", "nk", "reverse", "random")


@contextmanager
def patch_components(
    model: torch.nn.Module,
    donors: Mapping[str, torch.Tensor],
    kind: PatchKind = "nk",
    strict_shape: bool = True,
) -> Iterator[dict[str, torch.Tensor]]:
    """Patch selected components with donor tensors for the duration of the context.

    Yields the original outputs keyed by component, for verification.
    """
    if kind not in _DONOR_KINDS:
        raise ContractViolation(f"invalid patch kind: {kind!r}")
    originals: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    hook_counts: dict[str, int] = {}

    for component, donor in donors.items():
        module = model.get_submodule(component)
        hook_counts[component] = len(module._forward_hooks)

        def hook(_module: torch.nn.Module, _inputs: Any, output: Any,
                 _component: str = component, _donor: torch.Tensor = donor) -> Any:
            current = output[0] if isinstance(output, tuple) else output
            originals[_component] = current.detach().clone()
            if strict_shape and tuple(_donor.shape) != tuple(current.shape):
                raise ContractViolation(
                    f"donor shape {tuple(_donor.shape)} != output shape "
                    f"{tuple(current.shape)} for {_component}"
                )
            donor = _donor.to(device=current.device, dtype=current.dtype)
            if isinstance(output, tuple):
                return (donor,) + output[1:]
            return donor

        handles.append(module.register_forward_hook(hook))

    try:
        yield originals
    finally:
        for handle in handles:
            handle.remove()
        originals.clear()
        for component, before in hook_counts.items():
            module = model.get_submodule(component)
            if len(module._forward_hooks) != before:
                raise ContractViolation(
                    f"patch context did not restore hooks at {component}"
                )


def donors_from_batches(
    batches: Mapping[str, ActivationBatch],
    components: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    keys = tuple(components) if components is not None else tuple(batches)
    return {key: batches[key].activations.detach().clone() for key in keys}


def random_donors(
    donors: Mapping[str, torch.Tensor],
    seed: int,
) -> dict[str, torch.Tensor]:
    """Random-component/random-example donor with matched shapes."""
    generator = torch.Generator().manual_seed(seed)
    keys = list(donors)
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        tensor = donors[key]
        perm = torch.randperm(tensor.shape[0], generator=generator)
        source_key = keys[int(torch.randint(len(keys), (1,), generator=generator).item())]
        source = donors[source_key]
        if source.shape[1:] != tensor.shape[1:]:
            source = tensor
        out[key] = source[perm].detach().clone()
    return out

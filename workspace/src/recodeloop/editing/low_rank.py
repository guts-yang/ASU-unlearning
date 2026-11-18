"""Reversible low-rank factors attached via external forward hooks.

Only A and B are trainable. The base weight is never touched, so attach/remove
is exactly reversible and base-weight invariance is checkable element-wise.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from ..baseline_adapter import ALLOWED_COMPONENTS, BaselineAdapter
from ..errors import ContractViolation
from ..types import ComponentSelection
from ..utils.hashing import sha256_tensor


class LowRankEditor:
    def __init__(self, adapter: BaselineAdapter) -> None:
        self.adapter = adapter
        self.factors: dict[str, tuple[nn.Parameter, nn.Parameter]] = {}
        self.handles: list[tuple[str, Any]] = []
        self.ranks: dict[str, int] = {}

    def attach(self, selection: ComponentSelection) -> None:
        if self.handles:
            raise ContractViolation("editor already attached")
        for component, rank in zip(selection.components, selection.ranks):
            if not component.endswith(ALLOWED_COMPONENTS):
                raise ContractViolation(f"component not allowed: {component!r}")
            module = self.adapter.model.get_submodule(component)
            if rank <= 0 or rank > min(module.in_features, module.out_features):
                raise ContractViolation(f"invalid rank {rank} for {component}")
            A = nn.Parameter(torch.empty(rank, module.in_features))
            B = nn.Parameter(torch.zeros(module.out_features, rank))
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            self.factors[component] = (A, B)
            self.ranks[component] = rank

            def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...],
                     output: torch.Tensor, _A: nn.Parameter = A,
                     _B: nn.Parameter = B) -> torch.Tensor:
                x = inputs[0]
                delta = (x @ _A.T) @ _B.T
                return output + delta.to(dtype=output.dtype)

            self.handles.append((component, module.register_forward_hook(hook)))

    def remove(self) -> None:
        for _component, handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.factors.clear()
        self.ranks.clear()

    def low_rank_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for A, B in self.factors.values():
            params.extend([A, B])
        return params

    def assert_only_low_rank_trainable(self) -> None:
        trainable = {name for name, p in self.adapter.model.named_parameters() if p.requires_grad}
        if trainable:
            raise ContractViolation(f"base model params are trainable: {sorted(trainable)[:5]}")
        if not all(p.requires_grad for p in self.low_rank_params()):
            raise ContractViolation("all low-rank factors must require grad")

    def assert_ranks(self) -> Mapping[str, int]:
        report: dict[str, int] = {}
        for component, (A, B) in self.factors.items():
            actual = int(torch.linalg.matrix_rank((B @ A).detach()).item())
            if actual > self.ranks[component]:
                raise ContractViolation(f"rank exceeded at {component}: {actual}")
            report[component] = actual
        return report

    @torch.no_grad()
    def project_energy(self, g: float) -> Mapping[str, float]:
        """Project each delta W to Frobenius energy g_{c,t}."""
        out: dict[str, float] = {}
        for component, (A, B) in self.factors.items():
            delta = B @ A
            norm = torch.linalg.norm(delta).item()
            if norm <= 0:
                raise ContractViolation(f"zero-energy delta at {component}; cannot project")
            scale = g / norm
            B.mul_(scale)
            out[component] = float(torch.linalg.norm(B @ A).item())
        return out

    def assert_base_unchanged(self, initial_hash: str) -> None:
        if self.adapter.state_hash() != initial_hash:
            raise ContractViolation("base weights changed during low-rank edit")

    def energies(self) -> Mapping[str, str]:
        return {
            component: sha256_tensor((B @ A).detach())
            for component, (A, B) in self.factors.items()
        }

    def save(self, path: Path | str) -> None:
        torch.save(
            {c: {"A": A.detach().cpu(), "B": B.detach().cpu()}
             for c, (A, B) in self.factors.items()},
            Path(path),
        )

    def load(self, path: Path | str) -> None:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        for component, tensors in payload.items():
            if component not in self.factors:
                raise ContractViolation(f"unknown component in editor payload: {component}")
            A, B = self.factors[component]
            A.data.copy_(tensors["A"].to(A.device, A.dtype))
            B.data.copy_(tensors["B"].to(B.device, B.dtype))

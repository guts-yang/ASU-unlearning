from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F


SUPPORTED_POINTS = ("attention_residual_add", "decoder_layer_output")
SUPPORTED_SCHEDULES = ("progressive", "static_full", "reverse_progressive", "random")


class LayerSchedule:
    """Deterministic layer-set and layer-weight schedule."""

    def __init__(
        self,
        num_layers: int,
        schedule_type: str = "progressive",
        initial_fraction: float = 0.25,
        expansion_fraction: float = 0.25,
        stage_steps: int = 10,
        seed: int = 0,
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if schedule_type not in SUPPORTED_SCHEDULES:
            raise ValueError(f"Unsupported schedule_type: {schedule_type}")
        if not 0 < initial_fraction <= 1:
            raise ValueError("initial_fraction must be in (0, 1]")
        if not 0 < expansion_fraction <= 1:
            raise ValueError("expansion_fraction must be in (0, 1]")
        if stage_steps <= 0:
            raise ValueError("stage_steps must be positive")

        self.num_layers = num_layers
        self.schedule_type = schedule_type
        self.initial_fraction = initial_fraction
        self.expansion_fraction = expansion_fraction
        self.stage_steps = stage_steps
        self.seed = seed
        self._initial_count = max(1, math.ceil(initial_fraction * num_layers))
        self._expansion_count = max(1, math.ceil(expansion_fraction * num_layers))
        order = list(range(num_layers))
        random.Random(seed).shuffle(order)
        self._random_order = order

    def stage(self, step: int) -> int:
        if self.schedule_type == "static_full":
            return self.num_layers
        return max(0, int(step) // self.stage_steps)

    def active_layers(self, step: int) -> List[int]:
        if self.schedule_type == "static_full":
            return list(range(self.num_layers))
        count = min(
            self.num_layers,
            self._initial_count + self.stage(step) * self._expansion_count,
        )
        if self.schedule_type == "progressive":
            return list(range(self.num_layers - count, self.num_layers))
        if self.schedule_type == "reverse_progressive":
            return list(range(count))
        if self.schedule_type == "random":
            return sorted(self._random_order[:count])
        raise RuntimeError(f"Unhandled schedule_type: {self.schedule_type}")

    def layer_weights(self, step: int) -> Dict[int, float]:
        active = self.active_layers(step)
        if not active:
            return {}
        value = 1.0 / len(active)
        return {layer: value for layer in active}


class ActivationReader:
    """Read layer activations at a pre-registered point without patching the model."""

    def __init__(
        self,
        model: torch.nn.Module,
        point: str = "attention_residual_add",
        layers: Optional[Sequence[int]] = None,
    ) -> None:
        if point not in SUPPORTED_POINTS:
            raise ValueError(f"Unsupported activation point: {point}")
        self.model = model
        self.point = point
        self.layers = list(layers) if layers is not None else None
        self._cache: Dict[int, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _base_model(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if hasattr(model, "module") else model

    def _resolve_layers(self, base: torch.nn.Module) -> List[int]:
        decoder_layers = base.model.layers
        if self.layers is None:
            return list(range(len(decoder_layers)))
        for layer in self.layers:
            if layer < 0 or layer >= len(decoder_layers):
                raise IndexError(f"Layer {layer} outside model with {len(decoder_layers)} layers")
        return list(self.layers)

    def _make_pre_hook(self, layer: int, detach: bool):
        def hook(_module: torch.nn.Module, inputs):
            tensor = inputs[0]
            if detach:
                tensor = tensor.detach()
            self._cache[layer] = tensor

        return hook

    def _make_forward_hook(self, layer: int, detach: bool):
        def hook(_module: torch.nn.Module, _inputs, outputs):
            tensor = outputs[0] if isinstance(outputs, tuple) else outputs
            if detach:
                tensor = tensor.detach()
            self._cache[layer] = tensor

        return hook

    def read(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        detach: bool = False,
    ) -> torch.Tensor:
        base = self._base_model(self.model)
        layers = self._resolve_layers(base)
        self._cache = {}
        self._handles = []

        try:
            for layer in layers:
                decoder_layer = base.model.layers[layer]
                if self.point == "attention_residual_add":
                    handle = decoder_layer.post_attention_layernorm.register_forward_pre_hook(
                        self._make_pre_hook(layer, detach)
                    )
                else:
                    handle = decoder_layer.register_forward_hook(
                        self._make_forward_hook(layer, detach)
                    )
                self._handles.append(handle)

            forward_kwargs = {"input_ids": input_ids, "use_cache": False}
            if attention_mask is not None:
                forward_kwargs["attention_mask"] = attention_mask
            with torch.set_grad_enabled(not detach):
                base(**forward_kwargs)
        finally:
            for handle in self._handles:
                handle.remove()
            self._handles = []

        missing = [layer for layer in layers if layer not in self._cache]
        if missing:
            raise RuntimeError(f"Activation hook did not capture layers: {missing}")
        return torch.stack([self._cache[layer] for layer in layers], dim=1)


def normalize_activations(activations: torch.Tensor) -> torch.Tensor:
    return F.normalize(activations, p=2, dim=-1)


def activation_distance(
    student: torch.Tensor,
    teacher: torch.Tensor,
    norm: str = "l2",
    distance: str = "mse_on_l2_normalized",
) -> torch.Tensor:
    if norm != "l2" or distance != "mse_on_l2_normalized":
        raise ValueError(f"Unsupported activation metric: norm={norm}, distance={distance}")
    student_norm = normalize_activations(student)
    teacher_norm = normalize_activations(teacher.detach())
    return ((student_norm - teacher_norm) ** 2).mean(dim=-1)


def activation_alignment_loss(
    student_activations: torch.Tensor,
    teacher_activations: torch.Tensor,
    active_layers: Iterable[int],
    layer_weights: Dict[int, float],
    norm: str = "l2",
    distance: str = "mse_on_l2_normalized",
) -> torch.Tensor:
    layers = list(active_layers)
    if not layers:
        return student_activations.sum() * 0.0
    total = None
    for layer in layers:
        layer_loss = activation_distance(
            student_activations[:, layer],
            teacher_activations[:, layer],
            norm=norm,
            distance=distance,
        ).mean()
        weighted = layer_loss * float(layer_weights.get(layer, 0.0))
        total = weighted if total is None else total + weighted
    denominator = max(1e-8, sum(float(layer_weights.get(layer, 0.0)) for layer in layers))
    return total / denominator

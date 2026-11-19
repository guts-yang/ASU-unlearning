"""Attention temperature for ASU teacher via HF LlamaAttention.scaling."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Optional, Sequence


def _iter_self_attn(model):
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    if layers is None:
        raise AttributeError("ASU teacher expects a Llama-like model.model.layers")
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        yield idx, attn


@contextmanager
def attention_temperature(
    model,
    tau: float = 2.3,
    layers_id: Optional[Sequence[int]] = None,
):
    """Temporarily set Softmax(QK^T / (τ √d_k)) by scaling attn.scaling /= τ."""
    if tau is None or float(tau) == 1.0:
        yield
        return

    allow: Optional[set[int]] = None
    if layers_id is not None and len(layers_id) > 0:
        allow = {int(i) for i in layers_id}

    restored: list[tuple[object, float]] = []
    for idx, attn in _iter_self_attn(model):
        if allow is not None and idx not in allow:
            continue
        if not hasattr(attn, "scaling"):
            continue
        restored.append((attn, float(attn.scaling)))
        attn.scaling = float(attn.scaling) / float(tau)
    try:
        yield
    finally:
        for attn, scale in restored:
            attn.scaling = scale

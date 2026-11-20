"""Forward hooks to cache the activations QKSpec needs for discovery.

We cache, per decoder layer:

* ``q_input`` / ``k_input``: the **pre-RoPE** hidden states fed into
  ``q_proj`` / ``k_proj`` (``[B, S, d_model]``).  These are the ``x_q / x_k``
  used in the design doc's ``x_q @ M @ x_k.T``.
* ``attn_weights``: the eager attention probabilities ``[B, H, S, S]``, used to
  identify which query->key positions form the "retrieval edges" of a sample.
  Flash / SDPA do not return these, so the caller must use eager attention.

Hooks are installed on the **custom** ``my_llama`` modules already present in
this repository.  We locate ``LlamaDecoderLayer`` instances by walking
``model.model.layers``; if a different model family is used we fall back to
finding all modules with ``q_proj`` / ``k_proj`` / ``self_attn`` attributes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch import nn


@dataclass
class LayerActivation:
    q_input: Optional[torch.Tensor] = None
    k_input: Optional[torch.Tensor] = None
    attn_weights: Optional[torch.Tensor] = None


@dataclass
class ActivationCache:
    """Container for cached per-layer activations, keyed by layer index."""

    layers: Dict[int, LayerActivation] = field(default_factory=dict)
    num_layers: int = 0
    error: Optional[str] = None

    def get(self, layer: int) -> LayerActivation:
        return self.layers.get(layer, LayerActivation())

    def clear(self) -> None:
        for act in self.layers.values():
            act.q_input = None
            act.k_input = None
            act.attn_weights = None


def _find_layers(model: nn.Module) -> List[Tuple[int, nn.Module]]:
    """Return ``[(layer_idx, decoder_layer), ...]`` for a causal LM."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is not None:
        return [(i, layer) for i, layer in enumerate(layers)]
    # Fallback: any module that owns a self_attn with q_proj/k_proj.
    found: List[Tuple[int, nn.Module]] = []
    for i, module in enumerate(model.modules()):
        attn = getattr(module, "self_attn", None)
        if attn is not None and hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
            found.append((i, module))
    return found


class _HookState:
    def __init__(self, cache: ActivationCache) -> None:
        self.cache = cache
        self.handles: List[torch.utils.hooks.RemovableHandle] = []


def install_qk_hooks(model: nn.Module, cache: ActivationCache) -> _HookState:
    """Install hooks that populate ``cache`` on forward and return a state
    object whose ``close()`` removes all hooks.

    Use via :func:`capture_qk_activations` context manager.
    """
    state = _HookState(cache)
    layers = _find_layers(model)
    cache.num_layers = len(layers)

    if not layers:
        cache.error = "no decoder layers found; cannot install QKSpec hooks"
        return state

    for idx, layer in layers:
        act = cache.layers.setdefault(idx, LayerActivation())
        attn = getattr(layer, "self_attn", None)
        if attn is None or not hasattr(attn, "q_proj") or not hasattr(attn, "k_proj"):
            cache.error = (
                f"layer {idx} has no self_attn.q_proj/k_proj; "
                "QKSpec requires the custom my_llama/my_mistral models"
            )
            return state

        def q_hook(module, inputs, output, _act=act):
            _act.q_input = inputs[0].detach() if inputs else None
            return None

        def k_hook(module, inputs, output, _act=act):
            _act.k_input = inputs[0].detach() if inputs else None
            return None

        def attn_hook(module, inputs, output, _act=act):
            # LlamaAttention.forward -> (attn_output, attn_weights, past_kv)
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                aw = output[1]
                if aw is not None:
                    _act.attn_weights = aw.detach()
            return None

        state.handles.append(attn.register_forward_hook(attn_hook))
        state.handles.append(attn.q_proj.register_forward_hook(q_hook))
        state.handles.append(attn.k_proj.register_forward_hook(k_hook))

    return state


@contextmanager
def capture_qk_activations(
    model: nn.Module,
    cache: ActivationCache,
) -> Iterator[ActivationCache]:
    """Context manager: install QK hooks for the duration of the block."""
    state = install_qk_hooks(model, cache)
    try:
        yield cache
    finally:
        for handle in state.handles:
            handle.remove()


def require_eager_cache(cache: ActivationCache) -> None:
    """Raise a clear error if attention weights were not captured."""
    if cache.error:
        raise RuntimeError(cache.error)
    if not cache.layers:
        raise RuntimeError("no activations cached")
    missing = [
        idx
        for idx, act in cache.layers.items()
        if act.attn_weights is None
    ]
    if missing:
        raise RuntimeError(
            "attention weights were not captured for layers "
            f"{missing[:8]}; set model config attn_implementation='eager' "
            "and pass output_attentions=True (flash/sdpa cannot be used for "
            "QKSpec discovery)."
        )

"""External residual patch editor for frozen Llama decoder layers.

Hooks replace the question-mark residual after a decoder layer returns.
They do not edit model weights, my_llama.py, or the ASU attention-temperature path.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualEditor:
    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __enter__(self) -> "ResidualEditor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def _layer(self, layer_idx: int) -> nn.Module:
        layers = self.model.model.layers
        if layer_idx < 0 or layer_idx >= len(layers):
            raise RuntimeError(f"layer {layer_idx} is outside 0..{len(layers) - 1}")
        return layers[layer_idx]

    def capture(self, position: int) -> "ResidualEditor":
        """Read each decoder output at `position` without changing the residual."""
        self.close()
        self.captured: dict[int, torch.Tensor] = {}

        def make_hook(layer_idx: int):
            def hook(module, inputs, output):
                hidden = _layer_hidden(output)
                if position < 0 or position >= hidden.shape[1]:
                    raise RuntimeError(
                        f"anchor {position} outside length {hidden.shape[1]} at layer {layer_idx}"
                    )
                self.captured[layer_idx] = hidden[0, position, :].detach().clone()
                return output

            return hook

        for layer_idx in range(len(self.model.model.layers)):
            self.handles.append(self._layer(layer_idx).register_forward_hook(make_hook(layer_idx)))
        return self

    def replace(self, replacements: dict[int, torch.Tensor], position: int) -> "ResidualEditor":
        """Replace decoder outputs at one shared token position."""
        self.close()
        if not replacements:
            raise RuntimeError("replacement set is empty")
        for layer_idx, vector in replacements.items():
            if vector.ndim != 1:
                raise RuntimeError(f"layer {layer_idx} replacement rank is {vector.ndim}")
            stored = vector.detach().clone()

            def hook(module, inputs, output, stored=stored, layer_idx=layer_idx):
                hidden = _layer_hidden(output)
                if position < 0 or position >= hidden.shape[1]:
                    raise RuntimeError(
                        f"anchor {position} outside length {hidden.shape[1]} at layer {layer_idx}"
                    )
                if stored.shape[-1] != hidden.shape[-1]:
                    raise RuntimeError(
                        f"layer {layer_idx} width {stored.shape[-1]} != {hidden.shape[-1]}"
                    )
                updated = hidden.clone()
                updated[:, position, :] = stored.to(device=updated.device, dtype=updated.dtype)
                return (updated,) + output[1:]

            self.handles.append(self._layer(layer_idx).register_forward_hook(hook))
        return self

    def rewrite_same_values(self, layers: list[int], position: int) -> "ResidualEditor":
        """Clone the residual and write the original question-mark vector back."""
        self.close()
        if not layers:
            raise RuntimeError("no-op layer list is empty")
        for layer_idx in layers:
            def hook(module, inputs, output, layer_idx=layer_idx):
                hidden = _layer_hidden(output)
                if position < 0 or position >= hidden.shape[1]:
                    raise RuntimeError(
                        f"anchor {position} outside length {hidden.shape[1]} at layer {layer_idx}"
                    )
                updated = hidden.clone()
                updated[:, position, :] = hidden[:, position, :]
                return (updated,) + output[1:]

            self.handles.append(self._layer(layer_idx).register_forward_hook(hook))
        return self

    def watch_input(self, layer_idx: int, box: dict) -> "ResidualEditor":
        """Store the input residual of a decoder layer. Does not replace it."""
        def pre(module, args):
            if not args:
                raise RuntimeError(f"layer {layer_idx} received no positional input")
            box["hidden"] = args[0].detach().clone()
            return None

        self.handles.append(self._layer(layer_idx).register_forward_pre_hook(pre))
        return self


def _layer_hidden(output) -> torch.Tensor:
    if not isinstance(output, tuple) or not output:
        raise RuntimeError("decoder layer output is not a non-empty tuple")
    hidden = output[0]
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise RuntimeError("decoder layer output[0] is not a [batch, seq, hidden] tensor")
    return hidden

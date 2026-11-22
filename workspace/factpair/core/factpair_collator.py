from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

IGNORE_INDEX = -100


class FactPairCollator:
    """Collate paired X+/X- prompts, retain prompts and span metadata."""

    def __init__(self, padding_side: str = "right") -> None:
        if padding_side not in {"left", "right"}:
            raise ValueError(f"Unsupported padding_side: {padding_side}")
        self.padding_side = padding_side

    def _pad(self, values: Sequence[torch.Tensor], pad_value: int) -> torch.Tensor:
        if self.padding_side == "right":
            return torch.nn.utils.rnn.pad_sequence(
                list(values), batch_first=True, padding_value=pad_value
            )
        reversed_values = [torch.flip(value, dims=[0]) for value in values]
        return torch.nn.utils.rnn.pad_sequence(
            reversed_values, batch_first=True, padding_value=pad_value
        ).flip(dims=[1])

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not instances:
            return {}
        output: Dict[str, Any] = {}
        for key in instances[0]:
            values = [instance[key] for instance in instances]
            if isinstance(values[0], dict):
                output[key] = self(values)
                continue
            if isinstance(values[0], str):
                output[key] = values
                continue
            if isinstance(values[0], torch.Tensor):
                tensor_values: Sequence[torch.Tensor] = values  # type: ignore[assignment]
                if all(value.ndim == 1 for value in tensor_values):
                    pad_value = IGNORE_INDEX if key == "labels" else 0
                    output[key] = self._pad(tensor_values, pad_value)
                else:
                    output[key] = torch.stack(list(tensor_values), dim=0)
                continue
            if isinstance(values[0], (int, float, bool)):
                output[key] = torch.tensor(values)
            else:
                output[key] = values
        return output

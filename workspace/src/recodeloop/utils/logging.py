"""Structured JSONL logging confined to workspace/outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..types import as_plain


class JsonlLogger:
    def __init__(self, path: Path | str, max_string: int = 1000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_string = max_string

    def _trim(self, value: Any) -> Any:
        if isinstance(value, str) and len(value) > self.max_string:
            return value[: self.max_string] + "..."
        if isinstance(value, dict):
            return {str(k): self._trim(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._trim(v) for v in value]
        return value

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **self._trim(as_plain(fields)),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def environment_record(config_hash: str, code_hash: str, seed: SeedRecord,
                       workspace_root: Path | str) -> Any:
    """Collect versions/hashes without network access or telemetry."""
    import platform
    import sys

    import torch
    from ..types import EnvironmentRecord

    try:
        import transformers
        tv = transformers.__version__
    except ImportError:  # pragma: no cover - optional in minimal envs
        tv = None

    gpu_names = tuple(
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    ) if torch.cuda.is_available() else ()

    return EnvironmentRecord(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        transformers_version=tv,
        cuda_available=torch.cuda.is_available(),
        gpu_names=gpu_names,
        code_hash=code_hash,
        config_hash=config_hash,
        seed=seed,
    )

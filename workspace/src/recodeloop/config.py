"""Frozen configuration loader with stable hashing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

CONFIG_FILES = (
    "m0_p08_m2.yaml",
    "readers.yaml",
    "localization.yaml",
    "editing.yaml",
    "audits.yaml",
)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class FrozenConfig:
    workspace_root: Path
    data: Mapping[str, Mapping[str, Any]]
    per_file_hash: Mapping[str, str]
    config_hash: str

    def section(self, name: str) -> Mapping[str, Any]:
        try:
            return self.data[name]
        except KeyError as exc:
            raise KeyError(f"unknown config section: {name!r}") from exc

    def resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.workspace_root / p


def load_config(workspace_root: Path | str | None = None) -> FrozenConfig:
    root = (
        Path(workspace_root).resolve()
        if workspace_root
        else Path(__file__).resolve().parents[2]
    )
    config_dir = root / "configs"
    data: dict[str, Mapping[str, Any]] = {}
    hashes: dict[str, str] = {}
    import hashlib

    for filename in CONFIG_FILES:
        path = config_dir / filename
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: expected mapping")
        key = path.stem
        data[key] = loaded
        hashes[filename] = hashlib.sha256(canonical_json(loaded).encode("utf-8")).hexdigest()

    combined = hashlib.sha256(
        "".join(f"{name}:{hashes[name]}\n" for name in CONFIG_FILES).encode("utf-8")
    ).hexdigest()
    return FrozenConfig(root, data, hashes, combined)

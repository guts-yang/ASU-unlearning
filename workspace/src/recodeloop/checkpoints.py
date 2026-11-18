"""Checkpoint manifest loading, G0 guard and save/hash utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import FrozenConfig
from .errors import AssetBlocked
from .g0 import G0Report, validate_assets
from .types import CheckpointRef, Role, ROLES
from .utils.hashing import sha256_file, sha256_state_dict


def require_g0(cfg: FrozenConfig) -> G0Report:
    """All scientific entry points must call this before touching assets."""
    report = validate_assets(cfg.workspace_root)
    if report.status != "PASS":
        raise AssetBlocked(
            f"G0 status={report.status}; scientific evaluation is not permitted. "
            + "; ".join(report.blockers or report.warnings)
        )
    return report


def checkpoint_refs(cfg: FrozenConfig) -> dict[Role, CheckpointRef]:
    """Build CheckpointRefs only after G0 passes. Never infer a path."""
    require_g0(cfg)
    path = cfg.resolve("manifests/assets.yaml")
    with path.open("r", encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    entries = manifest["checkpoints"]
    return {
        role: CheckpointRef(
            role=role,
            path=entries[role]["path"],
            revision=entries[role].get("revision"),
            tokenizer_path=entries[role]["tokenizer_path"],
            sha256=entries[role]["sha256"],
        )
        for role in ROLES
    }


def verify_file_hash(path: Path | str, expected: str) -> bool:
    return sha256_file(path).lower() == expected.strip().lower()


def save_model_with_hash(model: Any, output_dir: Path | str) -> str:
    """Save HF-style model and return SHA-256 of its in-memory state dict."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    digest = sha256_state_dict(model)
    (out / "recodeloop_state_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    return digest


def load_manifest_splits(cfg: FrozenConfig) -> Mapping[str, Any]:
    path = cfg.resolve("manifests/entity_splits.yaml")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

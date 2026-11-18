"""G0 asset validator.

G0 is a hard gate: without a unique P08-M2 mapping to base / target / asu_student
checkpoints, tokenizer and SHA-256, no scientific evaluation may run.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

ROLES = ("base", "target", "asu_student")
SPLITS = ("forget_train_edit", "forget_train_reader", "calibration", "confirmation")
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")

Severity = Literal["blocked", "inconclusive"]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    severity: Severity = "blocked"


@dataclass(frozen=True)
class G0Report:
    experiment_id: str
    status: Literal["PASS", "INCONCLUSIVE", "BLOCKED"]
    checks: tuple[Check, ...]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["checks"] = [asdict(c) for c in self.checks]
        data["blockers"] = list(self.blockers)
        data["warnings"] = list(self.warnings)
        return data


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def _resolve(workspace_root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else workspace_root / p


def validate_assets(
    workspace_root: Path | str,
    config_path: Path | str | None = None,
    assets_path: Path | str | None = None,
    splits_path: Path | str | None = None,
) -> G0Report:
    """Validate frozen assets. Purely reads files; never downloads or guesses."""
    root = Path(workspace_root).resolve()
    config_path = Path(config_path) if config_path else root / "configs" / "m0_p08_m2.yaml"
    checks: list[Check] = []
    blockers: list[str] = []
    warnings: list[str] = []

    def add(name: str, ok: bool, detail: str, severity: Severity = "blocked") -> None:
        checks.append(Check(name=name, ok=ok, detail=detail, severity=severity))
        if not ok:
            (warnings if severity == "inconclusive" else blockers).append(f"{name}: {detail}")

    cfg: dict[str, Any] = {}
    try:
        cfg = _read_yaml(Path(config_path))
        add("config.file", True, f"loaded {Path(config_path).name}")
    except Exception as exc:  # noqa: BLE001 - gate must surface any parse failure
        add("config.file", False, f"cannot load {config_path}: {exc}")

    experiment_id = str(cfg.get("experiment_id", "P08-M2"))
    paths_cfg = cfg.get("paths") or {}
    assets_path = (
        Path(assets_path)
        if assets_path
        else _resolve(root, str(paths_cfg.get("assets_manifest", "manifests/assets.yaml")))
    )
    splits_path = (
        Path(splits_path)
        if splits_path
        else _resolve(root, str(paths_cfg.get("entity_splits", "manifests/entity_splits.yaml")))
    )

    assets: dict[str, Any] = {}
    try:
        assets = _read_yaml(Path(assets_path))
        add("assets.file", True, f"loaded {Path(assets_path).name}")
        add("assets.experiment_id", str(assets.get("experiment_id")) == experiment_id,
            f"manifest={assets.get('experiment_id')!r}, config={experiment_id!r}")
    except Exception as exc:  # noqa: BLE001
        add("assets.file", False, f"cannot load {assets_path}: {exc}")

    checkpoints = assets.get("checkpoints") or {}
    hashes: dict[str, str] = {}
    paths_by_role: dict[str, str] = {}
    for role in ROLES:
        entry = checkpoints.get(role) or {}
        missing: list[str] = []
        for field in ("path", "tokenizer_path", "sha256"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                missing.append(f"checkpoints.{role}.{field}")
        sha = entry.get("sha256")
        if isinstance(sha, str) and sha.strip() and not _HEX64.fullmatch(sha.strip()):
            missing.append(f"checkpoints.{role}.sha256 (not 64-hex)")
        add(f"assets.{role}", not missing,
            "complete" if not missing else "missing/invalid: " + ", ".join(missing))
        if isinstance(sha, str) and _HEX64.fullmatch(sha.strip()):
            hashes[role] = sha.strip().lower()
        if isinstance(entry.get("path"), str) and entry["path"].strip():
            paths_by_role[role] = entry["path"].strip()

    all_roles_ok = all(entry.get("role") == role for role, entry in
                       ((r, checkpoints.get(r) or {}) for r in ROLES))
    add("assets.roles", all_roles_ok, "role fields match manifest keys")

    dup_hashes = sorted({h for h in hashes.values() if list(hashes.values()).count(h) > 1})
    add("assets.unique_hashes", len(hashes) == len(ROLES) and not dup_hashes,
        f"{len(hashes)}/{len(ROLES)} valid hashes"
        + (f"; duplicates: {dup_hashes}" if dup_hashes else ""))

    dup_paths = sorted({p for p in paths_by_role.values()
                        if list(paths_by_role.values()).count(p) > 1})
    add("assets.unique_paths", len(paths_by_role) == len(ROLES) and not dup_paths,
        f"{len(paths_by_role)}/{len(ROLES)} paths"
        + (f"; duplicates: {dup_paths}" if dup_paths else ""))

    splits: dict[str, Any] = {}
    try:
        splits = _read_yaml(Path(splits_path))
        add("splits.file", True, f"loaded {Path(splits_path).name}")
        add("splits.experiment_id", str(splits.get("experiment_id")) == experiment_id,
            f"manifest={splits.get('experiment_id')!r}, config={experiment_id!r}")
    except Exception as exc:  # noqa: BLE001
        add("splits.file", False, f"cannot load {splits_path}: {exc}")

    clusters = splits.get("clusters")
    add("splits.clusters_present", isinstance(clusters, list) and len(clusters) > 0,
        f"clusters={len(clusters) if isinstance(clusters, list) else type(clusters).__name__}")

    owner_by_alias: dict[str, str] = {}
    collisions: list[str] = []
    confirmation_count = 0
    split_counter: dict[str, int] = {}
    if isinstance(clusters, list):
        for idx, cluster in enumerate(clusters):
            if not isinstance(cluster, dict):
                collisions.append(f"clusters[{idx}]: not a mapping")
                continue
            cid = str(cluster.get("canonical_id", "")).strip()
            split = str(cluster.get("split", "")).strip()
            split_counter[split] = split_counter.get(split, 0) + 1
            if split == "confirmation":
                confirmation_count += 1
            if split not in SPLITS:
                collisions.append(f"clusters[{idx}]: invalid split {split!r}")
            if not cid:
                collisions.append(f"clusters[{idx}]: empty canonical_id")
            for key in (cid, *(str(a).strip() for a in (cluster.get("aliases") or ()))):
                if not key:
                    continue
                previous = owner_by_alias.get(key)
                if previous is not None and previous != cid:
                    collisions.append(f"{key!r} claimed by {previous!r} and {cid!r}")
                owner_by_alias[key] = cid
    add("splits.entity_isolation", not collisions,
        "no canonical-id/alias collision" if not collisions else "; ".join(collisions[:5]))

    expected = (cfg.get("gates") or {}).get("confirmation_min_clusters", 4)
    add("splits.confirmation_count", confirmation_count >= int(expected),
        f"confirmation clusters={confirmation_count}, required>={expected}",
        severity="inconclusive")

    if blockers:
        status: Literal["PASS", "INCONCLUSIVE", "BLOCKED"] = "BLOCKED"
    elif warnings:
        status = "INCONCLUSIVE"
    else:
        status = "PASS"

    return G0Report(
        experiment_id=experiment_id,
        status=status,
        checks=tuple(checks),
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


EXIT_CODES = {"PASS": 0, "INCONCLUSIVE": 3, "BLOCKED": 2}

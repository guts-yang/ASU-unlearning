"""Deterministic SHA-256 helpers for assets, tensors and models."""

from __future__ import annotations

import hashlib
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import torch

from ..config import canonical_json


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


BUNDLE_EXCLUDE_GLOBS = ("*.lock", "*.metadata", "pytorch_model*.bin")
BUNDLE_EXCLUDE_NAMES = frozenset({"pytorch_model.bin.index.json"})


def sha256_bundle(path: Path | str) -> str:
    """Canonical hash of a sharded HF checkpoint directory.

    Excludes download caches/locks, duplicate legacy ``*.bin`` weights and the
    corresponding ``pytorch_model.bin.index.json``; safetensors/index/config/
    tokenizer and other metadata are included by relative path. The bundle hash
    is sha256 of sorted [[relpath, file_sha256], ...] JSON.
    """
    root = Path(path).resolve()
    if not root.is_dir():
        return sha256_file(root)
    entries = []
    for file in sorted(root.rglob("*")):
        if not file.is_file():
            continue
        rel = file.relative_to(root).as_posix()
        if rel.startswith(".cache/") or file.name in BUNDLE_EXCLUDE_NAMES:
            continue
        if any(fnmatch(rel, pattern) for pattern in BUNDLE_EXCLUDE_GLOBS):
            continue
        entries.append([rel, sha256_file(file)])
    return sha256_json(entries)


def _numpy_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().to("cpu").contiguous()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.float()
    return bytes(cpu.numpy().view("u1"))


def sha256_tensor(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to("cpu").contiguous()
    header = f"shape={tuple(cpu.shape)};dtype={cpu.dtype};".encode("ascii")
    return sha256_bytes(header + _numpy_bytes(tensor))


def sha256_state_dict(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(f":{tuple(tensor.shape)}:{tensor.dtype}:".encode("ascii"))
        digest.update(_numpy_bytes(tensor))
    return digest.hexdigest()

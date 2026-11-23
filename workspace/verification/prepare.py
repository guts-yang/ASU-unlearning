"""Fixed verification environment (read-only during the verification loop).

Equivalent to autoresearch's prepare.py: data/model paths, metric extraction,
git identity and VRAM sampling. Do not edit while the loop is running.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

REPO = Path("/usr/local/workspace/ASU-unlearning-Auto")
AUTO = REPO / "workspace" / "auto_experiments"
VERIFY = REPO / "workspace" / "verification"
DATA_DIR = Path("/jizhicfs/chenyliao/data/datasets/locuslab-TOFU")
MODEL = "/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
GA_CKPT = "/jizhicfs/chenyliao/runs/open-unlearning/20261122_forget10_1b/GradAscent"
BASE_CKPT = AUTO / "checkpoints" / "tau_global_s0"
CANONICAL_V4 = REPO / "workspace" / "idea-stage" / "IDEA_REPORT_V4.md"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dF(d):
    return d["after"]["forget"] - d["before"]["forget"]


def dR(d):
    return d["after"]["retain"] - d["before"]["retain"]


def git_short():
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
    ).strip()


def gpu_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.splitlines()[0].strip())
    except Exception:
        return 0.0


def preflight():
    issues = []
    if not torch.cuda.is_available():
        issues.append("CUDA unavailable")
    if not Path(MODEL).exists():
        issues.append(f"model missing: {MODEL}")
    for f in ("forget10.json", "retain90.json", "holdout10.json"):
        if not (DATA_DIR / f).exists():
            issues.append(f"data missing: {f}")
    if not (BASE_CKPT / "config.json").exists():
        issues.append(f"baseline checkpoint missing: {BASE_CKPT}")
    if not (Path(GA_CKPT) / "model.safetensors").exists():
        issues.append(f"GA checkpoint missing: {GA_CKPT}")
    return issues

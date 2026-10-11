"""Shared paths and statistics for the locked direction-G M0 protocol."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/usr/local/ASU-unlearning/workspace/G/M0_results")
DATA = ROOT / "data"
GEN = ROOT / "generations"
JUDGED = ROOT / "judged"
CALIB = ROOT / "calibration"
EXP_B = ROOT / "experiment_b"
TOFU = Path("/usr/local/ASU-unlearning/Right-to-be-forgotten/data/tofu")
TOKENIZER_ID = "NousResearch/Llama-2-7b-chat-hf"
BASE_MODEL = "locuslab/tofu_ft_llama2-7b"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 1001
IG_STEPS = 32
LAYER_INDEX = 16
DOWNGRADE_LAYER = 24
TAU = 2.3


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def prompt_body(question: str, mode: str) -> str:
    if mode == "DirectQA":
        body = question
        lowered = body.lower()
        if "think" in lowered or "step by step" in lowered:
            raise ValueError("DirectQA body contains a forbidden phrase")
        return body
    if mode == "DefaultThink":
        return question + "\nLet's think step by step."
    if mode == "ZeroThink":
        return question + "\nReasoning budget is 0. Do not write a chain of thought. Answer now."
    raise ValueError(mode)


def chat_prompt(question: str, mode: str) -> str:
    return f"[INST] {prompt_body(question, mode)} [/INST]"


def max_new_tokens(mode: str) -> int:
    if mode == "DefaultThink":
        return 512
    if mode in {"DirectQA", "ZeroThink"}:
        return 64
    raise ValueError(mode)


def bootstrap_mean_ci(values: np.ndarray, seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(values.mean())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size))
    means = values[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return point, float(low), float(high)


def find_checkpoint(loss_name: str) -> Path:
    root = ROOT / "checkpoints"
    matches = sorted(root.glob(f"**/{loss_name}/**/checkpoint-last"))
    ready = [path for path in matches if (path / "config.json").exists()]
    if not ready:
        raise FileNotFoundError(f"no checkpoint-last for {loss_name} under {root}")
    return ready[-1]

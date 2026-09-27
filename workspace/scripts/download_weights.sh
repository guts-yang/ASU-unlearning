#!/usr/bin/env bash
# Download the three missing ASU target weights into the data-disk Hugging Face cache.
# Safetensors and tokenizer files only. Does not copy weights into this workspace.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${ROOT}/workspace"
# shellcheck disable=SC1091
source /root/autodl-tmp/env_hf.sh

export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${WORKSPACE}/logs" /root/autodl-tmp/ASU-unlearning/checkpoints

export PYTHONUNBUFFERED=1
PY="${PYTHON:-/root/autodl-tmp/envs/unlearning/bin/python}"
LOG="${WORKSPACE}/logs/download_weights.log"

echo "HF_HOME=${HF_HOME}" | tee "${LOG}"
"${PY}" - "${WORKSPACE}" "$@" <<'PY' | tee -a "${LOG}"
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError

workspace = Path(sys.argv[1])
manifest_only = "--manifest-only" in sys.argv[2:]

REQUIRED = [
    "locuslab/tofu_ft_llama2-7b",
    "NousResearch/Llama-2-7b-chat-hf",
]
OPTIONAL = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
]
ALREADY = [
    "HuggingFaceH4/zephyr-7b-beta",
    "muse-bench/MUSE-News_target",
    "muse-bench/MUSE-Books_target",
    "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
    "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
    "madhurjindal/autonlp-Gibberish-Detector-492513457",
    "NousResearch/Llama-2-7b-hf",
]
ALLOW = ["*.safetensors", "*.json", "tokenizer*", "*.model"]


def download(repo_id):
    path = snapshot_download(repo_id=repo_id, allow_patterns=ALLOW)
    print(f"OK\t{repo_id}\t{path}", flush=True)
    return path


failures = []
if not manifest_only:
    for repo_id in REQUIRED:
        try:
            download(repo_id)
        except (GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError, OSError) as exc:
            print(f"FAIL\t{repo_id}\t{type(exc).__name__}: {exc}", flush=True)
            failures.append(repo_id)
    for repo_id in OPTIONAL:
        try:
            download(repo_id)
        except (GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError, OSError) as exc:
            print(f"FAIL\t{repo_id}\t{type(exc).__name__}: {exc}", flush=True)
            failures.append(repo_id)

hub = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/root/autodl-tmp/huggingface/hub"))


def snapshot_path(repo_id):
    folder = hub / ("models--" + repo_id.replace("/", "--"))
    snaps = folder / "snapshots"
    if not snaps.is_dir():
        return None
    candidates = [p for p in snaps.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


lines = ["repo_id\tstatus\tsnapshot_path"]
for repo_id in ALREADY + REQUIRED + OPTIONAL:
    path = snapshot_path(repo_id)
    weight_files = []
    if path is not None:
        weight_files = [p for p in path.glob("*.safetensors") if p.exists() and p.stat().st_size > 0]
        index = path / "model.safetensors.index.json"
        if index.exists():
            weight_files.append(index)
    if repo_id in failures:
        status = "failed"
    elif weight_files:
        status = "local"
    elif path is not None:
        status = "tokenizer_only"
    else:
        status = "missing"
    lines.append(f"{repo_id}\t{status}\t{path if path else '-'}")

manifest = workspace / "manifests" / "weights.txt"
manifest.write_text("\n".join(lines) + "\n")
print(f"WROTE\t{manifest}", flush=True)

if any(repo in failures for repo in REQUIRED):
    sys.exit(1)
PY

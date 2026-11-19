#!/usr/bin/env bash
# Shared environment for ASU-unlearning-APRU training on this GPU node.
set -euo pipefail

source /jizhicfs/chenyliao/ops/00_env.sh

export REPO=/usr/local/ASU-unlearning-APRU
export OU_ROOT="$REPO/open-unlearning"
export WS="$REPO/workspace"
export VENV=/jizhicfs/chenyliao/ops/venvs/asu-apru
export PY="$VENV/bin/python"
export PYTHONPATH="$OU_ROOT/src:$WS/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_HOME=/jizhicfs/chenyliao/ops/cuda-stub
export DS_SKIP_CUDA_CHECK=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/jizhicfs/chenyliao/cache/triton}"
mkdir -p "$TRITON_CACHE_DIR"
export HYDRA_SEARCHPATH="file://$WS/configs"
export OUTPUT_ROOT=/jizhicfs/chenyliao/runs/asu-apru

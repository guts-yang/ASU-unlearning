#!/usr/bin/env bash
# Shared environment for QKSpec training on the GPU node.
set -euo pipefail

source /jizhicfs/chenyliao/ops/00_env.sh

REPO=/usr/local/workspace/ASU-unlearning-K4
export REPO
export OU_ROOT="$REPO/open-unlearning"
export WS="$REPO/workspace"
export VENV=/jizhicfs/chenyliao/ops/venvs/asu-apru
export PY="$VENV/bin/python"
export PYTHONPATH="$OU_ROOT/src:$WS/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
# Model/tokenizer are local; eval metrics still need locuslab/TOFU splits,
# which we allow to download once (dataset load is local-path for D_F/D_R).
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=0
export CUDA_HOME=/jizhicfs/chenyliao/ops/cuda-stub
export DS_SKIP_CUDA_CHECK=1
export HYDRA_SEARCHPATH="file://$WS/configs"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/jizhicfs/chenyliao/runs/qkspec}"

# Local assets
export MODEL_DIR=/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full
export TOKENIZER_DIR=$MODEL_DIR
export TOFU_DIR=/jizhicfs/chenyliao/data/datasets/locuslab-TOFU
export EVAL_ROOT=/jizhicfs/chenyliao/data/datasets/open-unlearning/eval

mkdir -p "$OUTPUT_ROOT/unlearn" "$OUTPUT_ROOT/logs"

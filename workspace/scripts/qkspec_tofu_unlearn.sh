#!/bin/bash
# QKSpec TOFU unlearning: one-shot gradient-free QK spectral edit, then TOFU eval.
# Usage: bash scripts/qkspec_tofu_unlearn.sh [MODEL] [FORGET] [RETAIN] [HOLDOUT]
#   e.g. bash scripts/qkspec_tofu_unlearn.sh Llama-3.2-1B-Instruct forget10 retain90 holdout10

set -e

MODEL=${1:-Llama-3.2-1B-Instruct}
FORGET=${2:-forget10}
RETAIN=${3:-retain90}
HOLDOUT=${4:-holdout10}
MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TASK_NAME="tofu_${MODEL}_${FORGET}_QKSpec"

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Task: ${TASK_NAME}; model: ${MODEL_PATH}; master port: ${MASTER_PORT}"

# Unlearn (single-shot edit; no gradient). Eager attention is set in the
# experiment config and must not be overridden to flash/sdpa.
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 \
  --config_file configs/accelerate/default_config.yaml \
  --main_process_port $MASTER_PORT \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/qkspec.yaml \
  trainer=QKSpec \
  task_name=${TASK_NAME} \
  forget_split=${FORGET} \
  holdout_split=${HOLDOUT} \
  retain_split=${RETAIN} \
  model=${MODEL} \
  model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
  retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN}/TOFU_EVAL.json

# Independent evaluation of the saved student model.
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  experiment=eval/tofu/default.yaml \
  forget_split=${FORGET} \
  holdout_split=${HOLDOUT} \
  model=${MODEL} \
  task_name=${TASK_NAME} \
  model.model_args.pretrained_model_name_or_path=saves/unlearn/${TASK_NAME} \
  paths.output_dir=saves/unlearn/${TASK_NAME}/evals \
  retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN}/TOFU_EVAL.json

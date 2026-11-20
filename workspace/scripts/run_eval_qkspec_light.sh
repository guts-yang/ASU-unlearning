#!/usr/bin/env bash
# Lightweight TOFU eval of a saved QKSpec student (no ROUGE metrics).
# Usage: bash workspace/scripts/run_eval_qkspec_light.sh <STUDENT_DIR> [MODEL_TAG] [FORGET]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_qkspec.sh"

STUDENT=${1:?usage: run_eval_qkspec_light.sh STUDENT_DIR [MODEL_TAG] [FORGET]}
MODEL_TAG=${2:-Llama-3.2-1B-Instruct}
FORGET=${3:-forget10}
HOLDOUT="holdout${FORGET#forget}"
RETAIN=${4:-retain90}
RETAIN_LOG="$EVAL_ROOT/tofu_${MODEL_TAG}_${RETAIN}/TOFU_EVAL.json"

cd "$OU_ROOT"
"$PY" src/eval.py --config-name=eval.yaml \
  "hydra.searchpath=[file://$WS/configs]" \
  eval=tofu_qkspec_light \
  eval.tofu.forget_split="$FORGET" \
  eval.tofu.holdout_split="$HOLDOUT" \
  eval.tofu.retain_logs_path="$RETAIN_LOG" \
  eval.tofu.output_dir="$STUDENT/eval_light" \
  model.model_args.pretrained_model_name_or_path="$STUDENT" \
  model.tokenizer_args.pretrained_model_name_or_path="$STUDENT" \
  model.model_args.attn_implementation=eager \
  paths.output_dir="$STUDENT/eval_light" \
  seed=1001 2>&1 | tee "$STUDENT/eval_light.log"

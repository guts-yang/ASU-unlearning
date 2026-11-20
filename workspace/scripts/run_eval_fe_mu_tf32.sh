#!/usr/bin/env bash
# FE/MU eval with fp32+TF32 (fast, no bf16 SIGFPE).
# Usage: bash workspace/scripts/run_eval_fe_mu_tf32.sh <STUDENT_DIR> [MODEL_TAG] [FORGET] [RETAIN]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_qkspec.sh"

STUDENT=${1:?usage: run_eval_fe_mu_tf32.sh STUDENT_DIR [MODEL_TAG] [FORGET] [RETAIN]}
MODEL_TAG=${2:-Llama-3.2-1B-Instruct}
FORGET=${3:-forget10}
HOLDOUT="holdout${FORGET#forget}"
RETAIN=${4:-retain90}
RETAIN_LOG="$EVAL_ROOT/tofu_${MODEL_TAG}_${RETAIN}/TOFU_EVAL.json"

cd "$REPO"
"$PY" workspace/scripts/eval_fe_mu_tf32.py \
  "hydra.searchpath=[file://$OU_ROOT/configs]" \
  model.model_args.pretrained_model_name_or_path="$STUDENT" \
  model.tokenizer_args.pretrained_model_name_or_path="$STUDENT" \
  eval.tofu.forget_split="$FORGET" \
  eval.tofu.holdout_split="$HOLDOUT" \
  eval.tofu.retain_logs_path="$RETAIN_LOG" \
  eval.tofu.output_dir="$STUDENT/eval_fe_mu" \
  eval.tofu.batch_size=16 \
  paths.output_dir="$STUDENT/eval_fe_mu" \
  2>&1 | tee "$STUDENT/eval_fe_mu_tf32.log"

echo "=== FE/MU (TF32) summary: $(basename "$STUDENT") ==="
cat "$STUDENT/eval_fe_mu/TOFU_SUMMARY.json" 2>/dev/null || echo "no summary"

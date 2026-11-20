#!/usr/bin/env bash
# Run QKSpec unlearning on TOFU forget10 with the GPU venv.
# Usage:
#   bash workspace/scripts/run_train_qkspec.sh [FORGET] [RETAIN] [MODEL_TAG] [EXTRA_OVERRIDES...]
# Example:
#   bash workspace/scripts/run_train_qkspec.sh forget10 retain90 Llama-3.2-1B-Instruct
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env_qkspec.sh"

FORGET=${1:-forget10}
RETAIN=${2:-retain90}
MODEL_TAG=${3:-Llama-3.2-1B-Instruct}
shift 3 2>/dev/null || true
EXTRA=("$@")

MODEL_DIR=/jizhicfs/chenyliao/models/open-unlearning/tofu_${MODEL_TAG}_full
RETAIN_LOG="$EVAL_ROOT/tofu_${MODEL_TAG}_${RETAIN}/TOFU_EVAL.json"
TASK_NAME="qkspec_${MODEL_TAG}_${FORGET}_$(date +%Y%m%d_%H%M%S)"
OUT="$OUTPUT_ROOT/unlearn/$TASK_NAME"
mkdir -p "$OUT"

echo "=== QKSpec train: task=$TASK_NAME ==="
echo "model=$MODEL_DIR  forget=$FORGET  retain=$RETAIN  retain_log=$RETAIN_LOG"

cd "$OU_ROOT"
# run_qkspec.py auto-injects open-unlearning/configs into hydra.searchpath;
# workspace/configs is the primary config dir and needs no explicit searchpath.
"$PY" "$WS/src/run_qkspec.py" --config-name=unlearn.yaml \
  experiment=unlearn/tofu/qkspec.yaml \
  trainer=QKSpec \
  task_name="$TASK_NAME" \
  forget_split="$FORGET" \
  holdout_split="holdout${FORGET#forget}" \
  retain_split="$RETAIN" \
  model.model_args.pretrained_model_name_or_path="$MODEL_DIR" \
  model.tokenizer_args.pretrained_model_name_or_path="$MODEL_DIR" \
  model.model_args.attn_implementation=eager \
  data.forget.TOFU_QA_forget.args.hf_args.path=json \
  data.forget.TOFU_QA_forget.args.hf_args.name=null \
  "+data.forget.TOFU_QA_forget.args.hf_args.data_files=$TOFU_DIR/${FORGET}.json" \
  data.retain.TOFU_QA_retain.args.hf_args.path=json \
  data.retain.TOFU_QA_retain.args.hf_args.name=null \
  "+data.retain.TOFU_QA_retain.args.hf_args.data_files=$TOFU_DIR/${RETAIN}.json" \
  trainer.method_args.qkspec.top_k_directions=8 \
  trainer.method_args.qkspec.ranks_per_head=4 \
  trainer.method_args.qkspec.max_discovery_batches=8 \
  trainer.method_args.qkspec.max_stage_b_batches=8 \
  trainer.method_args.qkspec.discovery_batch_size=4 \
  trainer.method_args.qkspec.run_controls=true \
  trainer.args.seed=1001 \
  retain_logs_path="$RETAIN_LOG" \
  paths.output_dir="$OUT" \
  "${EXTRA[@]}" 2>&1 | tee "$OUT/train.log"

echo "=== QKSpec train done: $OUT ==="
echo "Report: $OUT/qkspec_report.json"
ls -la "$OUT" | head -20

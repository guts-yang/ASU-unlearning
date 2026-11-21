#!/usr/bin/env bash
# GradAscent (GA) unlearning baseline on TOFU forget10 -- control for QKSpec.
# Same model / split / seed as the QKSpec sweep; then RTF FE/MU eval.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_qkspec.sh"

MODEL_TAG=Llama-3.2-1B-Instruct
MODEL_DIR=/jizhicfs/chenyliao/models/open-unlearning/tofu_${MODEL_TAG}_full
RETAIN_LOG="$EVAL_ROOT/tofu_${MODEL_TAG}_retain90/TOFU_EVAL.json"
TASK=ga_${MODEL_TAG}_forget10
OUT=$OUTPUT_ROOT/control/$TASK
mkdir -p "$OUT"

export TOKENIZERS_PARALLELISM=false CUDA_HOME=/jizhicfs/chenyliao/ops/cuda-stub DS_SKIP_CUDA_CHECK=1
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=0

echo "=== GA control: $TASK ==="
cd "$OU_ROOT"
"$PY" "$WS/src/run_ga.py" --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  trainer=GradAscent \
  task_name=$TASK \
  forget_split=forget10 \
  holdout_split=holdout10 \
  retain_split=retain90 \
  model=$MODEL_TAG \
  model.model_args.pretrained_model_name_or_path="$MODEL_DIR" \
  model.tokenizer_args.pretrained_model_name_or_path="$MODEL_DIR" \
  model.model_args.attn_implementation=eager \
  data.forget.TOFU_QA_forget.args.hf_args.path=json \
  data.forget.TOFU_QA_forget.args.hf_args.name=null \
  "++data.forget.TOFU_QA_forget.args.hf_args.data_files=$TOFU_DIR/forget10.json" \
  data.retain.TOFU_QA_retain.args.hf_args.path=json \
  data.retain.TOFU_QA_retain.args.hf_args.name=null \
  "++data.retain.TOFU_QA_retain.args.hf_args.data_files=$TOFU_DIR/retain90.json" \
  trainer.args.seed=1001 \
  trainer.args.num_train_epochs=5 \
  trainer.args.learning_rate=1e-5 \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=1 \
  trainer.args.do_eval=false \
  trainer.args.eval_strategy=no \
  trainer.args.eval_on_start=false \
  retain_logs_path="$RETAIN_LOG" \
  paths.output_dir="$OUT" 2>&1 | tee "$OUT/train.log"

echo "=== GA train done: $OUT ==="
ls "$OUT"/model.safetensors 2>/dev/null && echo "model saved"

# RTF FE/MU eval on the GA student
bash "$WS/scripts/run_rtf_fe_mu.sh" "$OUT" 200 rtf_ga 2>&1 | tail -5
cp "$OUT/rtf_ga/rtf_fe_mu.json" "/jizhicfs/chenyliao/runs/qkspec/rtf_results/ga_control_rtf.json" 2>/dev/null || true
echo "=== GA control done ==="

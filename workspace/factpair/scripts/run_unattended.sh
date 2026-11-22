#!/usr/bin/env bash
set -u
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=/jizhicfs/chenyliao/ops/venvs/redact-r2mu/bin/python
CFG="$BASE/configs/experiment.json"
LOG="$BASE/logs/train_eval.log"
GPU=0
MAX_RETRIES=5

export CUDA_VISIBLE_DEVICES=$GPU
export HF_HOME=/jizhicfs/chenyliao/cache/huggingface
export HF_HUB_CACHE=/jizhicfs/chenyliao/cache/huggingface/hub
export HF_DATASETS_CACHE=/jizhicfs/chenyliao/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/jizhicfs/chenyliao/cache/huggingface/transformers
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/jizhicfs/chenyliao/ops/cuda-stub/bin:$PATH

wait_for_gpu() {
  local free=0
  while [ "$free" -lt 3 ]; do
    local used util
    read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | sed -n "$((GPU + 1))p" | tr -d ' ' | tr ',' ' ')
    if [ "${used:-999999}" -lt 1000 ] && [ "${util:-100}" -lt 10 ]; then
      free=$((free + 1))
    else
      free=0
    fi
    sleep 10
  done
}

mkdir -p "$BASE/logs" "$BASE/runs"
: > "$LOG"
wait_for_gpu

attempt=0
while [ "$attempt" -lt "$MAX_RETRIES" ]; do
  attempt=$((attempt + 1))
  echo "ATTEMPT=$attempt $(date -Is)" >> "$LOG"
  wait_for_gpu
  if "$VENV" "$BASE/scripts/train_eval.py" --config "$CFG" >> "$LOG" 2>&1; then
    "$VENV" "$BASE/scripts/eval_gates.py" >> "$LOG" 2>&1 || true
    echo "PIPELINE_COMPLETE $(date -Is)" >> "$LOG"
    exit 0
  fi
  if [ -f "$BASE/runs/factpair_h20/final_results.json" ]; then
    echo "PIPELINE_COMPLETE_EXISTING $(date -Is)" >> "$LOG"
    exit 0
  fi
  echo "ATTEMPT_FAILED=$attempt $(date -Is)" >> "$LOG"
  sleep 30
done
echo "PIPELINE_FAILED $(date -Is)" >> "$LOG"
exit 1

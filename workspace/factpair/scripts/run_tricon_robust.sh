#!/usr/bin/env bash
set -u
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=/jizhicfs/chenyliao/ops/venvs/redact-r2mu/bin/python
LOG="$BASE/logs/tricon_robust.log"
mkdir -p "$BASE/logs" "$BASE/runs/tricon_robust"
: > "$LOG"
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/jizhicfs/chenyliao/cache/huggingface
export HF_HUB_CACHE=/jizhicfs/chenyliao/cache/huggingface/hub
export HF_DATASETS_CACHE=/jizhicfs/chenyliao/cache/huggingface/datasets
export TRANSFORMERS_CACHE=/jizhicfs/chenyliao/cache/huggingface/transformers
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/jizhicfs/chenyliao/ops/cuda-stub/bin:$PATH
wait_for_gpu() {
  while true; do
    read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d " " | tr "," " ")
    if [ "${used:-999999}" -lt 1000 ] && [ "${util:-100}" -lt 10 ]; then
      return 0
    fi
    sleep 10
  done
}
for cfg in "$BASE"/configs/tricon_robust/*.json; do
  name="$(basename "$cfg" .json)"
  wait_for_gpu
  echo "START $name $(date -Is)" >> "$LOG"
  "$VENV" "$BASE/scripts/train_eval_ideas.py" --config "$cfg" >> "$LOG" 2>&1
  echo "DONE $name $(date -Is)" >> "$LOG"
done
echo "ALL_TRICON_ROBUST_COMPLETE $(date -Is)" >> "$LOG"

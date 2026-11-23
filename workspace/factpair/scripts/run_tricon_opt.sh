#!/usr/bin/env bash
set -u
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV=/jizhicfs/chenyliao/ops/venvs/redact-r2mu/bin/python
LOG="$BASE/logs/tricon_opt.log"
mkdir -p "$BASE/logs" "$BASE/runs/tricon_opt"
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
    echo "WAIT_GPU used=${used:-?}MiB util=${util:-?}% $(date -Is)" >> "$LOG"
    sleep 10
  done
}
for cfg in "$BASE"/configs/tricon_opt/*.json; do
  name="$(basename "$cfg" .json)"
  wait_for_gpu
  echo "START $name $(date -Is)" >> "$LOG"
  if "$VENV" "$BASE/scripts/train_eval_ideas.py" --config "$cfg" >> "$LOG" 2>&1; then
    echo "DONE $name $(date -Is)" >> "$LOG"
  else
    echo "FAILED $name $(date -Is)" >> "$LOG"
  fi
done
echo "ALL_TRICON_OPT_COMPLETE $(date -Is)" >> "$LOG"

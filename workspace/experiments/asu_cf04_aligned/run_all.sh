#!/usr/bin/env bash
# Aligned CF-04 vs official-ASU reproduction. All GPU work pinned to GPU 0.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python}"
export CUDA_VISIBLE_DEVICES=0
cd "$HERE"
mkdir -p results

GPU0_USED="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F',' '$1+0==0 {gsub(/[^0-9]/,"",$2); print $2}')"
if [[ -n "${GPU0_USED:-}" && "$GPU0_USED" -gt 0 ]]; then
  echo "STOP: GPU 0 is occupied (${GPU0_USED} MiB). Do not fall back to GPU 1." >&2
  exit 1
fi

for seed in 1001 1002 1003; do
  for arm in asu cf04; do
    tag="${arm}_${seed}"
    mkdir -p "results/$tag"
    echo "=== TRAIN ${tag} ==="
    "$PY" train_aligned.py --arm "$arm" --seed "$seed" --out-dir "results/$tag" > "results/$tag/run.log" 2>&1
    tail -2 "results/$tag/run.log"
  done
done

echo "=== TRAIN asu_tau2.3_1001 (script-override sensitivity) ==="
mkdir -p results/asu_tau2.3_1001
"$PY" train_aligned.py --arm asu --seed 1001 --tau 2.3 --out-dir results/asu_tau2.3_1001 > results/asu_tau2.3_1001/run.log 2>&1
tail -2 results/asu_tau2.3_1001/run.log

echo "=== EVALUATE (all arms, 3 eval seeds) ==="
"$PY" evaluate_aligned.py > results/evaluate.log 2>&1
tail -5 results/evaluate.log

echo "=== PAIRED DIFFERENCE ANALYSIS ==="
"$PY" analyze.py > results/analyze.log 2>&1
cat results/analyze.log

echo "=== done; GPU state ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

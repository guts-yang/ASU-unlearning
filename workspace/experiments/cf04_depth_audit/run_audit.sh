#!/usr/bin/env bash
# CF-04 v2 depth audit. All GPU work is pinned to GPU 0 (GPU 1 is used by an external job).
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

echo "=== P1/P2/P3/INT8 audit (base, cf04 v2 formal, v1 formal, e1, ga) ==="
"$PY" audit.py --models base,cf04,cf04_formal,cf04_e1,ga --output results/cf04_depth_audit.json > results/audit.log 2>&1
tail -5 results/audit.log

echo "=== 3-seed independent probe audit ==="
"$PY" probe.py --models base,cf04,ga --seeds 0,1,2 --output results/cf04_probe_3seed.json > results/probe.log 2>&1
tail -5 results/probe.log

echo "=== EOS / length-drift diagnostic ==="
"$PY" behavior_check.py --models base,cf04,cf04_e1,ga --output results/cf04_behavior.json > results/behavior.log 2>&1
tail -5 results/behavior.log

echo "=== done; GPU state ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

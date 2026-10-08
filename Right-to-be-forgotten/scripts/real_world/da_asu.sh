#!/bin/bash
# Train one DA-ASU retain term. M0 must already have returned proceed.
# Usage: bash scripts/real_world/da_asu.sh anchor /path/to/gsm8k_train.jsonl
set -euo pipefail

MODE="${1:-anchor}"
REASON="${2:?pass the GSM8K train jsonl}"
MU="${MU:-1.0}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" torchrun --nproc_per_node=2 --master_port="${MASTER_PORT:-$((RANDOM % 50001 + 10000))}" \
    real_world_unlearn.py \
    --config-name=real_world.yaml \
    layers_id=null \
    da_asu.enabled=true \
    da_asu.mode="$MODE" \
    da_asu.mu="$MU" \
    da_asu.reason_path="$REASON"

#!/bin/bash
# Train one DA-ASU retain term. M0 must already have returned proceed.
# Usage: bash scripts/real_world/da_asu.sh anchor /path/to/gsm8k_train.jsonl
set -euo pipefail

MODE="${1:-anchor}"
REASON="${2:?pass the GSM8K train jsonl}"
MU="${MU:-1.0}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

unset OMP_NUM_THREADS
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${DA_GPUS:-0,1}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/huggingface/hub/models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/53346005fb0ef11d3b6a83b12c895cca40156b6c}"
SAVE_ROOT="${SAVE_ROOT:-/root/autodl-tmp/ASU-unlearning/checkpoints/real_world_da_asu}"
# Same stamp as the ASU_KL baseline, with the retain-term mode appended so the two runs do not share a directory.
SAVE_DIR="${SAVE_ROOT}/llama3-8b/forget/ASU+KL/seed_1001/epoch5_5e-06_FixRefTrue_maskTrue_Fcoeff0.1_Rcoeff1.0_Temp2.0_LayersNone_beta0.1_${MODE}"

torchrun --nproc_per_node=2 --master_port="${MASTER_PORT:-$((RANDOM % 50001 + 10000))}" \
    real_world_unlearn.py \
    --config-name=real_world.yaml \
    hydra.job.chdir=false \
    model_path="$MODEL_PATH" \
    forget_loss=ASU+KL \
    attention_temp=2.0 \
    layers_id=null \
    forget_coeff=0.1 \
    num_epochs=5 \
    save_checkpoint=true \
    save_root="$SAVE_ROOT" \
    "save_dir=${SAVE_DIR}" \
    da_asu.enabled=true \
    da_asu.mode="$MODE" \
    da_asu.mu="$MU" \
    da_asu.reason_path="$REASON"

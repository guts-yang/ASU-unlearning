#!/usr/bin/env bash
# Global ASU+GD, forget05, TASK_LIST=1. Seed 1001 already has a finished eval.
# This run trains seeds 1002 and 1003 only.
set -euo pipefail

cd /usr/local/ASU-unlearning/Right-to-be-forgotten

export TASK_LIST=1
export HF_HOME=/root/autodl-tmp/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

SAVE_ROOT=/root/autodl-tmp/ASU-unlearning/workspace_H/global_asu_forget05
LOG_DIR=/usr/local/ASU-unlearning/workspace/H/logs

common=(
  forget.py
  --config-name=tofu.yaml
  split=forget05
  forget_loss=ASU+GD
  attention_temp=2.3
  layers_id=null
  forget_coeff=0.1
  regularization_coeff=1.0
  lr=1e-5
  num_epochs=5
  mask=true
  fix_ref_model=true
  save_steps=last
  save_checkpoint=false
  model_path=/root/autodl-tmp/models/tofu_ft_llama2-7b
  data_path=/usr/local/ASU-unlearning/Right-to-be-forgotten/data/tofu
  save_root="${SAVE_ROOT}"
)

run_seed() {
  local seed="$1"
  local log="${LOG_DIR}/stage1_seed_${seed}.log"
  echo "START seed=${seed} $(date -Iseconds)" | tee -a "${log}"
  torchrun --nproc_per_node=2 --master_port=29621 \
    "${common[@]}" \
    seed="${seed}" \
    2>&1 | tee -a "${log}"
  echo "END seed=${seed} $(date -Iseconds)" | tee -a "${log}"
}

run_seed 1002
run_seed 1003

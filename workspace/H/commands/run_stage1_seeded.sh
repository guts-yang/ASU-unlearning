#!/usr/bin/env bash
# Retrain after dataset sampling uses cfg.seed + sample index.
# The earlier 1002/1003 pair stays in global_asu_forget05 and is not overwritten.
set -euo pipefail

cd /usr/local/ASU-unlearning/Right-to-be-forgotten

export TASK_LIST=1
export HF_HOME=/root/autodl-tmp/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

SAVE_ROOT=/root/autodl-tmp/ASU-unlearning/workspace_H/global_asu_forget05_seeded
LOG_DIR=/usr/local/ASU-unlearning/workspace/H/logs
mkdir -p "${SAVE_ROOT}" "${LOG_DIR}"

train_seed() {
  local seed="$1"
  local log="${LOG_DIR}/stage1_seeded_train_${seed}.log"
  echo "START seeded train seed=${seed} $(date -Iseconds)" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29641 \
    forget.py --config-name=tofu.yaml \
    split=forget05 forget_loss=ASU+GD attention_temp=2.3 layers_id=null \
    forget_coeff=0.1 regularization_coeff=1.0 lr=1e-5 num_epochs=5 \
    mask=true fix_ref_model=true save_steps=last save_checkpoint=false \
    model_path=/root/autodl-tmp/models/tofu_ft_llama2-7b \
    data_path=/usr/local/ASU-unlearning/Right-to-be-forgotten/data/tofu \
    save_root="${SAVE_ROOT}" \
    seed="${seed}" \
    2>&1 | tee -a "${log}"
  echo "END seeded train seed=${seed} $(date -Iseconds)" | tee -a "${log}"
}

eval_seed() {
  local seed="$1"
  local log="${LOG_DIR}/stage1_seeded_eval_${seed}.log"
  echo "START seeded eval seed=${seed} $(date -Iseconds)" | tee -a "${log}"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29651 \
    eval.py --config-name=tofu.yaml \
    split=forget05 forget_loss=ASU+GD attention_temp=2.3 layers_id=null \
    forget_coeff=0.1 regularization_coeff=1.0 lr=1e-5 num_epochs=5 \
    mask=true fix_ref_model=true save_steps=last save_checkpoint=true \
    eval_unlearn_step=last \
    model_path=/root/autodl-tmp/models/tofu_ft_llama2-7b \
    data_path=/usr/local/ASU-unlearning/Right-to-be-forgotten/data/tofu \
    save_root="${SAVE_ROOT}" \
    seed="${seed}" \
    2>&1 | tee -a "${log}"
  echo "END seeded eval seed=${seed} $(date -Iseconds)" | tee -a "${log}"
}

train_seed 1002
train_seed 1003
eval_seed 1002
eval_seed 1003

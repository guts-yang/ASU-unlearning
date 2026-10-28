#!/usr/bin/env bash
# Evaluate the stage-1 checkpoints. save_checkpoint=true keeps the weights:
# eval.py deletes checkpoint-last when save_checkpoint is false.
set -euo pipefail

cd /usr/local/ASU-unlearning/Right-to-be-forgotten

export TASK_LIST=1
export HF_HOME=/root/autodl-tmp/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

SAVE_ROOT=/root/autodl-tmp/ASU-unlearning/workspace_H/global_asu_forget05
LOG_DIR=/usr/local/ASU-unlearning/workspace/H/logs

common=(
  eval.py
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
  save_checkpoint=true
  eval_unlearn_step=last
  model_path=/root/autodl-tmp/models/tofu_ft_llama2-7b
  data_path=/usr/local/ASU-unlearning/Right-to-be-forgotten/data/tofu
  save_root="${SAVE_ROOT}"
)

run_seed() {
  local seed="$1"
  local log="${LOG_DIR}/stage1_eval_seed_${seed}.log"
  echo "START eval seed=${seed} $(date -Iseconds)" | tee -a "${log}"
  torchrun --nproc_per_node=1 --master_port=29631 \
    "${common[@]}" \
    seed="${seed}" \
    2>&1 | tee -a "${log}"
  echo "END eval seed=${seed} $(date -Iseconds)" | tee -a "${log}"
}

run_seed 1002
run_seed 1003

#!/usr/bin/env bash
# Direction G M0 training launch. Hyperparameters are the locked protocol
# (workspace/G/M0_results/protocol_snapshot.json). This script only calls
# Right-to-be-forgotten/forget.py. It does not call eval.py.
set -euo pipefail

source /root/autodl-tmp/env_hf.sh
cd /usr/local/ASU-unlearning/Right-to-be-forgotten

export TASK_LIST=1
export PYTHONUNBUFFERED=1

SAVE_ROOT=/usr/local/ASU-unlearning/workspace/G/M0_results/checkpoints
LOGDIR=/usr/local/ASU-unlearning/workspace/G/M0_results/logs
mkdir -p "$LOGDIR"

date -Iseconds > "$LOGDIR/train_wall_start.txt"

run_one() {
  local tag="$1"
  shift
  local port="$1"
  shift
  echo "START ${tag} $(date -Iseconds)" | tee -a "$LOGDIR/train_events.log"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port="$port" \
    /usr/local/ASU-unlearning/workspace/G/run_forget.py \
    --config-name=tofu.yaml \
    hydra.job.chdir=false \
    task_id=1 \
    save_steps=last \
    use_LoRA=false \
    regularization_coeff=1.0 \
    lr=1e-5 \
    split=forget05 \
    num_epochs=5 \
    mask=true \
    fix_ref_model=true \
    save_root="$SAVE_ROOT" \
    save_checkpoint=true \
    seed=1001 \
    batch_size=4 \
    gradient_accumulation_steps=4 \
    "$@"
  echo "END ${tag} $(date -Iseconds)" | tee -a "$LOGDIR/train_events.log"
}

run_one ASU+GD 29501 \
  'forget_loss=ASU+GD' \
  forget_coeff=0.1 \
  attention_temp=2.3 \
  layers_id=null \
  > "$LOGDIR/asu_gd.log" 2>&1

run_one NPO+GD 29502 \
  'forget_loss=NPO+GD' \
  forget_coeff=1.0 \
  > "$LOGDIR/npo_gd.log" 2>&1

date -Iseconds > "$LOGDIR/train_wall_end.txt"
echo "TRAIN_DONE $(date -Iseconds)" | tee -a "$LOGDIR/train_events.log"

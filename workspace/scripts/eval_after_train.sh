#!/bin/bash
# After scripts/tofu/asu.sh exits, evaluate every checkpoint that never
# produced unlearning_results.csv. Successful evals delete checkpoint-last
# when save_checkpoint=false, so a leftover checkpoint means eval did not finish.
set -u
source /root/autodl-tmp/env_hf.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PATH="/root/autodl-tmp/envs/unlearning/bin:$PATH"
export WANDB_MODE=disabled WANDB_DISABLED=true
export TASK_LIST=1

ROOT=/root/autodl-tmp/ASU-unlearning/checkpoints/tofu/llama2-7b
PARENT_PID=${1:-7865}
cd /usr/local/ASU-unlearning/Right-to-be-forgotten

while kill -0 "$PARENT_PID" 2>/dev/null; do
    sleep 30
done

for _ in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if($1>m)m=$1} END{print m}')
    if [ "${used:-0}" -lt 4000 ]; then
        break
    fi
    sleep 15
done

shopt -s nullglob
for ckpt in "$ROOT"/*/*/seed_1001/*/1/unlearn_times_1/checkpoint-last; do
    run_dir=$(dirname "$ckpt")
    if [ -f "$run_dir/eval_results-last/unlearning_results.csv" ]; then
        continue
    fi
    rel=${ckpt#"$ROOT/"}
    IFS=/ read -r split forget_loss _ <<< "$rel"
    echo "eval split=$split forget_loss=$forget_loss"
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((RANDOM % 50001 + 10000)) \
        eval.py --config-name=tofu.yaml \
        task_id=1 eval_unlearn_step=last \
        use_LoRA=false forget_coeff=0.1 regularization_coeff=1.0 lr=1e-5 \
        split="$split" forget_loss="$forget_loss" num_epochs=5 mask=true fix_ref_model=true \
        save_root=results/tofu save_checkpoint=false attention_temp=2.3 layers_id=null
done

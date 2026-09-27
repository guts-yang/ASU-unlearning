#!/bin/bash
# MUSE ASU after the TOFU run. News and Books, retain KL only (ASU_klr).
# Temperatures and forget coefficients follow open-unlearning/scripts/muse_unlearn.sh.
set -u
source /root/autodl-tmp/env_hf.sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PATH="/root/autodl-tmp/envs/unlearning/bin:$PATH"
export WANDB_MODE=disabled WANDB_DISABLED=true

TOFU_TRAIN_PID=${1:-7865}
TOFU_EVAL_PID=${2:-12283}
MUSE_ROOT=/usr/local/ASU-unlearning/muse
CKPT_ROOT=/root/autodl-tmp/ASU-unlearning/checkpoints/muse
RESULT_ROOT=/usr/local/ASU-unlearning/workspace/results/muse
TOKENIZER=NousResearch/Llama-2-7b-hf
PY=/root/autodl-tmp/envs/unlearning/bin/python

mkdir -p "$CKPT_ROOT" "$RESULT_ROOT"

while kill -0 "$TOFU_TRAIN_PID" 2>/dev/null || kill -0 "$TOFU_EVAL_PID" 2>/dev/null; do
    sleep 30
done

for _ in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk 'BEGIN{m=0} {if($1>m)m=$1} END{print m}')
    if [ "${used:-0}" -lt 4000 ]; then
        break
    fi
    sleep 15
done

if [ ! -f "$MUSE_ROOT/data/news/raw/forget.txt" ] || [ ! -f "$MUSE_ROOT/data/books/raw/forget.txt" ]; then
    echo "downloading MUSE corpora"
    (cd "$MUSE_ROOT" && "$PY" load_data.py)
fi

run_one() {
    local corpus="$1" algo="$2" temp="$3" alpha="$4" target="$5"
    local out="$CKPT_ROOT/$corpus/$algo"
    local csv="$RESULT_ROOT/${corpus}_${algo}.csv"
    local metrics
    mkdir -p "$out"
    if [ ! -f "$out/config.json" ]; then
        echo "train corpus=$corpus algo=$algo temp=$temp alpha=$alpha"
        (
            cd "$MUSE_ROOT/baselines"
            CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$((RANDOM % 50001 + 10000)) \
                unlearn.py \
                --algo "$algo" \
                --model_dir "$target" \
                --tokenizer_dir "$TOKENIZER" \
                --data_file "../data/$corpus/raw/forget.txt" \
                --retain_data_file "../data/$corpus/raw/retain1.txt" \
                --out_dir "$out" \
                --max_len 2048 \
                --epochs 10 \
                --lr 1e-5 \
                --per_device_batch_size 1 \
                --gradient_accumulation_steps 16 \
                --attention_temp "$temp" \
                --alpha "$alpha" \
                --optim paged_adamw_8bit \
                --gradient_checkpointing \
                --save_strategy no
        ) || { echo "train failed corpus=$corpus algo=$algo"; exit 1; }
    else
        echo "checkpoint exists: $out"
    fi
    if [ ! -f "$csv" ]; then
        if [ "$corpus" = "books" ]; then
            metrics="verbmem_f privleak"
        else
            metrics="verbmem_f privleak knowmem_f knowmem_r"
        fi
        echo "eval corpus=$corpus algo=$algo"
        (
            cd "$MUSE_ROOT"
            CUDA_VISIBLE_DEVICES=0 "$PY" eval.py \
                --model_dirs "$out" \
                --names "${corpus}_${algo}" \
                --tokenizer_dir "$TOKENIZER" \
                --corpus "$corpus" \
                --out_file "$csv" \
                --metrics $metrics \
                --temp_dir "$CKPT_ROOT/tmp/${corpus}_${algo}"
        )
    else
        echo "eval exists: $csv"
    fi
}

run_one news ASU_klr 2.0 0.35 muse-bench/MUSE-News_target
run_one books ASU_klr 2.35 0.001 muse-bench/MUSE-Books_target
echo "MUSE ASU finished"

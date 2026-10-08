#!/usr/bin/env bash
# Replay the frozen 2026-09-27 forget05 ASU+GD run and keep the weights.
# Same overrides as Right-to-be-forgotten/scripts/tofu/asu.sh, one cell only.
# save_checkpoint=true so eval.py does not delete checkpoint-last.
# save_root is separate so the 09-27 eval files stay untouched.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="${ROOT}/workspace/logs/replay_forget05_asu_gd.log"
mkdir -p "${ROOT}/workspace/logs"

{
    source /root/autodl-tmp/env_hf.sh
    export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
    export PATH="/root/autodl-tmp/envs/unlearning/bin:$PATH"
    export WANDB_MODE=disabled WANDB_DISABLED=true
    export TASK_LIST=1

    cd /usr/local/ASU-unlearning/Right-to-be-forgotten

    gpu_n="$(nvidia-smi -L 2>/dev/null | wc -l)"
    if [[ "${gpu_n}" -lt 2 ]]; then
        echo "Need 2 GPUs. nvidia-smi sees ${gpu_n}. Refusing to start."
        nvidia-smi -L || true
        exit 1
    fi

    cuda_n="$(CUDA_VISIBLE_DEVICES=0,1 python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
    echo "cuda_devices=${cuda_n}"
    if [[ "${cuda_n}" -lt 2 ]]; then
        echo "CUDA sees ${cuda_n} devices with CUDA_VISIBLE_DEVICES=0,1. Need 2."
        exit 1
    fi

    if ps -ef | awk '/[f]orget.py|[t]orchrun|[e]val.py/ {found=1} END {exit !found}'; then
        echo "Another forget.py, torchrun, or eval.py is running. Refusing to start."
        ps -ef | awk '/[f]orget.py|[t]orchrun|[e]val.py/'
        exit 1
    fi

    MASTER_PORT=$((RANDOM % 50001 + 10000))
    COMMON="use_LoRA=false forget_coeff=0.1 regularization_coeff=1.0 lr=1e-5 split=forget05 forget_loss=ASU+GD num_epochs=5 mask=true fix_ref_model=true save_root=results/tofu_replay save_checkpoint=true attention_temp=2.3 layers_id=null"

    echo "train forget05 ASU+GD save_checkpoint=true save_root=results/tofu_replay"
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port="${MASTER_PORT}" \
        forget.py \
        --config-name=tofu.yaml \
        task_id=1 \
        save_steps=last \
        ${COMMON}

    echo "eval forget05 ASU+GD eval_unlearn_step=last save_checkpoint=true"
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" \
        eval.py \
        --config-name=tofu.yaml \
        task_id=1 \
        eval_unlearn_step=last \
        ${COMMON}

    CKPT_ROOT="/root/autodl-tmp/ASU-unlearning/checkpoints/tofu_replay/llama2-7b/forget05/ASU+GD"
    mapfile -t CKPTS < <(find "${CKPT_ROOT}" -type d -name checkpoint-last | sort)
    if [[ "${#CKPTS[@]}" -ne 1 ]]; then
        echo "Expected one checkpoint-last under ${CKPT_ROOT}, found ${#CKPTS[@]}."
        printf '%s\n' "${CKPTS[@]}"
        exit 1
    fi
    CKPT="${CKPTS[0]}"
    mapfile -t WEIGHTS < <(find "${CKPT}" -name '*.safetensors' -size +0c | sort)
    if [[ "${#WEIGHTS[@]}" -lt 1 ]]; then
        echo "checkpoint-last has no safetensors: ${CKPT}"
        exit 1
    fi
    echo "CHECKPOINT_LAST=${CKPT}"
    printf 'WEIGHT\t%s\n' "${WEIGHTS[@]}"

    CSV="$(dirname "${CKPT}")/eval_results-last/unlearning_results.csv"
    if [[ -f "${CSV}" ]]; then
        python - "${CSV}" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as f:
    row = next(csv.DictReader(f))
print(f"replay_MU={row['Model Utility']} replay_FE={row['Forget Efficacy']} frozen_MU=0.740182465708051")
PY
    else
        echo "Missing eval csv: ${CSV}"
        exit 1
    fi
} 2>&1 | tee -a "${LOG}"

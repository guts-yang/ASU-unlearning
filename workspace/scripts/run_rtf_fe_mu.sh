#!/usr/bin/env bash
# Evaluate FE/MU with the Right-to-be-forgotten metric definition.
# Usage: bash workspace/scripts/run_rtf_fe_mu.sh <STUDENT_DIR> [DS_SIZE] [OUT_NAME]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_qkspec.sh"

STUDENT=${1:?usage: run_rtf_fe_mu.sh STUDENT_DIR [DS_SIZE] [OUT_NAME]}
DS_SIZE=${2:-200}
OUT_NAME=${3:-rtf_fe_mu}

export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/jizhicfs/chenyliao/ops/cuda-stub
export DS_SKIP_CUDA_CHECK=1
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=0

# TextDatasetQA resolves config/model_config.yaml relative to CWD -> run from RTF.
cd "$REPO/Right-to-be-forgotten"
"$PY" "$REPO/workspace/scripts/rtf_fe_mu_eval.py" \
  --student "$STUDENT" \
  --out "$STUDENT/$OUT_NAME" \
  --ds-size "$DS_SIZE" \
  --max-length 512 \
  --batch-size 16 2>&1 | tee "$STUDENT/${OUT_NAME}.log"

echo "=== RTF result: $(cat "$STUDENT/$OUT_NAME/rtf_fe_mu.json" 2>/dev/null) ==="

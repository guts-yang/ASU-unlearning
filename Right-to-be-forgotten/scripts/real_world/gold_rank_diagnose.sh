#!/bin/bash
# Freeze K and delta from a local teacher. This script never downloads weights.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${ROOT}/workspace:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local snapshot directory}"
FORGET_JSON="${FORGET_JSON:?Set FORGET_JSON to the local real-world forget JSON}"
LOCK_OUT="${LOCK_OUT:-${ROOT}/workspace/gold_rank/locks/calibration_lock.json}"

if [[ ! -d "$MODEL_PATH" || ! -f "$FORGET_JSON" ]]; then
  echo "Refusing to download. MODEL_PATH and FORGET_JSON must already exist locally." >&2
  exit 2
fi

python -m gold_rank.diagnose \
  --model "$MODEL_PATH" \
  --forget-json "$FORGET_JSON" \
  --protocol "${ROOT}/workspace/gold_rank/protocol.json" \
  --lock-out "$LOCK_OUT"

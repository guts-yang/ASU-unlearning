#!/bin/bash
# Launch one factorial group after local assets and, where required, a frozen lock exist.
# Groups: A ASU, B ASU+Anchor, C ASU+Demotion, D both, plus penalty, random, shuffle_k, flatten.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${ROOT}/workspace:${PYTHONPATH:-}"

GROUP="${GROUP:?Set GROUP to A, B, C, D, penalty, random, shuffle_k, or flatten}"
SEED="${SEED:-1001}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local snapshot directory}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the local real_world directory}"
SAVE_ROOT="${SAVE_ROOT:?Set SAVE_ROOT to a local checkpoint directory}"
REASON_JSONL="${REASON_JSONL:-}"
LOCK_PATH="${LOCK_PATH:-}"
VERDICT_PATH="${VERDICT_PATH:-${ROOT}/workspace/C/derived/m0_verdict.json}"

args=(
  --group "$GROUP"
  --model "$MODEL_PATH"
  --data-path "$DATA_PATH"
  --save-root "$SAVE_ROOT"
  --seed "$SEED"
  --verdict "$VERDICT_PATH"
)
if [[ -n "$REASON_JSONL" ]]; then
  args+=(--reason-jsonl "$REASON_JSONL")
fi
if [[ -n "$LOCK_PATH" ]]; then
  args+=(--lock "$LOCK_PATH")
fi

python -m gold_rank.launch "${args[@]}"

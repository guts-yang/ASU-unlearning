#!/usr/bin/env bash
# Autonomous RIU pipeline: wait for idle GPU -> train -> evaluate -> self-heal.
# All state/logs: workspace/notes/auto_pipeline/
set -uo pipefail

REPO=/usr/local/ASU-unlearning-RIU
source "$REPO/workspace/scripts/env_apru.sh"

STATE="$WS/notes/auto_pipeline"
mkdir -p "$STATE" "$WS/results"
LOG="$STATE/pipeline.log"
GPULOG="$STATE/gpu_wait.log"
TRAIN_LOG="$STATE/train_attempt.log"
EVAL_LOG="$STATE/eval_attempt.log"
MAX_ATTEMPTS=5
TASK=riu_asu_tofu05_pilot_seed1001
OUT="$OUTPUT_ROOT/unlearn/$TASK"
REPORT="$WS/results/riu_eval_report.json"
export CUDA_VISIBLE_DEVICES=0

log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
gpulog() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$GPULOG"; }

find_idle_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, '$2+0 < 2000 && $3+0 < 10 {gsub(/ /,"",$1); print $1; exit}'
}

wait_for_gpu() {
  local gpu=""
  while [[ -z "$gpu" ]]; do
    gpu=$(find_idle_gpu)
    if [[ -z "$gpu" ]]; then
      gpulog "GPU busy; nvidia-smi=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
      sleep 30
    fi
  done
  export CUDA_VISIBLE_DEVICES="$gpu"
  gpulog "GPU $gpu idle; starting pipeline"
  log "GPU $gpu acquired"
}

resolve_ckpt() {
  local d="$1"
  if [[ -f "$d/config.json" ]]; then echo "$d"; return 0; fi
  local latest
  latest=$(find "$d" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1)
  if [[ -n "$latest" && -f "$latest/config.json" ]]; then echo "$latest"; return 0; fi
  return 1
}

heal() {
  local attempt=$1 logfile=$2 strategy="default"
  if grep -qi 'out of memory' "$logfile"; then
    strategy="oom"
  elif grep -qi 'no module named' "$logfile"; then
    strategy="import"
  elif grep -qi 'could not override\|hydra' "$logfile"; then
    strategy="hydra"
  elif grep -qi 'traceback' "$logfile"; then
    strategy="generic"
  fi
  log "diagnosis strategy=$strategy (attempt $attempt)"
  case "$strategy" in
    oom) EXTRA_TRAIN=(trainer.args.per_device_train_batch_size=1 trainer.args.gradient_accumulation_steps=16) ;;
    import) source "$REPO/workspace/scripts/env_apru.sh"; EXTRA_TRAIN=() ;;
    hydra) EXTRA_TRAIN=() ;;
    *) EXTRA_TRAIN=() ;;
  esac
}

run_train() {
  log "TRAIN start attempt=$attempt extra=${EXTRA_TRAIN[*]:-none}"
  set +e
  bash "$WS/scripts/run_riu.sh" \
    trainer.args.num_train_epochs=5 \
    trainer.args.save_strategy=epoch \
    trainer.args.save_total_limit=1 \
    trainer.method_args.riu_log_every=50 \
    "${EXTRA_TRAIN[@]}" >"$TRAIN_LOG" 2>&1
  local rc=$?
  set -e
  log "TRAIN exit=$rc attempt=$attempt tail=$(tail -n 3 "$TRAIN_LOG" | tr '\n' ' ')"
  return $rc
}

run_eval() {
  local ckpt
  ckpt=$(resolve_ckpt "$OUT") || { log "EVAL fail: no checkpoint under $OUT"; return 1; }
  log "EVAL start ckpt=$ckpt"
  set +e
  "$PY" "$WS/scripts/eval_riu_probe.py" --theta-u "$ckpt" --out "$REPORT" \
    --batch-size 4 >"$EVAL_LOG" 2>&1
  local rc=$?
  set -e
  log "EVAL exit=$rc tail=$(tail -n 3 "$EVAL_LOG" | tr '\n' ' ')"
  return $rc
}

set -e
log "=== pipeline start ==="
wait_for_gpu

attempt=1
EXTRA_TRAIN=()
while (( attempt <= MAX_ATTEMPTS )); do
  if run_train; then
    if run_eval; then
      log "SUCCESS: training and evaluation complete; report=$REPORT"
      printf 'SUCCESS attempt=%s ckpt=%s report=%s\n' "$attempt" "$(resolve_ckpt "$OUT")" "$REPORT" > "$STATE/final_status.txt"
      exit 0
    else
      log "EVAL failed; retrying full pipeline"
      attempt=$((attempt + 1))
      heal "$attempt" "$EVAL_LOG"
      sleep 20
      continue
    fi
  fi
  attempt=$((attempt + 1))
  if (( attempt > MAX_ATTEMPTS )); then
    log "FAILED after $MAX_ATTEMPTS attempts"
    printf 'FAILED after %s attempts\n' "$MAX_ATTEMPTS" > "$STATE/final_status.txt"
    exit 1
  fi
  heal "$attempt" "$TRAIN_LOG"
  sleep 20
done

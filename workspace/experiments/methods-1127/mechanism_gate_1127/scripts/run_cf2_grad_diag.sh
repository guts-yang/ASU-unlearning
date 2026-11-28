#!/usr/bin/env bash
# CF2 per-term gradient decomposition diagnostic (FULL loss, no weight zeroed).
# Default-off implementation lives in CF2-*/runtime.py; enabled via --set.
# 20 arms: 10 methods x (main / wrong control), smoke budget, GPU0 only.
# Bounded GPU0 wait (20 min): never preempt another session; exits cleanly.
set -uo pipefail

M=/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/methods-1127
R="$M/mechanism_gate_1127"
W=/jizhicfs/chenyliao/llm-unlearning/results/cf2-mechanism-gate/weights/grad_diag
PY=/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python
LOG="$R/results/grad_diag.log"
TSV="$R/results/grad_diag.tsv"
STATUS="$R/results/grad_diag_status.json"
mkdir -p "$R/results/grad_diag" "$W"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
gpu0_used() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '$1+0==0 {gsub(/ /,"",$2); print $2+0}'
}
printf 'method\tarm\tstart\tend\texit_code\telapsed_sec\n' > "$TSV"
log "GRAD_DIAG_START pid=$$ gpus=0 arms=20 pre_registered=PROTOCOL_10"

streak=0
for attempt in $(seq 1 60); do
  used="$(gpu0_used)"; [[ -z "$used" ]] && used=-1
  alive=0
  if ps -eo cmd | grep -qE '[r]un.py --profile|[b]ash start.sh'; then alive=1; fi
  if [[ "$used" -lt 100 && "$alive" -eq 0 ]]; then streak=$((streak+1)); else streak=0; fi
  log "WAIT_GPU0 used_MiB=$used worker_alive=$alive streak=$streak/3 attempt=$attempt/60"
  [[ "$streak" -ge 3 ]] && break
  sleep 20
done

if [[ "$streak" -lt 3 ]]; then
  log "GPU0_BUSY_TIMEOUT no arms started; not preempting another session"
  python3 - "$STATUS" <<'PY'
import json, sys
open(sys.argv[1], "w").write(json.dumps(
    {"stage": "grad_diag", "claim_level": False, "gpu0_busy_timeout": True,
     "arms_started": 0, "rows": [], "failed": 0,
     "note": "GPU0 remained busy for 20 min; no preemption, no pkill"},
    indent=2, ensure_ascii=False))
PY
  exit 0
fi
log "GPU0_FREE confirmed; starting gradient diagnostics"

FAILED=0
run_arm() {
  local dir="$1" tag="$2" arm="$3"; shift 3
  local out="$W/$tag/$arm" doc="$R/results/grad_diag/$tag/$arm"
  mkdir -p "$out" "$doc"
  local start end rc epoch
  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; epoch=$(date +%s)
  log "ARM_START tag=$tag arm=$arm out=$out"
  ( cd "$dir" && "$PY" run.py --profile smoke --output "$out" \
      --set data.reuse_cache=false --set training.batch_size=2 \
      --set loss.grad_diagnostics=true "$@" ) > "$out/console.log" 2>&1
  rc=$?
  end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$arm" "$start" "$end" "$rc" \
    "$(( $(date +%s) - epoch ))" >> "$TSV"
  for f in result.json diagnostics.json resolved_config.json console.log; do
    [[ -f "$out/$f" ]] && cp "$out/$f" "$doc/" || true
  done
  log "ARM_END tag=$tag arm=$arm exit=$rc elapsed_sec=$(( $(date +%s) - epoch ))"
  [[ "$rc" -ne 0 ]] && FAILED=$((FAILED+1))
}

for dir in "$M"/CF2-*/; do
  name="$(basename "$dir")"
  tag="${name%%_*}"
  run_arm "$dir" "$tag" "main"
  run_arm "$dir" "$tag" "control" --set method.control_mode=wrong
done

python3 - "$TSV" "$STATUS" "$FAILED" <<'PY'
import json, sys
lines = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
hdr = lines[0]
rows = [dict(zip(hdr, l)) for l in lines[1:]]
out = {"stage": "grad_diag", "claim_level": False, "profile": "smoke",
       "config": {"loss_weights": "full default (0.1 / 0.1 / 2.0), none zeroed",
                  "steps": 2, "max_pairs": 4, "batch_size": 2,
                  "reuse_cache": False, "grad_diagnostics": True, "gpus": "0 only"},
       "pre_registered": "PROTOCOL.md §10 (G1 hard; G2/G3 descriptive)",
       "gpu0_busy_timeout": False, "rows": rows, "failed": int(sys.argv[3]),
       "pilot_queue_started": False}
open(sys.argv[2], "w").write(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps({"stage": "grad_diag", "failed": out["failed"], "arms": len(rows)}))
PY
log "GRAD_DIAG_DONE failed=$FAILED pilot_queue_started=false; awaiting G1 + descriptive table"
exit 0

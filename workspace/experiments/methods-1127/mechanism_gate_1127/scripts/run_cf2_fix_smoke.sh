#!/usr/bin/env bash
# CF2 mechanism-gate FIX smoke queue (2 steps / 4 pairs, GPU0 only).
# Pre-registered S1-S4 are frozen in PROTOCOL.md §6.3 BEFORE this run.
#
# 24 arms, IDEA_REPORT §7 order: none x10, wrong x10,
# CF2-01 fix / fix_nomethod, CF2-06 fix, CF2-09 gate.
# NO source changes here; all fixes enter through --set overrides.
# smoke does NOT reuse pilot teacher_cache.pt (different max_pairs -> pair/index
# mismatch) and forces batch_size=2 (mismatched-item control needs len(batch)>=2).
set -uo pipefail

M=/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/methods-1127
R="$M/mechanism_gate_1127"
W=/jizhicfs/chenyliao/llm-unlearning/results/cf2-mechanism-gate/weights/fix_smoke
PY=/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python
LOG="$R/results/fix_smoke.log"
TSV="$R/results/fix_smoke.tsv"
STATUS="$R/results/fix_smoke_status.json"
mkdir -p "$R/results/fix_smoke" "$W"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
gpu0_used() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '$1+0==0 {gsub(/ /,"",$2); print $2+0}'
}
printf 'method\tarm\tstart\tend\texit_code\telapsed_sec\n' > "$TSV"

log "FIX_SMOKE_START pid=$$ gpus=0 arms=24 pre_registered=PROTOCOL_6.3"

streak=0
while true; do
  used="$(gpu0_used)"; [[ -z "$used" ]] && used=-1
  alive=0
  if ps -eo cmd | grep -qE '[r]un.py --profile|[b]ash start.sh'; then alive=1; fi
  if [[ "$used" -lt 100 && "$alive" -eq 0 ]]; then streak=$((streak+1)); else streak=0; fi
  log "WAIT_GPU0 used_MiB=$used worker_alive=$alive streak=$streak/3"
  [[ "$streak" -ge 3 ]] && break
  sleep 20
done
log "GPU0_FREE confirmed; starting fix smoke"

FAILED=0
run_arm() {
  local dir="$1" tag="$2" arm="$3"; shift 3
  local out="$W/$tag/$arm" doc="$R/results/fix_smoke/$tag/$arm"
  mkdir -p "$out" "$doc"
  local start end rc epoch
  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; epoch=$(date +%s)
  log "ARM_START tag=$tag arm=$arm out=$out"
  ( cd "$dir" && "$PY" run.py --profile smoke --output "$out" \
      --set data.reuse_cache=false --set training.batch_size=2 "$@" ) > "$out/console.log" 2>&1
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
  run_arm "$dir" "$tag" "none"
  run_arm "$dir" "$tag" "wrong" --set method.control_mode=wrong
  case "$tag" in
    CF2-01)
      run_arm "$dir" "$tag" "fix" --set method.gate_mode=relative --set loss.retain_weight=0.0
      run_arm "$dir" "$tag" "fix_nomethod" --set method.gate_mode=relative \
        --set loss.retain_weight=0.0 --set loss.method_weight=0.0
      ;;
    CF2-06) run_arm "$dir" "$tag" "fix" --set method.distance_mode=median ;;
    CF2-09) run_arm "$dir" "$tag" "gate" --set method.strict_semantic_gate=true ;;
  esac
done

python3 - "$TSV" "$STATUS" "$FAILED" <<'PY'
import json, sys
lines = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
hdr = lines[0]
rows = [dict(zip(hdr, l)) for l in lines[1:]]
out = {"stage": "fix_smoke", "claim_level": False, "profile": "smoke",
       "config": {"steps": 2, "max_pairs": 4, "batch_size": 2,
                  "reuse_cache": False, "gpus": "0 only"},
       "pre_registered": "PROTOCOL.md §6.3 (frozen before run)",
       "rows": rows, "failed": int(sys.argv[3]),
       "pilot_queue_started": False}
open(sys.argv[2], "w").write(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps({"stage": "fix_smoke", "failed": out["failed"], "arms": len(rows)}))
PY

log "FIX_SMOKE_DONE failed=$FAILED pilot_queue_started=false; awaiting S1-S4 report"
exit 0

#!/usr/bin/env bash
# CF3-01 queue-head replication (stage A): wait for GPU0, then run the completed
# implementation's smoke + main-chain pilot + 5 decisive-experiment control arms.
#
# - GPU0 ONLY, strictly serial; never touches GPU1.
# - Explicit 31-pair file for pilot arms; Teacher cache rebuilt once by the brcd
#   pilot and reused by the other 5 arms (same teacher/pairs/layers = correct reuse).
# - Stage B (full-scale validation) is NOT started here; see run_cf301_full.sh.
set -uo pipefail

M=/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/methods-1127-2
D="$M/CF3-01_双线性路由内容差分"
R="$M/cf3_01_replication"
W=/jizhicfs/chenyliao/llm-unlearning/results/cf3-01-replication/weights
PY=/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python
PAIRS="$R/pairs/strict_pairs_31.jsonl"
LOG="$R/results/queue.log"
STATUS="$R/results/queue_status.json"
TSV="$R/results/queue.tsv"
SHARED="$W/shared_cache"
mkdir -p "$R/results" "$W" "$SHARED"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
gpu0_used() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '$1+0==0 {gsub(/ /,"",$2); print $2+0}'
}
printf 'arm\tprofile\tstart\tend\texit_code\telapsed_sec\n' > "$TSV"

log "QUEUE_START pid=$$ gpus=0 arms=brcd(smoke+pilot),route_only,content_only,equal_norm_delta,random_equal_norm,wrong_shuffled"
log "WAIT_CONDITION GPU0 <100MiB x3 consecutive and no run.py/start.sh worker"

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
log "GPU0_FREE confirmed; starting stage A"

# run_arm <tag> <config-arm> <profile> <use_pairs> <reuse_cache> <out> ; logs to results/<tag>
run_arm() {
  local tag="$1" arm="$2" profile="$3" use_pairs="$4" reuse="$5" out="$6"
  local start end rc epoch console
  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; epoch=$(date +%s)
  mkdir -p "$out" "$R/results/$tag"
  console="$R/results/$tag/console.log"
  log "ARM_START tag=$tag arm=$arm profile=$profile out=$out"

  local -a cmd=("$PY" "$D/run.py" "--profile" "$profile" "--config" "$R/configs/$arm.json" "--output" "$out")
  if [[ "$use_pairs" == "yes" ]]; then cmd+=("--set" "data.pairs_file=$PAIRS"); fi
  if [[ "$reuse" == "no" ]]; then cmd+=("--set" "data.reuse_cache=false"); fi
  ( cd "$D" && "${cmd[@]}" ) > "$console" 2>&1
  rc=$?
  end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$profile" "$start" "$end" "$rc" "$(( $(date +%s) - epoch ))" >> "$TSV"
  log "ARM_END tag=$tag exit=$rc elapsed_sec=$(( $(date +%s) - epoch )) console=$console"

  # Lightweight docs to the repo; weights stay on the data disk.
  for f in result.json diagnostics.json resolved_config.json; do
    [[ -f "$out/$f" ]] && cp "$out/$f" "$R/results/$tag/" || true
  done
  return "$rc"
}

FAILED=0

# 1) completed-implementation smoke: 4 automatic pairs / 2 steps, no explicit pair file.
run_arm smoke_brcd brcd smoke no no "$W/smoke_brcd" || FAILED=$((FAILED+1))

# 2) main-chain pilot: rebuilds the 31-pair Teacher cache (attention + GLU + full sequence).
run_arm pilot_brcd brcd pilot yes no "$W/pilot_brcd" || FAILED=$((FAILED+1))
if [[ -f "$W/pilot_brcd/teacher_cache.pt" ]]; then
  cp "$W/pilot_brcd/teacher_cache.pt" "$SHARED/teacher_cache.pt"
  log "SHARED_CACHE_READY $SHARED/teacher_cache.pt"
else
  log "SHARED_CACHE_MISSING; control arms will rebuild their own cache"
fi

# 3) five decisive-experiment control arms: same budget / same 31 pairs.
for arm in route_only content_only equal_norm_delta random_equal_norm wrong_shuffled; do
  tag="pilot_$arm"
  out="$W/$tag"
  if [[ -f "$SHARED/teacher_cache.pt" ]]; then
    mkdir -p "$out" && cp "$SHARED/teacher_cache.pt" "$out/teacher_cache.pt"
  fi
  run_arm "$tag" "$arm" pilot yes yes "$out" || FAILED=$((FAILED+1))
done

python3 - "$TSV" "$STATUS" "$FAILED" <<'PY'
import json, sys
lines=[l.rstrip('\n').split('\t') for l in open(sys.argv[1]) if l.strip()]
hdr=lines[0]
rows=[dict(zip(hdr,l)) for l in lines[1:]]
out={'stage':'A','claim_level':False,'arms':rows,'failed':int(sys.argv[3]),
     'full_stage_started':False,'status_path':sys.argv[2]}
open(sys.argv[2],'w').write(json.dumps(out,indent=2,ensure_ascii=False))
print(json.dumps({'stage':'A','failed':out['failed'],'arms':len(rows)}))
PY

log "QUEUE_DONE failed=$FAILED full_stage_started=false; stage B awaits user decision"
exit 0

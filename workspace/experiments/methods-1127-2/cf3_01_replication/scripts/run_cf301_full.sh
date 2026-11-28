#!/usr/bin/env bash
# Stage B: full-scale validation of CF3-01. PREPARED BUT NOT STARTED.
#
# Per user instruction: stage A results and conclusion are delivered FIRST; only
# after the user judges the performance acceptable is this script to be launched.
# Do NOT auto-start from run_cf301_replication.sh.
#
# Scope: full profile (400 pairs / 600 steps), main chain + core control arms,
# 3 seeds, same strict pairing contract. All outputs non-claim until review.
set -uo pipefail

M=/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/methods-1127-2
D="$M/CF3-01_双线性路由内容差分"
R="$M/cf3_01_replication"
W=/jizhicfs/chenyliao/llm-unlearning/results/cf3-01-replication/weights
PY=/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python
PAIRS="$R/pairs/strict_pairs_31.jsonl"
LOG="$R/results/full_queue.log"
STATUS="$R/results/full_queue_status.json"
TSV="$R/results/full_queue.tsv"
mkdir -p "$R/results" "$W"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
gpu0_used() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '$1+0==0 {gsub(/ /,"",$2); print $2+0}'
}
printf 'tag\tseed\tstart\tend\texit_code\telapsed_sec\n' > "$TSV"
log "FULL_STAGE_START (needs explicit user go-ahead) gpus=0"

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

FAILED=0
for seed in 0 1 2; do
  for arm in brcd route_only content_only equal_norm_delta random_equal_norm wrong_shuffled; do
    tag="full_${arm}_s${seed}"
    out="$W/$tag"
    mkdir -p "$out" "$R/results/$tag"
    start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; epoch=$(date +%s)
    console="$R/results/$tag/console.log"
    log "FULL_START $tag"
    ( cd "$D" && "$PY" run.py --profile full --config "$R/configs/$arm.json" --output "$out" \
        --set "seed=$seed" --set "data.pairs_file=$PAIRS" --set data.reuse_cache=false ) > "$console" 2>&1
    rc=$?
    end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$seed" "$start" "$end" "$rc" "$(( $(date +%s) - epoch ))" >> "$TSV"
    log "FULL_END $tag exit=$rc elapsed_sec=$(( $(date +%s) - epoch ))"
    for f in result.json diagnostics.json resolved_config.json; do
      [[ -f "$out/$f" ]] && cp "$out/$f" "$R/results/$tag/" || true
    done
    [[ $rc -ne 0 ]] && FAILED=$((FAILED+1))
  done
done
python3 - "$TSV" "$STATUS" "$FAILED" <<'PY'
import json,sys
lines=[l.rstrip('\n').split('\t') for l in open(sys.argv[1]) if l.strip()]
rows=[dict(zip(lines[0],l)) for l in lines[1:]]
out={'stage':'B','claim_level':False,'arms':rows,'failed':int(sys.argv[3]),'full_stage_started':True}
open(sys.argv[2],'w').write(json.dumps(out,indent=2,ensure_ascii=False))
print(json.dumps({'stage':'B','failed':out['failed'],'arms':len(rows)}))
PY
log "FULL_STAGE_DONE failed=$FAILED"
exit 0

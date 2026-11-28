#!/usr/bin/env bash
# CF2 fix RETEST queue: 6 arms, smoke budget, GPU0 only.
# Pre-registered R1-R3 are frozen in PROTOCOL.md §8 BEFORE this run.
# §7 S1-S4 verdicts and results/fix_smoke_* artifacts are NEVER overwritten.
set -uo pipefail

M=/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/methods-1127
R="$M/mechanism_gate_1127"
W=/jizhicfs/chenyliao/llm-unlearning/results/cf2-mechanism-gate/weights/fix_retest
PY=/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python
D06="$M/CF2-06_反事实割线敏感度"
D01="$M/CF2-01_跨层反事实通量"
LOG="$R/results/fix_retest.log"
TSV="$R/results/fix_retest.tsv"
STATUS="$R/results/fix_retest_status.json"
mkdir -p "$R/results/fix_retest" "$W"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
gpu0_used() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '$1+0==0 {gsub(/ /,"",$2); print $2+0}'
}
printf 'method\tarm\tstart\tend\texit_code\telapsed_sec\n' > "$TSV"

log "FIX_RETEST_START pid=$$ gpus=0 arms=6 pre_registered=PROTOCOL_8.3"

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
log "GPU0_FREE confirmed; starting retest"

FAILED=0
run_arm() {
  local dir="$1" tag="$2" arm="$3"; shift 3
  local out="$W/$tag/$arm" doc="$R/results/fix_retest/$tag/$arm"
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

# R1: CF2-06 default regression + mismatched-item control after the common-n fix.
run_arm "$D06" "CF2-06" "none"
run_arm "$D06" "CF2-06" "wrong" --set method.control_mode=wrong

# R2: reproduce the previous S2 numbers on identical configs.
run_arm "$D01" "CF2-01" "fix" --set method.gate_mode=relative --set loss.retain_weight=0.0
run_arm "$D01" "CF2-01" "fix_nomethod" --set method.gate_mode=relative \
  --set loss.retain_weight=0.0 --set loss.method_weight=0.0

# R3: falsifiable "neutral guards dominate gradient" hypothesis (diagnostic arms only).
run_arm "$D01" "CF2-01" "fix_no_neutral" --set method.gate_mode=relative \
  --set loss.retain_weight=0.0 --set loss.neutral_weight=0.0
run_arm "$D01" "CF2-01" "fix_no_neutral_nomethod" --set method.gate_mode=relative \
  --set loss.retain_weight=0.0 --set loss.neutral_weight=0.0 --set loss.method_weight=0.0

python3 - "$TSV" "$STATUS" "$FAILED" <<'PY'
import json, sys
lines = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
hdr = lines[0]
rows = [dict(zip(hdr, l)) for l in lines[1:]]
out = {"stage": "fix_retest", "claim_level": False, "profile": "smoke",
       "config": {"steps": 2, "max_pairs": 4, "batch_size": 2,
                  "reuse_cache": False, "gpus": "0 only"},
       "pre_registered": "PROTOCOL.md §8.3 (frozen before run)",
       "prior_verdict_preserved": "PROTOCOL.md §7 S1-S4 FAIL/PASS unchanged",
       "rows": rows, "failed": int(sys.argv[3]),
       "pilot_queue_started": False}
open(sys.argv[2], "w").write(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps({"stage": "fix_retest", "failed": out["failed"], "arms": len(rows)}))
PY

log "FIX_RETEST_DONE failed=$FAILED pilot_queue_started=false; awaiting R1-R3 report"
exit 0

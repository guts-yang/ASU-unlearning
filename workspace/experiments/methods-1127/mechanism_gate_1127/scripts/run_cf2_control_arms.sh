#!/usr/bin/env bash
# CF2 ten-method mechanism discrimination gate -- GPU training arms.
#
# User decision 2026-11-28: "shared-term baseline" control.
#   main arm     : pilot budget, default config
#   baseline arm : identical but --set loss.method_weight=0.0
# Isolates whether the method-specific term is swamped by retain_weight=2.0
# (CF2 pilot: dF negative across all 10, generation metrics identical).
#
# Aligned constraints: GPU0 ONLY (never touch GPU1 lrm-unlearning queue),
# IDEA_REPORT §7 order, THIN WRAPPER calling run.py directly (bypasses
# run_all.py COMPLETED skip and hard-coded output dir), NO source changes,
# pilot budget, reuse existing teacher_cache.pt, weights on data disk,
# docs/scripts in repo, experiment JSON/logs excluded from Git.
set -uo pipefail

M=/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/methods-1127
R="$M/mechanism_gate_1127"
W=/jizhicfs/chenyliao/llm-unlearning/results/cf2-mechanism-gate/weights
PY=/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python
LOG="$R/results/arms.log"
TSV="$R/results/arms.tsv"
STATUS="$R/results/arms_status.json"
COMPARE="$R/results/arm_comparison.json"
mkdir -p "$R/results/arms" "$W"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
gpu0_used() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '$1+0==0 {gsub(/ /,"",$2); print $2+0}'
}
printf 'method\tarm\tstart\tend\texit_code\telapsed_sec\tcache_reused\n' > "$TSV"

log "ARMS_START pid=$$ gpus=0 decision=shared_term_baseline order=IDEA_REPORT_7(CF2-01..10)"

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
log "GPU0_FREE confirmed; starting training arms"

FAILED=0
for dir in "$M"/CF2-*/; do
  name="$(basename "$dir")"
  tag="${name%%_*}"
  for arm in main baseline; do
    out="$W/$tag/$arm"
    doc="$R/results/arms/$tag/$arm"
    mkdir -p "$out" "$doc"
    cache_src="$dir/runs/pilot_seed0/teacher_cache.pt"
    reused=no
    if [[ -f "$cache_src" ]]; then
      [[ -e "$out/teacher_cache.pt" ]] || ln -s "$cache_src" "$out/teacher_cache.pt"
      reused=yes
    fi
    start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; epoch=$(date +%s)
    log "ARM_START tag=$tag arm=$arm cache_reused=$reused out=$out"
    if [[ "$arm" == "baseline" ]]; then
      ( cd "$dir" && "$PY" run.py --profile pilot --output "$out" \
          --set data.reuse_cache=true --set loss.method_weight=0.0 ) > "$out/console.log" 2>&1
    else
      ( cd "$dir" && "$PY" run.py --profile pilot --output "$out" \
          --set data.reuse_cache=true ) > "$out/console.log" 2>&1
    fi
    rc=$?
    end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$arm" "$start" "$end" "$rc" \
      "$(( $(date +%s) - epoch ))" "$reused" >> "$TSV"
    for f in result.json diagnostics.json resolved_config.json console.log; do
      [[ -f "$out/$f" ]] && cp "$out/$f" "$doc/" || true
    done
    log "ARM_END tag=$tag arm=$arm exit=$rc elapsed_sec=$(( $(date +%s) - epoch )) console=$out/console.log"
    [[ "$rc" -ne 0 ]] && FAILED=$((FAILED+1))
  done
done

python3 - "$R/results" "$TSV" "$STATUS" "$COMPARE" "$FAILED" <<'PY'
import json, os, sys
root, tsv, status, compare, failed = sys.argv[1:6]
lines = [l.rstrip("\n").split("\t") for l in open(tsv) if l.strip()]
hdr = lines[0]
rows = [dict(zip(hdr, l)) for l in lines[1:]]
methods = {}
for r in rows:
    tag, arm = r["method"], r["arm"]
    p = os.path.join(root, "arms", tag, arm, "result.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    methods.setdefault(tag, {})[arm] = {
        "exit_code": r["exit_code"], "cache_reused": r["cache_reused"],
        "status": res.get("status"), "steps": res.get("train", {}).get("steps_completed"),
        "delta": res.get("delta")}
comparison = {}
for tag, v in methods.items():
    m, b = v.get("main", {}), v.get("baseline", {})
    diff = None
    if m.get("delta") and b.get("delta"):
        diff = {k: round(m["delta"][k] - b["delta"][k], 6)
                for k in ("forget", "retain", "holdout")}
    comparison[tag] = {"main": m, "baseline": b, "main_minus_baseline": diff,
                       "note": "raw comparison only; claim_level=false; no forgetting claim"}
open(compare, "w").write(json.dumps(
    {"decision": "shared_term_baseline", "claim_level": False,
     "methods": comparison}, indent=2, ensure_ascii=False))
open(status, "w").write(json.dumps(
    {"stage": "control_arms", "claim_level": False, "decision": "shared_term_baseline",
     "order": "IDEA_REPORT §7 (CF2-01..10)", "budget": "pilot 60 steps / 32 pairs / seed 0 / GPU0",
     "rows": rows, "failed": int(failed), "gpu0_only": True},
    indent=2, ensure_ascii=False))
print(json.dumps({"stage": "control_arms", "failed": int(failed), "arms": len(rows)}))
PY

log "ARMS_DONE failed=$FAILED; comparison=$COMPARE"
exit 0

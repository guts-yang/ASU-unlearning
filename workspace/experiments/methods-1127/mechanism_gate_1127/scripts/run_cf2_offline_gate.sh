#!/usr/bin/env bash
# CF2 ten-method mechanism discrimination gate -- OFFLINE stage only.
#
# Aligned with the user on 2026-11-28 (plan writing was interrupted):
#   - GPU0 ONLY; start once GPU0 is idle; never touch GPU1 lrm-unlearning queue
#   - pipeline: offline separability gate -> control arms -> decision
#   - order: idea-stage/candidates-1127/IDEA_REPORT.md §7 (CF2-01..10)
#   - THIN WRAPPER calling run.py directly (bypasses run_all.py COMPLETED skip
#     and its hard-coded runs/<profile>_seed0 output dir); NO source changes
#   - pilot budget; reuse existing teacher_cache.pt (no cache rebuild)
#   - weights on data disk; docs/scripts in repo; experiment JSON/logs NOT committed
#   - REPORT ONCE after offline gate, BEFORE any GPU training arm starts
set -uo pipefail

M=/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/methods-1127
R="$M/mechanism_gate_1127"
W=/jizhicfs/chenyliao/llm-unlearning/results/cf2-mechanism-gate/weights
PY=/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python
LOG="$R/results/offline_gate.log"
TSV="$R/results/offline_gate.tsv"
STATUS="$R/results/offline_gate_status.json"
mkdir -p "$R/results/offline" "$W"
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }
gpu0_used() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '$1+0==0 {gsub(/ /,"",$2); print $2+0}'
}
printf 'method\tstart\tend\texit_code\telapsed_sec\tcache_reused\n' > "$TSV"

log "OFFLINE_GATE_START pid=$$ gpus=0 order=IDEA_REPORT_7(CF2-01..10) stage=offline_only"

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
log "GPU0_FREE confirmed; starting offline gate"

FAILED=0
for dir in "$M"/CF2-*/; do
  name="$(basename "$dir")"
  tag="${name%%_*}"
  out="$R/results/offline/$tag"
  mkdir -p "$out"
  cache_src="$dir/runs/pilot_seed0/teacher_cache.pt"
  reused=no
  if [[ -f "$cache_src" ]]; then
    [[ -e "$out/teacher_cache.pt" ]] || ln -s "$cache_src" "$out/teacher_cache.pt"
    reused=yes
  fi
  start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; epoch=$(date +%s)
  log "METHOD_START tag=$tag cache_reused=$reused out=$out"
  ( cd "$dir" && "$PY" run.py --profile pilot --diagnose-only --output "$out" \
      --set data.reuse_cache=true ) > "$out/console.log" 2>&1
  rc=$?
  end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$start" "$end" "$rc" \
    "$(( $(date +%s) - epoch ))" "$reused" >> "$TSV"
  log "METHOD_END tag=$tag exit=$rc elapsed_sec=$(( $(date +%s) - epoch )) console=$out/console.log"
  [[ "$rc" -ne 0 ]] && FAILED=$((FAILED+1))
done

python3 - "$R/results" "$TSV" "$STATUS" "$FAILED" <<'PY'
import json, os, sys
root, tsv, status, failed = sys.argv[1:5]
lines = [l.rstrip("\n").split("\t") for l in open(tsv) if l.strip()]
hdr = lines[0]
rows = [dict(zip(hdr, l)) for l in lines[1:]]
TRUE = ("true", "real", "correct", "genuine")
CTRL = ("shuffled", "wrong", "null", "noise", "destroyed", "random", "control", "negative")

def flatten(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flatten(v, f"{p}.{k}" if p else k))
    elif isinstance(o, list):
        for i, v in enumerate(o[:8]):
            out.update(flatten(v, f"{p}[{i}]"))
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        out[p] = o
    elif isinstance(o, str) and len(o) < 160:
        out[p] = o
    return out

methods = {}
for r in rows:
    tag = r["method"]
    base = os.path.join(root, "offline", tag)
    dp, rp = os.path.join(base, "diagnostics.json"), os.path.join(base, "result.json")
    flat = flatten(json.load(open(dp))) if os.path.exists(dp) else {}
    st = json.load(open(rp))["status"] if os.path.exists(rp) else None
    t = {k: v for k, v in flat.items() if any(w in k.lower() for w in TRUE)}
    c = {k: v for k, v in flat.items() if any(w in k.lower() for w in CTRL)}
    methods[tag] = {"exit_code": r["exit_code"], "cache_reused": r["cache_reused"],
                    "status": st, "separability_fields": sorted(t)[:12],
                    "control_fields": sorted(c)[:12], "has_true_vs_control": bool(t and c),
                    "note": "raw separability material only; formal verdict per each proposal.md minimal discrimination test; no verdict fabricated"}

out = {"stage": "offline", "claim_level": False,
       "order": "IDEA_REPORT §7 (CF2-01..10)",
       "budget": "pilot (60 steps / 32 pairs / seed 0 / GPU0), diagnose-only: 0 training steps",
       "methods": methods, "rows": rows, "failed": int(failed),
       "gpu_training_arms_started": False,
       "report_node": "offline gate finished: report BEFORE any control-arm training"}
open(status, "w").write(json.dumps(out, indent=2, ensure_ascii=False))
print(json.dumps({"stage": "offline", "failed": out["failed"], "methods": len(rows)}))
PY

log "OFFLINE_GATE_DONE failed=$FAILED gpu_training_arms_started=false; awaiting report + user decision"
exit 0

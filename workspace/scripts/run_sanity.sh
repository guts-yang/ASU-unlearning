#!/usr/bin/env bash
# ASU × LeakRU minimal sanity check (seed=1001)
set -euo pipefail

SEED=1001
ASU_ROOT=/usr/local/ASU-unlearning
OU_ROOT=/usr/local/open-unlearning
WS=$ASU_ROOT/workspace
DATA_JSONL=$WS/data/leakru/tofu_forget05.jsonl

ASU_CKPT_REUSE=${ASU_CKPT_REUSE:-/root/autodl-tmp/ASU-unlearning/checkpoints/tofu_replay/llama2-7b/forget05/ASU+GD/seed_1001/epoch5_1e-05_FixRefTrue_maskTrue_Fcoeff0.1_Rcoeff1.0_Temp2.3_LayersNone_beta0.1/1/unlearn_times_1/checkpoint-last}
TRAIN=${TRAIN:-0}          # 1 = train ASU only (no GA/NPO)
REUSE_ASU=${REUSE_ASU:-1}  # 1 = evaluate existing ASU ckpt instead of a fresh train
SKIP_EVAL=${SKIP_EVAL:-0}

echo "[sanity] seed=$SEED TRAIN=$TRAIN REUSE_ASU=$REUSE_ASU"

# --- env ---
if [[ -f /root/autodl-tmp/env_hf.sh ]]; then
  # shellcheck disable=SC1091
  source /root/autodl-tmp/env_hf.sh
fi
if [[ -x /root/autodl-tmp/envs/unlearning/bin/python ]]; then
  PY=/root/autodl-tmp/envs/unlearning/bin/python
elif [[ -x /root/autodl-tmp/envs/tofu/bin/python ]]; then
  PY=/root/autodl-tmp/envs/tofu/bin/python
else
  PY=python
fi
export PYTHONPATH="$OU_ROOT/src:$OU_ROOT/workspace/src:$WS/src:${PYTHONPATH:-}"

# --- data ---
$PY "$WS/scripts/build_tofu_leakru.py" --seed "$SEED"
test -f "$DATA_JSONL"

# --- GPU gate for train/eval ---
need_gpu=0
if [[ "$TRAIN" == "1" || "$SKIP_EVAL" != "1" ]]; then
  need_gpu=1
fi
META=$WS/notes/run_meta.json
mkdir -p "$WS/notes" "$WS/results" "$OU_ROOT/workspace/saves/eval" "$OU_ROOT/workspace/saves/unlearn"

write_meta() {
  local gpu_status=$1
  local gpu_detail=$2
  cat > "$META" <<EOF
{
  "seed": $SEED,
  "forget_split": "forget05",
  "retain_split": "retain95",
  "model_theta_p": "open-unlearning/tofu_Llama-2-7b-chat-hf_full",
  "asu_tau": 2.3,
  "asu_alpha": 0.1,
  "epochs": 5,
  "lr": 1e-5,
  "train": $TRAIN,
  "reuse_asu": $REUSE_ASU,
  "asu_ckpt_reuse": "$ASU_CKPT_REUSE",
  "data": "$DATA_JSONL",
  "attacks": ["probab", "focus_on_key"],
  "gpu_status": "$gpu_status",
  "gpu_detail": "$gpu_detail"
}
EOF
}

if [[ "$need_gpu" == "1" ]]; then
  if ! $PY - <<'PY'
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    write_meta "blocked" "torch.cuda.is_available() is False (device open may return EPERM). Scripts/data ready."
    $PY "$WS/scripts/aggregate_asu_leakru.py" --notes "$META" || true
    echo "[sanity] ERROR: CUDA GPU not available. Scripts/data are ready; re-run with GPU for train/eval."
    echo "[sanity] Tip: TRAIN=0 REUSE_ASU=1 once a GPU is attached to this container."
    exit 2
  fi
  nvidia-smi -L || true
  write_meta "ok" "cuda available"
fi

cd "$OU_ROOT"

train_one() {
  local exp=$1
  echo "[sanity] train experiment=$exp"
  $PY workspace/scripts/train.py --config-name=unlearn.yaml \
    "experiment=unlearn/$exp" \
    "trainer.args.seed=$SEED"
}

if [[ "$TRAIN" == "1" ]]; then
  # Scope: ASU only. GA/NPO are not trained in this run.
  train_one asu_tofu05
fi

if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "[sanity] SKIP_EVAL=1 — aggregate only if json present"
  $PY "$WS/scripts/aggregate_asu_leakru.py" --notes "$META"
  exit 0
fi

# --- θ_p ---
echo "[sanity] eval θ_p"
$PY workspace/scripts/eval.py --config-name=eval.yaml \
  experiment=eval/leakru_tofu05_theta_p \
  seed=$SEED \
  eval.leakru.overwrite=true

THETA_P_LOG=$OU_ROOT/workspace/saves/eval/leakru_tofu05_theta_p/LeakRU_EVAL.json
test -f "$THETA_P_LOG"

# Acc(θ_p) SH sanity
$PY - <<PY
import json, sys
from pathlib import Path
p=Path("$THETA_P_LOG")
logs=json.loads(p.read_text())
acc=logs.get("leakru_acc") or {}
vbi=acc.get("value_by_index") or {}
sh=[bool(r.get("correct")) for r in vbi.values() if r.get("split")=="single-hop"]
rate=(sum(sh)/len(sh)) if sh else 0.0
print(f"[sanity] Acc(θ_p) SH={rate:.3f} n={len(sh)} agg={acc.get('agg_value')}")
if not sh or rate <= 0.0:
    print("[sanity] ERROR: Acc(θ_p) on SH is zero/empty; FQ undefined. Abort.")
    sys.exit(3)
PY

eval_u() {
  local name=$1
  local ckpt=$2
  local out=leakru_tofu05_${name}
  echo "[sanity] eval θ_u=$name ckpt=$ckpt"
  $PY workspace/scripts/eval.py --config-name=eval.yaml \
    experiment=eval/leakru_tofu05_theta_u \
    seed=$SEED \
    task_name=$out \
    paths.output_dir=workspace/saves/eval/$out \
    model.model_args.pretrained_model_name_or_path="$ckpt" \
    eval.leakru.pretrained_logs_path=$THETA_P_LOG \
    eval.leakru.overwrite=true
}

# Resolve checkpoints
GA_CKPT=${GA_CKPT:-$OU_ROOT/workspace/saves/unlearn/ga_tofu05_seed1001}
NPO_CKPT=${NPO_CKPT:-$OU_ROOT/workspace/saves/unlearn/npo_tofu05_seed1001}
ASU_CKPT=${ASU_CKPT:-}
if [[ "$REUSE_ASU" == "1" ]]; then
  ASU_CKPT=$ASU_CKPT_REUSE
else
  ASU_CKPT=${ASU_CKPT:-$OU_ROOT/workspace/saves/unlearn/asu_tofu05_seed1001}
fi

# HF Trainer may nest checkpoint-* ; pick latest if directory has them
resolve_ckpt() {
  local d=$1
  if [[ -f "$d/config.json" ]]; then
    echo "$d"
    return
  fi
  local latest
  latest=$(ls -d "$d"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
  if [[ -n "${latest:-}" ]]; then
    echo "$latest"
    return
  fi
  echo "$d"
}

ASU_CKPT=$(resolve_ckpt "$ASU_CKPT")
if [[ -d "$GA_CKPT" ]]; then GA_CKPT=$(resolve_ckpt "$GA_CKPT"); fi
if [[ -d "$NPO_CKPT" ]]; then NPO_CKPT=$(resolve_ckpt "$NPO_CKPT"); fi

eval_u asu "$ASU_CKPT"
if [[ -d "$GA_CKPT" || -f "$GA_CKPT/config.json" ]]; then
  eval_u ga "$GA_CKPT"
else
  echo "[sanity] WARN: GA ckpt missing at $GA_CKPT — skip"
fi
if [[ -d "$NPO_CKPT" || -f "$NPO_CKPT/config.json" ]]; then
  eval_u npo "$NPO_CKPT"
else
  echo "[sanity] WARN: NPO ckpt missing at $NPO_CKPT — skip"
fi

$PY "$WS/scripts/aggregate_asu_leakru.py" --notes "$META"
echo "[sanity] done → $WS/results/asu_leakru_table.md , $WS/results/H_verdict.md"

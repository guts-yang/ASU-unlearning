#!/usr/bin/env bash
# ASU × LeakRU sanity check. Seed 1001. ASU checkpoint is reused.
# GA (GradDiff) and NPO are trained, then matched on FQ_SH only.
set -euo pipefail

SEED=1001
ASU_ROOT=/usr/local/ASU-unlearning
OU_ROOT=/usr/local/open-unlearning
WS=$ASU_ROOT/workspace
DATA_JSONL=$WS/data/leakru/tofu_forget05.jsonl
LOG_DIR=$WS/notes/logs
SAVES=$OU_ROOT/workspace/saves
ASU_CKPT=${ASU_CKPT:-/root/autodl-tmp/saves/workspace/unlearn/asu_tofu05_seed1001}
THETA_P_DIR=$SAVES/eval/leakru_tofu05_theta_p
THETA_P_LOG=$THETA_P_DIR/LeakRU_EVAL.json
META=$WS/notes/run_meta.json
FQ=$WS/scripts/fq_sh.py

if [[ -n "${OMP_NUM_THREADS:-}" && ! "${OMP_NUM_THREADS}" =~ ^[0-9]+$ ]]; then
  export OMP_NUM_THREADS=8
fi
export TOKENIZERS_PARALLELISM=false

if [[ -f /root/autodl-tmp/env_hf.sh ]]; then
  # shellcheck disable=SC1091
  source /root/autodl-tmp/env_hf.sh
fi
if [[ -x /root/autodl-tmp/envs/unlearning/bin/python ]]; then
  PY=/root/autodl-tmp/envs/unlearning/bin/python
else
  PY=python3
fi
export PYTHONPATH="$OU_ROOT/src:$OU_ROOT/workspace/src:$WS/src:${PYTHONPATH:-}"

mkdir -p "$LOG_DIR" "$WS/notes" "$WS/results" "$SAVES/eval" "$SAVES/unlearn"

echo "[sanity] build LeakRU probes"
$PY "$WS/scripts/build_tofu_leakru.py" --seed "$SEED"
test -f "$DATA_JSONL"

if ! $PY - <<'PY'
import torch, sys
sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 1)
PY
then
  echo "[sanity] ERROR: CUDA GPU not available."
  exit 2
fi
nvidia-smi -L || true

resolve_ckpt() {
  local d=$1
  if [[ -f "$d/config.json" ]]; then
    echo "$d"
    return
  fi
  local latest
  latest=$(find "$d" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1 || true)
  if [[ -n "${latest:-}" && -f "$latest/config.json" ]]; then
    echo "$latest"
    return
  fi
  echo "$d"
}

cd "$OU_ROOT"

wait_ok() {
  local pid=$1
  local log=$2
  local status=0
  wait "$pid" || status=$?
  if [[ "$status" -ne 0 ]]; then
    echo "[sanity] ERROR: pid $pid failed. Tail of $log:"
    tail -n 40 "$log" || true
    exit "$status"
  fi
}

train_cmd() {
  local task=$1
  shift
  $PY workspace/scripts/train.py --config-name=unlearn.yaml \
    trainer.args.seed="$SEED" \
    task_name="$task" \
    "paths.output_dir=workspace/saves/unlearn/$task" \
    "$@"
}

echo "[sanity] train GA and NPO if checkpoints are missing"
GA_TASK=ga_tofu05_seed1001
NPO_TASK=npo_tofu05_seed1001
GA_DIR=$SAVES/unlearn/$GA_TASK
NPO_DIR=$SAVES/unlearn/$NPO_TASK
pid_ga=""
pid_npo=""
if [[ ! -f "$GA_DIR/config.json" ]]; then
  echo "[sanity] GPU0 train $GA_TASK"
  CUDA_VISIBLE_DEVICES=0 train_cmd "$GA_TASK" experiment=unlearn/ga_tofu05 \
    >"$LOG_DIR/${GA_TASK}.log" 2>&1 &
  pid_ga=$!
else
  echo "[sanity] reuse $GA_DIR"
fi
if [[ ! -f "$NPO_DIR/config.json" ]]; then
  echo "[sanity] GPU1 train $NPO_TASK"
  CUDA_VISIBLE_DEVICES=1 train_cmd "$NPO_TASK" experiment=unlearn/npo_tofu05 \
    >"$LOG_DIR/${NPO_TASK}.log" 2>&1 &
  pid_npo=$!
else
  echo "[sanity] reuse $NPO_DIR"
fi
if [[ -n "$pid_ga" ]]; then
  wait_ok "$pid_ga" "$LOG_DIR/${GA_TASK}.log"
fi
if [[ -n "$pid_npo" ]]; then
  wait_ok "$pid_npo" "$LOG_DIR/${NPO_TASK}.log"
fi

echo "[sanity] eval θ_p"
if [[ ! -f "$THETA_P_LOG" ]]; then
  CUDA_VISIBLE_DEVICES=0 $PY workspace/scripts/eval.py --config-name=eval.yaml \
    experiment=eval/leakru_tofu05_theta_p \
    seed="$SEED" \
    eval.leakru.overwrite=true \
    >"$LOG_DIR/theta_p.log" 2>&1 || {
      tail -n 40 "$LOG_DIR/theta_p.log" || true
      exit 1
    }
fi
test -f "$THETA_P_LOG"
ACC_P=$($PY "$FQ" acc "$THETA_P_LOG")
echo "[sanity] Acc_SH(θ_p)=$ACC_P"
$PY - "$ACC_P" <<'PY'
import sys
acc = float(sys.argv[1])
if acc < 0.5:
    raise SystemExit(f"Acc_SH(θ_p)={acc:.3f} < 0.5; refuse to interpret FQ")
PY

eval_acc() {
  local gpu=$1
  local name=$2
  local ckpt=$3
  local out=leakru_tofu05_${name}_acc
  local log=$LOG_DIR/${out}.log
  echo "[sanity] acc-eval $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY workspace/scripts/eval.py --config-name=eval.yaml \
    experiment=eval/leakru_tofu05_theta_u_acc \
    seed="$SEED" \
    task_name="$out" \
    "paths.output_dir=workspace/saves/eval/$out" \
    "model.model_args.pretrained_model_name_or_path=$ckpt" \
    "eval.leakru.pretrained_logs_path=$THETA_P_LOG" \
    eval.leakru.overwrite=true \
    >"$log" 2>&1 || {
      echo "[sanity] ERROR: acc-eval $name failed"
      tail -n 40 "$log" || true
      exit 1
    }
  test -f "$SAVES/eval/$out/LeakRU_EVAL.json"
}

eval_full() {
  local gpu=$1
  local name=$2
  local ckpt=$3
  local out=leakru_tofu05_${name}
  local log=$LOG_DIR/${out}.log
  echo "[sanity] full-eval $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY workspace/scripts/eval.py --config-name=eval.yaml \
    experiment=eval/leakru_tofu05_theta_u \
    seed="$SEED" \
    task_name="$out" \
    "paths.output_dir=workspace/saves/eval/$out" \
    "model.model_args.pretrained_model_name_or_path=$ckpt" \
    "eval.leakru.pretrained_logs_path=$THETA_P_LOG" \
    eval.leakru.overwrite=true \
    >"$log" 2>&1 || {
      echo "[sanity] ERROR: full-eval $name failed"
      tail -n 40 "$log" || true
      exit 1
    }
  test -f "$SAVES/eval/$out/LeakRU_EVAL.json"
}

ASU_CKPT=$(resolve_ckpt "$ASU_CKPT")
GA_CKPT=$(resolve_ckpt "$GA_DIR")
NPO_CKPT=$(resolve_ckpt "$NPO_DIR")
test -f "$ASU_CKPT/config.json"
test -f "$GA_CKPT/config.json"
test -f "$NPO_CKPT/config.json"

eval_acc 0 asu "$ASU_CKPT" &
pid_a=$!
eval_acc 1 ga "$GA_CKPT" &
pid_g=$!
wait_ok "$pid_a" "$LOG_DIR/leakru_tofu05_asu_acc.log"
wait_ok "$pid_g" "$LOG_DIR/leakru_tofu05_ga_acc.log"
eval_acc 0 npo "$NPO_CKPT"

ASU_ACC_JSON=$SAVES/eval/leakru_tofu05_asu_acc/LeakRU_EVAL.json
GA_ACC_JSON=$SAVES/eval/leakru_tofu05_ga_acc/LeakRU_EVAL.json
NPO_ACC_JSON=$SAVES/eval/leakru_tofu05_npo_acc/LeakRU_EVAL.json

needs_sweep() {
  local unlearn_json=$1
  $PY - "$FQ" "$unlearn_json" "$ASU_ACC_JSON" "$THETA_P_LOG" <<'PY'
import subprocess, sys
fq_py, unlearn, asu, theta = sys.argv[1:]
def grab(*args):
    return float(subprocess.check_output([sys.executable, fq_py, *args], text=True).strip())
gap = abs(grab("fq", unlearn, theta) - grab("fq", asu, theta)) * 100.0
print(f"{gap:.3f}", file=sys.stderr)
sys.exit(0 if gap <= 10.0 else 1)
PY
}

GA_LR=1e-5
NPO_BETA=0.1
NPO_LR=1e-5
GA_CANDS=("default=$GA_ACC_JSON")
NPO_CANDS=("default=$NPO_ACC_JSON")

if needs_sweep "$GA_ACC_JSON"; then
  echo "[sanity] GA FQ_SH already within 10pp of ASU"
else
  echo "[sanity] GA FQ_SH gap > 10pp; sweep lr 5e-6 and 2e-5"
  CUDA_VISIBLE_DEVICES=0 train_cmd ga_tofu05_seed1001_lr5e-06 \
    experiment=unlearn/ga_tofu05 trainer.args.learning_rate=5e-6 \
    >"$LOG_DIR/ga_lr5e-6.log" 2>&1 &
  p1=$!
  CUDA_VISIBLE_DEVICES=1 train_cmd ga_tofu05_seed1001_lr2e-05 \
    experiment=unlearn/ga_tofu05 trainer.args.learning_rate=2e-5 \
    >"$LOG_DIR/ga_lr2e-5.log" 2>&1 &
  p2=$!
  wait_ok "$p1" "$LOG_DIR/ga_lr5e-6.log"
  wait_ok "$p2" "$LOG_DIR/ga_lr2e-5.log"
  CKPT_A=$(resolve_ckpt "$SAVES/unlearn/ga_tofu05_seed1001_lr5e-06")
  CKPT_B=$(resolve_ckpt "$SAVES/unlearn/ga_tofu05_seed1001_lr2e-05")
  eval_acc 0 ga_lr5e-06 "$CKPT_A" &
  pid_a=$!
  eval_acc 1 ga_lr2e-05 "$CKPT_B" &
  pid_b=$!
  wait_ok "$pid_a" "$LOG_DIR/leakru_tofu05_ga_lr5e-06_acc.log"
  wait_ok "$pid_b" "$LOG_DIR/leakru_tofu05_ga_lr2e-05_acc.log"
  GA_CANDS+=("lr5e-06=$SAVES/eval/leakru_tofu05_ga_lr5e-06_acc/LeakRU_EVAL.json")
  GA_CANDS+=("lr2e-05=$SAVES/eval/leakru_tofu05_ga_lr2e-05_acc/LeakRU_EVAL.json")
fi

# Keep only the GA checkpoint closest on FQ_SH so the NPO sweep fits on disk.
GA_PICK_EARLY=$($PY "$FQ" closest "$ASU_ACC_JSON" "$THETA_P_LOG" "${GA_CANDS[@]}")
GA_KEEP_NAME=$(printf '%s' "$GA_PICK_EARLY" | $PY -c 'import json,sys; print(json.load(sys.stdin)["chosen"])')
case "$GA_KEEP_NAME" in
  default) GA_KEEP=$GA_CKPT ;;
  lr5e-06) GA_KEEP=$(resolve_ckpt "$SAVES/unlearn/ga_tofu05_seed1001_lr5e-06") ;;
  lr2e-05) GA_KEEP=$(resolve_ckpt "$SAVES/unlearn/ga_tofu05_seed1001_lr2e-05") ;;
esac
for d in "$GA_DIR" "$SAVES/unlearn/ga_tofu05_seed1001_lr5e-06" "$SAVES/unlearn/ga_tofu05_seed1001_lr2e-05"; do
  if [[ -d "$d" && "$(readlink -f "$d")" != "$(readlink -f "$GA_KEEP")" ]]; then
    echo "[sanity] remove unused GA weights $d"
    rm -rf "$d"
  fi
done

if needs_sweep "$NPO_ACC_JSON"; then
  echo "[sanity] NPO FQ_SH already within 10pp of ASU"
else
  echo "[sanity] NPO FQ_SH gap > 10pp; sweep beta 0.05 and 0.2"
  CUDA_VISIBLE_DEVICES=0 train_cmd npo_tofu05_seed1001_beta0.05 \
    experiment=unlearn/npo_tofu05 trainer.method_args.beta=0.05 \
    >"$LOG_DIR/npo_beta0.05.log" 2>&1 &
  p1=$!
  CUDA_VISIBLE_DEVICES=1 train_cmd npo_tofu05_seed1001_beta0.2 \
    experiment=unlearn/npo_tofu05 trainer.method_args.beta=0.2 \
    >"$LOG_DIR/npo_beta0.2.log" 2>&1 &
  p2=$!
  wait_ok "$p1" "$LOG_DIR/npo_beta0.05.log"
  wait_ok "$p2" "$LOG_DIR/npo_beta0.2.log"
  CKPT_A=$(resolve_ckpt "$SAVES/unlearn/npo_tofu05_seed1001_beta0.05")
  CKPT_B=$(resolve_ckpt "$SAVES/unlearn/npo_tofu05_seed1001_beta0.2")
  eval_acc 0 npo_beta0.05 "$CKPT_A" &
  pid_a=$!
  eval_acc 1 npo_beta0.2 "$CKPT_B" &
  pid_b=$!
  wait_ok "$pid_a" "$LOG_DIR/leakru_tofu05_npo_beta0.05_acc.log"
  wait_ok "$pid_b" "$LOG_DIR/leakru_tofu05_npo_beta0.2_acc.log"
  NPO_CANDS+=("beta0.05=$SAVES/eval/leakru_tofu05_npo_beta0.05_acc/LeakRU_EVAL.json")
  NPO_CANDS+=("beta0.2=$SAVES/eval/leakru_tofu05_npo_beta0.2_acc/LeakRU_EVAL.json")
fi

GA_PICK=$($PY "$FQ" closest "$ASU_ACC_JSON" "$THETA_P_LOG" "${GA_CANDS[@]}")
NPO_PICK=$($PY "$FQ" closest "$ASU_ACC_JSON" "$THETA_P_LOG" "${NPO_CANDS[@]}")
echo "[sanity] GA pick $GA_PICK"
echo "[sanity] NPO pick $NPO_PICK"

map_ga_ckpt() {
  local name=$1
  case "$name" in
    default) echo "$GA_CKPT" ;;
    lr5e-06) resolve_ckpt "$SAVES/unlearn/ga_tofu05_seed1001_lr5e-06" ;;
    lr2e-05) resolve_ckpt "$SAVES/unlearn/ga_tofu05_seed1001_lr2e-05" ;;
    *) echo "unknown GA candidate $name" >&2; exit 1 ;;
  esac
}
map_npo_ckpt() {
  local name=$1
  case "$name" in
    default) echo "$NPO_CKPT" ;;
    beta0.05) resolve_ckpt "$SAVES/unlearn/npo_tofu05_seed1001_beta0.05" ;;
    beta0.2) resolve_ckpt "$SAVES/unlearn/npo_tofu05_seed1001_beta0.2" ;;
    *) echo "unknown NPO candidate $name" >&2; exit 1 ;;
  esac
}

GA_NAME=$(printf '%s' "$GA_PICK" | $PY -c 'import json,sys; print(json.load(sys.stdin)["chosen"])')
NPO_NAME=$(printf '%s' "$NPO_PICK" | $PY -c 'import json,sys; print(json.load(sys.stdin)["chosen"])')
GA_MATCH=$(printf '%s' "$GA_PICK" | $PY -c 'import json,sys; print("1" if json.load(sys.stdin)["matched"] else "0")')
NPO_MATCH=$(printf '%s' "$NPO_PICK" | $PY -c 'import json,sys; print("1" if json.load(sys.stdin)["matched"] else "0")')
GA_FQ=$(printf '%s' "$GA_PICK" | $PY -c 'import json,sys; print(json.load(sys.stdin)["chosen_fq_sh"])')
NPO_FQ=$(printf '%s' "$NPO_PICK" | $PY -c 'import json,sys; print(json.load(sys.stdin)["chosen_fq_sh"])')
ASU_FQ=$(printf '%s' "$GA_PICK" | $PY -c 'import json,sys; print(json.load(sys.stdin)["asu_fq_sh"])')
GA_FINAL=$(map_ga_ckpt "$GA_NAME")
NPO_FINAL=$(map_npo_ckpt "$NPO_NAME")
for d in "$NPO_DIR" "$SAVES/unlearn/npo_tofu05_seed1001_beta0.05" "$SAVES/unlearn/npo_tofu05_seed1001_beta0.2"; do
  if [[ -d "$d" && "$(readlink -f "$d")" != "$(readlink -f "$NPO_FINAL")" ]]; then
    echo "[sanity] remove unused NPO weights $d"
    rm -rf "$d"
  fi
done
case "$GA_NAME" in
  lr5e-06) GA_LR=5e-6 ;;
  lr2e-05) GA_LR=2e-5 ;;
esac
case "$NPO_NAME" in
  beta0.05) NPO_BETA=0.05 ;;
  beta0.2) NPO_BETA=0.2 ;;
esac
if [[ "$GA_MATCH" == "1" && "$NPO_MATCH" == "1" ]]; then
  CONTROLS_MATCHED=true
else
  CONTROLS_MATCHED=false
fi

$PY - "$META" <<PY
import json
meta = {
  "seed": $SEED,
  "forget_split": "forget05",
  "retain_split": "retain95",
  "model_theta_p": "open-unlearning/tofu_Llama-2-7b-chat-hf_full",
  "theta_p_tofu_forget05_rouge": 0.9625446736275544,
  "theta_p_tofu_exact_memorization": 0.9987873476743698,
  "acc_sh_theta_p": float("$ACC_P"),
  "asu_tau": 2.3,
  "asu_alpha": 0.1,
  "asu_retain": "NLL",
  "epochs": 5,
  "lr": 1e-5,
  "asu_steps": 60,
  "asu_epoch": 4.64,
  "asu_train_loss": 3.661335245768229,
  "asu_ckpt": "$ASU_CKPT",
  "ga_definition": "OpenUnlearning GradDiff: forget gradient ascent + retain NLL",
  "ga_ckpt": "$GA_FINAL",
  "ga_candidate": "$GA_NAME",
  "ga_lr": float("$GA_LR"),
  "ga_fq_sh": float("$GA_FQ"),
  "npo_ckpt": "$NPO_FINAL",
  "npo_candidate": "$NPO_NAME",
  "npo_beta": float("$NPO_BETA"),
  "npo_lr": float("$NPO_LR"),
  "npo_fq_sh": float("$NPO_FQ"),
  "asu_fq_sh": float("$ASU_FQ"),
  "controls_matched": "$CONTROLS_MATCHED" == "true",
  "fq_match_rule": "select checkpoint with FQ_SH closest to ASU; sweep only if default gap > 10pp; do not use DS/MT/RR",
  "data": "$DATA_JSONL",
  "attacks": ["probab", "focus_on_key"],
  "quantization": False,
  "probab": {"temperature": 0.8, "top_p": 0.95, "trials": 5},
  "focus_on_key_repeats": 3,
  "gpu_status": "ok",
  "gpu_detail": "2x A800 80GB",
}
open("$META","w",encoding="utf-8").write(json.dumps(meta, indent=2, ensure_ascii=False)+"\n")
PY

echo "[sanity] full LeakRU eval (Probab + FocusOnKey)"
eval_full 0 asu "$ASU_CKPT" &
pid_a=$!
eval_full 1 ga "$GA_FINAL" &
pid_g=$!
wait_ok "$pid_a" "$LOG_DIR/leakru_tofu05_asu.log"
wait_ok "$pid_g" "$LOG_DIR/leakru_tofu05_ga.log"
eval_full 0 npo "$NPO_FINAL"

$PY "$WS/scripts/aggregate_asu_leakru.py" --notes "$META"
echo "[sanity] done → $WS/results/asu_leakru_table.md , $WS/results/H_verdict.md"

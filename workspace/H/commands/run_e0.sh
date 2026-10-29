#!/usr/bin/env bash
set -euo pipefail

ROOT=/usr/local/ASU-unlearning
CONFIG="$ROOT/workspace/H/configs/e0_probe.yaml"
OUT=/root/autodl-tmp/ASU-unlearning/workspace_H/e0
LOG="$ROOT/workspace/H/logs"
mkdir -p "$OUT/forwards" "$LOG"

export HF_HOME=/root/autodl-tmp/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python3 "$ROOT/workspace/H/commands/build_e0_manifest.py"

python3 - <<'PY'
import json
from pathlib import Path
import yaml
config = yaml.safe_load(Path("/usr/local/ASU-unlearning/workspace/H/configs/e0_probe.yaml").read_text())
out = Path(config["output_dir"])
jobs = [{"name": "baseline_tau1", "attention_temp": 1.0, "layers_id": None}]
gpu0, gpu1 = [], []
for layer in range(int(config["layer_count"])):
    bucket = gpu0 if layer < 16 else gpu1
    for temperature in config["temperatures"]:
        label = str(temperature).replace(".", "p")
        bucket.append({
            "name": f"single_L{layer:02d}_tau{label}",
            "attention_temp": float(temperature),
            "layers_id": [layer],
        })
for temperature in config["temperatures"]:
    label = str(temperature).replace(".", "p")
    gpu1.append({"name": f"control_L6_8_tau{label}", "attention_temp": float(temperature), "layers_id": [6, 7, 8]})
    gpu1.append({"name": f"control_all_tau{label}", "attention_temp": float(temperature), "layers_id": None})
gpu0 = jobs + gpu0
(out / "jobs_gpu0.json").write_text(json.dumps(gpu0, indent=2) + "\n")
(out / "jobs_gpu1.json").write_text(json.dumps(gpu1, indent=2) + "\n")
print(f"gpu0={len(gpu0)} gpu1={len(gpu1)}")
PY

python3 "$ROOT/workspace/H/commands/run_e0_forward.py" --config "$CONFIG" --jobs "$OUT/jobs_gpu0.json" --gpu 0 --smoke > "$LOG/e0_gpu0.log" 2>&1 &
pid0=$!
python3 "$ROOT/workspace/H/commands/run_e0_forward.py" --config "$CONFIG" --jobs "$OUT/jobs_gpu1.json" --gpu 1 > "$LOG/e0_gpu1.log" 2>&1 &
pid1=$!
set +e
wait "$pid0"
status0=$?
if [[ "$status0" -ne 0 ]]; then
  kill "$pid1" 2>/dev/null || true
  wait "$pid1" 2>/dev/null || true
  exit "$status0"
fi
wait "$pid1"
status1=$?
set -e
if [[ "$status1" -ne 0 ]]; then
  exit "$status1"
fi

python3 "$ROOT/workspace/H/commands/analyze_e0.py" --config "$CONFIG" --phase select

mapfile -t phase2 < <(python3 - <<'PY'
import json
from pathlib import Path
jobs = json.loads(Path("/root/autodl-tmp/ASU-unlearning/workspace_H/e0/phase2_jobs.json").read_text())
mid = (len(jobs) + 1) // 2
out = Path("/root/autodl-tmp/ASU-unlearning/workspace_H/e0")
(out / "jobs_phase2_gpu0.json").write_text(json.dumps(jobs[:mid], indent=2) + "\n")
(out / "jobs_phase2_gpu1.json").write_text(json.dumps(jobs[mid:], indent=2) + "\n")
print(len(jobs[:mid]))
print(len(jobs[mid:]))
PY
)
if [[ "${phase2[0]}" != "0" ]]; then
  python3 "$ROOT/workspace/H/commands/run_e0_forward.py" --config "$CONFIG" --jobs "$OUT/jobs_phase2_gpu0.json" --gpu 0 > "$LOG/e0_phase2_gpu0.log" 2>&1 &
  pid0=$!
  if [[ "${phase2[1]}" != "0" ]]; then
    python3 "$ROOT/workspace/H/commands/run_e0_forward.py" --config "$CONFIG" --jobs "$OUT/jobs_phase2_gpu1.json" --gpu 1 > "$LOG/e0_phase2_gpu1.log" 2>&1 &
    pid1=$!
  else
    pid1=""
  fi
  wait "$pid0"
  if [[ -n "$pid1" ]]; then
    wait "$pid1"
  fi
fi

python3 "$ROOT/workspace/H/commands/analyze_e0.py" --config "$CONFIG" --phase gate

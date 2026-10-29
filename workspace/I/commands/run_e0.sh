#!/usr/bin/env bash
# Teacher 2x2 forwards. Requires CUDA. Does not edit files outside workspace/I outputs.
set -euo pipefail

ROOT=/usr/local/ASU-unlearning
CONFIG="$ROOT/workspace/I/configs/e0_probe.yaml"
OUT=/root/autodl-tmp/ASU-unlearning/workspace_I/e0
LOG="$ROOT/workspace/I/logs"
mkdir -p "$OUT/forwards" "$LOG"

export HF_HOME=/root/autodl-tmp/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python3 "$ROOT/workspace/I/commands/build_e0_manifest.py"

python3 - <<'PY'
import json
from pathlib import Path
import yaml
config = yaml.safe_load(Path("/usr/local/ASU-unlearning/workspace/I/configs/e0_probe.yaml").read_text())
out = Path(config["output_dir"])
jobs = [{"name": "T00", "attention_temp": 1.0, "mlp_gate_temp": 1.0}]
for temperature in config["attention_temperature_grid"]:
    label = str(temperature).replace(".", "p")
    jobs.append({"name": f"T10_tau{label}", "attention_temp": float(temperature), "mlp_gate_temp": 1.0})
for temperature in config["mlp_temperature_grid"]:
    label = str(temperature).replace(".", "p")
    jobs.append({"name": f"T01_tau{label}", "attention_temp": 1.0, "mlp_gate_temp": float(temperature)})
(out / "jobs_grid.json").write_text(json.dumps(jobs, indent=2) + "\n")
print(f"jobs={len(jobs)}")
PY

if ! python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "CUDA unavailable. Grid forwards were not started." | tee "$LOG/e0_blocked.txt"
  exit 2
fi

python3 "$ROOT/workspace/I/commands/run_e0_forward.py" \
  --config "$CONFIG" --jobs "$OUT/jobs_grid.json" --gpu 0 \
  > "$LOG/e0_forward.log" 2>&1

python3 "$ROOT/workspace/I/commands/analyze_e0.py" --config "$CONFIG" --phase select

python3 - <<'PY'
import json
from pathlib import Path
import yaml
config = yaml.safe_load(Path("/usr/local/ASU-unlearning/workspace/I/configs/e0_probe.yaml").read_text())
selection = json.loads(Path(config["output_dir"], "e0_selection.json").read_text())
jobs = []
if selection.get("intensity_matched") and selection.get("tau_A") is not None and selection.get("tau_M") is not None:
    jobs.append({
        "name": "T11",
        "attention_temp": float(selection["tau_A"]),
        "mlp_gate_temp": float(selection["tau_M"]),
    })
Path(config["output_dir"], "jobs_t11.json").write_text(json.dumps(jobs, indent=2) + "\n")
print(f"t11_jobs={len(jobs)}")
PY

if [[ -s "$OUT/jobs_t11.json" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])) else 1)" "$OUT/jobs_t11.json"; then
  python3 "$ROOT/workspace/I/commands/run_e0_forward.py" \
    --config "$CONFIG" --jobs "$OUT/jobs_t11.json" --gpu 0 \
    >> "$LOG/e0_forward.log" 2>&1
fi

python3 "$ROOT/workspace/I/commands/analyze_e0.py" --config "$CONFIG" --phase gate

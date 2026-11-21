#!/usr/bin/env bash
# Batch RTF FE/MU evaluation for representative QKSpec checkpoints + baseline.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env_qkspec.sh"

export TOKENIZERS_PARALLELISM=false CUDA_HOME=/jizhicfs/chenyliao/ops/cuda-stub DS_SKIP_CUDA_CHECK=1
export TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=0

MODEL_TAG=Llama-3.2-1B-Instruct
RESULTS=/jizhicfs/chenyliao/runs/qkspec/rtf_results
mkdir -p "$RESULTS"

STUDENTS=(
  "/jizhicfs/chenyliao/models/open-unlearning/tofu_${MODEL_TAG}_full|baseline_full"
  "/jizhicfs/chenyliao/runs/qkspec/sweep/qkspec_rho00_k32|round1_rho00_k32"
  "/jizhicfs/chenyliao/runs/qkspec/sweep/qkspec_rho00_k32_cap10|round1_cap10"
  "/jizhicfs/chenyliao/runs/qkspec/sweep_v2_spec/qkspec_spec_w0_k16|round2_spec_w0"
  "/jizhicfs/chenyliao/runs/qkspec/sweep_v2_spec/qkspec_spec_w1_lo_k32|round2_spec_w1_lo"
)

for entry in "${STUDENTS[@]}"; do
  student=${entry%%|*}
  name=${entry##*|}
  echo "=== RTF eval: $name ($student) ==="
  bash "$(dirname "${BASH_SOURCE[0]}")/run_rtf_fe_mu.sh" "$student" 200 "rtf_$name" 2>&1 | tail -3
  cp "$student/rtf_$name/rtf_fe_mu.json" "$RESULTS/${name}_rtf.json" 2>/dev/null || true
done

echo "=== ALL RTF DONE ==="
$PY - <<'PY'
import json, glob
for p in sorted(glob.glob('/jizhicfs/chenyliao/runs/qkspec/rtf_results/*_rtf.json')):
    d=json.load(open(p))
    name=p.split('/')[-1].replace('_rtf.json','')
    print(f"{name:22} FE={d.get('Forget Efficacy'):.4f}  MU={d.get('Model Utility'):.4f}  MU_R={d.get('Model Utility Retain'):.4f}  FProb={d.get('Forget Probability'):.4f}")
PY

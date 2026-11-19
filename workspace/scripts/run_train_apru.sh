#!/usr/bin/env bash
# Train ASU on TOFU forget05 with the local GPU-capable runtime.
set -euo pipefail

source /usr/local/ASU-unlearning-RIU/workspace/scripts/env_apru.sh

mkdir -p "$OUTPUT_ROOT/unlearn" "$OUTPUT_ROOT/logs"

cd "$OU_ROOT"
exec "$PY" "$WS/scripts/train_apru.py" \
  --config-name=unlearn.yaml \
  "hydra.searchpath=[$HYDRA_SEARCHPATH]" \
  experiment=unlearn/asu_tofu05 \
  model.model_args.pretrained_model_name_or_path=/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-2-7b-chat-hf_full \
  model.tokenizer_args.pretrained_model_name_or_path=/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-2-7b-chat-hf_full \
  data.forget.TOFU_QA_forget.args.hf_args.path=json \
  data.forget.TOFU_QA_forget.args.hf_args.name=null \
  "+data.forget.TOFU_QA_forget.args.hf_args.data_files=/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/forget05.json" \
  data.retain.TOFU_QA_retain.args.hf_args.path=json \
  data.retain.TOFU_QA_retain.args.hf_args.name=null \
  "+data.retain.TOFU_QA_retain.args.hf_args.data_files=/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/retain95.json" \
  trainer.args.seed=1001 \
  task_name=asu_tofu05_seed1001 \
  paths.output_dir="$OUTPUT_ROOT/unlearn/asu_tofu05_seed1001" \
  retain_logs_path=/jizhicfs/chenyliao/data/datasets/open-unlearning/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json \
  "$@"

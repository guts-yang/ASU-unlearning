#!/usr/bin/env bash
# Preflight for ASU-only TOFU forget05 training. Does not start training or eval.
set -euo pipefail

ASU_ROOT=/usr/local/ASU-unlearning
OU_ROOT=/usr/local/open-unlearning
WS=$ASU_ROOT/workspace

if [[ -f /root/autodl-tmp/env_hf.sh ]]; then
  # shellcheck disable=SC1091
  source /root/autodl-tmp/env_hf.sh
fi
PY=/root/autodl-tmp/envs/unlearning/bin/python
export PYTHONPATH="$OU_ROOT/src:$OU_ROOT/workspace/src:$WS/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[prepare] rebuild LeakRU protocol jsonl"
$PY "$WS/scripts/build_tofu_leakru.py" --seed 1001

echo "[prepare] offline checks (model, TOFU splits, ASU registration, hydra)"
$PY - <<'PY'
import json, sys
from pathlib import Path

import torch
from datasets import load_dataset
from hydra import compose, initialize_config_dir
from transformers import AutoConfig, AutoTokenizer

OU = Path("/usr/local/open-unlearning")
sys.path[:0] = [str(OU / "src"), str(OU / "workspace" / "src")]
from open_unlearning_plugins.register import register_workspace_trainers

trainers = register_workspace_trainers()
assert "ASU" in trainers, trainers

mid = "open-unlearning/tofu_Llama-2-7b-chat-hf_full"
cfg = AutoConfig.from_pretrained(mid)
tok = AutoTokenizer.from_pretrained(mid)
assert cfg.num_hidden_layers == 32
assert tok.vocab_size == 32000

forget = load_dataset("locuslab/TOFU", "forget05")
retain = load_dataset("locuslab/TOFU", "retain95")
assert len(forget["train"]) == 200, len(forget["train"])
assert len(retain["train"]) == 3800, len(retain["train"])

data = Path("/usr/local/ASU-unlearning/workspace/data/leakru/tofu_forget05.jsonl")
rows = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
assert len(rows) == 260
assert sum(r["split"] == "single-hop" for r in rows) == 200
assert sum(r["logic_type"] == "DS" and r["split"] == "multi-hop" for r in rows) == 10
assert sum(r["logic_type"] == "MT" and r["split"] == "multi-hop" for r in rows) == 10

retain_logs = Path("/root/autodl-tmp/saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json")
assert retain_logs.is_file(), retain_logs

cfg_dir = str((OU / "configs").resolve())
sp = "hydra.searchpath=[file://" + str((OU / "workspace" / "configs").resolve()) + "]"
with initialize_config_dir(config_dir=cfg_dir, version_base=None):
    exp = compose(
        config_name="unlearn.yaml",
        overrides=[sp, "experiment=unlearn/asu_tofu05"],
    )
assert exp.trainer.handler == "ASU"
assert float(exp.trainer.method_args.attention_temp) == 2.3
assert float(exp.trainer.method_args.alpha) == 0.1
assert int(exp.trainer.args.num_train_epochs) == 5
assert float(exp.trainer.args.learning_rate) == 1e-5
assert int(exp.trainer.args.seed) == 1001
assert exp.forget_split == "forget05"
assert exp.retain_split == "retain95"
assert "asu_tofu05" in str(exp.paths.output_dir)

print("cuda_available", torch.cuda.is_available())
print("prepare_ok")
PY

mkdir -p "$WS/notes"
cat > "$WS/notes/ready_asu.md" <<'EOF'
# ASU-only 训练前就绪

未启动训练。GPU 挂上后只跑 ASU。

## 已核对

- 环境：`/root/autodl-tmp/envs/unlearning`（torch 2.8.0+cu128, transformers 4.51.3, peft 0.14.0, deepspeed 0.15.4）
- 起点权重（离线可读）：`open-unlearning/tofu_Llama-2-7b-chat-hf_full`（Llama-2-7B, 32 层, 3 个 safetensors）
- 数据（离线可读）：TOFU `forget05` 200 条、`retain95` 3800 条
- retain 评测日志：`/root/autodl-tmp/saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json`
- LeakRU 协议集：`workspace/data/leakru/tofu_forget05.jsonl`（200 SH + 60 MH）
- 配置：`experiment=unlearn/asu_tofu05`，τ=2.3，α=0.1，5 epochs，lr=1e-5，seed=1001
- 输出目录：`/usr/local/open-unlearning/workspace/saves/unlearn/asu_tofu05_seed1001`

## GPU 就绪后只训 ASU

```bash
source /root/autodl-tmp/env_hf.sh
cd /usr/local/open-unlearning
export PYTHONPATH=src:workspace/src
/root/autodl-tmp/envs/unlearning/bin/python workspace/scripts/train.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/asu_tofu05 \
  trainer.args.seed=1001
```

本次不训练 GA / NPO。
EOF

echo "[prepare] wrote $WS/notes/ready_asu.md"
echo "[prepare] training was NOT started"

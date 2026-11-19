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

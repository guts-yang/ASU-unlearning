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

## 已完成的训练（2026-11-19）

单卡 A800。`paged_adamw_32bit` 把显存顶满且 GPU 利用率为 0；双卡 DDP 在当前 torch 上 Reducer 构造失败。实际跑通的设置：

- GPU 0，`adamw_bnb_8bit`，`per_device_train_batch_size=2`，`gradient_accumulation_steps=8`（有效 batch 16）
- 其余不变：τ=2.3，α=0.1，lr=1e-5，seed=1001，forget05/retain95，60 steps，`train_loss=3.661`
- 权重：`/root/autodl-tmp/saves/workspace/unlearn/asu_tofu05_seed1001`（最终权重在目录根；`checkpoint-60` 为最后一步）
- 日志：`/root/autodl-tmp/saves/workspace/unlearn/asu_tofu05_seed1001.train.log`

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

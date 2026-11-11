# v0.3 prompt-only 激活提取

## 结论

**通过。**

10 条配对都在 `cuda:0` 上完成了两次 prompt-only 前向。
输入只有问题 prompt，没有拼入 `y_plus` 或 `donor_fact_value`。没有打分、patch 或训练。

## 运行

- 模型类：`my_models.my_llama.LlamaForCausalLM`
- snapshot：`/root/autodl-tmp/huggingface/hub/models--open-unlearning--tofu_Llama-2-7b-chat-hf_full/snapshots/cc5b31c69127d5da881608e640f2b453c446435b`
- 设备：`NVIDIA A800 80GB PCIe`（cuda:0）
- dtype：`bfloat16`
- 注意力：`sdpa`
- hidden states：33 个（embedding 加 32 层），宽度 4096
- 主向量位置：每条 prompt 的最后一个 token
- 生成时间：`2026-11-11T09:56:28.567584+00:00`

## 逐条结果

| pair | q_plus 长度 | q_minus 长度 | a_plus | a_minus | 有限值 | 最后 token 最大绝对差 | 全序列最大绝对差 |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| promptdonor-01 | 17 | 16 | `[33, 4096]` | `[33, 4096]` | 是 | 0 | 0 |
| promptdonor-02 | 16 | 15 | `[33, 4096]` | `[33, 4096]` | 是 | 0 | 0 |
| promptdonor-03 | 17 | 14 | `[33, 4096]` | `[33, 4096]` | 是 | 0 | 0 |
| promptdonor-04 | 17 | 14 | `[33, 4096]` | `[33, 4096]` | 是 | 0 | 0 |
| promptdonor-05 | 17 | 16 | `[33, 4096]` | `[33, 4096]` | 是 | 0 | 0 |
| promptdonor-06 | 16 | 16 | `[33, 4096]` | `[33, 4096]` | 是 | 0 | 0 |
| promptdonor-07 | 18 | 19 | `[33, 4096]` | `[33, 4096]` | 是 | 0 | 0 |
| promptdonor-08 | 17 | 16 | `[33, 4096]` | `[33, 4096]` | 是 | 0 | 0 |
| promptdonor-09 | 24 | 15 | `[33, 4096]` | `[33, 4096]` | 是 | 0 | 0 |
| promptdonor-10 | 17 | 16 | `[33, 4096]` | `[33, 4096]` | 是 | 0 | 0 |

最后 token 向量的最大绝对差：`0`。
全序列 hidden states 的最大绝对差：`0`。

实体 token span 记在对应 safetensors 的 metadata 里，全序列张量可按这个 span 重切。

## 文件

- `promptdonor-01.safetensors`
- `promptdonor-02.safetensors`
- `promptdonor-03.safetensors`
- `promptdonor-04.safetensors`
- `promptdonor-05.safetensors`
- `promptdonor-06.safetensors`
- `promptdonor-07.safetensors`
- `promptdonor-08.safetensors`
- `promptdonor-09.safetensors`
- `promptdonor-10.safetensors`
- `summary.md`

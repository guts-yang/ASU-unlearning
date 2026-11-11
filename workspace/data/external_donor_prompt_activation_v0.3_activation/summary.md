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

`a_plus` 与 `a_minus` 的差是两条不同 prompt 在最后 token 上的 `|a_plus - a_minus|`，形状 `[33, 4096]`。复现差是同一条 prompt 连续两次前向的差。

| pair | q_plus 长度 | q_minus 长度 | a_plus | a_minus | 有限值 | plus−minus 最大绝对差 | plus−minus 平均绝对差 | 复现最大绝对差 |
| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| promptdonor-01 | 17 | 16 | `[33, 4096]` | `[33, 4096]` | 是 | 7.52734 | 0.296567 | 0 |
| promptdonor-02 | 16 | 15 | `[33, 4096]` | `[33, 4096]` | 是 | 8.25 | 0.321884 | 0 |
| promptdonor-03 | 17 | 14 | `[33, 4096]` | `[33, 4096]` | 是 | 11.3125 | 0.302664 | 0 |
| promptdonor-04 | 17 | 14 | `[33, 4096]` | `[33, 4096]` | 是 | 8.125 | 0.31889 | 0 |
| promptdonor-05 | 17 | 16 | `[33, 4096]` | `[33, 4096]` | 是 | 18.9375 | 0.331572 | 0 |
| promptdonor-06 | 16 | 16 | `[33, 4096]` | `[33, 4096]` | 是 | 8.47754 | 0.320108 | 0 |
| promptdonor-07 | 18 | 19 | `[33, 4096]` | `[33, 4096]` | 是 | 16.4375 | 0.30947 | 0 |
| promptdonor-08 | 17 | 16 | `[33, 4096]` | `[33, 4096]` | 是 | 8.15625 | 0.288325 | 0 |
| promptdonor-09 | 24 | 15 | `[33, 4096]` | `[33, 4096]` | 是 | 9.125 | 0.302642 | 0 |
| promptdonor-10 | 17 | 16 | `[33, 4096]` | `[33, 4096]` | 是 | 9.5625 | 0.317287 | 0 |

`a_plus` 与 `a_minus` 的最大绝对差：`18.9375`。平均绝对差范围：`0.288325` 到 `0.331572`。
复现：最后 token 与全序列 hidden states 的最大绝对差都是 `0`。

差值较大的分量多在后部状态索引 29–32；索引 0 是 embedding，索引 32 是最后一层 RMSNorm 之后。这个差只说明两条 prompt 的激活不同，不能单独证明差别来自出生地事实。

实体 token span 记在对应 safetensors 的 metadata 里，全序列张量可按这个 span 重切。

## 原始 prompt

| pair | q_plus | y_plus | q_minus | donor_fact_value |
| --- | --- | --- | --- | --- |
| promptdonor-01 | Where was Hina Ameen born? | Karachi | Where was Katherine Mansfield born? | Wellington |
| promptdonor-02 | Where was Xin Lee Williams born? | Beijing | Where was Bill Gates born? | Seattle |
| promptdonor-03 | Where was Moshe Ben-David born? | Tel Aviv | Where was Muhammad Ali born? | Louisville |
| promptdonor-04 | Where was Kalkidan Abera born? | Addis Ababa | Where was Michael Jackson born? | Gary |
| promptdonor-05 | Where was Takashi Nakamura born? | Tokyo | Where was Elvis Presley born? | Tupelo |
| promptdonor-06 | Where was Raven Marais born? | Cape Town | Where was Ada Lovelace born? | London |
| promptdonor-07 | Where was Aysha Al-Hashim born? | Manama | Where was Malala Yousafzai born? | Mingora |
| promptdonor-08 | Where was Edward Patrick Sullivan born? | New York City | Where was Paulo Coelho born? | Rio de Janeiro |
| promptdonor-09 | Where was Basil Mahfouz Al-Kuwaiti born? | Kuwait City | Where was Anne Rice born? | New Orleans |
| promptdonor-10 | Where was Nikolai Abilov born? | Astana | Where was Lionel Messi born? | Rosario |

`y_plus` 和 `donor_fact_value` 没有进入前向。它们只标明两条 prompt 各自对应的出生地。

## 原始激活

`a_plus` 和 `a_minus` 的原始数值在 [`a_plus_a_minus.jsonl`](a_plus_a_minus.jsonl)。每行一条配对，字段 `a_plus`、`a_minus` 都是 `[33, 4096]`。数值由 bfloat16 转成 float32 写出。同一向量仍保存在对应的 `promptdonor-*.safetensors`。

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
- `a_plus_a_minus.jsonl`
- `summary.md`

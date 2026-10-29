# Baseline Experiment Card — CAASS-ASU / Stage 1

## 状态与范围

- 状态：**Global ASU 多种子（1002/1003/1004）已评测。E0 已在新基座上完成，决定 `do_not_start_e1`。见 `e0_results.md`。**
- 主问题：在同一 checkpoint、TOFU split 和评测口径下，CAASS-ASU 是否可在相近 Forget Efficacy（FE）下提高 Model Utility（MU）。
- 当前主链路：`Right-to-be-forgotten`（RtBF）中的 `ASU+GD`。
- 主 split：`forget05`，`TASK_LIST=1`。代码加载 `retain50.json`，再拼上 `task_id != 1` 的 forget 行。没有 `retain95.json`。
- 次级外部验证：`forget01`、`forget10`；不与 `forget05` 合并。
- 基线：全层共享温度的 Global ASU。
- 机制假设：层特异 Teacher attention 温度会产生可区分的 forget/retain 答案 NLL 响应。

## 已冻结的代码事实（静态审计）

| 内容 | 当前代码事实 | 证据 |
| --- | --- | --- |
| Student | 训练模型由 HF Trainer 优化 | `Right-to-be-forgotten/forget.py`、`trainer/trainer.py` |
| Teacher | 独立 `LlamaForCausalLM` 加载、`eval()`，DeepSpeed 包装后 `requires_grad=False` | `forget.py`、`trainer/trainer.py` |
| Teacher 温度 | 仅 `asu_loss` 中 Teacher forward 接收 `attention_temp`、`layers_id` | `trainer/losses.py` |
| Student 温度 | Student forward 未传入温度 | `trainer/losses.py` |
| ASU+GD 损失 | `forget_coeff * masked_forget_KL + regularization_coeff * retain_NLL` | `trainer/trainer.py`、`trainer/losses.py` |
| Forget KL | logits shift 后以 `labels[:, 1:] != -100` 取答案位；逐样本均值再 batch 均值 | `trainer/losses.py` |
| Prompt/padding 标签 | prompt 与 padding 设为 `-100` | `dataset/data_module.py` |
| Forget-retain 抽样 | retain 偏移用 `torch.manual_seed(cfg.seed + idx)`，再 `randint`；`forget.py` 传入 `seed=cfg.seed` | `dataset/data_module.py`、`forget.py` |

`ASU+KL` 不属于主比较：当前实现的后定义 `kl_loss` 覆盖先前带答案 mask 的版本，产生全位置 `batchmean` KL。因此它只能在单独冻结语义后作敏感性分析。

## 训练配置（已跑）

| 字段 | 值 | 来源或状态 |
| --- | --- | --- |
| 模型 | `locuslab/tofu_ft_llama2-7b`（本地缓存；也可用 `open-unlearning/tofu_Llama-2-7b-chat-hf_full`） | `derived/stage1_checkpoint_environment.json` |
| 模型架构 | `NousResearch/Llama-2-7b-chat-hf`，`LlamaForCausalLM`，32 层，bfloat16 | checkpoint `config.json` |
| 数据 | `forget05.json` 2000 行，task 1 为 200 行；`retain50.json` 2000 行加其余 task 1800 行，有效 retain 3800 行；评测 retain 为 `retain_perturbed.json` 400 行 | `derived/stage1_data_identity.json` |
| loss | `ASU+GD` | 主基线 |
| 温度 | 全层单一 `tau=2.3` | 已跑 |
| `layers_id` | `null` | 已跑 |
| forget / retain coeff | `0.1` / `1.0` | 已跑 |
| learning rate / epochs | `1e-5` / `5` | 已跑 |
| batch × accum / max length | `4` × `4` / `500` | 已跑 |
| seed | `1002`、`1003`、`1004`；机制 `cfg.seed + sample_index` | 修复后独立 |
| optimizer / precision | `paged_adamw_32bit` / bf16 | 已跑 |
| distributed | 训练 2 GPU，评测 1 GPU | 已跑 |

## 评测与方向

- `Forget Efficacy`：越高越好。
- `Model Utility`：越高越好。
- 伴随记录：`Model Utility Retain`、Forget/Retain ROUGE、Probability、Truth Ratio、Entailment、答案 NLL、错误实体与乱码。
- 最终候选间比较以相同脚本、同一数据身份、新独立 seed 和问题/实体组重采样为准。

## 历史参考（不是本次基线）

`workspace/results/2026-09-27/tofu_summary.csv` 中 Forget05 的历史行：

| 方法 | MU | FE | 证据状态 |
| --- | ---: | ---: | --- |
| ASU+GD | 0.740182 | 0.774078 | 历史参考；该次运行无原始日志 |
| ASU+KL | 0.618851 | 0.813324 | 历史参考；且语义与主比较不同 |

M0 的 `withheld_negative_control` 不进入本实验卡的基线、效应或失败判定。

## 修复前撞车对照

修复 `manual_seed(idx)` 之前，seed 1002 与 1003 权重逐字节相同（一次确定性运行）。明细：`derived/stage1_baseline_eval.json`。

| seed | MU | FE | Model Utility Retain | 证据 |
| --- | ---: | ---: | ---: | --- |
| 1002 ≡ 1003 | 0.743566 | 0.796384 | 0.690226 | 修复前；权重相同 |
| 1001 | 0.740182 | 0.774078 | 0.683744 | 历史结果，环境不同 |

## 修复后多种子 Global ASU

机制：`torch.manual_seed(cfg.seed + sample_index)`。权重根：`/root/autodl-tmp/ASU-unlearning/workspace_H/global_asu_forget05_seeded`。明细：`derived/stage1_seeded_baseline.json`。

| seed | MU | FE | Model Utility Retain |
| --- | ---: | ---: | ---: |
| 1002 | 0.739407 | 0.783372 | 0.694880 |
| 1003 | 0.740865 | 0.786116 | 0.690072 |
| 1004 | 0.748917 | 0.793032 | 0.707015 |
| mean ± stdev | 0.743063 ± 0.005122 | 0.787507 ± 0.004978 | 0.697322 ± 0.008731 |

## 仍未解除 / 备注

1. E0 未启动；是否用上述波动估 Pareto 允许偏差，由协议与 gate 另行裁定。
2. 当前 `transformers==4.46.3`，`requirements.txt` 仍锁定 `4.43.0`。

数据路径、SHA-256 和 checkpoint snapshot 见 `configs/global_asu_forget05.template.yaml` 与 `derived/`。

## Smoke-test 通过标准

在真实输入到位后，依次记录并要求全部通过：

1. checkpoint/tokenizer 与 JSON split 加载；
2. 一个 forget/retain batch 的 shape、answer mask 与 padding mask 审计；
3. Teacher 温度 forward 与标准 Student forward；
4. ASU+GD 局部 loss、`backward()` 和仅 Student 参数更新；
5. 单次评测入口生成结果；
6. Teacher 梯度全为 `None`，参数哈希不变。

任何一项失败，停止 E0/E1/E2，并在 `gates/` 记录根因与最小修复。

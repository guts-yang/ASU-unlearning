# Baseline Experiment Card — CAASS-ASU / Stage 1

## 状态与范围

- 状态：**阻塞，未运行**。
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
| Forget-retain 抽样 | retain 为按 `idx` 固定种子后随机偏移的样本。`forget05` 的 retain 文件名由 `100 - min(10 * 5, 90)` 得到 `retain50`；`TASK_LIST=1` 时再拼上其余 task | `dataset/data_module.py`、`forget.py` |

`ASU+KL` 不属于主比较：当前实现的后定义 `kl_loss` 覆盖先前带答案 mask 的版本，产生全位置 `batchmean` KL。因此它只能在单独冻结语义后作敏感性分析。

## 拟冻结的训练配置（不得将其当作已跑超参）

| 字段 | 拟用值 / 规则 | 来源或状态 |
| --- | --- | --- |
| 模型 | `locuslab/tofu_ft_llama2-7b` | 路径与三份分片哈希见 `derived/stage1_checkpoint_environment.json`；尚未加载 |
| 模型架构 | `NousResearch/Llama-2-7b-chat-hf`，`LlamaForCausalLM`，32 层，bfloat16 | checkpoint `config.json` |
| 数据 | `forget05.json` 2000 行，task 1 为 200 行；`retain50.json` 2000 行加其余 task 1800 行，有效 retain 3800 行；评测 retain 为 `retain_perturbed.json` 400 行 | `derived/stage1_data_identity.json`。`retain95.json` 不存在 |
| loss | `ASU+GD` | 主基线 |
| 温度 | 全层单一 `tau=2.3` | 历史脚本参考，非本地最优值 |
| `layers_id` | 命令必须传 `null`。`tofu.yaml` 默认 `[]` 不会给任何层加温度 | `config/tofu.yaml`、`my_models/my_llama.py` |
| forget coefficient | `0.1` | 当前脚本 |
| retain coefficient | `1.0` | 当前脚本 |
| learning rate | `1e-5` | 当前脚本 |
| epochs | `5` | 当前脚本 |
| per-device batch | `4` | 当前配置 |
| gradient accumulation | `4` | 当前配置 |
| max sequence length | `500` | 当前代码 |
| seed | 预注册至少 3 个；具体列表须在运行前写入配置快照 | 尚未冻结 |
| optimizer | `paged_adamw_32bit` | 当前代码 |
| precision | bf16 | 当前代码 |
| distributed | 训练 2 GPU，评测 1 GPU | 当前脚本 |

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

## 运行前阻塞项

1. 评测依赖仍缺 `jsonlines`、`sentence_transformers`。当前 `transformers==4.46.3`，`requirements.txt` 锁定 `4.43.0`。这次检查没有安装缺包。
2. 可见 GPU 型号、数量、显存和作业时限。`nvidia-smi` 存在，但 `nvidia-smi -L` 没有设备，`/dev/nvidia*` 不存在。
3. 运行前固定的至少三个 seed、评测保存路径和可用 GPU-hours。配置模板里这两项仍是 null。

数据路径、SHA-256 和 checkpoint snapshot `8fa500e8f345f1dd9cfe95bb4689878c944c9cbd` 已写入 `configs/global_asu_forget05.template.yaml`。这只是身份核对，不是训练结果。

## Smoke-test 通过标准

在真实输入到位后，依次记录并要求全部通过：

1. checkpoint/tokenizer 与 JSON split 加载；
2. 一个 forget/retain batch 的 shape、answer mask 与 padding mask 审计；
3. Teacher 温度 forward 与标准 Student forward；
4. ASU+GD 局部 loss、`backward()` 和仅 Student 参数更新；
5. 单次评测入口生成结果；
6. Teacher 梯度全为 `None`，参数哈希不变。

任何一项失败，停止 E0/E1/E2，并在 `gates/` 记录根因与最小修复。

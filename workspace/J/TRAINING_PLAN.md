# 训练规划：M0 之后的单侧 gold 上限

日期：2026-10-30。资产以 [`PREP_READINESS.md`](PREP_READINESS.md) 为准。先做离线 M0。只有书面结论为 GO 时才启动训练。训练是在现有 ASU+GD 上加一个单侧 gold 上限，不覆盖已有 checkpoint，不从 Hub 下载。

## 固定资产

| 角色 | 路径 |
|---|---|
| 初始化 Student / Teacher | `/root/autodl-tmp/models/tofu_ft_llama2-7b`（open-unlearning `tofu_Llama-2-7b-chat-hf_full`，snapshot `cc5b31c69127d5da881608e640f2b453c446435b`） |
| 对照 Student | `.../tofu_replay/.../ASU+GD/seed_1001/.../checkpoint-last` |
| 遗忘数据 | `Right-to-be-forgotten/data/tofu/forget05.json`，`task_id=1` 为 200 题 |
| 保留数据 | `data/tofu/retain50.json`（1999 行）。`forget.py` 对 `forget05` 会取 `retain50`，再并入 `task_id != 1` 的 forget 题 |
| 对话模板 | `model_config.yaml` 的 `llama2-7b`：`[INST] ` … ` [/INST]` |
| tokenizer / config 缓存 | `NousResearch/Llama-2-7b-chat-hf` snapshot `351844e75ed0bcbbe3f10671b3c808d2b83894ee` 已在本机。`forget.py` 用 `model_family` 的 `hf_key` 读 tokenizer 和 config，用 `model_path` 读权重 |

历史 `config.yaml` 里的 `model_path: locuslab/tofu_ft_llama2-7b` 不改、不用于加载。该 Hub 缓存已删除。

硬件：2× A800 80GB，当前空闲。`/root/autodl-tmp` 剩余约 166GB。一份新 checkpoint 约 13.5GB。

## 阶段 A：M0（不训练）

对 `task_id=1` 的 200 题做 teacher-forcing。编码与训练一致：`logits[:, :-1]` 对 `labels[:, 1:]`，只统计 `labels != -100` 的答案位。Teacher 使用 `attention_temp=2.3`、`layers_id=null`；Student 不使用温度。

比较两个 Student，Teacher 都是上面的 open-unlearning 基座：

1. 初始化基座自己作为 Student。
2. 已有 ASU+GD `checkpoint-last`。

记录每个有效答案位的 \(p_T(g)\)、\(p_S(g)\)、解析梯度 \(p_S(g)-p_T(g)\)，以及 Teacher / Student 的 gold Top-1/5/10。另抽小 batch，对 `asu_loss` 的标量做 autograd，与解析梯度的相对误差阈值 `1e-4`。

聚合按作者，不把 token 当独立样本。`task_id=1` 是 10 位作者 × 20 题，作者从问句解析，方式与方向 I 相同。作者 bootstrap 1000 次，种子写入诊断配置。输出放到 `workspace/J/derived/`，并写 `workspace/J/结论.md`。

预估：单卡 teacher-forcing，约 20–40 分钟。

### 门控

- **GO**：ASU Student 上，作者级 \(p_S(g)<p_T(g)\) 比例的 bootstrap 下界 **> 0.05**，且 Teacher gold Top-10 的作者级均值 **≥ 0.20**，并且 KL API 与小样本 autograd 一致。含义只是：值得做下面这一次最小训练。
- **NO-GO**：下界 ≤ 0.05，或 Teacher Top-10 作者均值 < 0.20，或梯度符号与 `KL(p_T \|\| p_S)` 不一致。到此停止。
- **INCONCLUSIVE**：权重、mask 或作者切块对不上。到此停止。

GO 之前不改 `losses.py`，不调用 `forget.py`。

## 阶段 B：最小训练（仅 GO 之后）

对照是已有的 ASU+GD checkpoint，不重训。新跑只加一项单侧上限，其余超参与那次 ASU+GD 相同。

在答案位上，Teacher 概率停止梯度：

\[
\mathrm{relu}\big(\log p_S(g)-\log p_T(g)\big)
\]

按与 `asu_loss` 相同的方式做序列平均再 batch 平均。\(p_S(g)\le p_T(g)\) 时这项为 0。系数固定为 1.0，这一轮不做网格。保留项仍是 GD。遗忘项仍是原来的 `KL(p_T \|\| p_S)`，Teacher logits 不做 gold-rank 编辑。非 gold 条件分布留到这次结果出来之后再单独立项。

| 项 | 值 |
|---|---|
| `model_path` | `/root/autodl-tmp/models/tofu_ft_llama2-7b` |
| `forget_loss` | 新名称 `ASU+GD+Cap`，与旧的 `ASU+GD` 目录分开 |
| `split` / `task_id` / `seed` | `forget05` / `1` / `1001` |
| `attention_temp` / `layers_id` / `mask` | `2.3` / `null` / `true` |
| `fix_ref_model` / `use_LoRA` | `true` / `false` |
| `lr` / `epochs` / `batch` / `grad_accum` | `1e-5` / `5` / `4` / `4` |
| `forget_coeff` / `regularization_coeff` / `beta` | `0.1` / `1.0` / `0.1` |
| 进程 | `torchrun --nproc_per_node=2` |
| 环境 | 启动前设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1` |

`task_id=1` 的 forget 集长度为 200。`forget.py` 里

`max_steps = int(5 * 200) // (4 * 4 * 2) = 31`。

保存根目录：

`/root/autodl-tmp/ASU-unlearning/checkpoints/dard_oneside`

`overwrite_dir=false`。禁止写入 `checkpoints/tofu_replay` 和 `Right-to-be-forgotten/results/tofu_replay`。

预估：31 步双卡约 25–45 分钟。评测用现有 `eval.py` 的四项（retain、real authors、world facts、forget），对照 checkpoint 与新 checkpoint 各评一次，约再加 30–60 分钟。GO 之后这一轮合计大约 1.5–2.5 小时。只跑 seed 1001，这一轮不登记多种子的非劣界。

## 停手条件

- M0 不是 GO。
- 新目录里已经有 `checkpoint-last`。
- 加载路径解析到 `locuslab/tofu_ft_llama2-7b` 或任何 Hub id。
- 训练过程要求下载权重、tokenizer 或数据。

## 这轮不声称的事

单侧上限跑完后，只比较它和已有 ASU+GD 在同一评测上的描述统计。不把 \(p_S(g)<p_T(g)\) 写成遗忘硬下限，不把这一次训练写成 DARD 已经成立，也不报告零泄漏。

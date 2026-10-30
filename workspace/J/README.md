# DARD-ASU：阶段 1（M0）离线诊断

本目录独立维护方向 J 的代码审计和 M0 Teacher-Forcing 离线诊断。它不实现 DARD-ASU loss，不启动训练，不写入或覆盖 checkpoint，也不下载模型或数据。

## 固定口径

- 路径：`Right-to-be-forgotten/real_world_unlearn.py` 与 `trainer/losses.py`
- 数据：`data/real_world/forget.json`
- 模型：Llama-3-8B 的同一初始化 Student、ASU Teacher 和可选 ASU 后 Student
- 遗忘项：`F.kl_div(log p_S, log p_T, log_target=True)`，即 `KL(p_T || p_S)`
- Teacher：`attention_temp=2.0`、`layers_id=null`（全部层）；Student 不使用 attention temperature
- 预测位置：`logits[:, :-1, :]` 预测 `labels[:, 1:]`；只统计 `labels[:, 1:] != -100` 的答案位

因此，在未编辑 Teacher logits 的原始 ASU 项中，每个有效答案位的 Student gold-logit 导数为

`dL/dz_S(g) = (p_S(g) - p_T(g)) / (batch_size * valid_answer_tokens_in_example)`。

比例因子为正，所以 `p_S(g) < p_T(g)` 当且仅当该位置的实际遗忘项 gold-logit 梯度为负；梯度下降将提高该 gold logit。这是指定 checkpoint、数据和 teacher-forcing 前缀下的局部现象，不是遗忘下限或 DARD 有效性的证明。

## 运行前置条件

不提供或下载任何资产时，当前状态是 **INCONCLUSIVE**。诊断需要以下已有的本地绝对路径：

1. `--forget-json`：real-world `forget.json`；
2. `--teacher-model`：包含 `config.json` 的本地 Llama-3-8B snapshot；
3. `--student-initial`：包含 `config.json` 的同一初始化 Student checkpoint/snapshot；
4. 可选 `--student-asu`：ASU 训练后 Student checkpoint/snapshot。

路径也可先记录在 [`assets.manifest.example.json`](assets.manifest.example.json) 的副本中。不得将 Hub ID、远程路径或替代模型传入。脚本在导入 Transformers 前设置离线环境变量，并在前置检查失败时退出。

## 命令

从仓库根目录执行；以下全为占位路径，不能原样运行：

```bash
python workspace/J/m0_diagnose.py \
  --forget-json /absolute/path/data/real_world/forget.json \
  --teacher-model /absolute/path/llama3-base \
  --student-initial /absolute/path/initial-student \
  --student-asu /absolute/path/asu-student \
  --output-dir /absolute/path/diagnostics/dard_asu_m0 \
  --batch-size 1 \
  --max-length 300
```

`--student-asu` 缺失时只报告 initial Student，并在结论中明确 checkpoint 缺失。输出目录必须不存在，以避免覆盖既有证据。

## 输出和统计

`m0_summary.json`、`m0_summary.md` 和 `per_question.jsonl` 记录：

- 有效答案 token、题目和可识别 author/entity group 数；
- `p_T(g)`、各 Student 的 `p_S(g)` 分布；
- `p_S(g) < p_T(g)`、解析 gold-gradient `< 0`、及小 batch 实际 autograd gold-gradient `< 0` 的比例；
- Teacher/Student gold rank 的 Top-1/5/10、rank 中位数；
- resolved 参数、资产 SHA-256、PyTorch/Transformers 版本，以及解析和 autograd 的误差。

置信区间使用题目为 cluster 的非参数 bootstrap；token 只是在题内聚合，绝不当作独立样本。若数据没有可用的 `author`、`entity` 或显式 `--cluster-key` 字段，只报告题目聚类。所有数值是描述统计，并非显著性检验。

## 当前审计结论

本仓库未包含 `data/real_world/forget.json`、本地 Llama-3 权重、初始化 Student 或 ASU 后 Student checkpoint；历史 C/H checkpoint 已删除。因此当前不能生成真实 M0 表、不能给出 GO/NO-GO。资产按上述口径就绪后，运行此脚本才可将结论从 **INCONCLUSIVE** 更新为基于证据的判断。

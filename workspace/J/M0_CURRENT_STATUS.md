# DARD-ASU M0 当前状态

生成日期：2026-10-30。范围为 Real-world / Llama-3-8B / ASU+KL；未训练、未下载、未替代 checkpoint 或数据。

## 代码审计

- 遗忘 loss：[`Right-to-be-forgotten/trainer/losses.py`](../../Right-to-be-forgotten/trainer/losses.py) 的 `asu_loss`。
- 实际 API：`F.kl_div(input=log p_S, target=log p_T, reduction="none", log_target=True)`；故遗忘项为 `KL(p_T || p_S)`。
- 归约：词表求和，`labels[:, 1:] != -100` 答案 mask，逐题有效 token 平均，再 batch 平均。
- 对齐：`logits[:, :-1, :]` 预测 `labels[:, 1:]`。忘却数据的 labels 来自 [`workspace/gold_rank/masking.py`](../gold_rank/masking.py) 的 `encode_forget_example`，prompt 与 padding 均不会进入分母。
- Teacher：[`Right-to-be-forgotten/config/real_world.yaml`](../../Right-to-be-forgotten/config/real_world.yaml) 配置 `attention_temp=2.0`、`layers_id=null`；Student forward 不传该参数。
- 对每个有效位，实际 batch loss 的 gold-logit 导数为 `(p_S(g)-p_T(g))/(batch_size × 该题有效答案 token 数)`。正比例不改变符号，因此 `p_S(g)<p_T(g)` 表示梯度下降会提高 gold logit。

## M0 结果表

| 指标 | 当前结果 |
|---|---:|
| 有效答案 token | N/A |
| 题目数 / author 或 entity group 数 | N/A |
| `p_S(g) < p_T(g)` | N/A |
| 实际 autograd `dL/dz_S(g) < 0` | N/A |
| Teacher Top-1 / Top-5 / Top-10 | N/A |
| 初始 Student Top-1 / Top-5 / Top-10 | N/A |
| ASU 后 Student Top-1 / Top-5 / Top-10 | N/A |

不能将 N/A 当作零或负结果。

## 结论：INCONCLUSIVE

当前仓库没有同口径的 `data/real_world/forget.json`、本地 Llama-3-8B snapshot、初始化 Student 或 ASU 后 Student checkpoint；历史 Direction C/H checkpoint 已删除。用户也明确要求此时不下载数据或 checkpoint。因此不存在可用于同前缀 Teacher-Forcing 比较的资产，不能产生 M0 数值、置信区间或 GO/NO-GO 性能判断。

资产到位后运行 [`m0_diagnose.py`](m0_diagnose.py)。该脚本只接受本地路径，强制离线，使用题目 cluster bootstrap（默认 2,000 次）报告描述性 95% 区间；token 仍嵌套在题目内，绝不被当作独立样本。

## 当前验证

- `python -m py_compile workspace/J/m0_core.py workspace/J/m0_diagnose.py workspace/J/tests/test_m0_core.py`：通过。
- `git diff --check`：通过。
- 缺失 forget JSON 的 CLI preflight：按设计拒绝，并输出“refusing download or substitute data”。
- 单元测试文件 `workspace/J/tests/test_m0_core.py` 覆盖 KL autograd/解析梯度、causal shift、padding/`-100`、rank tie-break 和空答案位；当前 Python 环境未安装 `torch`，因此无法执行。真实 M0 也同样被依赖和资产共同阻塞。

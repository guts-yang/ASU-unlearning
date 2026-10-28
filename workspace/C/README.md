# 方向 C — DA-ASU（统一归档）

**本目录是方向 C 的唯一维护入口。** 结论、判决 JSON、评测副本、设计报告与日志都放这里；不要再分散到 `workspace/logs/`、`workspace/da_asu/`、`workspace/results/direction_c/` 等处各自改。

- 方法全称：Derivation-Anchored Attention Smoothing Unlearning
- 核心一句话：ASU 升温会误伤依赖精确注意力的数学推导；在保留侧把推导位置的注意力锚定回基座，试图在不放松遗忘的前提下拉回 GSM8K。
- 代码历史：`origin/C` 上的 `workspace/da_asu/`（当前工作树里多半只剩 `__pycache__`）；训练入口在 `Right-to-be-forgotten/real_world_unlearn.py`。
- 大权重在数据盘，见 `checkpoints_pointer/`。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| [`结论.md`](结论.md) | 预实验结论（读这个） |
| [`derived/`](derived/) | M0 / M0b / M1 结构化结果与评测文本副本 |
| [`design/`](design/) | 设计报告与 `protocol.py` 快照（自 `origin/C`） |
| [`logs/`](logs/) | 原始运行日志副本 |
| [`checkpoints_pointer/`](checkpoints_pointer/) | 数据盘权重路径说明 |

## 状态（2026-10-08 跑完，2026-10-28 归档）

| 阶段 | 判决 | 含义 |
| --- | --- | --- |
| M0 | `proceed` | 基座上推导 token 比虚词更怕升温（H1） |
| M0b | `proceed` | ASU 学生在推导行上注意力熵高于基座（H2）；特异性 `not_selective` |
| M1 | `fail_m1` | anchor 拉回 GSM8K（+3.8pt）且高于 logit 对照，但 FE 相对基线微降，三条门未齐过 |

详见 [`结论.md`](结论.md) 与 [`derived/m1_results.json`](derived/m1_results.json)。

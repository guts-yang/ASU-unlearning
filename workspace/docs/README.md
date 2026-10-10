# 方向 G：ChainBreak——LRM 长思维链"认知短路"注意力阻断遗忘

> Chain-of-Thought Attention Circuit Breaker Unlearning
> **当前状态（2026-10-10）：立项，方案级（无实验）。** idea-evaluator 判决 **Strong Accept（worth pursuing, pending the validation experiment）**，是三个 ASU 衍生候选（TopoWash / ChainBreak / LeakShield）中唯一 Strong Accept。M0 生死实验前置，未通过不得进入方法开发。
>
> **诚实边界：**本文档所有效果均为假设，尚无任何实验数字。文献检索（2026-10-09）确认赛道已有占位工作（R-TOFU EMNLP 2025、STaR AAAI 2026、CiPO），首创性声明收窄为"唯一在**参数层面**剪断 LRM 推演通路的方法"，与推理时抑制/拒绝式方法（STaR）和偏好优化方法（CiPO / Reasoned IDK）机制轴不同。命名需与 Zou et al. Circuit Breakers（NeurIPS 2024）明确区隔，见差异化表。

## 一句话

大推理模型（LRM）的长 CoT 具有强"因果动量"——即使输入端被 ASU 烫平，模型仍可凭推理链中间自己写下的半成品顺推召回被遗忘实体；ChainBreak 用因果显著性定位极少数**支点 Token（Pivot-tokens）**，仅在这些支点被下游查询时施加动态指数温度断路，在保留通用推理语法的同时切断事实性闭环。

## 核心假说（The Circuit-Breaker Hypothesis）

LRM 推演遵循"探索（Exploration）→ 认知跃迁（Cognitive Jump / Pivot）→ 确定性收敛（Deterministic Convergence）"的相变结构；链条的存续取决于极少数承上启下的支点 Token。遗忘无需抹除整条链（会导致逻辑崩塌为胡言乱语），只需在支点上制造注意力断路，使上游事实证据无法传递给下游结论。

## 文件

| 文件 | 内容 |
|---|---|
| `方向G_ChainBreak_概念大纲.md` | 完整方案：立论、假说、三阶流水线、数学表述、差异化、实验设计、里程碑与风险登记 |
| `方向G_创新评估报告.md` | idea-evaluator 全量评估（五维评分/范式探针/致命缺陷审计）+ 2026-10-09 文献检索锚定（STaR/R-TOFU/CiPO/Circuit Breakers delta 表） |
| `方向G-ASU_核心思想逻辑流程图.{drawio,png,audit.md}` | 机制逻辑图三件套：ASU 基线逐字保留 + 方向G 扩展层（M0 闸门 × pivot 定位 × 断路算子 × 双轨损失 × 对抗评测闭环） |

## 与 A–F 的边界

- A = 权重端残余泄漏审计（已触发 kill-switch，遗留"teacher→student 秩恢复"现象）；G 不依赖 A 结论，但攻击评测套件可复用 A 的 RR 协议。
- B = 多模态 MSAU；C = AM-ASU retain 侧效用修复；D（已停止）= RAG 证据回流；E = 进化搜索平滑程序；F = DIP-ASU 教师投影。
- **G = LRM 生成过程级遗忘**：干预对象不是输入 Prompt—短输出关联（ASU/基座），也不是检索证据（D），而是长 CoT 内部因果推演通路。与 A–F 无机制重叠。

## 最近邻工作（检索锚定，2026-10-09）

| 工作 | 出处 | 与 G 的差异轴 |
|---|---|---|
| R-TOFU | Yoon, Jeung & No, EMNLP 2025（aclanthology 2025.emnlp-main.265，代码 ai-isl.github.io/r-tofu） | 基准 + Reasoned IDK（偏好优化/拒绝式）；G 为参数级机制干预，且把 R-TOFU 的 ZeroThink decoding-trap 作为主攻击场景 |
| STaR | Sensitive Trajectory Regulation, AAAI 2026（doi 10.1609/aaai.v40i41.40818） | 推理时语义检测+安全前缀+轨迹抑制+token 过滤（输出侧）；G 为训练时参数化温度断路（权重侧） |
| CiPO | arXiv 2604.15847 | 反事实路径偏好优化；G 不改写推理内容，只阻断事实传递 |
| Circuit Breakers | Zou et al., NeurIPS 2024, arXiv 2406.04313 | 表征重路由（RepE/LoRRA，运行时对抗防御）；G 为注意力温度（遗忘任务），术语需区隔 |

## M0 生死实验（2 周，前置）

1. 在 R-TOFU（TOFU Forget10 + CoT 标注）上复现 ASU / NPO / Reasoned IDK，量化"推理动量绕过"：CoT 中 pivot 前后 token 的 gold 恢复率与 ZeroThink 型 decoding-trap 泄漏，确认断路靶位存在。
2. Pivot 定位可行性：积分梯度 vs 因果中介在 ≥20 条样本上的一致性（Top-K 重合率）；定位不稳定则降级为"层×位置条件化温度"退路或判死。
3. 判死条件：M0 显示现有基线在 R-TOFU 上已无 bypass（CoT 残迹≈0）或 pivot 定位与随机断路无显著差异 → 方向 G 转入判死归档。

# 方向 G：ChainBreak 创新评估报告（idea-evaluator）

> 评估日期：2026-10-09（检索）/ 2026-10-10（归档）。评估对象：基于 ASU（ICLR 2026）衍生的 Idea 2 "ChainBreak——大推理模型（LRM）思维链中的'认知短路'注意力阻断遗忘"。
> 检索纪律：新颖性判定基于 2026-10-09 web 检索锚定；"未检索到直接重叠"不等于新颖性证明，只作 metadata 级判断。

---

## 1. 首印象与论文类型定位

- **Paper type**: Novel Method + New Setting。
- **一句话故事**："在长 CoT 的因果关键跃迁点上做注意力断路，让推理模型的'推演动量'无法绕过遗忘。"故事一句话成立，且指向已文档化的失败模式（R-TOFU 的 decoding-trap）。

## 2. 致命缺陷审计（early gate）

| # | 缺陷 | 严重度 | 防御 |
|---|---|---|---|
| 1 | **"首创 LRM 认知链遗忘"不成立，赛道已被占**：R-TOFU（Yoon, Jeung & No, EMNLP 2025，基准 + Reasoned IDK）；**STaR**（Sensitive Trajectory Regulation, AAAI 2026, doi 10.1609/aaai.v40i41.40818——推理时对整条轨迹做敏感内容抑制，机制叙事最接近）；CiPO（arXiv 2604.15847，反事实偏好优化）。机制轴差异仍存在：STaR 为推理时 token 过滤 + 安全前缀（输出侧），ChainBreak 为因果中介定位 pivot + **训练时参数化**注意力温度断路（权重侧）——对象与机制均不同，但必须在 related work 放显式 delta 表 | MAJOR | delta 表三列对齐 R-TOFU / STaR / CiPO；卖点收窄为"唯一在参数层面剪断推演通路的方法"，以 ZeroThink 型 decoding-trap 绕过为主战场 |
| 2 | **与 Circuit Breakers（Zou et al., NeurIPS 2024, arXiv 2406.04313）的区隔风险**：命名与"断点阻断有害信息流"叙事高度相似；RR 是表征重路由（RepE/LoRRA）+运行时对抗防御，ChainBreak 是注意力温度 + 遗忘任务。审稿人若认为"换个地方做 circuit breaker"会直接拒 | MAJOR | 全程改用 ASU 语境术语（断路算子 = 条件化温度张量），机制对比实验加 RR 基线 |

无 CRITICAL 缺陷（未被自有数据或检索结果直接否证），继续评分。

## 3. 生命周期与能力匹配

| 方面 | 输入 | 评估 |
|---|---|---|
| Idea 类别 | Frontier Exploration | 基准（R-TOFU 公开：ai-isl.github.io/r-tofu）+ 方法双组件 |
| 生命周期 | 3–5 个月 | 覆盖一个顶会周期；M0 2 周即可出判决 |
| 有效工时 | 研究生 + 双卡 + OpenUnlearning/OpenUnlearning 级工程经验 | R1-Distill-7B 级模型可训（LoRA + teacher 前向缓存） |
| Fit | **Green** | 唯一新组件为 pivot 定位器（积分梯度/因果中介），工具链成熟 |

## 4. 五维评分

| 维度 | 分 | 依据 |
|---|---|---|
| Higher | 7 | R-TOFU 已证：answer-only 遗忘在 CoT 轨迹留大量 step-wise 残迹；机制层修复有真实提升空间（机制分，未验证） |
| Faster | 5 | pivot 因果中介定位贵，训练时摊销后中性 |
| Stronger | **8（机制分）** | 直击已文档化 bypass（R-TOFU ZeroThink/LessThink decoding trap）；命名验证实验 = M0 |
| Cheaper | 5 | 无依据 |
| Broader | **8（机制分）** | 框架可推广到一切"生成过程级"安全遗忘（agentic 轨迹、工具调用痕迹） |

## 5. 范式探针

| 探针 | 判定 | 理由 |
|---|---|---|
| First Principles | **Yes** | 挑战"遗忘 = 改最终答案分布"的默认假设 |
| Elephant in the Room | **Yes** | LRM 时代的遗忘是全行业可见但方法缺位的问题 |
| Technology Cycle | **Yes** | LRM 普及使该问题新近可行/必要 |
| Hamming's Rule | No | 解决后主要影响 LRM 安全治理子领域 |

3/4 yes，颠覆潜力：**strong**。

## 6. 可行性

| 风险 | 等级 | 缓解 |
|---|---|---|
| 计算 | Green | R-TOFU Forget10 + 7B distill，双卡可行 |
| 数据 | Green | R-TOFU / WMDP / MATH-500 / GSM8K / BBH 全开源 |
| 工程 | Yellow | causal mediation / 积分梯度实现是唯一新组件；M0 先验证定位一致性 |
| 时间 | Green | 3–5 个月可覆盖投稿周期 |

## 7. 判决

**Strong Accept（worth pursuing, pending the validation experiment）**

Top-3 首要行动：
1. **M0（2 周）**：在 R-TOFU 上复现 ASU / NPO / Reasoned-IDK，量化 CoT 推理动量绕过（pivot 前后 token gold 恢复率 + ZeroThink 型 decoding-trap 泄漏），确认断路靶位存在。
2. **Pivot 定位可行性**：积分梯度 vs 因果中介在 ≥20 条样本上的一致性（Top-K 重合率）；不稳定则降级为"层×位置条件化温度"退路或判死。
3. **Delta 表**：钉死 vs STaR / CiPO / R-TOFU / Circuit Breakers（NeurIPS 2024）——这是本文存活的 reviewer 第一问。

### 同批候选横向对比（TopoWash / ChainBreak / LeakShield）

| 维度 | TopoWash | ChainBreak | LeakShield |
|---|---|---|---|
| 判决 | Accept w/ Revisions（需重锚"teacher→student 秩恢复"） | **Strong Accept（pending M0）** | Accept w/ Revisions（需换 Forman-Ricci） |
| 拥挤度 | 低（TDA×遗忘无人做） | 中（STaR/R-TOFU 占赛道，机制轴可区分） | 高（基准/诊断/推理门控全齐，缺参数级方法） |
| 前提证据 | ⚠️ 与方向 A 五通道审计部分冲突 | ✅ R-TOFU 已文档化 bypass | ✅ 多跳泄漏效应量已被多篇证实 |
| 与 A–F 冲突 | 无（可复用 A 审计脚本） | 无 | 与 A 方法论互补 |

优先级结论：**ChainBreak > LeakShield ≈ TopoWash（改写版）**。ChainBreak 是唯一同时满足"问题已文档化 + 机制轴未被占 + 算力可达"的候选，且 M0 最便宜。

### 检索记录（2026-10-09）

关键词组：① persistent homology unlearning / attention topology；② relearning attack LoRA recovery；③ Circuit Breaker representation rerouting；④ unlearning large reasoning models R-TOFU；⑤ multi-hop leakage unlearning；⑥ Ricci curvature attention；⑦ selective attention head unlearning。命中即上表所列；"persistent homology 驱动遗忘干预"与"曲率驱动的参数级多跳遗忘"两个组合均未检索到直接重叠（metadata 级判断）。

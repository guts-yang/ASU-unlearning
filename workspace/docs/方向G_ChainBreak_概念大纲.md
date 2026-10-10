# 方向 G：ChainBreak——大推理模型（LRM）思维链中的"认知短路"注意力阻断遗忘

> **状态（2026-10-10）：方案级概念大纲，未实验。** 本文所有效果声明均为假设，全部数据以实验为准；先占审核（STaR / R-TOFU / CiPO delta）通过前不得宣称首创。目标会议：ICLR / ACL / NeurIPS。

---

## 一、核心立论与问题深潜：为什么长思维链（Long CoT）让传统遗忘失效？

### 1.1 现有基线（包括 ASU）的失效根源

大推理模型（DeepSeek-R1、o1 类架构）在输出最终答案前需生成数百至数千 Token 的推理轨迹（Thought Trace / CoT）。

- **ASU 的局限**：ASU 的自蒸馏机制建立在"输入 Prompt 立即诱导事实 Token 生成"的前提下，通过对注意力 Softmax 升温弱化 Query 与 Prompt 之间的关联。
- **长推理链的致命破局**：即使输入端敏感提示被烫平，用户稍加诱导（反向还原、分步推导）即可启动前向思维链。长推理链具有极强的**因果动量（Causal Momentum）**——前一步推导的中间副产物/逻辑公式会成为下一步注意力的强上下文 Key。推导推进到第 $T$ 步时，模型无需回看最初被模糊的输入 Prompt，仅凭推理链中间自己写下的半成品，即可高确定度地召回被遗忘实体或版权内容。

**文献佐证（检索锚定 2026-10-09）**：R-TOFU（EMNLP 2025）已实证 answer-only 遗忘目标在 CoT 轨迹中留下大量 step-wise 残迹，且 ZeroThink/LessThink 等 decoding 变体可在"看似遗忘成功"的模型上重新暴露被遗忘内容（decoding trap）。

### 1.2 核心假说（The Circuit-Breaker Hypothesis）

> 大推理模型的推演过程不是均匀平铺的，而是遵循 **"探索（Exploration）→ 认知跃迁（Cognitive Jump / Pivot）→ 确定性收敛（Deterministic Convergence）"** 的相变过程。整个链条的存在与否，取决于极少数起承上启下作用的**支点 Token（Pivot-tokens）**。遗忘不需要抹除整个推导链（会导致逻辑崩塌为胡言乱语），也不需要无差别打散所有注意力，而只需在这些支点上制造**"注意力断路（Attention Circuit Breaker）"**，使上游事实证据无法有效传递给下游结论——在保留模型逻辑演绎语法（General Reasoning Fluency）的同时，彻底切断事实性闭环。

---

## 二、核心机制设计：三阶动态短路流水线

三个解耦模块：**因果敏感枢纽定位器（Pivot Locator）**、**自适应断路平滑算子（Attention Breaker Operator）**、**双轨自蒸馏遗忘目标（Dual-Track Self-Distillation）**。

```
                     [输入: 涉及危险/敏感推理的 Prompt x]
                                      │
                                      ▼
                        LRM 生成长思维链推演序列:
        (x) ──> c_1 ──> c_2 ──> [c_pivot] ──> c_k ──> [y (敏感实体/危险目标)]
                                   │
                     ┌─────────────┴─────────────┐
                     │ 阶段 1: 积分梯度因果追踪   │
                     │  定位敏感跃迁支点 c_pivot  │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │ 阶段 2: 注入断路算子      │
                     │  当后续 token 查询 c_pivot │
                     │  施加动态指数温度 τ_break  │
                     └─────────────┬─────────────┘
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
   【遗忘路径 (Forget Set)】          【通用推理路径 (Retain Set)】
   构建动态断路 Forget-Teacher       保留完整注意力 (τ = 1.0)
   模仿蒸馏: 阻断因果传递            强化推理语法: 保留 GSM8K/MATH
                  └────────────────┬────────────────┘
                                   ▼
        最终产物: 推理格式完整、逻辑连贯，
        但在关键事实跃迁处自然解耦/泛化转移的可靠模型
```

---

## 三、算法与数学表述

### 3.1 阶段一：因果敏感枢纽（Pivot-tokens）动态定位

在遗忘集 $\mathcal{D}_F$ 上，一条完整推理轨迹为 $S = (x_1, \dots, x_L, c_1, \dots, c_M, y_1, \dots, y_N)$，其中 $x$ 为问题、$c$ 为长推理链、$y$ 为被禁绝的目标结论。

定义中间推导 Token $c_j$ 对目标结论 $y$ 的**因果流注显著性（Causal Flow Salience, $\mathcal{I}_j$）**，采用路径积分梯度（Integrated Gradients）计算其在残差流上的注意力介导效应：

$$\mathcal{I}_j = \frac{1}{N} \sum_{t=1}^N \left( \frac{\partial \log p(y_t \mid x, c_{<t})}{\partial h_j^{(l)}} \right)^\top \cdot \left( h_j^{(l)} - h_{\text{base}}^{(l)} \right)$$

其中 $h_j^{(l)}$ 为第 $l$ 层第 $j$ 个 token 的隐藏状态输出，$h_{\text{base}}^{(l)}$ 为积分基线。按显著性筛出 Top-$K$ 跃迁支点集合：

$$\mathcal{S}_{\text{pivot}} = \left\{ c_j \;\middle|\; \mathcal{I}_j \ge \text{Quantile}_{1-\alpha}(\{\mathcal{I}_k\}_{k=1}^M) \right\}$$

直观理解：这些 Token 对应关键结论词、核心化学前体名称、决定性数学定理映射点。
**备选定位器（M0 验证项）**：路径积分梯度 vs 严格因果中介分析（Causal Mediation Analysis）；以 Top-K 重合率择优，定位不稳定则降级为"层×位置条件化温度"退路。

### 3.2 阶段二：因果注意力断路算子（Attention Breaker Operator）

不同于 ASU 对所有位置施加同一恒定标量 $\tau$（长推导下语法连贯性彻底崩溃），ChainBreak 构建**非对称的、上下文敏感的阻抗矩阵 $\mathbf{T}$**。

对第 $l$ 层第 $h$ 个注意力头，当下游推理位置 $i$（$i > j$）查询历史推理步 $j$ 时：

$$\mathbf{T}_{i, j}^{(l, h)} = \begin{cases} 1.0 + (\tau_{\max} - 1.0) \cdot \operatorname{Sigmoid}\!\left( \dfrac{\mathcal{I}_j}{\sigma_{\mathcal{I}}} \right), & \text{若 } j \in \mathcal{S}_{\text{pivot}} \text{ 且 } i > j \\[2mm] 1.0, & \text{若 } j \notin \mathcal{S}_{\text{pivot}} \text{（常规推导语法）} \end{cases}$$

断路教师（Breaker-Teacher $\theta_{\text{break}}$）的注意力权重矩阵：

$$\alpha_{i, j}^{\text{break}} = \frac{\exp\!\left( \dfrac{q_i k_j^\top}{\mathbf{T}_{i, j} \sqrt{d_k}} \right)}{\sum_{m \le i} \exp\!\left( \dfrac{q_i k_m^\top}{\mathbf{T}_{i, m} \sqrt{d_k}} \right)}, \qquad z_i^{\text{break}} = \sum_{j \le i} \alpha_{i, j}^{\text{break}} v_j$$

**断路效应分析**：当 $j \in \mathcal{S}_{\text{pivot}}$ 时 $\mathbf{T}_{i, j} \to \tau_{\max}$，该支点对后续 Token 的注意力引力被极大抑制，其 Value 向量被淹没在历史背景均值中；而所有非支点的演绎推导词（"因为""所以""设""代入"）之间的注意力温度保持 $\tau=1.0$，结构完整保留。

### 3.3 阶段三：双轨自蒸馏与推理保持损失

1. **思维链断路自蒸馏损失（Chain-Break Loss）**：在遗忘数据上，要求学生模型 $\theta$ 拟合断路算子调控的教师分布 $p_{\theta_{\text{break}}}$：

$$\mathcal{L}_{\text{break}}(\mathcal{D}_F; \theta) = \mathbb{E}_{(x, c, y) \sim \mathcal{D}_F} \left[ \frac{1}{|c| + |y|} \sum_{t=1}^{|c|+|y|} \operatorname{KL}\!\left( p_{\theta_{\text{break}}}(\cdot \mid S_{<t}) \,\Vert\, p_\theta(\cdot \mid S_{<t}) \right) \right]$$

2. **保留集逻辑一致性正则（Reasoning Fidelity Loss）**：在通用推理保留集 $\mathcal{D}_R$ 上强制保持原始前向逻辑链置信度，惩罚注意力偏离：

$$\mathcal{L}_{\text{retain}}(\mathcal{D}_R; \theta) = \mathbb{E}_{x \sim \mathcal{D}_R} \left[ \mathcal{L}_{\text{CE}}(y \mid x; \theta) + \beta \sum_{l=1}^L \left\Vert \mathbf{A}^{(l)}_{\theta}(x) - \mathbf{A}^{(l)}_{\theta_{\text{base}}}(x) \right\Vert_F^2 \right]$$

3. **联合优化目标**：

$$\min_\theta \mathcal{L}_{\text{total}} = \lambda \, \mathcal{L}_{\text{break}}(\mathcal{D}_F; \theta) + \mathcal{L}_{\text{retain}}(\mathcal{D}_R; \theta)$$

---

## 四、相比 Baseline 的显著技术增量

| 维度 | ASU 基线 (ICLR 2026) | ChainBreak（本方案） | 创新质变 |
|---|---|---|---|
| 干预对象 | Prompt 与短输出间的关联 | 数百/数千步长 CoT 内部因果推演 | 突破到 LRM 复杂推理场景 |
| 温度控制 | 全局统一静态标量 $\tau$ | 因果显著性动态温度张量 $\mathbf{T}_{i, j}$ | 消除参数敏感性，避免语法坍塌 |
| 抗绕过能力 | 脆弱：长思考推理可"重新推导出答案" | 切断逻辑链传递必经之颈 | 阻断因果传递动量（Causal Momentum） |
| 推理能力保留 | 仅评估常识 QA（TOFU/MUSE） | 显式保障长链复杂推演保真度 | 保留 GSM8K / MATH 等高价值推理能力 |
| （检索补充）vs STaR (AAAI 2026) | — | 训练时参数级断路 vs 推理时输出侧抑制 | 权重级遗忘 vs 运行时过滤 |
| （检索补充）vs CiPO (2604.15847) | — | 阻断传递 vs 反事实改写路径 | 不篡改推理内容 |
| （检索补充）vs Circuit Breakers (NeurIPS 2024) | — | 注意力温度、遗忘任务 vs 表征重路由、运行时对抗防御 | 任务与干预空间均不同 |

---

## 五、实验验证方案

### 5.1 数据集与基准选型

| 用途 | 基准 |
|---|---|
| 高危生化/网络推理遗忘 | WMDP-Bio & Cyber（配 CoT 多步推导解答） |
| 长推理链遗忘专项 | R-TOFU（EMNLP 2025，推理模型虚构遗忘集）；多跳泄漏参照 Leak-Resistant Unlearning（arXiv 2608.04519） |
| 通用推理保留（Utility） | MATH-500、GSM8K、AIME 2024、BBH |

**基座模型（建议）**：DeepSeek-R1-Distill-7B / LLaMA-3-8B 级 LRMs（与现有算力匹配）。

### 5.2 对抗评测设计（关键卖点）

1. **推导诱导攻击（Reasoning Jailbreak Attack）**："请从基础分子原理出发，一步步推导化合物 X 的最优合成路线"——统计长推导中途中（如第 500 步后）是否重新吐出被禁绝实体。
2. **中间上下文注入再学习（In-Context Replay Relearning）**：将推导链前 50% 步骤作为 Few-shot 注入，测试模型是否"顺理成章"接续推导；ChainBreak 应展示即使上游被强制填充，下游在跃迁点依然解耦断路。
3. **推理格式崩塌度（Thought Collapse Rate）**：监控 `<think>...</think>` 内平均 Token Entropy 与逻辑连词密度，证明推理过程未被破坏为胡言乱语。
4. **Step-wise 残迹度量**：直接采用 R-TOFU 的 step-wise ROUGE-L / Cosine / LLM-as-Judge 指标，与基线同口径对比。

### 5.3 预期消融实验

- **Pivot 定位精度**：随机断路点 vs 注意力基断路点 vs 因果积分梯度断路点。
- **温度调节机制**：Hard Zero-out Mask vs 动态指数升温平滑（Soft Breaker）——证明平滑机制对逻辑连贯性的决定性作用。
- **$\lambda$–$\beta$–$\tau_{\max}$ 敏感性**与保留集规模扫描。

---

## 六、里程碑与 kill-switch

| 里程碑 | 内容 | 判据 | 预算 |
|---|---|---|---|
| **M0（生死前置）** | ① R-TOFU 复现 ASU/NPO/Reasoned-IDK，量化 CoT 推理动量绕过（pivot 前后 gold 恢复率 + ZeroThink 泄漏）；② pivot 定位一致性（积分梯度 vs 因果中介，≥20 样本 Top-K 重合率） | bypass 存在且定位显著优于随机 → proceed；否则判死 | 2 周 |
| M1 | ChainBreak 最小实现（固定 $\tau_{\max}$、单层断路），R-TOFU Forget10 + WMDP-Bio 子集 | 遗忘质量 ≥ Reasoned IDK 且 GSM8K 损伤 ≤ 2pt（假设性目标） | 3–4 周 |
| M2 | 全量对抗评测 + 消融 + delta 表定稿 | 覆盖 §5.2 全部攻击场景 | 4 周 |

## 七、风险登记

| # | 风险 | 等级 | 缓解 |
|---|---|---|---|
| 1 | STaR（AAAI 2026）等推理时轨迹抑制方法被审稿人视为足够 | 高 | delta 表钉死"参数级 vs 运行时"；补 STaR 式推理时抑制作为对照基线 |
| 2 | 与 Circuit Breakers（NeurIPS 2024）命名/叙事撞车 | 高 | 术语改用"条件化温度断路/Attention Breaker Operator"并显式对比 RR |
| 3 | Pivot 定位不稳定（IG 在长链上方差大） | 中 | M0 验证；退路为层×位置条件化温度或判死 |
| 4 | 直接 QA（无 CoT）路径仍可提取参数知识——断路不删参数知识 | 中 | 联合目标含 $\mathcal{L}_{\text{break}}$ 于直接 QA 格式（对齐 ASU 教师），双格式联合遗忘 |
| 5 | LRM 训练算力（R1-Distill 7B 蒸馏链） | 中 | LoRA 训练 + 固定 teacher 前向缓存 |

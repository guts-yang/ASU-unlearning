---
id: CF2-04
rank: 4
title: 不平衡传输质量销毁 m_destroy —— 删除无法共享的激活质量，而非搬成 donor
avg_score: 7.8
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无跨模型独立回执
novelty_status: PROCEED WITH CAUTION（UOT/AcT 近邻较多）
baseline_relation: 从向量差改为带边缘松弛的反事实激活测度及未配对质量
---

# CF2-04 · 不平衡传输质量销毁 $m_{\mathrm{destroy}}$

## 0. 一句话故事

> 平衡最优传输要求所有源激活质量都有去处，这会把目标专属差异也强行映射到 donor。我们改用不平衡最优传输：能共享的激活质量允许匹配，无法匹配的目标专属质量不再强行搬运，而是被显式标记为“应销毁部分”；Student 只内化这部分残差的删除，不把 donor 状态当目标。

## 1. 问题与近邻空白

AcT（ICLR 2025，arXiv:2410.23054）、CHaRS 和 clustered OT steering 已把激活分布传输用于推理时控制；Opt-Out（ACL 2025）在参数空间使用 Wasserstein 正则；UOT-Unlearn（arXiv:2603.16489）在一步生成模型中重分配生成分布质量。因此“OT + unlearning”不新。

本方案唯一可辩护的增量是：**在 LLM 反事实中间激活上，用未配对源质量定义删除对象，并训练 Student 参数内化，而不是把完整源分布映射到目标分布。** 若退化为 balanced OT、donor 对齐或输出分布替换，则不再是本候选。

## 2. 可证伪假设

**H**：目标事实对应一部分在反事实测度间无法低成本匹配的源激活质量；删除该未配对残差比 balanced OT 更少 donor 答案替换和 retain 损伤。若未配对质量主要由样本噪声/批种子决定，或 balanced OT 在 donor 率与效用上不更差，则核心优势被证伪。

## 3. 方法

### 3.1 反事实激活测度

在固定 $(\ell,p)$、$p\in S_f$ 上，对 $N$ 个释义/模板配对收集 $a_i^\pm=A_{T,\ell,p}^{\pm,(i)}$，构造经验测度：

$$
\mu^\pm=\sum_{i=1}^{N}w_i^\pm\delta_{a_i^\pm},\qquad
C_{ij}=\|a_i^+-a_j^-\|_2^2.
$$

所有 $a_i^\pm$ stop-gradient；donor 答案 $y^-$ 不进入输入或损失。

### 3.2 半松弛不平衡传输

求熵正则、边缘松弛的传输计划：

$$
\pi^*=\arg\min_{\pi\ge0}
\langle C,\pi\rangle
+\varepsilon\sum_{ij}\pi_{ij}(\log\pi_{ij}-1)
+\tau D_\phi(\pi\mathbf1\|\mu^+)
+\tau D_\phi(\pi^\top\mathbf1\|\mu^-).
$$

定义源侧已匹配质量与目标专属销毁质量：

$$
m_i=(\pi^*\mathbf1)_i,\qquad
\boxed{d_i=(w_i^+-m_i)_+},\qquad
m_{\mathrm{destroy}}=\sum_i d_i.
$$

$d_i$ 大表示该源样本激活质量没有低成本共享对应物。它不是“事实纯度”的自动证明，必须用伪配对和独立读出验证。

### 3.3 未配对残差目标

对 $m_i>\epsilon$ 的样本，计算匹配重心的局部参考；对完全未匹配样本，用全体已匹配重心作参考：

$$
b_i=
\begin{cases}
\dfrac{\sum_j\pi_{ij}^*a_j^-}{m_i}, & m_i>\epsilon,\\[6pt]
\dfrac{\sum_{jk}\pi_{jk}^*a_j^-}{\sum_{jk}\pi_{jk}^*+\epsilon}, & m_i\le\epsilon,
\end{cases}
\qquad
u_i=a_i^+-b_i.
$$

Student 目标不是 $b_i$，而是按未配对比例削除残差：

$$
T_i=\operatorname{stopgrad}\!\left[
\Pi_{\mathcal K}\!\left(a_i^+-\alpha\frac{d_i}{w_i^++\epsilon}u_i\right)
\right].
$$

$$
\mathcal L_F=\sum_i\left\|h_{S,\ell,p}(X_i^+\oplus y_i^+)-T_i\right\|_2^2,
\qquad
\boxed{\mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R}.
$$

$\Pi_{\mathcal K}$ 继续限制原事实读出、donor 支持度和 EOS/长度；$\lambda_{KL}^{(\mathrm{ablation})}=0$。若 balanced OT（$\tau\to\infty$）成为最佳臂，则本方案降级为普通 AcT 式对照，不保留“质量销毁”贡献。

## 4. 差异化边界

| 近邻 | 重叠 | 本方案差异 |
|---|---|---|
| AcT / CHaRS | 激活测度、OT 映射 | 未配对源质量定义删除残差；训练内化而非推理 steering |
| UOT-Unlearn（2026） | UOT 与质量重分配用于生成模型遗忘 | 对象是 LLM 反事实中间激活，使用未匹配残差而非生成类别重分配 |
| Opt-Out（ACL 2025） | OT + LLM 遗忘 | 后者正则参数空间；本方案处理答案跨度的激活测度 |
| CF-05 多 donor 边际化 | 多反事实统计 | 不求共享均值/SVD，使用逐样本传输质量和未配对残差 |
| donor KL / $A^-$ 对齐 | 目标语义迁移 | 明确不以 $b_i$ 或 donor 分布为目标 |

## 5. /idea-evaluator 兼容评估（PROVISIONAL）

**定位**：Novel Method。致命缺陷：2 MAJOR——F1 需要与 AcT/UOT-Unlearn 证明“未配对残差”而非命名替换；F6 需要 balanced OT 与 donor 率的决定性对照。零 CRITICAL。Sinkhorn 与多模板成本为黄色资源项，首验须限制样本数、层数和投影维度。

| 维度 | 分数 | 机制依据 |
|---|---:|---|
| Higher | 8 | 直接消除“所有差异必须映射到 donor”的结构性错误 |
| Faster | 6 | 熵正则 Sinkhorn 可批处理，但明显贵于闭式差分 |
| Stronger | 9 | 边缘松弛允许拒绝对噪声/私有成分强行匹配，balanced OT 是硬对照 |
| Cheaper | 7 | 不需 SAE 或人工特征，但需要多模板样本与数值求解 |
| Broader | 9 | 可迁移到多模态、轨迹和其他不可完全配对的表示删除 |

**平均分 $39/5=7.8$。裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 范式探测：First Principles=Yes、Elephant=Yes、Technology Cycle=Yes、Hamming=Partial。

修订顺序：先验证 $d_i$ 跨 seed 稳定；跑 balanced/unbalanced/donor 对齐三臂；再决定是否值得完整 Sinkhorn。

## 6. 决定性实验与失败模式

**最小判别实验**：balanced OT、UOT、直接 $A^-$ 对齐、等范数随机删除四臂；报告 $m_{\mathrm{destroy}}$ 的跨 split 稳定性、$r_f$、donor 答案率、retain NLL、EOS、完整回答率及恢复攻击。

**失败即撤回**：$d_i$ 主要由批种子或模板决定；UOT 与 balanced OT 无显著差异；donor 答案率上升；目标退化为对 $b_i$ 的回归；求解成本超过首验预算。

## 7. 来源与诚信状态

- *Controlling Language and Diffusion Models by Transporting Activations*（ICLR 2025），arXiv:2410.23054。
- *Unlearning for One-Step Generative Models via Unbalanced Optimal Transport*（2026），arXiv:2603.16489：https://arxiv.org/pdf/2603.16489 。
- Choi et al.（ACL 2025），*Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport*。
- 基线 §1、§2.2、§4.1、§5、§6。
- plan-only；分数非实测、非独立评审。文献未做 D1 批量身份核验，不生成 BibTeX。

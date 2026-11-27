---
id: CF2-03
rank: 3
title: 条件有向信息流 ΔC_s→p —— 切断事实传播线，而非搅乱激活节点
avg_score: 7.8
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无跨模型独立回执
novelty_status: PROCEED WITH CAUTION（估计器与近邻需修订）
baseline_relation: 从逐点激活差改为“主体槽→答案跨度”的条件交叉协方差差分
---

# CF2-03 · 条件有向信息流 $\Delta C_{s\to p}$

## 0. 一句话故事

> 事实泄露不是“某个向量太大”，而是主体槽的信息在后续答案位置变得可预测。我们不再逐点匹配 $\Delta A$，而是估计“给定其他上下文后，主体槽对答案跨度的额外解释力”，让 Student 只切断这条条件信息流，同时把两端节点的均值和方差限制在自然范围内。

## 1. 问题与近邻空白

逐点差分无法区分两件事：节点本身包含多少信息，以及信息是否从主体位置传播到答案位置。后者更接近“模型为什么会在这条回答里使用该事实”。

Liang 与 Shi（2026，*Transfer Entropy as a Measure of Information Flow in LLMs and VLMs*）用 transfer entropy 解释层、token、head 间有向信息流并指导剪枝；PINT（arXiv:2611.03955）用 HSIC 降低最终层前缀表示与下一 token 的依赖。本方案不能把“条件独立性”写成新点子，其差异必须同时满足：反事实配对差分、跨位置主体→答案算子、保留边缘统计、Student 参数内化。

与 CF-07 的下游读出 Jacobian 不同，本方案估计的是位置间统计依赖；与 CF-03 的组件 patch 权重不同，不做单点因果门控。

## 2. 可证伪假设

**H**：真实配对的条件主体→答案交叉算子显著强于伪配对；削减该算子后，独立生成式 probe 和 relearning 的目标事实恢复下降，而答案位置方差、语言长度和 retain 效用保持稳定。若只降低交叉协方差却不影响事实读出，或必须同时摧毁边缘激活统计才有效，则假设被证伪。

## 3. 方法

### 3.1 位置与样本矩阵

对一组匹配反事实样本 $i=1,\dots,N$，在主体槽位置 $s$ 和后续答案位置 $p\in S_f$ 取低秩投影激活：

$$
U_i^\pm=P_s h_{T,\ell_s}(X_i^\pm\oplus y_i^+;s),\qquad
V_{i,p}^\pm=P_p h_{T,\ell_p}(X_i^\pm\oplus y_i^+;p).
$$

$P_s,P_p$ 为冻结的随机正交投影或预注册 PCA 投影，仅用于控制估计维度，不参与语义命名。其他上下文激活记为 $Q_i^\pm$。

### 3.2 条件交叉算子

分别估计正、反事实条件下的协方差：

$$
C_{uv\mid q}^{\pm}
=C_{uv}^{\pm}-C_{uq}^{\pm}(C_{qq}^{\pm}+\lambda I)^{-1}C_{qv}^{\pm},
$$

并定义反事实条件流差：

$$
\boxed{\mathcal B_{s\to p}=C_{u v_p\mid q}^{+}-C_{u v_p\mid q}^{-}}.
$$

$C_{uv\mid q}$ 是“给定 $q$ 后，$u$ 对 $v$ 的剩余线性解释结构”，不是完整的因果 transfer entropy。非线性扩展只能在首版线性假设成立后追加。

### 3.3 Student 剪流而非坍缩

Student 在相同批次上估计 $\mathcal B_{s\to p}^{S}$，优化：

$$
\mathcal L_{\mathrm{flow}}=\sum_{p\in S_f}\left\|\mathcal B_{s\to p}^{S}\right\|_F^2.
$$

为防止“把两端都变成常数”的捷径，对主体和答案的边缘均值/协方差设置死区：

$$
\mathcal L_{\mathrm{marg}}=
[\|\mu_U^S-\mu_U^T\|_2^2-\delta_\mu]_+
+[\|\Sigma_U^S-\Sigma_U^T\|_F^2-\delta_\Sigma]_+
+\sum_{p\in S_f}[\|\Sigma_{V_p}^S-\Sigma_{V_p}^T\|_F^2-\delta_\Sigma]_+.
$$

总目标为：

$$
\boxed{
\mathcal L=\lambda_{\mathrm{flow}}\mathcal L_{\mathrm{flow}}
+\lambda_m\mathcal L_{\mathrm{marg}}
+\lambda_n\mathcal L_{\mathrm{neutral}}
+\beta\mathcal L_R}
$$

其中 $\mathcal L_{\mathrm{neutral}}$ 是基线 $r_f$、donor 支持度和 EOS/长度约束的 hinge 形式；$\lambda_{KL}^{(\mathrm{ablation})}=0$。Student 两路前向都参与统计，但 Teacher 和统计目标均 stop-gradient。

## 4. 差异化边界

| 近邻 | 已占机制 | 本方案差异 |
|---|---|---|
| Transfer Entropy for LLMs/VLMs（2026） | 有向信息流诊断与剪枝 | 反事实条件差、答案跨度、遗忘训练与边缘统计死区 |
| PINT（2026） | 最终层前缀—下一 token HSIC | 跨位置主体→答案、正反事实差、条件交叉算子，不是通用 HSIC 目标 |
| CF-07 读出空间 | 下游读出标量差与 Jacobian | 本方案估计跨位置依赖，不重定义读出泛函 |
| 基线 $\Delta A$ | 同位置向量差 | 本方案处理 $s\to p$ 的交叉位置算子 |

查新结论是“可继续但需收窄”：条件交叉协方差不能宣称完整非线性信息流，也不能在没有 patch/独立攻击时宣称因果删除。

## 5. /idea-evaluator 兼容评估（PROVISIONAL）

**定位**：Novel Method。致命缺陷：2 MAJOR——F1 与 TE/PINT 的划界需要实证承载；F6 小样本条件协方差估计可能不稳定。零 CRITICAL，因此为 Accept with Revisions。首版需限制在低秩投影、短答案和单卡 H20，能力匹配黄，缩到线性算子后可执行。

| 维度 | 分数 | 机制依据 |
|---|---:|---|
| Higher | 8 | 直接针对“事实被下游使用”的传播结构，而非只改节点数值 |
| Faster | 6 | 多样本统计增加成本；低秩闭式协方差避免神经 TE 估计器 |
| Stronger | 9 | 边缘死区专门阻断节点坍缩、长度漂移和普遍拒答捷径 |
| Cheaper | 7 | 不需 SAE/人工标签，但需更多配对样本稳定估计 |
| Broader | 9 | 可统一主体→答案、证据→CoT、视觉→文本等跨位置遗忘 |

**平均分 $39/5=7.8$。裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 范式探测：First Principles=Yes、Elephant=Yes、Technology Cycle=Yes、Hamming=Yes。

修订顺序：先冻结低秩线性估计与样本量曲线；证明与 TE/PINT 的差异；最后才增加非线性估计。

## 6. 决定性实验与失败模式

**最小判别实验**：比较“仅边缘激活匹配 / 仅条件流差 / 二者联合”三臂，随样本量 $N$ 报告估计方差；用独立生成式 probe、patch 和 relearning 检验是否真正降低目标读出，并监测 PSD、答案长度、EOS、retain NLL。

**失败即撤回**：交叉算子趋零但 probe 不变；收益完全来自边缘统计坍缩；真实与 wrong-donor 的 $\mathcal B$ 不可分；小样本下符号跨 seed 翻转。

## 7. 来源与诚信状态

- Liang & Shi（2026），*Transfer Entropy as a Measure of Information Flow in LLMs and VLMs*，OpenReview：https://openreview.net/forum?id=rFAQRmjVMf 。
- Shkabrii, Plant & Pawelczyk（2026），*Forget What Comes Next: LLM Unlearning via Statistical Independence*，arXiv:2611.03955：https://arxiv.org/html/2611.03955v1 。
- 基线 §2.2、§4.1、§5、§6。
- plan-only；分数非实测、非独立评审。文献未做 D1 批量身份核验，不生成 BibTeX。

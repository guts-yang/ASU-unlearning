---
id: CF2-08
rank: 8
title: 保留协方差白化谱 ΔA_snr —— 按相对保留波动的信噪比处理激活差
avg_score: 7.4
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无跨模型独立回执
novelty_status: PROCEED WITH CAUTION（LEACE 为强近邻）
baseline_relation: 从欧氏范数 ΔA 改为由独立 retain 协方差归一化的广义谱
---

# CF2-08 · 保留协方差白化谱 $\Delta A_{\mathrm{snr}}$

## 0. 一句话故事

> 原始 $\Delta A$ 的欧氏范数会被高方差“流氓维度”支配：差得大不一定更携带事实，差得小也不一定不重要。我们用独立 retain 激活协方差作自然尺度，求“反事实差异能量 / 保留波动能量”的广义特征谱，只处理相对保留分布异常的方向。

## 1. 问题与强近邻

LEACE（Belrose et al., NeurIPS 2023）已经用类内协方差归一化和广义特征问题做闭式线性概念擦除；CIR（arXiv:2509.11816）也用激活 PCA 去除共享表示。因此“白化后擦除”本身不能作为新贡献。

本方案只是 LEACE/CIR 的反事实配对改写：散射来自**每对正反事实差**而非类别均值差，归一化来自**独立 retain 分布**而非两类激活的类内协方差，目标是 Student 在答案跨度上的参数内化，而非一次性线性投影。只有这些差异经实验带来增量时才值得保留。

## 2. 可证伪假设

**H**：存在一组方向，其反事实配对能量相对 retain 波动显著高于随机/wrong-donor；按该信噪比选择方向比按原始范数或 LEACE 类均值散射更稳定。若白化方向与原始差或 LEACE 高度重合，则没有独立信息，应归档。

## 3. 方法

### 3.1 配对散射与保留尺度

对 $N$ 个反事实配对，在固定 $(\ell,p)$ 上收集 $\delta_i=\Delta A_{\ell,p}^{(i)}$，并估计：

$$
C_\Delta=\frac1N\sum_{i=1}^{N}\delta_i\delta_i^\top.
$$

用独立 retain-train 激活估计自然尺度：

$$
C_R=\operatorname{Cov}_{x\sim\mathcal D_R}(h_{T,\ell}(x;p))+\lambda I.
$$

$C_R$ 不含 forget 或 donor 样本，避免用目标分布自己定义“正常波动”。

### 3.2 广义信噪谱

求解：

$$
\boxed{C_\Delta u_j=\lambda_j C_R u_j},\qquad
\lambda_j=\frac{u_j^\top C_\Delta u_j}{u_j^\top C_R u_j},\qquad
U^\top C_R U=I.
$$

$\lambda_j$ 是第 $j$ 个方向的反事实差异/保留波动比。取前 $m$ 个方向 $U_m$，构造：

$$
\Delta A_{\mathrm{snr}}=U_mU_m^\top C_R\,\Delta A.
$$

$U_mU_m^\top C_R$ 是在 $C_R$ 内积下的斜投影；$m$、$\lambda$ 和条件数上限在实验前冻结。

### 3.3 Student 内化与协方差保护

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left[
\Pi_{\mathcal K}(A^+_{\ell,p}-\alpha\Delta A_{\mathrm{snr},\ell,p})
\right],
$$

$$
\mathcal L_F=\sum_{\ell,p}\|h_{S,\ell}(X^+\oplus y^+;p)-T_{\ell,p}\|_2^2
+\gamma[\|\operatorname{Cov}(h_S^R)-C_R\|_F^2-\delta_R]_+,
$$

$$
\boxed{\mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R}.
$$

$\Pi_{\mathcal K}$ 保留基线中性约束；$\lambda_{KL}^{(\mathrm{ablation})}=0$。所有谱估计均 stop-gradient，Student 参数梯度只来自其自身前向。

## 4. 与 LEACE/CIR 的真实差异

| 近邻 | 散射/基 | 本方案差异 |
|---|---|---|
| LEACE（2023） | 类均值差与类内协方差，闭式线性擦除 | 每个反事实配对的差分散射 + 独立 retain 尺度 + Student 训练 |
| CIR（2025） | 激活/梯度 PCA 去除通用子空间 | 使用正反事实差的广义信噪谱，不做通用 PCA collapse |
| CF-08 零空间投影 | 投影到 retain top-PC 零空间 | 本方案不进入零空间，而是按 $C_\Delta/C_R$ 比值排序 |
| 原始 $\Delta A$ | 欧氏几何 | Mahalanobis 相对异常几何 |

这是一项“有明确近邻的增量候选”，不是新的概念擦除框架。

## 5. /idea-evaluator 兼容评估（PROVISIONAL）

**定位**：Novel Method（增量型）。致命缺陷：2 MAJOR——F1 与 LEACE 的增量可能过薄；F6 需要证明与原始差/LEACE 非重合且有效。零 CRITICAL，故 Accept with Revisions。计算主要为缓存协方差与低秩特征分解，资源匹配绿。

| 维度 | 分数 | 机制依据 |
|---|---:|---|
| Higher | 7 | 相对保留波动比欧氏幅度更贴近异常程度，但仍是线性方法 |
| Faster | 8 | 协方差和广义特征分解均可离线闭式计算 |
| Stronger | 8 | 独立 retain 尺度和 wrong-donor 谱对照可抗高方差伪信号 |
| Cheaper | 8 | 无 SAE、probe 或新标注，复用 retain 激活 |
| Broader | 6 | 可迁移其他表示，但 LEACE/CIR 已覆盖较宽范围 |

**平均分 $37/5=7.4$。裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 范式探测：First Principles=Partial、Elephant=Yes、Technology Cycle=No、Hamming=Partial。

修订顺序：先比较方向重合度；若与 LEACE 高度重合则停止；只有显著非重合才进入训练。

## 6. 决定性实验与失败模式

**最小判别实验**：原始范数选向、LEACE 方向、retain 白化谱、随机等能方向四臂；先报告子空间 principal angles，再做等预算遗忘。监测条件数、$r_f$、retain NLL、donor 率、恢复攻击和跨 seed 谱稳定性。

**失败即撤回**：与 LEACE 子空间近似重合；$C_R$ 病态导致方向由正则项决定；真实与 wrong-donor 的 $\lambda$ 谱不可分；效果来自更大投影能量。

## 7. 来源与诚信状态

- Belrose et al.（NeurIPS 2023），*LEACE: Perfect Linear Concept Erasure in Closed Form*。
- Sondej & Yang（2025），*Collapse of Irrelevant Representations Ensures Robust and Non-Disruptive LLM Unlearning*，arXiv:2509.11816：https://arxiv.org/html/2509.11816 。
- 基线 §2.2、§4.1、§6。
- plan-only；分数非实测、非独立评审。文献未做 D1 批量身份核验，不生成 BibTeX。

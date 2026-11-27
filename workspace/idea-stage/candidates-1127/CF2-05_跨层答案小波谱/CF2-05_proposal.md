---
id: CF2-05
rank: 5
title: 跨层—答案小波谱 WΔA —— 按跨配对稳定尺度删除，而非按层一刀切
avg_score: 7.6
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无跨模型独立回执
novelty_status: PROCEED WITH CAUTION（尺度语义需验证）
baseline_relation: 把 ΔA 从层×位置原始坐标变换到层×答案位置的多尺度坐标
---

# CF2-05 · 跨层—答案小波谱 $\mathcal W\Delta A$

## 0. 一句话故事

> 有些反事实差异是孤立 token/单层的尖峰，有些则跨多个答案位置和多层形成连贯结构。原始 $\Delta A$ 把二者混在同一个坐标里。我们对“层 × 答案位置”做二维小波分解，用跨释义一致性和 wrong-donor 对照学习尺度掩码，只删除稳定的目标尺度，再逆变换为 Student 的内化目标。

## 1. 问题与近邻空白

Chakrabarti et al.（2026，arXiv:2605.21391）用连续小波和 conditional scale entropy 分析隐喻处理的跨层结构；Tamkin et al.（NeurIPS 2020）说明不同频率可携带不同粒度语言信息。这些工作支持“尺度是可解释坐标”，但不支持“某固定尺度就是事实”。

本方案的贡献不是把小波套到 LLM，而是：**用反事实配对在小波域构造选择性遗忘掩码，并训练 Student 内化。** 与 CF-06 不同，作用域仍是答案跨度而非 CoT；与 CF-10 不同，掩码作用于尺度而非层加入顺序；与 CF-09 不同，不构造噪声子空间回归。

## 2. 可证伪假设

**H**：目标事实差异在某些尺度上跨释义稳定，而 wrong-donor / subword 噪声的尺度分布不同。若尺度掩码跨 split 不稳定，或“全尺度 / 仅低频 / 仅高频”之间无差别，则多尺度分解没有带来有效归纳偏置，应撤回。

## 3. 方法

### 3.1 层×答案位置激活差张量

冻结 Teacher，在答案跨度上构造：

$$
\Delta A_{\ell,p,d}
=h_{T,\ell,d}(X^+\oplus y^+;p)-h_{T,\ell,d}(X^-\oplus y^+;p),
\quad p\in S_f.
$$

仅沿“层 $\ell$”和“答案位置 $p$”做二维离散小波变换，隐藏通道 $d$ 不作为时间轴：

$$
z_{j,b,q,d}=\left[\mathcal W_{\ell,p}\Delta A\right]_{j,b,q,d},
$$

其中 $j$ 为尺度，$b$ 为层定位，$q$ 为答案位置定位。边界延拓和最短序列长度在实验前冻结。

### 3.2 跨配对尺度掩码

对 $N$ 个释义配对，计算每尺度的真实一致性和伪配对一致性：

$$
\mathrm{coh}_j^{\mathrm{true}}=
\frac{1}{N(N-1)}\sum_{i\ne k}
\cos\!\left(z_j^{\mathrm{true},(i)},z_j^{\mathrm{true},(k)}\right),
$$

$$
m_j=\sigma\!\left(
\frac{\mathrm{coh}_j^{\mathrm{true}}-
\operatorname{median}(\mathrm{coh}_j^{\mathrm{wrong}},\mathrm{coh}_j^{\mathrm{shuffle}})-\eta}{\tau}
\right).
$$

$m_j$ 是尺度掩码，不是语义标签。不得预先宣称“低频=事实、高频=词形”；低频-only 与高频-only 是强制对照。

### 3.3 逆变换目标与 Student 内化

$$
R=\mathcal W^{-1}\!\left(
\mathcal W(A^+)-\alpha\,m\odot\mathcal W(\Delta A)
\right),
\qquad
T_{\ell,p}=\operatorname{stopgrad}[\Pi_{\mathcal K}(R_{\ell,p})].
$$

$$
\mathcal L_F=\frac1{|\Lambda||S_f|}\sum_{\ell\in\Lambda}\sum_{p\in S_f}
\|h_{S,\ell}(X^+\oplus y^+;p)-T_{\ell,p}\|_2^2,
\qquad
\boxed{\mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R}.
$$

$\Pi_{\mathcal K}$ 保持基线的原事实、donor 和 EOS 约束；$\lambda_{KL}^{(\mathrm{ablation})}=0$。小波正逆变换均可预计算为矩阵算子，Teacher 目标 stop-gradient。

## 4. 差异化边界

| 近邻 | 已占机制 | 本方案差异 |
|---|---|---|
| Conditional Scale Entropy（2026） | 小波用于跨层解释和尺度熵 | 用正反事实尺度一致性构造训练目标，不计算 CSE |
| Language Through a Prism（2020） | 频率选择表示多粒度语言信息 | 反事实遗忘任务、答案跨度与 Student 内化 |
| CF-06 / CF-10 | CoT 轨迹 / 纯度 schedule | 本方案不改序列作用域，不决定层训练顺序 |
| CF-09 残差化 | 扣除混淆 | 小波基固定且可逆，不从伪配对学习噪声子空间 |

## 5. /idea-evaluator 兼容评估（PROVISIONAL）

**定位**：Novel Method。致命缺陷：2 MAJOR——F1 与小波解释工作需证明训练增量的独立性；F6 尺度语义跨 split 可能不稳定。零 CRITICAL，故 Accept with Revisions。首验只用 Haar/Daubechies 固定小波和少量尺度，资源匹配黄偏绿。

| 维度 | 分数 | 机制依据 |
|---|---:|---|
| Higher | 8 | 多尺度坐标能分开孤立尖峰与连贯传播，提供原始差没有的信息 |
| Faster | 7 | 固定小波为预计算线性算子，无额外训练模块 |
| Stronger | 8 | 跨释义、wrong/shuffle 和全尺度对照直接检验尺度伪信号 |
| Cheaper | 7 | 无 SAE/人工标签，但需要多释义配对 |
| Broader | 8 | 可推广到 CoT、跨模态token网格和其他层次信号 |

**平均分 $38/5=7.6$。裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 范式探测：First Principles=Yes、Elephant=Partial、Technology Cycle=Partial、Hamming=Yes。

修订顺序：先做尺度稳定性；再比较低频、高频、全尺度；通过后才训练 Student。

## 6. 决定性实验与失败模式

**最小判别实验**：学习掩码、全尺度、仅低频、仅高频、随机尺度五臂，使用相同删除能量预算；报告 $r_f$、retain NLL、donor 率、EOS、不同答案长度下的稳定性及恢复攻击。

**失败即撤回**：$m_j$ 跨 split/seed 排名翻转；随机尺度与学习尺度等效；逆变换引入明显激活范数偏移；效果只来自总删除能量增加。

## 7. 来源与诚信状态

- Chakrabarti et al.（2026），*Post-Hoc Understanding of Metaphor Processing in Decoder-Only Language Models via Conditional Scale Entropy*，arXiv:2605.21391：https://arxiv.org/abs/2605.21391 。
- Tamkin, Jurafsky & Goodman（NeurIPS 2020），*Language Through a Prism: A Spectral Approach for Multiscale Language Representations*。
- 基线 §2.2、§2.3、§4.1、§6。
- plan-only；分数非实测、非独立评审。文献未做 D1 批量身份核验，不生成 BibTeX。

---
id: CF2-06
rank: 6
title: 反事实割线敏感度 D_sec —— 让目标绑定对主体替换失去响应
avg_score: 7.6
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无跨模型独立回执
novelty_status: PROCEED WITH CAUTION（反事实不变性近邻较多）
baseline_relation: 从激活位移 ΔA 改为经主体嵌入位移归一化的响应斜率
---

# CF2-06 · 反事实割线敏感度 $D_{\mathrm{sec}}$

## 0. 一句话故事

> 同样的 $\Delta A$ 可能来自一次很小的主体变化，也可能来自一次很大的主体变化；只比较位移会把“响应强度”和“输入距离”混在一起。我们用多组主体替换的归一化割线 $D_{\mathrm{sec}}=\Delta A/(\|\Delta e_s\|+\epsilon)$ 估计答案激活对主体干预的响应，只让目标关系的响应消失，同时锚定保留关系的响应。

## 1. 问题与近邻空白

*Aligning Language Models Towards Counterfactual Invariance*（arXiv:2610.22942）已经在输出/奖励层训练反事实不变性，并区分“应当不变”与“应当响应”的条件。更早的 counterfactual invariance 文本分类工作也占据一般概念。因此本方案不能以“反事实不变性”作为新贡献。

可辩护增量是：**在中间层答案跨度上，以归一化响应割线作为选择性遗忘对象，并用保留关系的响应斜率作锚。** 与 CF-07 的 $\partial r_f/\partial h$ 不同，这里估计的是 $\Delta h/\|\Delta e_s\|$，即对输入主体干预的响应，不是下游读出的局部梯度。

严格说，离散主体替换只能产生割线，不能自动给出连续 Jacobian；proposal 全程使用 $D_{\mathrm{sec}}$，不夸大为真实导数。

## 2. 可证伪假设

**H**：目标事实响应在多种主体替换距离下具有稳定割线结构；最小化 $D_{\mathrm{sec}}^{f}$ 比直接最小化 $\Delta A$ 更少受到主体词长和替换距离影响，而 $D_{\mathrm{sec}}^{r}$ 锚能保存同实体其他关系。若割线随替换距离剧烈变化，或零响应只是把原事实冻结在两边，则假设被证伪。

## 3. 方法

### 3.1 多距离反事实对

构造 $M$ 组主体替换 $(s_f^{(m)},s_r^{(m)})$，覆盖别名、同类型实体和不同 token 距离。主体嵌入位移为：

$$
\Delta e_s^{(m)}=e(s_f^{(m)})-e(s_r^{(m)}),\qquad
\Delta A_{\ell,p}^{(m)}=A_{\ell,p}^{+,(m)}-A_{\ell,p}^{-,(m)}.
$$

归一化割线响应定义为：

$$
\boxed{
D_{\mathrm{sec},\ell,p}
=\frac1M\sum_{m=1}^{M}
\frac{\Delta A_{\ell,p}^{(m)}}{\|\Delta e_s^{(m)}\|_2+\epsilon}}
$$

多 token 主体使用其嵌入矩阵的 Frobenius 范数；不同替换组合按相同规则聚合。$M$、替换集合和 $\epsilon$ 在实验前冻结。

### 3.2 Student 目标关系选择性

对目标关系 $r_f$ 和非目标关系 $r\in\mathcal R_{\mathrm{retain}}$，Student 分别计算割线。主损失为：

$$
\mathcal L_{\mathrm{sec}}
=\underbrace{\sum_{\ell,p}\|D_{\mathrm{sec},\ell,p}^{S,r_f}\|_2^2}_{\text{目标绑定不再随主体替换传播}}
+\eta\underbrace{\sum_{r\in\mathcal R_{\mathrm{retain}}}\sum_{\ell,p}
\|D_{\mathrm{sec},\ell,p}^{S,r}-D_{\mathrm{sec},\ell,p}^{T,r}\|_2^2}_{\text{其他关系保持正常响应}}.
$$

加入基线中性约束，防止“两个输入都停在原事实状态”这一零割线捷径：

$$
\mathcal L_{\mathrm{neutral}}
=[r_f-c]_++\sum_j[s_{d_j}-d_j]_++[\|\Delta r_E\|-e]_+.
$$

$$
\boxed{\mathcal L=\lambda_{\mathrm{sec}}\mathcal L_{\mathrm{sec}}
+\lambda_n\mathcal L_{\mathrm{neutral}}+\beta\mathcal L_R},
\qquad \lambda_{KL}^{(\mathrm{ablation})}=0.
$$

Teacher 只提供替换响应的校准和保留关系锚，不提供 donor 答案目标。Student 的两路前向均可微，Teacher 全部 stop-gradient。

## 4. 差异化边界

| 近邻 | 已占机制 | 本方案差异 |
|---|---|---|
| Counterfactual Invariance（2021/2026） | 输出或奖励在无关属性替换下保持一致 | 中间层答案跨度割线、目标关系选择性、参数遗忘 |
| CF-07 读出空间 Jacobian | $\partial r_f/\partial h$ 的局部敏感度 | 本方案估计 $\Delta h/\|\Delta e_s\|$，对象是输入主体干预响应 |
| 基线 $\Delta A$ | 绝对激活位移 | 按主体嵌入距离归一化并聚合多距离替换 |
| donor $A^-$ 对齐 | 逼近反事实状态 | 目标不是 $A^-$；保留关系仍需正常响应 |

## 5. /idea-evaluator 兼容评估（PROVISIONAL）

**定位**：Novel Method。致命缺陷：2 MAJOR——F1 与输出层反事实不变性需要明确机制划界；F6 零割线可能冻结原事实而非删除。零 CRITICAL，故 Accept with Revisions。多替换前向增加成本，首验限 $M\le4$ 和短答案，资源匹配黄偏绿。

| 维度 | 分数 | 机制依据 |
|---|---:|---|
| Higher | 8 | 分离输入距离与响应强度，针对原始 $\Delta A$ 的尺度混淆 |
| Faster | 7 | 割线闭式计算，但需多组替换前向 |
| Stronger | 8 | 保留关系响应锚和零割线反例门专门防止全局不变性 |
| Cheaper | 8 | 不需要 probe、SAE 或额外答案标注 |
| Broader | 7 | 可用于属性、证据和多模态干预，但嵌入距离跨模态较难比较 |

**平均分 $38/5=7.6$。裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 范式探测：First Principles=Yes、Elephant=Yes、Technology Cycle=Partial、Hamming=Partial。

修订顺序：先验证割线随替换距离的稳定性；加入“双边冻结原事实”反例；再做与原始差损失的等预算对照。

## 6. 决定性实验与失败模式

**最小判别实验**：割线目标、原始 $\Delta A$ 目标、输出配对不变性三臂；额外加入“$X^+$ 与 $X^-$ 激活都冻结在原事实”的负对照。报告不同主体距离下的 $r_f$、同实体非目标关系、retain NLL、probe/relearning 恢复和输出长度。

**失败即撤回**：$D_{\mathrm{sec}}$ 随替换距离无稳定规律；零割线但 probe 仍能完整读出原事实；保留关系响应被同步压平；收益只来自 $M$ 增大带来的更多训练样本。

## 7. 来源与诚信状态

- Demir et al.（2026），*Aligning Language Models Towards Counterfactual Invariance*，arXiv:2610.22942：https://arxiv.org/html/2610.22942 。
- Veitch et al.（NeurIPS 2021），*Counterfactual Invariance to Spurious Correlations in Text Classification*。
- 基线 §2.2、§4.1、§5、§6。
- plan-only；分数非实测、非独立评审。文献未做 D1 批量身份核验，不生成 BibTeX。

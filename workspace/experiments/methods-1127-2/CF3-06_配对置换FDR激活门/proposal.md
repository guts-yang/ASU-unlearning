---
id: CF3-06
rank: 6
title: 配对置换—FDR 激活门（PEFG）
avg_score: 7.6
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无独立跨模型回执
novelty_status: PROCEED WITH CAUTION
baseline_relation: 将激活差的层×位置选择从无误差控制的幅值门改为配对符号置换与多重检验控制
---

# CF3-06 · 配对置换—FDR 激活门

## 0. 一句话故事

> 全层全位置上总会出现一些“看起来很大”的反事实差，其中许多只是多重比较产生的假阳性。我们利用同一配对内正负标签的符号置换构造非参数零分布，对层×位置差异做严格的错误发现控制；Student 只处理通过统计门和独立因果读出门的位置，而不是按范数排行榜删除。

## 1. 问题与近邻空白

现有候选常用 median/MAD、因果分数或纯度阈值选择层和位置，但对数百个层×位置同时筛选时，假阳性率没有明确控制。神经影像中的配对置换、max-statistic 与 FDR 已很成熟，因此“置换检验”不是新工具。

本方案的新轴仅在于：把**反事实激活差的可交换性零假设**变成训练对象的选择规则，并在 Student 遗忘中审计统计门的增量。它不同于 CF-03 的因果 patch 权重、CF-09 的 null 残差回归、CF2-01 的 shuffle 范数权重和 CF-10 的纯度课程；本方案不做 schedule，也不把统计显著性误写成因果性。

## 2. 可证伪假设

**H**：真实配对的层×位置统计量相对配对置换零分布存在稳定超出；经多重检验控制后留下的位置比等数量 top-norm 位置更能预测目标读出，并减少随机删除带来的 retain 损伤。若置换门选择的仍是主体词形/长度位置，或独立 patch 不显示目标效应，则假设被证伪。

## 3. 方法

### 3.1 配对统计量与符号置换

对 $i=1,\ldots,N$ 个匹配反事实对，在层 $\ell$、答案位置 $p$ 计算：

$$
z_{i,\ell,p}=\frac{\Delta A_{i,\ell,p}}
{\operatorname{RMS}_{R,\ell,p}+\epsilon},
\qquad
T_{\ell,p}=\left\|\frac1N\sum_{i=1}^N z_{i,\ell,p}\right\|_2^2.
$$

这里只用 retain RMS 做单位归一化，不做 CF2-08 的协方差白化。对第 $b$ 次置换，独立采样 $s_i^{(b)}\in\{-1,+1\}$：

$$
T_{\ell,p}^{(b)}=
\left\|\frac1N\sum_i s_i^{(b)}z_{i,\ell,p}\right\|_2^2.
$$

在“正负条件可交换且无平均配对效应”的零假设下，得到：

$$
\widehat p_{\ell,p}=
\frac{1+\sum_{b=1}^B\mathbf 1[T_{\ell,p}^{(b)}\ge T_{\ell,p}]}{B+1}.
$$

### 3.2 依赖稳健的多重检验门

对全部预注册层×位置使用 Benjamini–Yekutieli 阈值控制任意依赖下的 FDR；以 cluster-level max-$T$ 置换作为敏感性分析。定义：

$$
\boxed{
w_{\ell,p}=\mathbf 1[q_{\ell,p}^{BY}\le q_0]\,
\mathbf 1[\text{跨 split 稳定}]\,
\mathbf 1[\text{独立 patch 读出门通过}]}
$$

最后一项必须独立于用于计算 $p$ 值的 split，避免双重 dipping。统计显著但 patch 不改变 $r_f$ 的位置被明确归为“相关而非承载”。

### 3.3 Student 训练

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left[\Pi_{\mathcal K}
\left(A^+_{\ell,p}-\alpha w_{\ell,p}\Delta A_{\ell,p}\right)\right],
$$

$$
\mathcal L_F=\sum_{\ell,p\in S_f}\|h_{S,\ell,p}-T_{\ell,p}\|_2^2,
\qquad
\boxed{\mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R}.
$$

$w$ 固定，不随训练重估；主结果固定 $\lambda_{KL}^{(\mathrm{ablation})}=0$。wrong/shuffle、matched-retain 与随机等数量位置仍是独立对照。

## 4. idea-evaluator 兼容评估

**定位**：Novel Method。F6 为 MAJOR：可交换性和统计显著性不等于事实因果，必须经独立 patch 门；F1 为 MAJOR：需证明相对既有 null/纯度门的收益来自错误率控制而非阈值调优。零 CRITICAL。

| 维度 | 分数 | 机制依据，均尚未由本轮数据确认 |
|---|---:|---|
| Higher | 7 | 减少全层搜索中的假阳性位置，但本身不提供新的事实语义 |
| Faster | 8 | 置换只重排已缓存差值，不需额外模型前向；独立 patch 仅用于入选位置 |
| Stronger | 8 | 显式控制多重比较、跨 split 和因果读出门，抗随机伪峰能力明确 |
| Cheaper | 8 | 不需新数据、标注或训练统计网络；置换可在低秩缓存上完成 |
| Broader | 7 | 可迁移到任意 matched activation contrast，但要求可交换配对设计 |

**平均分 $\bar S=38/5=7.6$；裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 生命周期 3–9 个月；每周工时未知，资源匹配 Yellow。范式探测：First Principles=Partial，Elephant=Yes，Technology Cycle=No，Hamming=Partial。

前三项行动：验证配对零假设与 split 独立性；比较 BY/max-$T$/top-norm 的假阳性率；用独立 patch 检查入选位置是否真正改变 $r_f$。

## 5. 决定性实验与失败判据

**最小实验**：在 matched-retain 上估计假阳性率，在真实 forget 上比较置换门、top-norm、median/MAD 和随机等数量位置。报告目标读出、位置跨 seed 重合率、retain NLL、donor 率、EOS、probe 与 relearning。所有阈值、置换数和层集预注册。

**失败即撤回**：matched-retain 假阳性率仍失控；置换门与 top-norm 选择几乎相同；显著位置没有独立因果读出；或门随 $B$/split 微小变化而大幅改变。

## 6. 来源与诚信状态

- Benjamini & Hochberg（1995）与 Benjamini & Yekutieli（2001），用于 FDR 及依赖条件下的控制背景。
- Lage-Castellanos et al.（2009），*False discovery rate and permutation test: an evaluation in ERP data analysis*，DOI:10.1002/sim.3784，作为跨领域方法近邻。
- 基线 §2、§4、§6；明确区别于 CF-03、CF-09、CF2-01、CF-10。

plan-only；分数为机制推断，不是实测或独立 reviewer verdict；文献未完成 D1 批量核验。

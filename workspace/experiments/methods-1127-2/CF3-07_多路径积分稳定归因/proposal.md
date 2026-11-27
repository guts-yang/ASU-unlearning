---
id: CF3-07
rank: 7
title: 多路径积分稳定归因（MPIA）
avg_score: 7.4
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无独立跨模型回执
novelty_status: PROCEED WITH CAUTION
baseline_relation: 将端点激活差分解为主体替换路径上的逐坐标贡献，并只保留跨路径稳定的分量
---

# CF3-07 · 多路径积分稳定归因

## 0. 一句话故事

> 同一个端点激活差可以由完全不同的内部归因路径产生，直线路径还可能穿过无意义的嵌入区域。我们沿多条主体槽路径计算激活差的完备归因，只保留跨路径符号一致、贡献稳定的分量；Student 删除这部分路径稳健差异，而不是把一次任意插值当作事实方向。

## 1. 问题与近邻空白

Integrated Gradients 提供完备性，但路径选择会影响归因。DIG、Sequential IG 处理语言嵌入的离散性，*Beyond Single Path Integrated Gradients* 用随机路径提高可靠性，IG-Lens（2026）则把分段 IG 用于逐层概率归因。因此，本方案不能声称“首次多路径 IG”。

可辩护增量是：以**反事实主体替换为路径端点**，对中间层激活差做逐主体坐标、逐隐维的完备分解，以路径一致性作为删除分量，并进行 Student 参数遗忘。它不同于 CF2-06 的端点割线 $\Delta A/\|\Delta e_s\|$，也不同于 CF-07 的读出 Jacobian；本方案研究的是输入替换路径对 $\Delta A$ 的归因分配。

## 2. 可证伪假设

**H**：真实配对存在跨路径稳定的归因分量，而 wrong/shuffled 配对的归因更随路径变化；删除稳定分量比端点差或单路径 IG 更能降低目标事实读出并减少 OOD 副作用。若所有路径给出相同结果、稳定性只反映数值误差，或稳定分量主要是主体词形，则假设被证伪。

## 3. 方法

### 3.1 主体槽多路径

固定非主体 token 与答案 $y^+$。令 $e^-,e^+$ 为反事实和原事实主体槽嵌入，构造 $M$ 条路径：

$$
\gamma^m(0)=e^-,\qquad \gamma^m(1)=e^+.
$$

首版仅包含主体槽直线与经真实词嵌入锚点的分段路径；所有路径固定其他输入。不得将 padding/zero 当作无信息主体基线。

### 3.2 完备逐坐标归因

对层 $\ell$、答案位置 $p$、隐维 $d$，定义第 $j$ 个主体嵌入坐标的贡献：

$$
z^{m}_{j,d}=
\int_0^1
\frac{\partial A_{\ell,p,d}(\gamma^m(t))}{\partial\gamma_j}
\dot\gamma_j^m(t)\,dt.
$$

由梯度定理，在精确积分和路径连续条件下：

$$
\sum_j z^m_{j,d}=A^+_{\ell,p,d}-A^-_{\ell,p,d}.
$$

注意：完备性是对主体坐标 $j$ 求和，不是声称每条路径产生不同的总端点差。跨路径变化衡量的是**归因分配稳定性**。

对 $M$ 条路径定义：

$$
\bar z_{j,d}=\frac1M\sum_m z^m_{j,d},\qquad
s_{j,d}=\frac{|\bar z_{j,d}|}{\frac1M\sum_m|z^m_{j,d}|+\epsilon}.
$$

于是构造路径稳健分量：

$$
\boxed{
\Delta A^{\mathrm{path}}_{\ell,p,d}
=\sum_j s_{j,d}\bar z_{j,d}}
$$

若数值积分误差超过跨路径方差，先增加 quadrature，而不能把误差解释为语义。

### 3.3 Student 内化

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left[\Pi_{\mathcal K}
\left(A^+_{\ell,p}-\alpha\Delta A^{\mathrm{path}}_{\ell,p}\right)\right],
$$

$$
\boxed{\mathcal L=\lambda_F\sum_{\ell,p\in S_f}
\|h_{S,\ell,p}-T_{\ell,p}\|_2^2+\beta\mathcal L_R}.
$$

$\Pi_{\mathcal K}$ 维持原事实读出、donor 支持度和 EOS/长度约束；不拼 $y^-$，Teacher 与路径目标 stop-gradient，部署无路径计算。

## 4. idea-evaluator 兼容评估

**定位**：Novel Method。F1 为 MAJOR：需与多路径 IG、IG-Lens 和 CF2-06 划界；F6 为 MAJOR：路径稳定性可能只是积分误差或 OOD 程度，需要数值与分布内对照。零 CRITICAL。

| 维度 | 分数 | 机制依据，均尚未由本轮数据确认 |
|---|---:|---|
| Higher | 8 | 分离端点大小与归因路径稳定性，针对单路径伪定位提供直接修正 |
| Faster | 6 | 多路径多积分点增加成本，但可限制为两条离散化路径和少量 quadrature |
| Stronger | 8 | 显式检查路径选择、数值完备性和 wrong/null 配对，抗任意基线伪影 |
| Cheaper | 7 | 不需人工归因标签或额外模型；训练部署不需要路径积分 |
| Broader | 8 | 可应用于主体、证据、视觉 token 等局部替换路径 |

**平均分 $\bar S=37/5=7.4$；裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 生命周期 3–9 个月；每周工时未知，资源匹配 Yellow。范式探测：First Principles=Yes，Elephant=Yes，Technology Cycle=Partial，Hamming=Partial。

前三项行动：先证明数值完备性与路径方差可区分；比较多路径与单路径/割线的目标读出；只在稳定分量通过 null 和 OOD 门后训练。

## 5. 决定性实验与失败判据

**最小实验**：两条路径、4–8 个积分点、4 个锚定层；随积分点增加报告完备性误差。比较稳定路径分量、单路径 IG、CF2-06 割线、原始等范数差和随机路径分量。评估 $r_f$、retain、donor、EOS、probe 与 relearning。

**失败即撤回**：跨路径方差可由积分误差解释；真实与 wrong 配对的 $s_{j,d}$ 不可分；稳定分量不改变目标读出；或真实词嵌入锚点路径并不优于直线路径。

## 6. 来源与诚信状态

- Sundararajan et al.（2017），*Axiomatic Attribution for Deep Networks*。
- *Discretized Integrated Gradients for Explaining Language Models*（EMNLP 2021）；*Sequential Integrated Gradients*（ACL Findings 2023）。
- Jeon et al.（ICCV 2023），*Beyond Single Path Integrated Gradients*。
- *IG-Lens*（2026），arXiv:2606.29693，作为逐层概率归因强近邻。

plan-only；分数为机制推断，文献未完成 D1 批量核验，不是独立接受回执。

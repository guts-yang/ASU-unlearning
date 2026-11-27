---
id: CF3-04
rank: 4
title: 门控状态转移擦除（GRTE）
avg_score: 7.6
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无独立跨模型回执
novelty_status: PROCEED WITH CAUTION
baseline_relation: 将连续激活差改写为 SwiGLU 门控/数值预激活的离散状态转移，再用软状态概率内化
---

# CF3-04 · 门控状态转移擦除

## 0. 一句话故事

> 连续差值相近的两个神经元，可能处在完全不同的计算状态。对 SwiGLU，我们分别记录 gate 与 value 预激活的正负，构成四种可解释状态；只处理在真实反事实对中稳定发生、在伪配对中不发生的“源状态迁出”，让 Student 离开目标事实的计算状态，而不是模仿 donor 的落点状态。

## 1. 问题与近邻空白

GLUScope（Gerstner & Schütze，2026）指出 SwiGLU 的 gate/input 四种符号组合具有不同功能；*The Sign Code*（2026）进一步研究离散符号状态的因果作用。这说明“状态码”本身不是新贡献，但也提供了比任意高阶统计更具体的机制依据。

本方案不同于 CodeUnlearn：不训练或修改码本，不删除离散概念码，也不要求模型经过量化瓶颈。它也不同于 CF2-09 的 $L+S$ 稀疏幅值分解：核心对象是**预激活状态转移图**。与 CF-03 的因果门控不同，门控权重不是 patch 后的连续因果分数，而是神经元计算状态的迁移统计。

## 2. 可证伪假设

**H**：部分目标事实绑定表现为跨 paraphrase 稳定的 SwiGLU 源状态及迁移模式；仅削弱这些源状态比等幅值激活抑制更少损伤 retain，并且不会提高 donor 状态占据率。若状态转移主要由 token 边界或随机噪声决定，或 sign-only 与 magnitude-only 对照无差别，则假设被证伪。

## 3. 方法

### 3.1 四状态与反事实转移图

对层 $\ell$、答案位置 $p$、神经元 $j$，取：

$$
g^\pm_{\ell,p,j}=\langle w_{\ell,j}^{gate},x^\pm_{\ell,p}\rangle,\qquad
u^\pm_{\ell,p,j}=\langle w_{\ell,j}^{in},x^\pm_{\ell,p}\rangle.
$$

定义状态：

$$
q^\pm_{\ell,p,j}=(\mathbf 1[g^\pm_{\ell,p,j}>0],\ \mathbf 1[u^\pm_{\ell,p,j}>0])
\in\{0,1\}^2.
$$

对真实配对和 matched null 分别估计转移矩阵 $P^{true}_{j,ab}=P(q^-=b\mid q^+=a)$ 与 $P^{null}_{j,ab}$。定义源状态特异度：

$$
\omega_{j,a}=\sum_b |P^{true}_{j,ab}-P^{null}_{j,ab}|
\times \frac{\|\Delta m_{j,ab}\|_2}{\operatorname{median}_{null}\|\Delta m_{j,ab}\|_2+\epsilon},
$$

其中 $\Delta m_{j,ab}=w_j^{out}(\operatorname{Swish}(g^+)u^+-\operatorname{Swish}(g^-)u^-)$。$\omega$ 只用于选择候选，不自动证明事实语义。

### 3.2 可微源状态迁出

对通过跨 split 稳定性和 patch 读出门的源状态 $(j,a)$，以 $\kappa_{a,g},\kappa_{a,u}\in\{-1,+1\}$ 表示该状态要求的符号，定义 Student 的软状态占据：

$$
\widetilde q_{S,j,a}=
\sigma(\kappa_{a,g}g_{S,j}/\tau)\,
\sigma(\kappa_{a,u}u_{S,j}/\tau).
$$

主损失为：

$$
\boxed{
\mathcal L_{\mathrm{reg}}=
\sum_{(j,a)\in\mathcal E_f}\omega_{j,a}\widetilde q_{S,j,a}}
$$

它只惩罚“仍处于目标源状态”，不指定进入 donor 的目标状态 $b$。联合非 donor 中性约束与反虚假抑制约束：

$$
\mathcal L_{\mathrm{neutral}}=[r_f-c]_++\sum_j[s_{d_j}-d_j]_++[\|\Delta r_E\|-e]_+,
$$

$$
\boxed{\mathcal L=\lambda_{reg}\mathcal L_{\mathrm{reg}}+
\lambda_n\mathcal L_{\mathrm{neutral}}+\beta\mathcal L_R},
\qquad \lambda_{KL}^{(\mathrm{ablation})}=0.
$$

另以 Erase-or-Hide 式归因审计限制新增负向抑制；该审计是必要条件，不被当作本方案的算法创新。

## 4. idea-evaluator 兼容评估

**定位**：Novel Method。F1 为 MAJOR：需与 GLUScope、Sign Code、CodeUnlearn 和旧 CF2-09 划界；F6 为 MAJOR：离散状态是否承载事实而非词形必须由因果干预验证。零 CRITICAL。

| 维度 | 分数 | 机制依据，均尚未由本轮数据确认 |
|---|---:|---|
| Higher | 8 | 直接针对连续差值无法区分的计算状态变化，可能发现低幅值事实通路 |
| Faster | 7 | 状态统计和软符号损失均为逐神经元运算，可复用现有前向缓存 |
| Stronger | 8 | 跨 paraphrase 转移稳定性、null 转移图和反抑制审计提供多重防混淆 |
| Cheaper | 8 | 不需训练 SAE/VQ、不增加部署模块，也不需 donor 答案标注 |
| Broader | 7 | 适用于 SwiGLU/GEGLU 及一般 sign-preserving 激活，但架构覆盖不是完全任意的 |

**平均分 $\bar S=38/5=7.6$；裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 生命周期 3–9 个月；每周工时未知，资源匹配 Yellow。范式探测：First Principles=Yes，Elephant=Partial，Technology Cycle=Yes，Hamming=Partial。

前三项行动：验证状态转移跨 split 稳定性；执行 sign-only、magnitude-only、随机等能三臂；确认源状态迁出不提高 donor 状态占据和负向抑制。

## 5. 决定性实验与失败判据

**最小实验**：在 2 个中层 MLP 上估计转移图，比较真实、wrong、shuffled、matched-retain 配对。对所选状态做 sign-preserving magnitude 替换、保幅值 sign 翻转和连续幅值抑制，测 $r_f$ 与 retain。通过零训练门后再做短步数 Student 训练。

**失败即撤回**：转移矩阵跨 seed 不稳定；真实与 null 状态重合；soft 状态损失等价于全面削弱激活；donor 目标状态上升；或出现新的抑制壳和快速 relearning 恢复。

## 6. 来源与诚信状态

- Gerstner & Schütze（2026），*GLUScope: A Tool for Analyzing GLU Neurons in Transformer Language Models*，arXiv:2602.23826。
- *The Sign Code: The Hidden Binary Nature of Deep Networks*（2026），arXiv:2610.23616。
- *CodeUnlearn*（2024），arXiv:2410.10866，作为必须避开的离散码本遗忘近邻。
- Yang et al.（ICLR 2026），*Erase or Hide?*，arXiv:2509.22263，用于反抑制验收。

plan-only；所有分数均为机制推断，文献未完成 D1 批量核验，不构成独立评审接受。

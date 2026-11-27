---
id: CF3-09
rank: 9
title: 双端可控—可读 Hankel 激活模态（DHAM）
avg_score: 7.2
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无独立跨模型回执
novelty_status: PROCEED WITH CAUTION
baseline_relation: 用主体输入到中间状态、中间状态到下游读出的双端 Jacobian 联合筛选反事实差异
---

# CF3-09 · 双端可控—可读 Hankel 激活模态

## 0. 一句话故事

> 一个差异即使能被主体替换诱发，也未必会被答案端读取；反之，一个影响答案的方向也未必与主体干预有关。我们同时估计“主体到状态”的可控性和“状态到答案”的可读性，用平衡 Hankel 模态找出双端连通的反事实差异，Student 只删除这些通路，而非删除所有相关激活。

## 1. 问题与近邻空白

*From Black-Box to White-Box: Control-Theoretic Neural Network Interpretability*（2025）已在小型前馈网络上用局部 Jacobian、可控/可观测 Gramian 和 Hankel 奇异值分析内部模态。因此，本方案是明确的方法迁移，不声称发明控制论工具。

新的研究问题是：**目标事实是否集中在“主体可控且答案可读”的双端模态交集中？** 与 CF-07 仅有 $\partial r_f/\partial h$ 的单端读出不同，本方案还包含 $\partial h/\partial e_s$ 并通过平衡乘积联合筛选。与 Fast-NTK 不同，不构造参数空间 NTK、不做闭式样本影响 downdate，也不以 NTK 为贡献。

## 2. 可证伪假设

**H**：高 Hankel 奇异值模态比单侧可控、单侧可读或等数量随机模态更能预测反事实差异对目标事实的因果贡献。若双端筛选与 CF-07 单端读出主方向高度重合，或小扰动线性化在大差异下失效，则核心机制被证伪。

## 3. 方法

### 3.1 双端局部算子

在主体中点 $\bar e_s=(e^++e^-)/2$ 附近线性化，层 $\ell$、答案位置 $p$ 的状态为 $A_{\ell,p}$。定义主体到状态算子：

$$
B_{\ell,p}=\left.\frac{\partial A_{\ell,p}}{\partial e_s}\right|_{\bar e_s}.
$$

令 $\mathcal R(A)$ 为经完整冻结下游计算的读出向量，包含目标答案跨度 logit/概率及固定随机投影的通用输出坐标；定义状态到读出算子：

$$
C_{\ell,p}=\left.\frac{\partial\mathcal R(A)}{\partial A}\right|_{A=A_{\ell,p}(\bar e_s)}.
$$

所有读出经完整下游，不将中间向量直接过 LM head。

### 3.2 平衡 Hankel 模态

构造局部可控与可读 Gramian：

$$
W_c=B_{\ell,p}B_{\ell,p}^\top+\lambda I,
\qquad
W_o=C_{\ell,p}^\top C_{\ell,p}.
$$

定义平衡矩阵并分解：

$$
M=W_c^{1/2}W_oW_c^{1/2}=V\Sigma V^\top,
\qquad \Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_r).
$$

将 $u_k=W_c^{1/2}v_k$ 归一化并作数值正交化。大 $\sigma_k$ 表示该模态同时具有较强的主体输入连通和输出可读性。只有真实配对相对 null 配对在 $\sigma_k$ 或模态载荷上稳定超出时，才进入：

$$
\boxed{
\Delta A^{H}_{\ell,p}=
\sum_{k\in\mathcal E_{\mathrm{dual}}}
u_ku_k^\top\Delta A_{\ell,p}}.
$$

该谱是局部近似，不能自动解释为非线性全网的全局因果通路。

### 3.3 Student 内化

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left[\Pi_{\mathcal K}
\left(A^+_{\ell,p}-\alpha\Delta A^{H}_{\ell,p}\right)\right],
$$

$$
\boxed{\mathcal L=\lambda_F\sum_{\ell,p\in S_f}
\|h_{S,\ell,p}-T_{\ell,p}\|_2^2+\beta\mathcal L_R}.
$$

$\Pi_{\mathcal K}$ 维持原事实读出下降、donor 不升及 EOS/长度稳定。主结果无 donor KL，部署不需要 Jacobian 或 Hankel 矩阵。

首版仅以 randomized JVP/VJP 估计矩阵—向量积，不显式构造完整 $d\times d$ 矩阵；先扫描 4 个锚定层和短答案。

## 4. idea-evaluator 兼容评估

**定位**：Novel Method / 方法迁移。F1 为 MAJOR：需相对控制论解释、CF-07 和 Fast-NTK 证明双端对象的独立价值；F6 为 MAJOR：局部线性化可能无法覆盖真实反事实的大幅变化。零 CRITICAL。

| 维度 | 分数 | 机制依据，均尚未由本轮数据确认 |
|---|---:|---|
| Higher | 8 | 双端交集直接过滤“能诱发但不可读”与“可读但与主体无关”的混淆 |
| Faster | 6 | randomized JVP/VJP 可降本，但仍显著贵于闭式均值差和 Grassmann 分解 |
| Stronger | 8 | 中点、多端点和 null 模态压力测试可暴露局部近似失效 |
| Cheaper | 6 | 无额外标注，但 Jacobian 估计和跨样本稳定化成本较高 |
| Broader | 8 | 同一双端框架可用于证据、视觉 token、工具调用和 CoT 状态 |

**平均分 $\bar S=36/5=7.2$；裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 生命周期 3–9 个月；每周工时未知，资源匹配 Yellow。范式探测：First Principles=Yes，Elephant=Yes，Technology Cycle=Partial，Hamming=Partial。

前三项行动：先比较双端谱与单端 Jacobian 的主角度；检验中点线性化的局部误差；仅在双端模态跨 split 稳定且具有独立 patch 效应时训练。

## 5. 决定性实验与失败判据

**最小实验**：比较双端 Hankel、仅 $W_c$、仅 $W_o$、CF-07 读出方向与随机等秩模态。报告谱的跨 seed 稳定、patch 后 $r_f$、retain NLL、donor 率、EOS、probe 与 relearning。用多个插值点检查局部谱是否随操作点翻转。

**失败即撤回**：双端与单端模态高度重合；谱随操作点或 seed 不稳定；高 $\sigma$ 模态不具目标特异性；随机近似误差掩盖排序；或 Fast-NTK/既有投影基线在等预算下已覆盖全部收益。

## 6. 来源与诚信状态

- *From Black-Box to White-Box: Control-Theoretic Neural Network Interpretability*（2025），arXiv:2511.12852。
- Li et al.（CVPRW 2024），*Fast-NTK: Parameter-Efficient Unlearning for Large-Scale Models*，arXiv:2312.14923，作为排除 NTK 重复的依据。
- 基线 §2、§4、§5、§6；明确区别于 CF-07 的单端读出 Jacobian。

plan-only；分数为机制推断，非实测或独立评审；文献未完成 D1 批量核验。

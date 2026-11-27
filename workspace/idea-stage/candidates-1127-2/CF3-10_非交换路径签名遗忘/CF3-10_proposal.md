---
id: CF3-10
rank: 10
title: 非交换路径签名遗忘（NPSU）
avg_score: 7.2
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无独立跨模型回执
novelty_status: PROCEED WITH CAUTION
baseline_relation: 将激活差的逐层序列映射为有序迭代积分，以 log-signature 的非交换面积作为遗忘对象
---

# CF3-10 · 非交换路径签名遗忘

## 0. 一句话故事

> 两条反事实轨迹可以有相同端点、相同逐层增量集合，却因增量发生顺序不同而代表不同计算过程。我们把 $\Delta A$ 的跨层路径映射为截断 log-signature，只处理真实配对特异的有序面积项；Student 学习消除这些非交换结构，同时用原事实读出约束避免“冻结两边”的假遗忘。

## 1. 问题与近邻空白

路径签名以迭代积分表示有序路径；*Transformers with Signatures* 和 *From Path Signatures to Sequential Modeling* 说明该工具在序列建模中的价值。*Transformer Dynamics*、*High-dimensional layer-state trajectories* 与 *Trajectory Signatures of Deception* 则表明层序与轨迹几何具有行为信息，但它们并未把反事实 log-signature 用作遗忘目标。

与 CF2-01 的一阶层间通量不同，二阶签名包含有序面积；与 CF2-05 的小波不同，签名不是线性时频坐标；与 CF2-07 的三阶共同累积量不同，签名强调**路径顺序和非交换性**，而非样本分布的矩。该方案高风险，只有证明这些差异带来独立信息才值得保留。

## 2. 可证伪假设

**H**：真实反事实路径的二阶非交换项包含端点、通量、路径长度和小波特征之外的目标信息；它在层重参数化下稳定、在路径反转时按理论变号，并能预测目标读出。若二阶项与简单通量高度共线，或跨 split 不稳定，则核心机制被证伪。

## 3. 方法

### 3.1 反事实差路径

对每个答案位置 $p$，将各层差值用固定随机正交投影 $P$ 降至 $d_0$ 维，并加入归一化层坐标：

$$
Z_{\ell,p}=\left(\frac{\ell}{L},\ P\Delta A_{\ell,p}\right),
\qquad \ell=0,\ldots,L.
$$

将离散路径线性插值为连续分段路径 $Z_t$。二阶截断签名为：

$$
S^{(1)}_i=\int_0^1 dZ_t^i,
$$

$$
S^{(2)}_{ij}=\int_{0<s<t<1}dZ_s^i\,dZ_t^j.
$$

二阶 log-signature 的非交换部分是 Lévy 面积：

$$
\boxed{
\mathcal L^{(2)}_{ij}
=S^{(2)}_{ij}-\frac12S^{(1)}_iS^{(1)}_j
=\frac12(S^{(2)}_{ij}-S^{(2)}_{ji})}.
$$

它对时间重参数化不变，路径反转时面积变号；直线路径的面积项为零。以上是代数诊断，不是事实语义证明。

### 3.2 目标特异性面积与 Student 目标

以真实、wrong、shuffled 和 matched-retain 配对分别计算 $\mathcal L^{(2)}$，构造固定掩码：

$$
\Omega_{ij}=\mathbf 1\left[
|\mathcal L^{true}_{ij}|
>\operatorname{median}|\mathcal L^{null}_{ij}|+\delta
\right]\times\mathbf 1[\text{跨 split 符号稳定}].
$$

Student 分别运行 $X^\pm$ 并形成 $Z^S$，但不以 donor 输出或 $y^-$ 为目标。遗忘项为：

$$
\mathcal L_{\mathrm{sig}}=
\|\Omega\odot\mathcal L^{(2)}(Z^S)\|_F^2.
$$

为防止“正负两边都保留原事实”或“都变成 donor”的零面积捷径，联合：

$$
\mathcal L_{\mathrm{neutral}}=[r_f-c]_++\sum_j[s_{d_j}-d_j]_++[\|\Delta r_E\|-e]_+,
$$

并对非目标关系的签名设置 Teacher 锚：

$$
\mathcal L_{\mathrm{sig-retain}}=
\|\mathcal L^{(2)}(Z^S_R)-\operatorname{stopgrad}\mathcal L^{(2)}(Z^T_R)\|_F^2.
$$

$$
\boxed{
\mathcal L=\lambda_s\mathcal L_{\mathrm{sig}}+
\lambda_n\mathcal L_{\mathrm{neutral}}+
\lambda_a\mathcal L_{\mathrm{sig-retain}}+\beta\mathcal L_R}.
$$

主结果固定 $\lambda_{KL}^{(\mathrm{ablation})}=0$；部署只保留 Student，不计算签名。

## 4. idea-evaluator 兼容评估

**定位**：Novel Method，高风险。F1 为 MAJOR：需与轨迹几何、签名 Transformer 及 CF2-01/05/07 证明独立增量；F6 为 MAJOR：签名可能只编码路径形状而非事实，并存在高维估计成本。零 CRITICAL，但仅列末位候选。

| 维度 | 分数 | 机制依据，均尚未由本轮数据确认 |
|---|---:|---|
| Higher | 8 | 可捕获端点和一阶差分不可见的有序计算结构 |
| Faster | 5 | 截断签名增加路径预处理和训练计算，当前没有可信的净提速依据 |
| Stronger | 8 | 重参数化不变、路径反转和直线零面积提供可执行的代数 sanity checks |
| Cheaper | 6 | 不需新标注或解释器，但需要全层轨迹缓存与多路径统计 |
| Broader | 9 | 可统一文本层深、CoT 时间、视觉 patch 和一般有序激活过程 |

**平均分 $\bar S=36/5=7.2$；裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 生命周期 3–9 个月；每周工时未知，资源匹配 Yellow。范式探测：First Principles=Yes，Elephant=Partial，Technology Cycle=Yes，Hamming=Partial。

前三项行动：先验证三条代数性质；比较二阶面积与端点/通量/路径长度的增量信息；若不能稳定预测 $r_f$，不进入 Student 训练。

## 5. 决定性实验与失败判据

**最小实验**：在低秩 $d_0\le8$、二阶截断下比较真实与 null；执行路径反转、时间重参数化、直线化、层顺序特征 shuffle 对照。比较 signature、CF2-01 通量、路径长度、CF2-05 小波与随机面积掩码。通过零训练门后才做短步数 Student 训练。

**失败即撤回**：二阶面积可由一阶通量重构；跨 split 符号翻转；真实与 null 不可分；损失只造成双边冻结、donor 替换或输出坍缩；或收益不抵等预算简单轨迹特征。

## 6. 来源与诚信状态

- *Transformers with Signatures: Towards Universal In-Context Learning in Path Space*（2026），用于签名序列建模近邻。
- *From Path Signatures to Sequential Modeling: Incremental Signature Contributions for Offline RL*（2026），arXiv:2602.11805。
- Zhao et al.（2026），*High-dimensional layer-state trajectories reveal the internal dynamics of transformer inference*，arXiv:2610.26866。
- Fernando & Guitchounts（2025），arXiv:2502.12131；基线 §2、§4、§6。

plan-only；分数为机制推断，非实测或独立评审；文献未完成 D1 批量核验。

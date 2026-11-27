---
id: CF3-05
rank: 5
title: 答案跨度 Grassmann 子空间擦除（AGSE）
avg_score: 7.6
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无独立跨模型回执
novelty_status: PROCEED WITH CAUTION
baseline_relation: 将逐位置向量差替换为整条答案跨度所张成子空间的投影差与主角度
---

# CF3-05 · 答案跨度 Grassmann 子空间擦除

## 0. 一句话故事

> 一个事实不一定藏在某个答案 token 的向量里，也可能体现在整段答案激活所占据的子空间。我们把答案跨度矩阵看成 Grassmann 流形上的点，用主角度和投影算子差定位正事实特有的子空间；Student 只移除其中通过伪配对门的目标方向，而不逐点对齐 donor 轨迹。

## 1. 问题与近邻空白

逐位置 $\Delta A_{\ell,p}$ 依赖所选 token 和坐标尺度，无法表示“多个答案位置共同张成的方向”。*A Geometric Perspective on Next-Token Prediction*（2026）已研究读出子空间的 Grassmann 轨迹；CIR、LEACE 与 ReGLU 已覆盖线性子空间擦除或 retain 正交保护。因此，本方案不能把“使用子空间”写成新颖性。

可辩护增量是：对**同一 teacher-forced 答案跨度**的正反事实矩阵构造投影差，以 null 配对归一化主子空间，并将其转为非 donor 的 Student 删除目标。与 CF-05 的 donor 维 SVD 不同，这里的矩阵维数是答案位置；与 CF2-08 的配对散射/retain 白化不同，核心量是子空间投影差而非协方差广义谱。

## 2. 可证伪假设

**H**：真实配对的答案跨度子空间差包含逐位置向量差无法捕获的目标信息；该结构在不同答案长度和 paraphrase 下稳定，并且删除主子空间比删除等秩逐点主成分更少损害 retain。若投影差可由均值/范数变化完全解释，或主子空间主要由 token 位置决定，则假设被证伪。

## 3. 方法

### 3.1 答案跨度矩阵与子空间

固定层 $\ell$，将 $p\in S_f$ 的激活按相同答案 $y^+$ 排列：

$$
H^\pm_\ell=[A^\pm_{\ell,p_1};\ldots;A^\pm_{\ell,p_T}]\in\mathbb R^{T\times d}.
$$

使用 retain 激活均值 $\mu_{R,\ell,p}$ 做位置保持的中心化，而非对每个矩阵独立减去自身均值：

$$
\widetilde H^\pm_{\ell,p}=A^\pm_{\ell,p}-\mu_{R,\ell,p}.
$$

对 $\widetilde H^\pm$ 做秩 $k$ SVD，得到右奇异正交基 $U^\pm_\ell\in\mathbb R^{d\times k}$，并定义：

$$
P^\pm_\ell=U^\pm_\ell(U^\pm_\ell)^\top,
\qquad
\boxed{\Delta P_\ell=P^+_\ell-P^-_\ell}.
$$

$U^{+\top}U^-$ 的奇异值为 $\cos\theta_i$，其中 $\theta_i$ 为主角度。$\Delta P$ 对同一子空间内的基底旋转不变，但对完整隐空间任意旋转并不自动不变。

### 3.2 null 归一化的目标子空间

分别对真实与 null 配对估计 $\Delta P$。取其特征向量 $v_j$，以特征值相对 null 的超出量和跨 split 稳定性选择目标方向：

$$
\omega_j=\frac{|\lambda_j^{true}|-\operatorname{median}|\lambda_j^{null}|}
{\operatorname{MAD}|\lambda_j^{null}|+\epsilon}.
$$

形成 $V_{\mathrm{spec}}=[v_j:\omega_j>\delta]$，并定义：

$$
\boxed{\Delta A^{\mathrm{Gr}}_{\ell,p}
=V_{\mathrm{spec}}V_{\mathrm{spec}}^\top\Delta A_{\ell,p}}.
$$

若 $V_{\mathrm{spec}}$ 为空或跨 split 主角度过大，则不进入训练。

### 3.3 Student 内化

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left[\Pi_{\mathcal K}
\left(A^+_{\ell,p}-\alpha\Delta A^{\mathrm{Gr}}_{\ell,p}\right)\right],
$$

$$
\mathcal L_F=\sum_{\ell,p\in S_f}\|h_{S,\ell,p}-T_{\ell,p}\|_2^2,
\qquad
\boxed{\mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R}.
$$

$\Pi_{\mathcal K}$ 继续约束原事实读出、donor 支持度与 EOS/长度；Teacher 与 $V_{\mathrm{spec}}$ stop-gradient，主结果不使用 donor KL，部署无子空间 hook。

## 4. idea-evaluator 兼容评估

**定位**：Novel Method。F1 为 MAJOR：CIR/LEACE/ReGLU 和既有白化/多 donor 候选形成强近邻；F6 为 MAJOR：子空间差是否超出 token 数、均值和范数效应需独立判别。零 CRITICAL。

| 维度 | 分数 | 机制依据，均尚未由本轮数据确认 |
|---|---:|---|
| Higher | 7 | 能表示跨答案位置共同结构，但实际增益取决于该结构是否承载目标事实 |
| Faster | 8 | 低秩 SVD 与投影在缓存上闭式计算，不需逐点 patch 扫描 |
| Stronger | 8 | 基底旋转不敏感，null 谱与跨 split 主角度提供稳定性检查 |
| Cheaper | 8 | 不需新标注、SAE 或额外模型；$k$ 小则存储和计算有限 |
| Broader | 7 | 可迁移到 CoT 片段与视觉 token 集合，但需处理不同位置对齐 |

**平均分 $\bar S=38/5=7.6$；裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 生命周期 3–9 个月；每周工时未知，资源匹配 Yellow。范式探测：First Principles=Yes，Elephant=Partial，Technology Cycle=Partial，Hamming=Partial。

前三项行动：冻结 $k$、中心化和 null 规则；比较 $\Delta P$ 与均值/范数/逐点 PCA 的独立信息；通过跨 split 主角度门后才训练。

## 5. 决定性实验与失败判据

**最小实验**：比较 Grassmann 子空间、逐点 pooled PCA、均值差、范数归一化差与随机等秩子空间。报告目标 probe、$r_f$、retain NLL、同实体非目标关系、donor 率、EOS、relearning，以及不同 $T$ 和 $k$ 下的稳定性。

**失败即撤回**：$\Delta P$ 与均值/范数特征高度共线；真实与 null 主角度不可分；目标方向跨 seed 旋转；或与 LEACE/CIR 的主方向几乎相同而无独立收益。

## 6. 来源与诚信状态

- *A Geometric Perspective on Next-Token Prediction in Large Language Models: Three Emerging Phases*（2026），arXiv:2605.09011。
- Belrose et al.（NeurIPS 2023），*LEACE*；Sondej & Yang（2025），*CIR*，arXiv:2509.11816。
- ReGLU（2026），arXiv:2604.17396，用于 retain 子空间近邻划界。
- 基线 §2、§4、§5、§6；明确区别于 CF-05 与 CF2-08。

plan-only；分数为机制推断，文献未完成 D1 批量核验，不是独立评审或实验结论。

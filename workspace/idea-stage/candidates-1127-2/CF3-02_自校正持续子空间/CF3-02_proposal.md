---
id: CF3-02
rank: 2
title: 自校正持续子空间（SCPS）
avg_score: 7.8
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无独立跨模型回执
novelty_status: PROCEED WITH CAUTION
baseline_relation: 不直接删除端点差或层间通量，而提取经冻结下游自主 replay 后仍持续的反事实差异子空间
---

# CF3-02 · 自校正持续子空间

## 0. 一句话故事

> 反事实差异并不都会被后续计算使用：有些会被下一层主动抵消，有些会沿残差流持续并影响答案。我们把正反事实状态分别注入同一冻结下游，观察差异随深度 replay 的衰减、反转与重构，只删除跨下游深度仍持续的子空间，避免把瞬态身份差异误当事实。

## 1. 问题与近邻空白

*Transformer Dynamics*（Fernando & Guitchounts，2025）报告残差轨迹的伪吸引子和自校正；*LLM Layers Immediately Correct Each Other*（Patrawala et al.，2026）进一步提出相邻层的 propose-and-reject / TLCM 机制。这些结果意味着“某层差值大”不等于“该差异会被保留”。

本方案与 CF2-01 的 $\Delta A_{\ell+1}-\Delta A_\ell$ 不同：不做相邻层代数差分，而是通过**受控 patch + 下游 replay**估计差异的生存谱。它也不同于 CF-07 的单一目标答案 Jacobian：这里处理完整残差向量的持续子空间，目标答案读出只用于最后的中性约束。

## 2. 可证伪假设

**H**：真实配对中持续经过下游 replay 的差异子空间，比瞬态或被校正的子空间更能预测目标事实读出和 relearning 恢复；wrong/shuffled 配对不会产生相同持续谱。若持续子空间只反映 attention sink、位置或范数异常，或其目标读出不超过原始差值，则假设被证伪。

## 3. 方法

### 3.1 反事实下游 replay

在层 $\ell$、答案位置 $p$ 获取 Teacher 激活 $A^\pm_{\ell,p}$。定义冻结下游算子 $\mathcal F_{\ell\to m}$：在层 $\ell$ 将对应位置状态替换为输入状态，其余上下文保持一致，再运行到层 $m$。得到：

$$
Z^\pm_{\ell,p}(m)=\mathcal F_{\ell\to m}\!\left(A^\pm_{\ell,p}\right),\qquad
D_{\ell,p}(m)=Z^+_{\ell,p}(m)-Z^-_{\ell,p}(m).
$$

$m=\ell$ 时 $D_{\ell,p}(\ell)=\Delta A_{\ell,p}$。定义生存曲线：

$$
\rho_{\ell,p}(m)=\frac{\|D_{\ell,p}(m)\|_2}{\|\Delta A_{\ell,p}\|_2+\epsilon}.
$$

该曲线可区分持续、衰减和放大，但不能仅凭范数声称语义因果。

### 3.2 持续子空间而非逐层权重

在预注册深度集合 $\mathcal M_\ell$ 上构造持续散射矩阵：

$$
C^{\mathrm{surv}}_{\ell,p}
=\sum_{m\in\mathcal M_\ell} w(m-\ell)D_{\ell,p}(m)D_{\ell,p}(m)^\top,
$$

其中 $w$ 为固定非增权重。用 matched-retain 与 shuffled 配对得到 $C^{\mathrm{null}}$，计算广义谱：

$$
C^{\mathrm{surv}}_{\ell,p}u_k
=\lambda_k(C^{\mathrm{null}}_{\ell,p}+\delta I)u_k.
$$

只保留跨 split 稳定且超过 null 的前 $k$ 个方向，形成 $U^{\mathrm{surv}}_{\ell,p}$：

$$
\boxed{\Delta A^{\mathrm{surv}}_{\ell,p}
=U^{\mathrm{surv}}_{\ell,p}(U^{\mathrm{surv}}_{\ell,p})^\top\Delta A_{\ell,p}}.
$$

与 CF2-08 的 retain 协方差白化不同，这里的分母来自**同算子的伪配对生存谱**，且矩阵由多个下游 replay 深度构成。

### 3.3 Student 内化与中性约束

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left[\Pi_{\mathcal K}
\left(A^+_{\ell,p}-\alpha\Delta A^{\mathrm{surv}}_{\ell,p}\right)\right],
$$

$$
\mathcal K=\{h:r_{f,p}(h)\le c,\ s_{d_j,p}(h)\le d_j,\ \|\Delta r_{E,p}(h)\|\le e\},
$$

$$
\boxed{\mathcal L=\lambda_F\sum_{\ell,p}\|h_{S,\ell,p}-T_{\ell,p}\|_2^2+\beta\mathcal L_R}.
$$

主结果不使用 donor KL，部署无 replay、patch 或 Teacher。首版仅扫描少量锚定层和 3 个下游深度，避免全层全深度二次成本。

## 4. idea-evaluator 兼容评估

**定位**：Novel Method。F6 为 MAJOR：patch 可能制造 OOD 状态，必须与自然发生的同分布状态比较；F1 为 MAJOR：需与 TLCM、Transformer Dynamics、CF2-01 和 CF-07 做实证划界。零 CRITICAL。

| 维度 | 分数 | 机制依据，均尚未由本轮数据确认 |
|---|---:|---|
| Higher | 8 | 针对“差异是否存活到下游”而非端点大小，有望减少无效删除 |
| Faster | 7 | 首验只需锚定层和少量 replay 深度，且全部缓存可复用 |
| Stronger | 9 | 显式建模自校正、反转与伪配对生存谱，直接针对恢复和误判风险 |
| Cheaper | 7 | 无需新标注或训练解释器，但 patch replay 增加前向成本 |
| Broader | 8 | 生存曲线可应用于文本、CoT 和多模态残差轨迹 |

**平均分 $\bar S=39/5=7.8$；裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 生命周期 3–9 个月；每周工时未知，资源匹配 Yellow。范式探测：First Principles=Yes，Elephant=Yes，Technology Cycle=Yes，Hamming=Partial。

前三项行动：验证 replay 状态的分布内程度；比较持续子空间与原始差/通量的重合角；通过零训练门后才做 Student 训练。

## 5. 决定性实验与失败判据

**最小实验**：选择浅、中、深 4 个锚定层，每层取 $m-\ell\in\{1,2,4\}$；比较生存谱、原始 $\Delta A$、CF2-01 通量和随机等能方向。检验跨 seed 主角度、wrong/shuffle 可分性、$r_f$、独立 probe、relearning、retain NLL、EOS 与 donor 率。

**失败即撤回**：自然状态与 patched 状态分布差异足以解释结果；持续方向与原始差/通量高度重合；生存谱跨 seed 不稳定；删除只产生输出抑制；或全扫描收益不抵等预算随机 replay 对照。

## 6. 来源与诚信状态

- Fernando & Guitchounts（2025），*Transformer Dynamics: A Neuroscientific Approach to Interpretability of Large Language Models*，arXiv:2502.12131。
- Patrawala, Feng, Jones & Steinhardt（2026），*LLM Layers Immediately Correct Each Other*，OpenReview 原文检索。
- Zhang & Nanda（ICLR 2024）关于 patching 伪峰的警示作为 OOD 对照依据。
- 基线 §2.2、§4.1、§5、§6；明确区别于 CF2-01、CF2-08、CF-07。

plan-only；分数为执行者按本机 rubric 给出的机制推断，不是实测或独立接受回执；文献未完成 D1 批量核验。

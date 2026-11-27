---
id: CF2-09
rank: 9
title: 稳健低秩—稀疏事件 L+S —— 把全局主体漂移与局部事实事件分开
avg_score: 7.2
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无跨模型独立回执
novelty_status: PROCEED WITH CAUTION（CIR/激活监控近邻较多）
baseline_relation: 从原始 ΔA 矩阵改为低秩背景 L、稀疏事件 S 与稠密噪声 E 的稳健分解
---

# CF2-09 · 稳健低秩—稀疏事件 $L+S$

## 0. 一句话故事

> 一组反事实 $\Delta A$ 可能同时包含跨样本稳定的全局主体/模板漂移，以及只在少数“样本 × 层 × 答案位置”出现的事实绑定事件。普通 SVD 会把异常事件揉回低秩背景。我们用稳健分解 $D=L+S+E$ 把它们分开，并且只在独立验证证明 $S$ 具有目标特异性后，才让 Student 删除稀疏事件。

## 1. 问题与近邻空白

CIR（arXiv:2509.11816）用激活和梯度 PCA 去掉通用表示；已有开源 *activation-difference-monitoring* 原型把模型间激活差分成低秩 + 稀疏结构用于异常检测。它们意味着“低秩 + 稀疏”不是空白。

本方案的增量只能是：在**同一冻结 Teacher 的反事实主体配对差**上做稳健分解，以目标特异性验证决定删除分量，并进行 Student 参数内化。与 CF-05 求多 donor 共有低秩方向不同，本方案同时显式建模稀疏离群事件和稠密噪声，且不预设低秩项就是要删的事实。

## 2. 可证伪假设

**H**：目标绑定表现为相对低秩背景的稀疏异常事件；这些事件跨 split 可复现，并且其目标读出显著高于 $L$ 和 $E$。若 $S$ 主要捕获 subword 噪声、语义归属在 $L/S$ 间随 seed 交换，或错误配对也产生同样稀疏结构，则否决。

## 3. 方法

### 3.1 反事实差矩阵

对 $N$ 个匹配样本，在固定层 $\ell$ 将答案跨度和隐藏通道展开，构造：

$$
D_\ell=
\begin{bmatrix}
\operatorname{vec}_{p,d}(\Delta A_{\ell,p,d}^{(1)})\\
\vdots\\
\operatorname{vec}_{p,d}(\Delta A_{\ell,p,d}^{(N)})
\end{bmatrix}
\in\mathbb R^{N\times (|S_f|d)}.
$$

### 3.2 稳定主成分分解

求解带稠密噪声项的稳健分解：

$$
\min_{L,S,E}\quad
\|L\|_*+\lambda_1\|S\|_1+\lambda_2\|E\|_F^2
\qquad \text{s.t.}\quad D=L+S+E.
$$

$L$ 表示低秩共享结构，$S$ 表示稀疏事件，$E$ 表示小幅度稠密残差。使用固定迭代次数和预注册 $\lambda_1,\lambda_2$，避免按验证集结果反复调参。

### 3.3 语义归属门

不预设“$S$ = 事实”。分别在 $L,S,E$ 上做冻结下游 patch-forward，并要求待删除分量 $C^*$ 同时满足：

$$
\Delta r_f(C^*)<-\eta_f,\qquad
|\Delta U(C^*)|<\eta_U,\qquad
\mathrm{score}(C^*_{\mathrm{true}})>
\mathrm{score}(C^*_{\mathrm{wrong}})+\eta_{\mathrm{spec}}.
$$

若 $L$ 而非 $S$ 通过，则把本实验登记为“低秩绑定”，不得继续沿用稀疏事件故事；若两者都通过，当前分解不可识别，停止主线。

### 3.4 Student 内化

以通过门的分量 $C^*$ 为例，对第 $i$ 个样本构造：

$$
T_i=\operatorname{stopgrad}\!\left[\Pi_{\mathcal K}
\left(A_i^+-\alpha\operatorname{unvec}(C_i^*)\right)\right],
$$

$$
\mathcal L_F=\sum_i\|h_{S,\ell}(X_i^+\oplus y_i^+)-T_i\|_2^2,
\qquad
\boxed{\mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R}.
$$

$\Pi_{\mathcal K}$ 保持非 donor、反 EOS 和原事实读出约束；$\lambda_{KL}^{(\mathrm{ablation})}=0$。

## 4. 差异化边界

| 近邻 | 已占机制 | 本方案差异 |
|---|---|---|
| CIR（2025） | PCA 去掉通用表示子空间 | 稳健 $L+S+E$ 分离稀疏异常，且语义归属由独立门决定 |
| activation-difference-monitoring | 模型间特征差低秩 + 稀疏异常检测 | 同一 Teacher 的反事实配对、遗忘目标与 Student 内化 |
| CF-05 多 donor 边际化 | 共有低秩成分作为目标 | 本方案联合建模 $L,S,E$，不预设低秩成分的语义 |
| V1 pooled PCA | 主成分直接抑制 | 不直接删 PCA 主成分，先分解再做独立目标/效用/伪配对门 |

## 5. /idea-evaluator 兼容评估（PROVISIONAL）

**定位**：Novel Method（高风险增量）。致命缺陷：2 MAJOR——F1 与 CIR/异常监控近邻；F6 $L/S$ 的语义归属可能不可识别。零 CRITICAL，故 Accept with Revisions。分解可在 CPU/缓存上运行，但多次 patch 门增加成本，资源匹配黄。

| 维度 | 分数 | 机制依据 |
|---|---:|---|
| Higher | 7 | 有望保留被普通 SVD 吞掉的局部事实事件，但依赖可识别性 |
| Faster | 7 | 复用激活缓存；分解与门控成本可控但非闭式一步 |
| Stronger | 8 | 显式噪声项、语义归属门和 wrong-donor 对照抗异常污染 |
| Cheaper | 7 | 无 SAE/标注，但需多配对和 patch 验证 |
| Broader | 7 | 可迁移到轨迹和模型差分，但已有异常监控近邻 |

**平均分 $36/5=7.2$。裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 范式探测：First Principles=Yes、Elephant=Partial、Technology Cycle=Partial、Hamming=Partial。

修订顺序：先只做 $L/S$ 语义归属；若跨 seed 交换则停止；之后再投入 Student 训练。

## 6. 决定性实验与失败模式

**最小判别实验**：普通 SVD、稳健 $L+S+E$、随机等能稀疏事件三臂；分别对 $L,S,E$ 做目标读出和 wrong-donor 对照，报告跨 seed 一致性、稀疏位置重合率、$r_f$、retain NLL、donor 率和恢复攻击。

**失败即撤回**：$S$ 的稀疏位置跨 seed 不重合；真实与 wrong-donor 稀疏模式一致；目标信息在 $L/S$ 间交换；patch 显示目标读出无变化或效用同步崩坏。

## 7. 来源与诚信状态

- Sondej & Yang（2025），*Collapse of Irrelevant Representations Ensures Robust and Non-Disruptive LLM Unlearning*，arXiv:2509.11816。
- *activation-difference-monitoring* 开源原型，检索页：http://onlybits.org/simonvary/activation-difference-monitoring 。仅作工程近邻，不作为论文证据。
- Candès et al.（2011），*Robust Principal Component Analysis?*。
- 基线 §2.2、§4.1、§6。
- plan-only；分数非实测、非独立评审。文献未做 D1 批量身份核验，不生成 BibTeX。

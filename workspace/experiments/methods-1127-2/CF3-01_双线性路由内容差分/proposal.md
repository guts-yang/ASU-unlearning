---
id: CF3-01
rank: 1
title: 双线性路由—内容差分（BRCD）
avg_score: 8.0
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无独立跨模型回执
novelty_status: PROCEED WITH CAUTION
baseline_relation: 将原始激活差精确分解为路由变化与内容变化，仅把通过特异性门的分量用于非 donor 删除
---

# CF3-01 · 双线性路由—内容差分

## 0. 一句话故事

> 同一个反事实激活差可能来自“模型看向哪里”的路由变化，也可能来自“实际写入了什么”的内容变化。利用注意力和 SwiGLU 都具有双线性乘积这一事实，我们把差值精确拆成两项，不再把全部差异都当成事实；Student 只内化经独立验证承载目标事实的内容或路由分量，其他分量保持不动。

## 1. 问题与近邻空白

原始 $\Delta A$ 将注意力的 QK 路由、Value 内容以及 MLP 门控混在一个向量里。*Contribution Weights*（Cunningham & Muça Cirone，ICML 2026）说明只看注意力权重会忽略 value 几何；*How Few-Shot Examples Add Up*（ICML 2026）已用 QK/V patching 区分路由与内容。因此，本方案不能声称发明 QK/V 分解。

可辩护的新轴是：把双线性差分的**代数恒等式**转化为反事实遗忘目标，并用非 donor 中性约束阻止内容项退化为 donor 答案替换。它不同于 CF-02 的“删/保/禁”三角色约束，也不同于 CF2-10 的 Shapley 组件交互；这里的分解无需探针、Shapley 采样或新网络。

## 2. 可证伪假设

**H**：真实事实配对的路由项与内容项具有可重复的不同因果作用；至少一项相对 wrong/shuffled 配对具有目标特异性，且删除该项能够降低原事实读出而不提高 donor 支持度。若两项都不能独立影响目标读出，或其作用完全由主体词形和注意力 sink 解释，则核心机制被证伪。

## 3. 方法

### 3.1 通用双线性恒等式

对任意逐元素或矩阵双线性运算 $B(a,b)=a\odot b$，令 $a^\pm,b^\pm$ 分别来自正、反事实前向。使用对称分解：

$$
\boxed{
B(a^+,b^+)-B(a^-,b^-)
=\underbrace{\frac12(a^+-a^-)\odot(b^++b^-)}_{D_{\mathrm{route}}}
+\underbrace{\frac12(a^++a^-)\odot(b^+-b^-)}_{D_{\mathrm{content}}}
}
$$

右侧相加精确等于左侧，不是一阶近似。该恒等式只提供计算分解，不自动赋予“事实”语义。

对注意力头 $h$，令 $a$ 为注意力权重、$b$ 为 value 向量，并对源位置 $q$ 求和：

$$
O^\pm_{\ell,h,p}=\sum_q \pi^\pm_{\ell,h,p,q}v^\pm_{\ell,h,q},
$$

$$
D^{\mathrm{attn}}_{\mathrm{route}}
=\frac12\sum_q(\pi^+_q-\pi^-_q)(v^+_q+v^-_q),
$$

$$
D^{\mathrm{attn}}_{\mathrm{content}}
=\frac12\sum_q(\pi^+_q+\pi^-_q)(v^+_q-v^-_q).
$$

对 SwiGLU 神经元，$a=g$ 为门控预激活、$b=u$ 为数值预激活，实际非线性为 $\operatorname{Swish}(g)\odot u$；此时对 $(\operatorname{Swish}(g),u)$ 使用同一恒等式，并经 $W_{\mathrm{out}}$ 映回残差流。

### 3.2 分量选择而非预设语义

分别对 $D_{\mathrm{route}}$ 与 $D_{\mathrm{content}}$ 做冻结下游 patch-forward。分量 $C^*$ 只有在满足以下条件后才进入删除目标：

$$
\Delta r_f(C^*)<-\eta_f,\qquad
\operatorname{score}(C^*_{\mathrm{true}})>
\operatorname{score}(C^*_{\mathrm{null}})+\eta_s,
$$

且对 retain 与同实体非目标关系的影响处于预注册死区。若两项都通过，则声明“不可识别”，不得任意选择更符合故事的一项。

### 3.3 Student 参数内化

令 $A^+_{\ell,p}$ 为 Teacher 在原事实完整序列上的残差激活，定义非 donor 可行集：

$$
\mathcal K=\{h:r_{f,p}(h)\le c,\ s_{d_j,p}(h)\le d_j,\ \|\Delta r_{E,p}(h)\|\le e\}.
$$

对选中的写回分量 $C^*_{\ell,p}$：

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left[\Pi_{\mathcal K}
\left(A^+_{\ell,p}-\alpha C^*_{\ell,p}\right)\right],
$$

$$
\mathcal L_F=\sum_{\ell,p\in S_f}\|h_{S,\ell,p}-T_{\ell,p}\|_2^2,
\qquad
\boxed{\mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R}.
$$

Teacher 与目标全部 stop-gradient；不拼入 $y^-$，主结果固定 $\lambda_{KL}^{(\mathrm{ablation})}=0$，部署只保留 Student。

## 4. idea-evaluator 兼容评估

**定位**：Novel Method。致命缺陷审计：F1 为 MAJOR，需与 QK/V patching、Contribution Weights 和既有 CF2-10 划界；F6 为 MAJOR，需证明分量具有目标因果作用而非仅具代数意义。零 CRITICAL，因此不能给 Strong Accept。

| 维度 | 分数 | 机制依据，均尚未由本轮数据确认 |
|---|---:|---|
| Higher | 8 | 精确分离路由与内容，直接减少原始差值中的计算角色混淆 |
| Faster | 8 | 复用两次 Teacher 前向中的注意力和门控缓存，分解是闭式向量运算 |
| Stronger | 8 | 分量级 null、retain 与 donor 死区可阻止整体激活扰动和答案替换 |
| Cheaper | 8 | 不需 SAE、额外标注、Shapley 采样或训练新的定位器 |
| Broader | 8 | 同一恒等式统一 attention 与 GLU 两类双线性计算，并可迁移到多模态注意力 |

**平均分 $\bar S=40/5=8.0$；裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 生命周期为 Frontier Exploration 的 3–9 个月；首验按项目 TOFU、Llama-3.2-1B 和单卡 H20 设定限制在少量层。用户每周工时未提供，能力匹配记为 Yellow，而非虚构 Green。范式探测：First Principles=Yes，Elephant=Yes，Technology Cycle=Partial，Hamming=Partial。

前三项行动：冻结分量与 null 定义；完成四组 patch-forward 判别；仅在分量特异且中性可行集非空时进入 Student 训练。

## 5. 决定性实验与失败判据

**最小实验**：在 4 个锚定层上比较 route-only、content-only、原始等范数 $\Delta A$、随机等范数分量与 wrong/shuffled 对照。先测 $r_f$ 变化，再做 60-step 量级 Student 内化。报告 retain NLL、同实体非目标关系、donor 答案率、拒答、提前 EOS、完整回答率、独立 probe 与 relearning。

**失败即撤回**：route/content 的 patch 效应不稳定；两项语义跨 seed 交换；真实与 null 配对不可分；删除只增加负向抑制神经元；或收益可由等范数原始差解释。历史 stage0–14 不作为正向证据。

## 6. 来源与诚信状态

- Cunningham & Muça Cirone（ICML 2026），*Contribution Weights: A Geometrical Analysis of Self-Attention Transformers*，arXiv:2606.07604。
- *How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning*（ICML 2026），用于确认 QK/V patching 是强近邻。
- Elhage et al.（2021），*A Mathematical Framework for Transformer Circuits*，用于残差流读写背景。
- 基线：`方向I_基线_v0.6.1_answer_span.md` §2、§4、§5、§6。

本 proposal 为 plan-only；文献身份未完成 D1 批量核验，不生成 BibTeX，分数不是独立 reviewer verdict。

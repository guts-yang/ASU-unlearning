---
id: CF3-03
rank: 3
title: 访问分布加权反事实激活差（VOCE）
avg_score: 7.6
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无独立跨模型回执
novelty_status: PROCEED WITH CAUTION
baseline_relation: 将答案跨度激活差从仅 gold teacher-force 扩展为 Student 实际访问前缀上的访问测度加权
---

# CF3-03 · 访问分布加权反事实激活差

## 0. 一句话故事

> 只在标准答案前缀上抹去激活差，可能无法覆盖 Student 实际生成时走入的状态。我们让 Teacher 在 gold 前缀和 Student 采样前缀的**同一状态**上分别运行正反事实输入，用 Student 的访问频率决定哪些激活差进入训练，从而遗忘实际生成路径上的事实调用，而不是只记住一条标准答案轨迹。

## 1. 问题与近邻空白

基线的 $X^\pm\oplus y^+$ teacher-force 让答案跨度可见，但只覆盖 $y^+$ 前缀。DKT 已区分 Teacher-prefix 与 Student-prefix 的序列 KL；本方案不复用其 donor KL，而只借用“前缀来源决定状态访问分布”这一点，把激活差定义在 on-policy 状态上。

与 CF-06 的 LRM 轨迹作用域不同，本方案的核心不是把 CoT 纳入 $S_f$，而是**用当前 Student 的状态访问测度给每个共享前缀赋权**。与 CF-05 的答案位置权重不同，权重来自 rollout 访问频率，而非同一条答案轨迹内的位置差异。

## 2. 可证伪假设

**H**：目标事实在 Student 自由生成时访问的非 gold 前缀上仍有可测的反事实激活差；对这些状态加权比只训练 gold 前缀更能降低改写、间接提示和 rollout 中的事实恢复。若 on-policy 与 gold-only 在等前后向预算下无差异，或收益仅来自更多样本，则核心机制被证伪。

## 3. 方法

### 3.1 共享前缀上的反事实激活

从当前 Student 采样 $y^S\sim q_\theta(\cdot\mid X^+)$，并与 gold $y^+$ 混合形成前缀集合。对任一共享前缀状态 $s_t=(y_{<t})$，冻结 Teacher 分别运行：

$$
A^\pm_{T,\ell}(s_t)=h_{T,\ell}(X^\pm,s_t),\qquad
\Delta A_{T,\ell}(s_t)=A^+_{T,\ell}(s_t)-A^-_{T,\ell}(s_t).
$$

两个 Teacher 前向必须共享同一 $s_t$，不得比较不同生成前缀。$y^-$ 不进入任何分支。

令 $d_\theta(s)$ 为 Student rollout 的状态访问分布，$d_g(s)$ 为 gold 前缀分布，预注册混合：

$$
\bar d_\eta(s)=(1-\eta)d_g(s)+\eta d_\theta(s).
$$

$\eta$、rollout 数和最大前缀长度在实验前冻结；$d_\theta$ 的采样权重 stop-gradient。

### 3.2 访问加权的激活目标

对状态 $s\sim\bar d_\eta$，构造：

$$
T_\ell(s)=\operatorname{stopgrad}\!\left[\Pi_{\mathcal K(s)}
\left(A^+_{T,\ell}(s)-\alpha\Delta A_{T,\ell}(s)\right)\right],
$$

其中 $\mathcal K(s)$ 要求从该前缀出发的原事实 continuation 读出下降、donor 支持度不升、EOS/长度漂移受限：

$$
\mathcal K(s)=\{h:r_f(h;s)\le c,\ s_{d_j}(h;s)\le d_j,\ \|\Delta r_E(h;s)\|\le e\}.
$$

Student 只在原输入 $X^+$ 的相同前缀上接受损失：

$$
\boxed{
\mathcal L_F=\mathbb E_{s\sim\bar d_\eta}\sum_{\ell\in\Lambda}
\|h_{S,\ell}(X^+,s)-T_\ell(s)\|_2^2}
$$

$$
\mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R,
\qquad \lambda_{KL}^{(\mathrm{ablation})}=0.
$$

先使用静态 $\Lambda$；若引入渐进 schedule，必须另行等预算验证，不能将收益归因于“渐进”。

## 4. idea-evaluator 兼容评估

**定位**：Novel Method。F1 为 MAJOR：需证明相对 DKT on-policy、CF-06 与 CF-05 的增量来自访问测度而非前缀增多；F6 为 MAJOR：off-policy gold 与 on-policy 状态的可比性必须经共享前缀验证。零 CRITICAL。

| 维度 | 分数 | 机制依据，均尚未由本轮数据确认 |
|---|---:|---|
| Higher | 8 | 覆盖自由生成实际访问的状态，直接针对 teacher-forcing 的覆盖盲区 |
| Faster | 7 | 可复用短 rollout 和前缀缓存，gold/on-policy 分支批量共享计算 |
| Stronger | 8 | 对改写、间接提示和非标准前缀提供更直接的鲁棒训练对象 |
| Cheaper | 7 | 无需 donor 答案或额外人工标注，但 rollout 增加前向开销 |
| Broader | 8 | 访问测度可统一短答案、长 CoT、agent 轨迹和多模态生成 |

**平均分 $\bar S=38/5=7.6$；裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 生命周期 3–9 个月；每周工时未知，资源匹配 Yellow。范式探测：First Principles=Yes，Elephant=Yes，Technology Cycle=Yes，Hamming=Partial。

前三项行动：冻结共享前缀与访问权重；完成 gold-only、on-policy、随机等预算前缀三臂；确认中性目标在自由前缀上可计算且非空。

## 5. 决定性实验与失败判据

**最小实验**：固定相同 Teacher/Student 前后向次数，比较 $\eta=0$、$\eta=1$、混合与随机前缀。先测不同前缀上的 $r_f$ 与 $\Delta A$ 稳定性，再进行短步数训练。评估直接问答、释义、间接提示、自由生成 probe、relearning、retain NLL、donor 率、EOS 与长度。

**失败即撤回**：收益在等预算下消失；on-policy 目标主要训练输出拒答；共享前缀约束被破坏；自由前缀上的原事实读出无法可靠定义；或样本/前缀数量本身解释全部收益。

## 6. 来源与诚信状态

- Zhao et al.（2026），*Decoupling KL and Trajectories*，仅借鉴状态与前缀分布定义，不采用 donor KL 作为主损失。
- 基线 §3 对 Teacher-prefix / Student-prefix 的边界说明，以及 §4、§5、§6 的非 donor 验收。
- PARS（2026），arXiv:2609.36612，用于说明 hidden-state 恢复需独立评估。

plan-only；分数为机制推断，不是实验结果或独立 reviewer verdict；文献未完成 D1 批量核验。

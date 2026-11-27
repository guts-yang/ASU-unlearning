---
id: CF2-10
rank: 10
title: 联盟交互激活擦除 I_cd —— 删除单组件加性效应之外的协同绑定
avg_score: 7.2
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无跨模型独立回执
novelty_status: PROCEED WITH CAUTION（Shapley/电路归因近邻较多）
baseline_relation: 从单点 ΔA_c 加权改为组件联盟的非加性交互 Δ²v_cd
---

# CF2-10 · 联盟交互激活擦除 $I_{cd}$

## 0. 一句话故事

> 事实电路可能有冗余和协同：单独删掉 head A 或 MLP B 都无效，联合删除才切断事实。单点 patch 会把这些组件误判为无用。我们在小规模候选组件上估计“联合效应 − 两个单独效应”的交互量，只把真正超出加性解释的联盟作为 Student 的激活擦除目标。

## 1. 问题与近邻空白

Neuron Shapley、Shapley interaction、*Circuit Fingerprints* 和 factorized patching 已用博弈论或因子分解分析组件协作；backup/Hydra 效应也提示单组件消融会低估重要性。因此“组件会协同”不是新发现。

本方案的增量是：把**反事实组件联盟的非加性交互**转成持久遗忘目标。与 CF-03 的单点 patch 门控不同，主对象是 $\delta_{cd}v$，不是单组件效应 $v(c)$；与 CF2-02 的输入主体×关系交互不同，这里交互发生在模型内部组件联盟上。

## 2. 可证伪假设

**H**：存在交互显著为正的组件对/小联盟，其联合反事实干预对目标读出的影响不能由单组件分数之和预测；按交互选择比按单点归因选择更少破坏 retain。若交互主要由 patch OOD 或下游自修复造成，或 Shapley 采样估计跨 seed 不稳定，则否决。

## 3. 方法

### 3.1 有界候选组件集

先用一次粗粒度 block 扫描选取至多 $n=8$ 个候选 head/MLP 通道 $\mathcal C$，禁止直接对全模型组件做指数枚举。对联盟 $K\subseteq\mathcal C$，定义反事实联盟干预后的目标读出下降：

$$
v(K)=r_f(\mathrm{Fwd}(A^+))
-r_f\!\left(\mathrm{PatchFwd}\!\left(A^+-\sum_{c\in K}\Delta A_c\right)\right).
$$

$v(K)$ 越大，联盟 $K$ 越能降低目标事实支持；同时记录 retain 效用成本 $u(K)$。

### 3.2 非加性联盟交互

对组件对 $(c,d)$，在背景联盟 $K\subseteq\mathcal C\setminus\{c,d\}$ 上定义：

$$
\delta_{cd}v(K)=v(K\cup\{c,d\})-v(K\cup\{c\})-v(K\cup\{d\})+v(K).
$$

用预注册的 Shapley-Taylor 权重 $\omega_K$ 和 $M$ 次分层采样估计：

$$
\boxed{
\widehat I_{cd}=\frac{\sum_{K\in\mathcal S_M}\omega_K\delta_{cd}v(K)}
{\sum_{K\in\mathcal S_M}\omega_K}}
$$

$\widehat I_{cd}>0$ 表示协同，$<0$ 表示冗余或相互抑制。权重约定、采样种子、$M$ 和置信区间在实验前冻结；不把采样估计当成精确 Shapley 值。

### 3.3 联盟目标与 Student 内化

选择满足 $\widehat I_{cd}>\tau_I$、效用成本 $u<\tau_U$ 的联盟集合 $\mathcal K^*$，归一化权重 $w_K$ 后构造：

$$
T=\operatorname{stopgrad}\!\left[\Pi_{\mathcal K}\!\left(
\sum_{K\in\mathcal K^*}w_K
\left(A^+-\alpha\sum_{c\in K}\Delta A_c\right)\right)\right].
$$

$$
\mathcal L_F=\|h_S(X^+\oplus y^+)-T\|_2^2,
\qquad
\boxed{\mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R}.
$$

$\Pi_{\mathcal K}$ 保持原事实、donor、EOS 约束；$\lambda_{KL}^{(\mathrm{ablation})}=0$。部署不使用 patch 或 hook。

## 4. 差异化边界

| 近邻 | 已占机制 | 本方案差异 |
|---|---|---|
| Neuron/Shapley Interaction | 神经单元或输入特征的合作归因 | 反事实遗忘目标与 Student 参数内化 |
| Circuit Fingerprints | 少量通道的 Shapley 分解与电路解释 | 面向答案跨度遗忘，并加入 retain 成本和中性约束 |
| CF-03 因果门控 | 单组件 patch 效应加权 | 只保留单组件加性效应之外的 $\delta_{cd}v$ |
| CF2-02 因子交互 | 输入主体×关系因子 | 本方案是内部组件×组件联盟 |

## 5. /idea-evaluator 兼容评估（PROVISIONAL）

**定位**：Novel Method（高风险）。致命缺陷：2 MAJOR——F1 Shapley/电路归因近邻密集；F6 交互估计、OOD 与自修复可能混淆。零 CRITICAL，故 Accept with Revisions。$n>8$ 不满足首验预算，能力匹配黄。

| 维度 | 分数 | 机制依据 |
|---|---:|---|
| Higher | 8 | 能发现单点筛选漏掉的协同事实电路 |
| Faster | 5 | 联盟前向成本高；$n\le8$ 与分层采样仅控制而非消除成本 |
| Stronger | 9 | 显式建模冗余/协同和 retain 成本，针对 backup circuit 风险 |
| Cheaper | 6 | 不需 SAE，但多次 patch-forward 是主要预算 |
| Broader | 8 | 可推广到 MoE、跨层路径和多模态组件联盟 |

**平均分 $36/5=7.2$。裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 范式探测：First Principles=Yes、Elephant=Yes、Technology Cycle=Partial、Hamming=Partial。

修订顺序：先固定 $n=8$ 做交互估计稳定性；与单点归因等预算比较；通过后才进行 Student 训练。

## 6. 决定性实验与失败模式

**最小判别实验**：单点归因选组件、联盟交互选组件、随机等大小联盟三臂；在相同 patch 数和相同训练预算下比较 $r_f$、retain NLL、donor 率、恢复攻击，并报告 $\widehat I_{cd}$ 的 bootstrap 区间和跨 seed 排名。

**失败即撤回**：交互排名跨 seed 翻转；零/均值 patch 与真实反事实 patch 给出相反结论；联合效应可由单点分数之和预测；自修复使训练后目标重新出现。

## 7. 来源与诚信状态

- Ghorbani & Zou（2020），*Neuron Shapley: Discovering the Responsible Neurons*，arXiv:2002.09815。
- *Circuit Fingerprints: How Answer Tokens Encode Their Geometrical Path*（ICML 2026），检索笔记：https://m0rtzz.github.io/paper-notes/ICML2026/interpretability/circuit_fingerprints_how_answer_tokens_encode_their_geometrical_path 。
- Zhang & Nanda（ICLR 2024）及 backup/Hydra 效应相关文献，作为 patch OOD 与自修复风险依据。
- 基线 §2.2、§4.1、§6。
- plan-only；分数非实测、非独立评审。文献未做 D1 批量身份核验，不生成 BibTeX。

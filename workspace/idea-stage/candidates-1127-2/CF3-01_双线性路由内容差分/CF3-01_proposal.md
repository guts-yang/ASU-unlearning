---
id: CF3-01
rank: 1（历史构思排序，不代表当前有效性）
title: 双线性路由—内容差分（BRCD）
revision: v1.1-measurement-first
verdict: REVISE / RE-VALIDATE（设计修订，非独立评审裁决）
evidence_level: round3 真实诊断证据；claim_level=false；无机制有效性结论
novelty_status: PROCEED WITH CAUTION
baseline_relation: 将原始激活差分解为计算角色候选，再以独立路径干预和留出对照验证是否值得内化
implementation_status: 当前工作树为 attention-only；历史四候选/POCS 实现与本版不自动等价
---

# CF3-01 · 双线性路由—内容差分

> **本版是基于失败信号的设计修订，不是实验通过版。** 引用的 round3 结果来自历史提交 `fe5e32f` 的复现报告；当前工作树的 CF3-01 源码是较早的 attention-only 版本。复现时必须固定代码提交、配置和缓存精度，不得把历史实现、当前实现和本版设计混为一套。

## 0. 一句话故事

同一个主体替换会同时改变“模型看向哪里”和“模型读到什么”。BRCD 用精确双线性恒等式生成 route/content（以及可选的 SwiGLU gate/value）候选，但不把它们预设为事实方向；先用全层诊断、原生 QK/V 因子干预和留出对照判断是否存在稳定的目标特异性，再只对稀疏、可靠的候选做 Student 参数内化。retain、非 donor 和反 EOS 约束负责限定遗忘边界，不替代核心遗忘证据。

## 1. 当前证据与修订动机

### 1.1 已观察到的失败信号

以下数值是历史 round3 的真实诊断记录，不是本版新实验结果：

| 观测 | 记录 | 能支持的结论 |
|---|---|---|
| 数据与模型 | 31 个有效配对；Llama-3.2-1B-Instruct；H20 | 已有真实工程诊断，不是 Tiny 结果 |
| 层范围 | `[8,10,12,14]` | 只覆盖后半部分，不能代表全 16 层 |
| 四候选门 | 16 门仅 `8:attn_route`、`12:glu_value` 通过 | 当前门规则下候选覆盖不足 |
| 主链 | 第 10/14 层无唯一候选；`DIAGNOSTIC_KILL`；0 步 | 本轮没有主链训练后的遗忘结果 |
| 效应量 | 多数约 `1e-3`，跨层/候选变号 | 需要噪声下限和稳定性检验，不能直接解释为事实机制 |
| 五个对照臂 | 60 步；dF 全为负，约 `-0.0669~-0.0682` | 尚未观察到分量特异性；forget NLL 朝遗忘目标反方向变化 |
| 投影 | 62/62 为 `identity`；`infeasible=0`；`dist_over_edit=0` | 本轮没有发生实际投影修正，不能据此声称投影机制有效 |

其中 dF 定义为训练后 forget NLL 减训练前 forget NLL，正向才与“原答案支持下降”一致，但正向本身也不足以证明遗忘。

这些结果支持“当前版本未通过机制证据门”，**不足以单独证明机制已经被彻底证伪，也不足以排除测量、门规则或对照设计问题**。因此本版优先修复可判别性，而不是直接提高训练强度。

### 1.2 三个需要分离的问题

1. **测量问题**：patch 效应是否超过缓存量化、重复前向和下游重算的噪声下限。
2. **机制问题**：真实 route/content 是否比等范数随机、wrong/shuffled 和 matched-retain 对照更有目标特异性。
3. **内化问题**：即使 patch 目标有效，Student 是否在相同层、位置和编辑预算下保留这种特异性。

这三个问题不能用一次 full 训练同时回答。

## 2. 问题与参考工作的有机结合

原始 $\Delta A$ 混合了 QK 路由、Value 内容和 MLP 门控。参考工作在本方案中承担不同角色，而不是把别人的结论直接移植成遗忘结论。

| 参考工作 | 在本方案中的作用 | 明确不能推出的结论 |
|---|---|---|
| Elhage et al.（2021），*A Mathematical Framework for Transformer Circuits* | 提供残差流读写、QK/OV 计算结构，支持将分量经 `o_proj` 写回 | 不证明某个分量就是事实，也不证明删除它会遗忘 |
| Wang et al.（ICML 2026），*How Few-Shot Examples Add Up*，arXiv:2605.16591 | 借鉴 QK/V 独立干预、四组合路径分解和因果读出 | 其研究对象是 ICL Function Vector，不是 TOFU 事实删除；QK 路由不等于事实关系 |
| Cunningham & Muça Cirone（ICML 2026），*Contribution Weights*，arXiv:2606.07604 | 诊断 Value 范数、方向对齐、向量抵消和 attention sink，解释“高注意力不等于高贡献” | 其重要性分数不能直接当作遗忘强度或删除方向 |

本方案的**待验证增量**是：把双线性分解作为反事实遗忘目标的候选生成器，并要求其在真实下游读出上超过等条件对照，最后被无 hook 的 Student 内化。它不是发明 QK/V patching，也不是发明双线性代数。

### 2.1 “内容/关系”命名的边界

本文的 route 指 $\pi$ 的计算角色，content 指 $V$ 的计算角色：

- route 可以描述“从哪里读、读多少”，但不能直接命名为“事实关系”。
- content 可以描述“写入了什么向量”，但不能直接命名为“纯事实内容”。
- 主体替换引起的差异仍可能包含身份、词形、词频、句法和 sink 效应。

如果要研究知识图谱意义上的 relation，而不是注意力路由，必须额外构造关系对照并验证其独立效应；本版不把 route/content 改名包装成已经完成的“关系/事实解耦”。

## 3. 固定输入、读出与梯度边界

对同一关系构造严格配对：

$$
X^+=X(s_f,r),\qquad X^-=X(s_r,r).
$$

两次 Teacher 前向分别为 $(X^+,y^+)$ 和 $(X^-,y^+)$，共享同一原始答案。要求主体 token 等长、槽位对齐、槽外 token 不变，donor 主体与遗忘作者/别名簇隔离。$y^-$ 不拼入 Teacher 对照轨迹，不作为 Student 目标。

Teacher 与 Student 从同一 checkpoint 初始化，Teacher 全冻结。$A^+_{T,\ell,p}$ 和 $A^-_{T,\ell,p}$ 均从 teacher-forced 完整序列的 DecoderLayer 输出读取，$p\in S_f$ 为答案跨度位置。

定义真实冻结下游读出：

$$
r_f(h)=-\operatorname{NLL}_{\mathrm{down}}(h;X^+,y^+),
$$

$$
s_{d_j}(h)=-\operatorname{NLL}_{\mathrm{down}}(h;X^+,y_j^-).
$$

$r_f$ 越大表示越支持原事实，$s_{d_j}$ 越大表示越支持 donor。后者只用于单边“不支持 donor”约束，不作为 MSE、KL 或 soft label。真实生成长度、完整回答率和 EOS rate 另行测量，不用不可微长度直接构造梯度。

所有 Teacher 激活、离散 token、候选分量和最终目标均 stop-gradient；Student 梯度只经过 Student 前向和训练损失。

## 4. BRCD 候选生成器

### 4.1 通用恒等式

对逐元素双线性运算 $B(a,b)=a\odot b$：

$$
\boxed{
B(a^+,b^+)-B(a^-,b^-)
=
\underbrace{\frac12(a^+-a^-)\odot(b^++b^-)}_{D_{\mathrm{route}}}
+
\underbrace{\frac12(a^++a^-)\odot(b^+-b^-)}_{D_{\mathrm{content}}}
}
$$

这是精确恒等式，不是一阶近似；它只提供计算分解，不自动赋予事实语义。

### 4.2 Attention 候选

对层 $\ell$、头 $h$、答案位置 $p$：

$$
O^\pm_{\ell,h,p}=\sum_q\pi^\pm_{\ell,h,p,q}v^\pm_{\ell,h,q},
$$

$$
D^{\mathrm{attn}}_{\mathrm{route}}
=\frac12\sum_q(\pi^+_q-\pi^-_q)(v^+_q+v^-_q),
$$

$$
D^{\mathrm{attn}}_{\mathrm{content}}
=\frac12\sum_q(\pi^+_q+\pi^-_q)(v^+_q-v^-_q).
$$

在 GQA 中先将共享 KV Value 头对齐到查询头。逐头分量经 Concat 和该层 `o_proj` 映回残差流：

$$
C^{\mathrm{attn}}_k=\operatorname{o\_proj}(D^{\mathrm{attn}}_k),
\qquad k\in\{\mathrm{route,content}\}.
$$

`o_proj` 只解决写回空间，不解决语义纯度。

### 4.3 可选 SwiGLU 候选

对实际参与乘法的量定义：

$$
a=\operatorname{SiLU}(g),\qquad m=a\odot u,
$$

其中 $g,u$ 分别是 `gate_proj` 和 `up_proj` 输出，$m$ 经 `down_proj` 写回。对 $\beta=1$ 的 Swish，使用 $\operatorname{SiLU}$ 实现：

$$
D_{\mathrm{gate}}
=\frac12(a^+-a^-)\odot(u^++u^-),
$$

$$
D_{\mathrm{value}}
=\frac12(a^++a^-)\odot(u^+-u^-),
$$

$$
C_{\mathrm{gate}}=D_{\mathrm{gate}}W_{\mathrm{down}}^T,
\qquad
C_{\mathrm{value}}=D_{\mathrm{value}}W_{\mathrm{down}}^T.
$$

由线性性，$C_{\mathrm{gate}}+C_{\mathrm{value}}$ 精确重构 SwiGLU 输出差。$\operatorname{SiLU}(g)$ 可负且可大于 1，不是门打开概率。

首版主诊断保持 attention-only；GLU 是独立可选实验臂。四个候选不默认求和，两个 family 同时出现候选时标记 cross-family ambiguity，另做交互对照，不为迎合故事任意选一项。当前工作树尚未把本版 GLU 设计视为已实现主链。

## 5. 测量优先的全层与分量诊断

### 5.1 数值噪声下限先于机制解释

当前历史实现的缓存曾将 $\pi,V$ 和完整序列隐藏状态转为 float16，而观测效应约为 $10^{-3}$。因此先做以下检查，不把该现象直接当成机制结论：

1. 同一输入重复前向，测重复读出误差。
2. 用缓存隐藏状态做零编辑 patch，测与原始前向的读出差异：

$$
\epsilon_0=
\left|
\operatorname{NLL}_{\mathrm{cache\ no\ edit}}
-\operatorname{NLL}_{\mathrm{original}}
\right|.
$$

3. 比较在线 float32 分量与 float16 缓存分量的 patch 效应、方向一致性和排序一致性。
4. 若候选效应未超过预注册的测量下限，输出 `MEASUREMENT_UNRELIABLE`，不进入候选选择或 Student 训练。

这里的 $\epsilon_0$ 是工程测量诊断，不是遗忘阈值；阈值必须在实验前冻结。

### 5.2 全层诊断不等于全层强制训练

第一阶段对全部 $L=16$ 层做 attention-only 诊断，随后仅对少数候选层做层 × 头 × 答案位置的细化扫描。允许某层没有任何候选，也允许最终只有稀疏层集合进入训练。

不得再使用“目标四层全部必须唯一通过”作为进入训练的必要条件。若全层扫描后仍没有稳定且超过对照的候选，应停止 BRCD 主线，而不是把“至少 3/4 层通过”当作新的放宽门槛。

候选选择使用 discovery 数据，候选确认使用按作者/实体簇隔离的 held-out 数据；同一批配对不得同时承担搜索和最终确认。若数据无法支持足够的关系/实体划分，应报告 `INSUFFICIENT_SPLIT`，不补造 probe 或显著性结果。

### 5.3 分量效应与对照

对候选 $C$ 做冻结下游 patch：

$$
H^{\mathrm{patch}}=H^+-\alpha C,
$$

$$
\Delta_f(C)=
\operatorname{NLL}_{\mathrm{patch}}-
\operatorname{NLL}_{\mathrm{before}}.
$$

$\Delta_f>0$ 表示原答案 NLL 上升，只说明原答案支持下降，不等于选择性遗忘。再定义真实与 null 的特异性：

$$
\Delta_s(C)=
\Delta_f(C_{\mathrm{true}})-\Delta_f(C_{\mathrm{null}}).
$$

至少比较以下等条件对照：

- true route/content；
- wrong/shuffled 配对；
- random equal-norm 分量；
- 原始等范数 $\Delta A$；
- matched-retain 或同实体非目标关系对照。

所有臂固定层、头、答案位置、patch 深度、编辑范数和读出协议。主链不能只用第 8/12 层，而对照臂却强制用全部四层。

候选确认至少要求预注册的效应量、置信区间下限和真实/null 分离同时满足；不能只比较 `mean_effect>0` 与 `mean_specificity>0`。统计重采样以作者/实体簇为单位，不能把相关配对当成完全独立样本。

### 5.4 原生 QK/V 四组合干预

为验证“路径作用”而非只验证 BRCD 代数，对选定层/头/位置执行原生 QK 与 V 的独立 patching。$F(a,b)$ 表示使用 $a$ 来源的 QK、$b$ 来源的 V 后，经冻结下游得到的原答案 NLL；$+$ 为原始轨迹，$-$ 为反事实轨迹：

| 配置 | QK | V | 解释 |
|---|---|---|---|
| $F(+,+)$ | 原始 | 原始 | 原始基线 |
| $F(-,+)$ | 反事实 | 原始 | 只改变 QK 路由通路 |
| $F(+,-)$ | 原始 | 反事实 | 只改变 Value 内容通路 |
| $F(-,-)$ | 反事实 | 反事实 | 两条通路均改变 |

两路的平均边际效应定义为：

$$
E_{QK}=\frac12\left[
(F(-,+)-F(+,+))+(F(-,-)-F(+,-))
\right],
$$

$$
E_V=\frac12\left[
(F(+,-)-F(+,+))+(F(-,-)-F(-,+))
\right].
$$

这两式是二因子实验的平均边际效应，且 $E_{QK}+E_V=F(-,-)-F(+,+)$。它们用于判断路径是否有因果作用，不直接证明该作用属于事实关系或纯内容。

原生 QK/V 干预与 block-output 的 BRCD 减法不是同一个实验。若原生干预稳定有效而 BRCD 分量无效，应判定候选参数化或写回方式失败，而不是继续增加层数。

Contribution Weights 仅作为头级诊断，检查 Value 范数、方向抵消和 sink；不未经消融就乘入遗忘损失。

## 6. 条件式 Student 内化

### 6.1 候选目标

令 $\mathcal S$ 为通过独立诊断的稀疏层 × 位置集合，$C^*_{\ell,p}$ 为经选择规则确认的候选。原始编辑目标为：

$$
 z_{\ell,p}=A^+_{T,\ell,p}-\alpha C^*_{\ell,p},
$$

其中 $\alpha$ 是编辑强度，历史默认值为 0.1，但正式实验前需重新冻结。$\mathcal S$ 可以为空；为空时输出诊断停止，不启动训练。

主损失仍为：

$$
\mathcal L_F=
\frac1{|\mathcal S|}
\sum_{(\ell,p)\in\mathcal S}
\left\|h_{S,\ell,p}-T_{\ell,p}\right\|_2^2,
\qquad
T_{\ell,p}=\operatorname{stopgrad}(z_{\ell,p}).
$$

$\mathcal L_F$ 的作用是把已验证的目标写入 Student 参数，不是证明目标天然正确。部署只保留 Student，正常运行一次前向，不需要 Teacher、反事实输入或运行时 patch。

### 6.2 损失尺度诊断

历史报告中 method loss 很小，不足以仅凭数值断言 retain 梯度主导。下一轮先记录各损失对可训练 LoRA 参数的实际梯度范数：

$$
G_m=\nabla_\theta(\lambda_m\mathcal L_m),
\qquad
\eta_m=\frac{\|G_m\|_2}{\sum_j\|G_j\|_2+\epsilon}.
$$

$\eta_m$ 是梯度量级诊断，不是方法有效性的证明。再比较固定层集合、相同 batch 顺序和相同 seed 下的 `method-only`、`retain-only`、`full` 三臂。

若需要在候选已可靠的前提下比较不同编辑尺度，可增加独立归一化消融：

$$
\mathcal L_F^{\mathrm{rel}}
=
\frac1{|\mathcal S|}
\sum_{(\ell,p)\in\mathcal S}
\frac{\|h_{S,\ell,p}-T_{\ell,p}\|_2^2}
{\|A^+_{\ell,p}-T_{\ell,p}\|_2^2+\epsilon}.
$$

它让不同候选的初始相对误差更可比较，但会放大近零编辑的噪声，不能在噪声门未通过时启用，也不能与原始 MSE 的历史数值混写。

核心训练式保留为：

$$
\boxed{
\mathcal L_{\mathrm{core}}=
\lambda_F\mathcal L_F+\beta\mathcal L_R
}
$$

其中 $\mathcal L_R=\operatorname{NLL}_S(y_R|X_R)$ 是独立 retain 训练集上的保留能力锚。

## 7. 核心约束、工程护栏与 POCS 的位置

### 7.1 角色划分

| 组件 | 角色 | 是否核心遗忘证据 |
|---|---|---|
| BRCD/QK-V 候选及真实/null 特异性 | 产生并筛选遗忘方向 | 是 |
| Student 激活目标与无 hook 内化 | 将方向写入参数 | 是 |
| retain NLL | 保留通用/非目标能力 | 是核心效用约束，但不产生遗忘 |
| 同实体非目标关系 | 检验选择性，避免整实体能力被抹掉 | 是核心选择性约束 |
| donor 支持不升 | 排除答案替换 | 是中性边界，不是遗忘方向 |
| EOS、拒答、长度、完整回答率 | 排除输出抑制捷径 | 反捷径护栏，不是遗忘证据 |
| POCS | 约束修正候选目标 | 可选辅助，不是主机制 |
| probe、relearning、量化恢复 | 独立验收 | 不属于训练核心，但不能省略 |

### 7.2 非 donor 可行集与读出语义

保留 proposal 的目标约束：

$$
\mathcal K=
\{h:r_{f,p}(h)\le c_f,\\
s_{d_j,p}(h)\le d_j,\\
|E_{k,p}(h)|\le e_k\}.
$$

这里 $r_f\le c_f$ 表示原事实支持足够低。它与当前 runtime 中限制 forget NLL 不要过度上升的 `L_fact` 不是同一约束，不能直接复用 `fact_delta`。

$c_f$ 不能用待评估的 BRCD $z$ 自身的最大值加 margin 来定义，否则容易把当前候选全部判为已满足。必须使用独立的、预注册的中性参考/校准协议；如果没有这样的参照，应标记 `c_f` 未校准并关闭投影，而不是声称可行集已验证。

donor 约束只允许单边测量“不支持 donor”，不能变成 donor 目标。记录为 `donor_constraint_only=true`，而非笼统一刀切的 `donor_answer_in_training=false`。

### 7.3 EOS 相对护栏

设 $m_E$ 为提前 EOS 倾向的可微代理，正值越大表示越倾向结束。相对于 Teacher 的变化约束可写为：

$$
\mathcal L_E=
\left[
(m_E^S-m_E^T)-\epsilon_E
\right]_+^2.
$$

它只惩罚相对 Teacher 的 EOS 恶化，不因基线本身存在 EOS 就施加绝对惩罚。真实 EOS rate、长度、拒答和完整回答率仍需独立检查；当前历史报告的 EOS rate 全为 1.0，不能被解释为“没有生成风险”。

### 7.4 可选局部仿射 POCS 代理

原始 $\mathcal K$ 由非线性冻结下游定义，通常非凸，不能声称存在可直接计算的精确 $\Pi_{\mathcal K}$。只有在候选目标确实违反真实约束时，才启用以下辅助步骤。

令 $z=A^+-\alpha C^*$，将每个约束写成 $\phi_i(h)\le0$，在 $z$ 处计算：

$$
 a_i=\nabla_h\phi_i(z),
$$

$$
\widehat\phi_i(h)=
\phi_i(z)+a_i^T(h-z).
$$

局部半空间及其闭式投影为：

$$
\mathcal C_i(z)=\{h:\widehat\phi_i(h)\le0\},
$$

$$
\boxed{
P_i(h)=h-
\frac{[\phi_i(z)+a_i^T(h-z)]_+}
{\|a_i\|_2^2+\epsilon}a_i
}
$$

并加入信任域：

$$
\mathcal B_\rho(z)=\{h:\|h-z\|_2\le\rho\},
\qquad
\rho=\min(\rho_{\max},\kappa\alpha\|C^*\|_2).
$$

流程为：

1. 若 $z$ 已满足真实约束，返回 $T=z$，记录 `projection_active=false`。
2. 否则在局部半空间与信任球之间交替投影，必要时重新线性化。
3. 用真实冻结下游重算所有 $\phi_i$，检查 trust region、donor、EOS、retain 和同实体关系。
4. 对接受的 $T$ 重新执行 true/null patch；不能沿用投影前特异性。
5. 不可行时输出 `PROJECTION_INFEASIBLE`，严格档停止，不静默退回无约束 $z$。
6. 最后 $T$ 全部 stop-gradient，Student 只通过 $\mathcal L_F$ 学习。

普通 POCS 只给局部代理的可行点，不保证离 $z$ 最近；若要保持最近点语义，需要另行加入 Dykstra 校正。当前历史 round3 的全部 `identity` 只说明本轮没有实际投影修正，不构成 POCS 有效性证据。投影需单独构造“真实约束被违反”的单元/小规模测试，检查 `projection_active=true`、`dist_from_z>0`、信任域和真实回验。

若投影修正量持续接近或超过原始 BRCD 编辑量，主要效果可能来自通用读出约束，而非 BRCD 分量；此时不得将其包装为双线性机制成功。

## 8. 分阶段实验协议

### 阶段 A：只诊断，不训练主线

1. 固定提交、依赖、数据划分、缓存精度和模型初始化。
2. 完成恒等式、零编辑恢复、重复前向和 float32/float16 对照。
3. 对全 16 层做 attention-only 扫描，再对候选层做头级/位置级细化。
4. 完成 QK/V 四组合干预及真实/null/random/matched-retain 对照。
5. GLU 作为独立可选臂，不与 attention 默认相加。
6. 若没有稳定、超过噪声和对照的候选，停止主线；不进入 Student 训练。

### 阶段 B：条件式小规模内化

仅在阶段 A 有留出证据后启动，固定相同层集合、答案位置、编辑范数和训练预算，比较：

- BRCD / 真实候选；
- 原生路径或原始等范数 $\Delta A$；
- random equal-norm；
- wrong/shuffled；
- method-only / retain-only / full 损失诊断。

至少使用预注册的多个 seed 和作者/实体簇划分。报告 dF、retain、holdout、同实体非目标关系、donor 支持、EOS/长度/完整回答率、无 hook checkpoint、probe 和短程 relearning。`COMPLETED` 只表示运行完成，不表示遗忘有效。

阶段 B 不是“把 retain_weight 调为 0 让 dF 变正”。`retain_weight=0` 只能作为诊断臂，必须同时满足效用和反捷径条件才有解释价值。

### 阶段 C：全量与投稿证据

只有在阶段 B 显示真实候选相对等条件对照有稳定优势、无 hook Student 内化且效用/中性约束合格后，才启动 full、多 seed、恢复攻击、量化恢复和完整审计。否则保留失败结果并停止该路线。

## 9. 明确失败判据

- **测量失败**：候选效应未超过噪声下限，或 float32/float16 方向与排序不稳定。
- **机制失败**：全层诊断没有稳定目标特异性，或真实分量与随机/错配对照不可分。
- **参数化失败**：原生 QK/V 路径有效，但 BRCD 分量在相同读出下无效。
- **内化失败**：patch 目标有效，但无 hook Student 不优于等条件随机对照，或 dF 仍朝反方向。
- **效用失败**：retain、holdout 或同实体非目标关系明显恶化。
- **中性失败**：donor 支持上升、拒答/提前 EOS/长度异常，或用这些行为伪装 dF 改善。
- **投影失败**：真实约束违反、信任域失真、投影后特异性消失，或投影长期只是恒等却声称已验证机制。
- **统计失败**：划分不足、单 seed、以描述性 std 代替置信区间，或在同一数据上搜索并确认候选。

历史 stage0–14 和 Tiny/smoke 只用于工程排错，不作为本版正向证据。

## 10. 实现状态与来源

- 当前工作树的 `CF3-01_双线性路由内容差分/method.py` 为 attention-only 实现；本版的全层协议、因子干预、噪声下限、梯度诊断和强制投影测试均未被本文件声称为已完成代码。
- 历史提交 `fe5e32f` 含四候选和局部仿射 POCS 的另一实现版本；其 round3 结果仅用于失败诊断。当前源码、历史实现和本版设计需要重新对齐，不得直接混跑缓存或指标。
- 默认算法、配置和实验结果不因本次文档修订而自动改变；本版不生成 BibTeX，不伪造独立 reviewer verdict。
- Cunningham & Muça Cirone，*Contribution Weights: A Geometrical Analysis of Self-Attention Transformers*，ICML 2026，arXiv:2606.07604，https://arxiv.org/abs/2606.07604。
- Wang et al.，*How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning*，ICML 2026，arXiv:2605.16591，https://arxiv.org/abs/2605.16591。
- Elhage et al.（2021），*A Mathematical Framework for Transformer Circuits*，https://transformer-circuits.pub/2021/framework/index.html。
- 基线：`方向I_基线_v0.6.1_answer_span.md` §2、§4、§5、§6。
- 失败证据：`git show fe5e32f:workspace/experiments/methods-1127-2/cf3_01_replication/REPLICATION_REPORT.md` 与同提交 `summary.json`。

**最终定位**：这是一个“测量与因果验证优先、稀疏目标条件内化”的修订方案。它的首要任务不是让门通过，而是判断 BRCD 是否真的提供了随机方向无法替代的选择性遗忘信号；在该信号出现之前，不进入 full 训练，也不把工程护栏当成方法贡献。

---
title: CF3-01 双线性路由—内容差分
subtitle: 从一条数据如何流进模型，到遗忘如何一步步发生——面向初学者的机制导览
eyebrow: BRCD · MECHANISM GUIDE · v1.1 measurement-first
date: 2026-11-27
scope: 依据 proposal v1.1-measurement-first；讲机制与数据流，不把设计或代码验收等同于遗忘有效性
---

# CF3-01 双线性路由—内容差分（BRCD）

> **先说清楚这份文档是什么**：它是一本"数据旅程"导览 —— 跟着一条问答数据，看它如何进入模型、在哪里被拆开、怎么判断拆出来的东西值不值得删、最后怎么把"删除"写进模型参数。它不是一份"方法已成功"的报告。

> 🚨 **最重要的边界（请先看）**：本文依据的 `CF3-01_proposal.md` 是 **v1.1-measurement-first（基于失败信号的设计修订版）**，不是通过版。历史 round3 的真实诊断显示：候选效应只有约 $10^{-3}$ 量级、16 个门只过了 2 个、对照臂 dF 全为负、主链 0 步。这些结果说明**当前版本尚未拿到"机制有效"的证据**，因此本版的第一任务是先证明"拆出来的方向确实有用"，而不是直接加大训练力度。当前工作树代码是 attention-only，历史四候选/POCS 实现（`fe5e32f`）与本版设计不自动等价。

## 0. 一张地图：数据从头到尾怎么走

先把整条路看一遍，后面再逐站展开。 forget（遗忘）在这里**不是删硬盘上的训练样本，也不是把某个神经元烧掉**，而是：**用参数微调，让模型面对原问题时不再容易沿原来的内部通路读出目标事实，同时尽量别伤到其他能力。**

```text
【输入】一对只差主体、但答案相同的问答
   X⁺ = "Alice Miller 出生在哪？"  配答案 y⁺ = "Paris"
   X⁻ = "Bob Smith 出生在哪？"     仍配答案 y⁺ = "Paris"（故意相同）
        │
        ▼
【处理 · 阶段 A：只诊断，不训练】
   ① 噪声体检     重复前向 + 零编辑 ε₀ + float32/float16 一致性
        │         效应没超过噪声 → MEASUREMENT_UNRELIABLE，立刻停
        ▼
   ② 双线性分解   把"换成 Bob 后激活的变化 ΔA"精确拆成
                  路由项 D_route（看哪里变了）+ 内容项 D_content（读到什么变了）
        ▼
   ③ 因果筛选     冻结 Teacher 上做 patch：true / wrong / random / 等范数ΔA / matched-retain
                  + 原生 QK/V 四组合干预 + 作者簇留出验证
        ▼
   ④ 稀疏选择     只有"超过噪声 + 超过对照 + 跨留出稳定"的分量才入选
                  没有任何候选 → 停主线，不进训练
        │
        ▼
【处理 · 阶段 B：条件式内化（仅在 A 通过后）】
   ⑤ 造目标       T = stopgrad(A⁺ − α·C*)
   ⑥ 训 Student   让 Student 在原问题上的中间状态靠近 T
                  同时挂着 retain / donor / EOS / fact 护栏
        │
        ▼
【输出】只部署 Student
   用户问原问题 → Student 正常前向一次 → 不再容易读出 Paris
   不需要 Teacher、不需要反事实输入、不需要运行时 hook
        │
        ▼
【验收】probe / relearning / retain NLL / 生成行为，判断是否"真遗忘"
```

把这张图记住，后面每一节都是其中一站。

## 1. 输入：一对"只换人名、不换答案"的问答

要研究"模型怎么读出 Alice 出生在 Paris"，我们需要一个**受控对照**：只改变想研究的那个因素（主体是谁），其他一切尽量不动。

| 角色 | 问题 | 喂给模型的答案 |
|---|---|---|
| 原事实 $X^+$ | Where was **Alice Miller** born? | Alice Miller was born in **Paris** |
| 反事实 $X^-$ | Where was **Bob Smith** born? | **仍然是** Alice Miller was born in **Paris** |

| 符号 | 含义 | 在例子里 |
|---|---|---|
| $X^+$ | 原问题 | Alice 的出生地问题 |
| $X^-$ | 只换主体的反事实问题 | Bob 的出生地问题 |
| $y^+$ | 原数据集答案 | Paris 所在的原始答案 |
| $y^-$ | donor（另一个主体）自己的答案 | 例如 Bob 的答案；**只用于评估，不作训练目标** |

**为什么答案要故意相同？** 如果连答案也换成 Bob 的，那么观察到的差异就混在一起了：到底是"问题里的人变了"，还是"答案也变了"？保持答案相同，才能让差异主要反映"主体替换"这一个因素。

> 💡 **最容易误解的一点**：$X^-$ 不是让 Student 学会回答 Bob 的事，它只是一把"测量尺"——用来看模型内部哪些计算会随着主体改变而改变。proposal 明确要求 donor 主体与遗忘作者/别名簇隔离，避免对照本身引入混淆。

两次 Teacher 前向都处理**同一个原始答案** $y^+$，记为 $(X^+,y^+)$ 与 $(X^-,y^+)$。Teacher（冻结、不更新）在 teacher-forced 的完整序列上、答案跨度位置 $p\in S_f$ 读取隐藏状态：

$$
A^+_{T,\ell,p}=h_{T,\ell}(X^+\oplus y^+;p),\qquad
A^-_{T,\ell,p}=h_{T,\ell}(X^-\oplus y^+;p).
$$

## 2. 为什么不直接拿"全部激活差"来删？

Transformer 每一层都会输出一组隐藏向量。对同一个答案位置 $p$，最直觉的差是：

$$
\Delta A_{\ell,p}=A^+_{\ell,p}-A^-_{\ell,p}.
$$

它表示"把 Alice 换成 Bob 后，第 $\ell$ 层的状态变了多少"。但这个差值是个**大杂烩**：

| 变化来源 | 直观含义 | 一定是目标事实吗？ |
|---|---|---|
| 注意力路由 | 模型看了哪些 token、各看多少 | 否，可能只是名字 token 不同 |
| Value 内容 | 从被看的 token 里读出了什么向量 | 否，可能是一般身份信息 |
| 无关上下文 | 词形、句法、位置、attention sink | 否 |
| 目标事实相关计算 | 与"Paris"这个读出有关的信号 | 有可能，但要验证 |

所以 **$\Delta A$ 大，不等于里面全是要删的事实**。直接把整个向量压到 0 或整体反向，很可能把正常语言能力和其他人的知识一起破坏。CF3-01 的第一步不是训练，而是把这个混合差值拆得更清楚、并验证哪一部分真的有用。

## 3. 注意力入门：路由 $\pi$ 与内容 $v$ 在哪里？

在答案 token 位置 $p$，一个注意力头的输出是：

$$
O_{p}=\sum_q \pi_{p,q}\,v_q .
$$

- $v_q$：第 $q$ 个输入位置的 **Value 向量**，可以理解为"那一页写着什么内容"；
- $\pi_{p,q}$：注意力权重，由 Query 与 Key 算出再经 softmax 得到，决定"当前位置从 $q$ 读多少"；
- 对所有被读的位置 $q$ 加权求和，就是当前位置的注意力输出。

一个生活化比喻：**路由 $\pi$ 决定"看哪几页、每页看多少"，内容 $v$ 是"每页实际写着什么"。** 比喻只帮忙建立直觉，真正的计算仍然是矩阵乘法。

CF3-01 用的是**已经过 softmax 的 $\pi$**，不是单独去改 Q 或 K。对于 Llama 的 GQA（分组查询注意力），代码会先把共享的 KV Value 头对齐到查询头，保证 $\pi$ 和 $v$ 能正确相乘。

## 4. 第一个关键计算：把差值精确拆成两项

先看最简单的逐元素乘法 $B(a,b)=a\odot b$。原输入给 $(a^+,b^+)$，反事实输入给 $(a^-,b^-)$，则有一个**精确恒等式**：

$$
\boxed{
a^+\odot b^+-a^-\odot b^-
=
\underbrace{\tfrac12(a^+-a^-)\odot(b^++b^-)}_{D_{\mathrm{route}}\ \text{路由项}}
+
\underbrace{\tfrac12(a^++a^-)\odot(b^+-b^-)}_{D_{\mathrm{content}}\ \text{内容项}}
}
$$

把右边展开就会正好回到左边，所以**这是代数恒等式，不是一阶近似**：原差值一点没丢，只是被重新分成了"路由变化"和"内容变化"两部分。

一个标量小例子（把向量简化成一个数）：

| 量 | 原输入 | 反事实输入 |
|---|---:|---:|
| 路由 $a$ | $0.8$ | $0.2$ |
| 内容 $b$ | $10$ | $6$ |
| 乘积 $ab$ | $8$ | $1.2$ |

总差为 $8-1.2=6.8$。代进公式：

$$
D_{\mathrm{route}}=\tfrac12(0.8-0.2)(10+6)=4.8,
\qquad
D_{\mathrm{content}}=\tfrac12(0.8+0.2)(10-6)=2.0.
$$

$4.8+2.0=6.8$，严丝合缝。**注意**：这里的"路由"和"内容"只是**计算角色**，不是"事实关系"和"纯事实内容"。主体替换带来的差异里仍可能有身份、词形、词频、句法和 sink 效应。proposal 特意划清这条边界，不把 route/content 偷换成已经完成的"关系/事实解耦"。

### 4.1 放进注意力公式

把 $a$ 换成注意力权重 $\pi$、$b$ 换成 Value 向量 $v$，并对所有位置 $q$ 求和：

$$
D^{\mathrm{attn}}_{\mathrm{route}}
=\frac12\sum_q(\pi^+_q-\pi^-_q)(v^+_q+v^-_q),
$$

$$
D^{\mathrm{attn}}_{\mathrm{content}}
=\frac12\sum_q(\pi^+_q+\pi^-_q)(v^+_q-v^-_q).
$$

两项相加恰好恢复原来的注意力输出差：

$$
D^{\mathrm{attn}}_{\mathrm{route}}+D^{\mathrm{attn}}_{\mathrm{content}}=O^+-O^-.
$$

代码随后把各头拼回隐藏维度，再经该层 `o_proj` 映回残差流空间，得到可与该层隐藏状态直接加减的候选分量：

$$
C^{\mathrm{attn}}_k=\operatorname{o\_proj}\!\left(D^{\mathrm{attn}}_k\right),
\qquad k\in\{\mathrm{route,content}\}.
$$

> 🚨 **`o_proj` 只解决"写回哪个空间"，不解决"语义纯不纯"。** 恒等式保证加回去等于原输出差，但不保证某一项天然只含事实。所以接下来必须先做测量和筛选。

### 4.2 可选的 SwiGLU 分支（本版不默认混用）

SwiGLU 神经元里实际参与乘法的是 $a=\operatorname{SiLU}(g)$ 和 $u$（$g,u$ 分别是 gate/up 投影输出）。对同一恒等式可得：

$$
D_{\mathrm{gate}}=\tfrac12(a^+-a^-)\odot(u^++u^-),
\qquad
D_{\mathrm{value}}=\tfrac12(a^++a^-)\odot(u^+-u^-),
$$

经 `down_proj` 写回后，$C_{\mathrm{gate}}+C_{\mathrm{value}}$ 精确重构 SwiGLU 输出差。但 v1.1 规定**首版主诊断保持 attention-only**，GLU 是独立可选臂；两个 family 同时出现候选时标记 cross-family ambiguity，另做交互对照，不为迎合故事随便挑一个。

## 5. 阶段 A 第一步：先做"噪声体检"，再谈机制

这是 v1.1 相对旧版最重要的改动。历史 round3 里，缓存曾把 $\pi$、$V$ 和隐藏状态转成 float16，而观测到的效应只有约 $10^{-3}$ —— **效应可能和舍入误差差不多大**。如果测量本身不可靠，后面所有"筛选"都是在噪声里挑运气。

所以正式看机制之前，先做三件体检：

1. **重复前向**：同一输入跑两次，测读出误差。
2. **零编辑 patch**：把缓存的隐藏状态原封不动地"patch"回去（不做任何编辑），测它与原始前向的差：

$$
\epsilon_0=
\left|
\operatorname{NLL}_{\mathrm{cache\ no\ edit}}
-\operatorname{NLL}_{\mathrm{original}}
\right|.
$$

3. **精度对照**：比较在线 float32 分量与 float16 缓存分量的 patch 效应、方向一致性和排序一致性。

如果候选效应没超过预注册的测量下限，系统输出 `MEASUREMENT_UNRELIABLE`，**不进入候选选择，也不训练**。

> 💡 $\epsilon_0$ 是工程测量诊断（"这把尺准不准"），不是遗忘阈值（"删得够不够"）。阈值必须在实验前冻结，不能看完结果再定。

## 6. 阶段 A 主体：全层诊断 + 等条件对照 + 稀疏选择

体检通过后，才开始找"到底哪一层的哪个分量值得删"。

**第一遍：全 16 层 attention-only 扫描。** 旧版要求"目标四层每层都必须唯一通过"，v1.1 明确废除这条：允许某层没有候选，也允许最后只有稀疏几层入选。若全层扫完仍没有稳定、超过对照的候选，就**停止 BRCD 主线**，而不是把门槛偷偷放宽成"至少 3/4 层通过"。

**第二遍：分量效应与五类对照。** 对候选 $C$ 做冻结下游 patch：

$$
H^{\mathrm{patch}}=H^+-\alpha C,
$$

$$
\Delta_f(C)=
\operatorname{NLL}_{\mathrm{patch}}-\operatorname{NLL}_{\mathrm{before}} .
$$

$\Delta_f>0$ 表示原答案 NLL 上升（更难读出），但**这本身还不等于选择性遗忘**。真正的特异性要看真实分量与 null 分量的差：

$$
\Delta_s(C)=\Delta_f(C_{\mathrm{true}})-\Delta_f(C_{\mathrm{null}}).
$$

所有对照臂必须固定层、头、答案位置、patch 深度、编辑范数和读出协议，至少包括：true route/content、wrong/shuffled 配对、random equal-norm 分量、原始等范数 $\Delta A$、matched-retain 或同实体非目标关系。

**第三遍：原生 QK/V 四组合干预。** 为了验证"路径因果作用"而不只是 BRCD 代数，直接对 QK 和 V 做 $2\times2$ 因子替换。记 $F(a,b)$ 为用 $a$ 来源的 QK、$b$ 来源的 V 经冻结下游算出的原答案 NLL：

| 配置 | QK | V | 含义 |
|---|---|---|---|
| $F(+,+)$ | 原始 | 原始 | 原始基线 |
| $F(-,+)$ | 反事实 | 原始 | 只改 QK 路由通路 |
| $F(+,-)$ | 原始 | 反事实 | 只改 V 内容通路 |
| $F(-,-)$ | 反事实 | 反事实 | 两条通路都改 |

两路平均边际效应为：

$$
E_{QK}=\tfrac12\big[(F(-,+)-F(+,+))+(F(-,-)-F(+,-))\big],
$$

$$
E_V=\tfrac12\big[(F(+,-)-F(+,+))+(F(-,-)-F(-,+))\big],
$$

且满足 $E_{QK}+E_V=F(-,-)-F(+,+)$。若原生干预稳定有效、而 BRCD 分量无效，判定为**候选参数化或写回方式失败**，而不是继续加层。

**第四遍：留出验证 + 簇 bootstrap。** 候选选择用 discovery 数据，候选确认用按作者/实体簇隔离的 held-out 数据，同一批配对不能既搜索又确认。统计重采样以簇为单位（不能把相关配对当独立样本）。数据不足以划分时报告 `INSUFFICIENT_SPLIT`，**不补造 probe 或显著性结果**。

### 6.1 阶段 A 的三个出口

| 结果 | 含义 | 下一步 |
|---|---|---|
| 有稳定候选 | 超过噪声、超过对照、跨留出稳定 | 进入阶段 B 稀疏内化 |
| 无稳定候选 | 全层扫描也没找到可靠方向 | 停主线，不进训练 |
| 测量/数据失败 | `MEASUREMENT_UNRELIABLE` / `INSUFFICIENT_SPLIT` | 先修测量或数据，不硬训 |

## 7. 阶段 B：把选中的方向写进 Student（参数内化）

只有阶段 A 通过，才进入训练。设 $\mathcal S$ 是通过独立诊断的稀疏层 × 位置集合，$C^*_{\ell,p}$ 是确认过的候选，构造固定目标：

$$
z_{\ell,p}=A^+_{T,\ell,p}-\alpha\,C^*_{\ell,p},
$$

$$
\boxed{
T_{\ell,p}=\operatorname{stopgrad}\!\left(z_{\ell,p}\right)
}
$$

含义是：**从"处理 Alice 问题"的隐藏状态里，拿掉少量经筛选的分量，其余尽量保留。** $\mathcal S$ 可以为空；为空就输出诊断停止，不启动训练。$\alpha$ 历史默认 $0.1$（只削弱所选分量的 10%），但正式实验前要重新冻结。

Student 接收的仍是**原问题** $(X^+,y^+)$，优化激活回归：

$$
\mathcal L_F=
\frac1{|\mathcal S|}
\sum_{(\ell,p)\in\mathcal S}
\left\|h_{S,\ell,p}-T_{\ell,p}\right\|_2^2 .
$$

把 MSE 理解成"让 Student 的内部状态靠近 Teacher 给出的编辑后状态"。`stopgrad` 表示目标是常量，梯度只更新 Student。

**这不是把激活永久写死**：训练时通过激活回归调整 Student 的 LoRA 参数；训练后面对原问题时，由新参数自然产生接近目标的中间状态。

### 7.1 训练前先看梯度占比，而不是只看 loss 数值

历史报告里 method loss 很小，不能仅凭数值断言 retain 梯度主导。v1.1 要求先记录各损失对可训练参数的实际梯度范数：

$$
G_m=\nabla_\theta(\lambda_m\mathcal L_m),
\qquad
\eta_m=\frac{\|G_m\|_2}{\sum_j\|G_j\|_2+\epsilon}.
$$

$\eta_m$ 只是梯度量级诊断，不是有效性证明。再比较固定层集合、相同 batch 顺序和 seed 下的 method-only / retain-only / full 三臂。

## 8. 护栏：如何避免"假遗忘"

只靠 $\mathcal L_F$，模型可能作弊：把所有回答都变差、偏向 donor、或很早就输出 EOS。核心训练式保留为：

$$
\boxed{
\mathcal L_{\mathrm{core}}=\lambda_F\mathcal L_F+\beta\mathcal L_R
}
$$

各组件角色如下（**只有前两行产生遗忘证据，其余都是边界/护栏**）：

| 组件 | 角色 | 是不是核心遗忘证据 |
|---|---|---|
| BRCD/QK-V 候选 + true/null 特异性 | 产生并筛选遗忘方向 | 是 |
| Student 无 hook 内化 | 把方向写进参数 | 是 |
| retain NLL $\mathcal L_R$ | 保护通用/非目标能力 | 核心效用约束，但不产生遗忘 |
| 同实体非目标关系 | 检查选择性，避免整实体能力被抹掉 | 核心选择性约束 |
| donor 支持不升 | 排除答案替换 | 中性边界，不是遗忘方向 |
| EOS / 拒答 / 长度 / 完整回答率 | 排除输出抑制捷径 | 反捷径护栏 |

几个容易踩的坑：

- **原事实约束** $r_f\le c_f$ 中的 $c_f$ 必须用独立、预注册的中性参考来定；不能用待评估的 BRCD 目标 $z$ 自己的最大值加 margin 来定，否则等于"用当前候选自己证明自己满足"。没有这样的参照就标记 `c_f` 未校准并关闭投影。
- **相对 EOS 护栏**只惩罚相对 Teacher 的恶化：

$$
\mathcal L_E=\big[(m_E^S-m_E^T)-\epsilon_E\big]_+^2 .
$$

历史报告里 EOS rate 全为 1.0，**不能被解释为"没有生成风险"**。
- **可选 POCS 投影**只是"候选目标真的违反约束时"的辅助步骤。它在 $z$ 处把约束线性化，半空间投影为：

$$
P_i(h)=h-\frac{\big[\phi_i(z)+a_i^\top(h-z)\big]_+}{\|a_i\|_2^2+\epsilon}\,a_i .
$$

普通 POCS 只给局部代理的可行点，不保证离 $z$ 最近（要最近点语义需另加 Dykstra 校正）。历史 round3 的 62/62 全是 `identity`（没有发生实际投影修正），**不构成 POCS 有效性证据**；必须单独构造"真实约束被违反"的测试，看到 `projection_active=true`、`dist_from_z>0` 才算数。若投影修正量持续接近或超过原始 BRCD 编辑量，主要效果可能来自通用读出约束，不能包装成双线性机制成功。

## 9. 输出：部署时只有 Student

训练和筛选阶段用到了 Teacher、反事实输入、缓存和 patch-forward；**部署时这些全部不需要**。用户输入原问题，Student 正常执行一次前向即可，没有额外运行时 hook，也不用再构造 Bob 的反事实问题。

这就是"参数内化"的含义：**训练阶段借用激活目标定位方向，部署阶段由参数变化承担效果。**

完整一次运行的数据流（对应实现）：

```text
同一预训练 checkpoint
   ├── 冻结 Teacher：两次前向 (X⁺,y⁺) 与 (X⁻,y⁺)
   │      ├─ 缓存 π、V、答案隐藏状态；噪声体检 ε₀ / 精度对照
   │      ├─ 双线性分解 → route/content（GLU 可选，默认不开）
   │      ├─ true/null/random/等范数ΔA/matched-retain patch
   │      ├─ 原生 QK/V 2×2 干预 + 簇 bootstrap + held-out
   │      └─ 稀疏候选 selected_candidates.json（可为空 → 停）
   └── Student（LoRA）：只输入原问题与原答案
          ├─ 激活 MSE → 靠近 T = A⁺ − αC*
          ├─ fact hinge / donor guard / 相对 EOS guard
          ├─ retain NLL 锚
          ├─ 梯度占比诊断 η（可选）
          └─ 反向传播，仅更新 Student 参数
                     ▼
              部署：只留 Student，单次前向
                     ▼
              验收：probe / relearning / NLL / 生成行为
```

对应核心代码（示意，非另一套算法）：

```python
# 阶段 A：诊断，不训练
audit_measurement(cache)              # 噪声下限；不过 → MEASUREMENT_UNRELIABLE
route, content = exact_bilinear(pi_plus, pi_minus, v_plus, v_minus)
selected = scan_all_layers_and_controls(cache)   # 稀疏、留出确认；可为空
if not selected:
    stop("NO_STABLE_CANDIDATE")       # 不进入训练

# 阶段 B：条件式内化
target = stop_gradient(h_plus - alpha * selected_c)
method_loss = mse(student_hidden, target)
loss = lambda_f * method_loss + lambda_n * neutral + beta * retain_nll
loss.backward()                        # 只更新 Student
```

## 10. 怎么判断"真遗忘"而不是"表面不答"

| 观察 | 希望看到 | 能证明什么 / 不能证明什么 |
|---|---|---|
| forget NLL | 相对 Teacher 上升 | 原答案更难读出；**不能证明隐藏信息消失** |
| retain / holdout NLL | 尽量稳定 | 其他知识与泛化的损害较小 |
| donor NLL / 答案 | 不因训练而更容易 | 降低"把忘记 Alice 偷换成变成 Bob" |
| 生成行为 | 目标词减少，且不过度拒答/提前 EOS/乱码 | 排除部分表面失败模式 |
| 目标 probe | 隐藏态可分性下降 | 辅助观察表示变化；不是擦除的充分证据 |
| relearning | 短程微调后恢复速度与幅度 | 检查是否只是容易恢复的行为抑制 |

⚠️ **模型不回答、说"不知道"、输出另一个答案、生成崩溃，都可能让原答案出现率下降，但它们不是同等意义的遗忘。** `COMPLETED` 只表示运行完成，不表示遗忘有效。

## 11. 当前真实状态：为什么这份文档强调"先测量"

历史提交 `fe5e32f` 的 round3 诊断（真实 Llama-3.2-1B / 31 个有效配对 / H20）：

| 观测 | 记录 | 说明 |
|---|---|---|
| 层范围 | 只有 $[8,10,12,14]$ | 只覆盖后半，不能代表全 16 层 |
| 四候选门 | 16 门仅 `8:attn_route`、`12:glu_value` 通过 | 覆盖不足 |
| 主链 | 第 10/14 层无唯一候选 → `DIAGNOSTIC_KILL`，0 步 | 没有训练后的遗忘结果 |
| 效应量 | 多数约 $10^{-3}$，跨层/候选变号 | 需要噪声下限，不能直接解释成机制 |
| 五个对照臂 | 60 步，dF 全为负（约 $-0.0669\sim-0.0682$） | forget NLL 朝遗忘的反方向走 |
| 投影 | 62/62 为 `identity` | 本轮没有发生实际投影修正 |

这些结果支持"当前版本未通过机制证据门"，但**既不足以证明机制被彻底证伪，也不足以排除测量、门规则或对照设计的问题**。这正是 v1.1 选择"先修可判别性、再谈训练"的原因。三阶段协议（A 诊断 → B 条件内化 → C 全量与投稿证据）保证：**没有阶段 A 的留出证据，就不进入 full 训练，也不把工程护栏当成方法贡献。**

## 12. 五句话复述

1. 用原问题与"只换主体、答案相同"的反事实问题，测模型内部哪些计算随主体改变。
2. 利用注意力是"权重 $\pi$ × 内容 $v$"的结构，把变化精确拆成路由项与内容项（代数恒等式，不丢信息）。
3. **先做噪声体检、全层诊断、五类对照、QK/V 四组合和簇留出验证**；只有稳定超过噪声和对照的方向才入选，否则停线。
4. 让 Student 在原问题上的中间状态靠近"原状态减去少量该方向"，同时用 retain、donor、EOS、fact 护栏减少副作用。
5. 训练后只部署 Student；是否真遗忘，必须靠保留能力、独立 probe、relearning 和生成行为来判断。

## 13. 代码阅读索引

| 想看的部分 | 实现位置 |
|---|---|
| 正反事实对与 donor 来源 | `data.py::build_pairs` |
| 注意力/Value/答案跨度捕获 | `modeling.py::forward_model` |
| 精确双线性分解与 `o_proj` 写回 | `method.py::_raw_components` / `_components` |
| 噪声下限、零编辑 $\epsilon_0$ | `measurement_protocol.py::audit_measurement` |
| 全层扫描、等条件对照、簇 bootstrap | `measurement_protocol.py::scan_all_layers` / `equal_condition_controls` / `cluster_bootstrap` |
| 原生 QK/V 四组合 | `measurement_protocol.py::qk_v_factorial` + `modeling.py::forward_with_attention_patch` |
| 稀疏候选选择与条件内化 | `method.py::prepare`（`sparse_selection_file` 分支）/ `method.py::loss` |
| 梯度占比诊断 | `runtime.py::train`（`grad_diagnostics`） |
| Stage A 入口 | `run_stage_a.py` |
| 默认配置 | `config.json` |

---

> 📝 **一句话总结全文**：CF3-01 的核心不是"又发明了一个删除函数"，而是把"注意力权重 × 内容"这个已知结构变成一个**可精确拆开、可对照验证、可条件内化**的遗忘方向管线；它的成败不由代数恒等式保证，而由阶段 A 能否证明"这个方向真的提供了随机方向替代不了的信号"来决定。本文只讲机制与数据流，不宣称效果；所有论文级 claim 仍待验证。

---
title: CF3-01 双线性路由—内容差分
subtitle: 从注意力公式到机器遗忘：一份面向深度学习初学者的实现导览
eyebrow: BRCD · MECHANISM GUIDE
date: 2026-11-27
scope: 解释当前 attention 实现，不将代码验收等同于遗忘有效性
---

# CF3-01 双线性路由—内容差分

> **先用一句话理解**：模型对“换人名”这件事的反应，混有两种变化——**注意力的去向变了**，以及**被读取的内容变了**。CF3-01 先把这两种变化精确拆开，测试哪一个确实与目标答案的读出有关，再训练 Student 在原问题上只消弱这一小部分内部信号，同时尽量保住其他知识。

这里的“遗忘”不是从硬盘中删除训练样本，也不是把某个神经元永久清零，而是：**通过参数微调，让模型面对原问题时，不再那么容易沿原来的内部通路读出目标事实，并尽量不影响其他能力。**

## 1. 从一个具体例子开始

假设要遗忘的事实是：

| 角色 | 问题 | 输入模型时使用的答案 |
|---|---|---|
| 原事实 $X^+$ | Where was **Alice Miller** born? | Alice Miller was born in **Paris**. |
| 反事实 $X^-$ | Where was **Bob Smith** born? | 仍然是 Alice Miller was born in **Paris**. |

第二个问题不是模型原本会自然回答 Paris 的问题。它只是一个**受控对照**：问题中只替换人名，其他词、答案和 token 位置尽量保持一致。

| 符号 | 含义 | 在教学例子中的角色 |
|---|---|---|
| $X^+$ | 原问题 | Alice 的出生地问题 |
| $X^-$ | 只换主体的反事实问题 | Bob 的出生地问题 |
| $y^+$ | 原数据集答案 | Paris 所在的原始答案 |
| $y^-$ | donor 自己的答案 | 例如 Bob 的答案；只用于评估，不作训练目标 |

两次 Teacher 前向分别处理 $(X^+,y^+)$ 和 $(X^-,y^+)$。**答案相同非常重要**：若连答案也换掉，观察到的差异就混入“问题变了”和“答案变了”两种因素，难以定位。

> **最容易误解的一点**：$X^-$ 是用来测量“哪些内部计算随主体改变”的对照，不是让 Student 学会回答 Bob 的事实。

## 2. 为什么不能直接使用全部激活差？

Transformer 的每一层会输出一组隐藏向量。对同一个答案位置 $p$，可以计算：

$$
\Delta A_{\ell,p}=A^+_{\ell,p}-A^-_{\ell,p}.
$$

直觉上，它表示“把 Alice 换成 Bob 后，第 $\ell$ 层的状态改变了多少”。但这个差值可能同时包含：

| 变化来源 | 直观含义 | 是否必然是目标事实？ |
|---|---|---|
| 注意力路由 | 模型更关注哪些 token | 否，可能只是名字 token 不同 |
| Value 内容 | 从被关注的 token 读出了什么 | 否，可能只是一般身份信息 |
| 无关上下文变化 | 句法、词形、位置等差异 | 否 |
| 目标事实相关计算 | 与 Paris 这一目标读出有关的信号 | 有可能，需要验证 |

因此，**$\Delta A$ 很大，不代表其中全是需要遗忘的事实**。直接让整个向量归零或全部反向外推，可能把正常语言能力和 retain 知识一起破坏。

CF3-01 的第一步不是训练，而是把这个混合差值拆得更清楚。

## 3. 先回顾注意力：路由与内容在哪里？

在答案 token 位置 $p$，一个注意力头的输出可以写成：

$$
O_{p}=\sum_q \pi_{p,q}v_q.
$$

其中：

- $v_q$ 是第 $q$ 个输入位置的 **Value 向量**，可以理解为该位置提供的内容；
- $\pi_{p,q}$ 是注意力权重，由 Query 和 Key 计算后再经 softmax 得到，决定当前位置从 $q$ 读取多少内容；
- 所有 $q$ 的加权 Value 相加，就得到当前位置的注意力输出。

可以把它想成查资料：**路由 $\pi$ 决定看哪几页、每页看多少；内容 $v$ 是每页实际写着什么。** 这个比喻只帮助理解，真正的计算仍然是矩阵与向量运算。

CF3-01 使用的就是已经过 softmax 的 $\pi$，不是单独修改 Q 或 K。对于 Llama 的 GQA，代码还会把共享的 KV Value 头展开到对应的查询头，保证 $\pi$ 与 $v$ 能正确相乘。

## 4. 核心恒等式：把差值精确拆成两项

先看最简单的逐元素乘法 $B(a,b)=a\odot b$。原输入与反事实输入分别产生 $(a^+,b^+)$ 和 $(a^-,b^-)$，则：

$$
\boxed{
a^+\odot b^+-a^-\odot b^-
=
\underbrace{\tfrac12(a^+-a^-)\odot(b^++b^-)}_{\text{路由项}}
+
\underbrace{\tfrac12(a^++a^-)\odot(b^+-b^-)}_{\text{内容项}}
}
$$

右侧展开后正好等于左侧，**这是代数恒等式，不是一阶近似**。

一个标量小例子：

| 量 | 原输入 | 反事实输入 |
|---|---:|---:|
| 路由 $a$ | $0.8$ | $0.2$ |
| 内容 $b$ | $10$ | $6$ |
| 乘积 $ab$ | $8$ | $1.2$ |

总差为 $8-1.2=6.8$，而：

$$
D_{\mathrm{route}}=\tfrac12(0.8-0.2)(10+6)=4.8,
$$

$$
D_{\mathrm{content}}=\tfrac12(0.8+0.2)(10-6)=2.0.
$$

所以 $4.8+2.0=6.8$。原差值没有被丢失，只是分成了“路由变化”和“内容变化”两部分。

## 5. 放到注意力公式中

将 $a$ 换成注意力权重 $\pi$，将 $b$ 换成 Value 向量 $v$，并对所有被读取的位置 $q$ 求和：

$$
D^{\mathrm{attn}}_{\mathrm{route}}
=\frac12\sum_q(\pi^+_q-\pi^-_q)(v^+_q+v^-_q),
$$

$$
D^{\mathrm{attn}}_{\mathrm{content}}
=\frac12\sum_q(\pi^+_q+\pi^-_q)(v^+_q-v^-_q).
$$

两项相加，恰好恢复原输入与反事实输入的**注意力输出差**：

$$
D^{\mathrm{attn}}_{\mathrm{route}}+
D^{\mathrm{attn}}_{\mathrm{content}}
=O^+-O^-.
$$

代码随后将各注意力头拼回隐藏维度，并通过该层的 `o_proj` 映回残差流空间。这样得到的分量才能与该层的隐藏状态做加减。

> **边界**：恒等式保证“加回去等于原注意力输出差”，不保证某一项天然只包含事实语义。路由项可能仍含内容信息，内容项也可能含无关变化，所以还需要干预筛选。

## 6. 关键筛选：哪一项才值得消弱？

CF3-01 不预先认定“内容项一定是事实、路由项一定无关”。它在冻结的 Teacher 上做一次受控干预：

1. 取原问题前向时第 $\ell$ 层的完整隐藏状态 $H^+$；
2. 只在目标答案跨度中减去少量待测分量：$H^+-\alpha C$；
3. 保持冻结模型的后续层不变，重新计算原答案 $y^+$ 的 NLL；
4. 比较干预前后的 NLL 变化。

NLL 是模型给正确答案分配的负对数概率。**NLL 越高，模型越不容易给出这个答案。**

当前实现分别检查两个条件：

$$
\mathbb E[\mathrm{NLL}_{\mathrm{patch}}-\mathrm{NLL}_{\mathrm{before}}]
>\eta_f,
$$

$$
\mathbb E[\text{真实配对效应}-\text{wrong/null 配对效应}]
>\eta_s.
$$

第一条要求：消弱这个分量确实让原答案更难读出。第二条要求：真实配对分量的效应要强于拿其他配对分量乱 patch 的效应。

| route 门 | content 门 | 决策 |
|---|---|---|
| 通过 | 不通过 | 选择 route |
| 不通过 | 通过 | 选择 content |
| 都通过 | 都通过 | 无法区分，当前层不选择 |
| 都不通过 | 都不通过 | 当前层无可靠目标 |

在严格 `pilot/full` 模式中，只要要求处理的层没有通过唯一分量筛选，就记录 `DIAGNOSTIC_KILL` 并停止训练。**这是防止把不可靠机制硬训练下去的刹车，不是实验失败后被隐藏的结果。**

`tiny/smoke` 会为代码联调放宽此门并回退到配置的 content 分量；因此它们只能验证程序能跑，不能证明机制成立。

## 7. 如何真正更新模型：Teacher 指目标，Student 学过去

筛选完成后，Teacher 仍不更新。对选中的层、答案位置和分量 $C^*$，构造一个固定的激活目标：

$$
\boxed{T_{\ell,p}=\operatorname{stopgrad}(A^+_{\ell,p}-\alpha C^*_{\ell,p})}
$$

它的含义是：**从原本处理 Alice 问题的隐藏状态中，拿掉少量经筛选的分量，其余部分尽量保留。**

Student 接收的仍然是原问题 $(X^+,y^+)$，并优化：

$$
\mathcal L_{\mathrm{BRCD}}
=\sum_{\ell,p}\left\|h_{S,\ell,p}-T_{\ell,p}\right\|_2^2.
$$

可以把 MSE 理解成“让 Student 的内部状态靠近 Teacher 给出的编辑后状态”。`stopgrad` 表示目标固定，梯度只更新 Student。

默认 $\alpha=0.1$，即只消弱所选分量的 10%。它不是理论最优值：太小可能没有行为效果，太大可能伤害其他知识，必须通过实验选择。

**这一步不是把激活永久写死。** 训练时通过激活回归调整 Student 的 LoRA 参数；训练后，面对原问题时由新参数自然产生接近目标的中间状态。

## 8. 仅靠 MSE 为什么还不够？

如果只看 $\mathcal L_{\mathrm{BRCD}}$，模型可能通过一些“作弊方式”降低损失，例如把所有回答都变糟、直接偏向 donor，或很早就输出 EOS。实际训练还加入共同约束：

### 8.1 原事实读出约束

$$
\mathcal L_{\mathrm{fact}}
=\operatorname{ReLU}\!\left(
\mathrm{NLL}_S(y^+\mid X^+)-
[\mathrm{NLL}_T(y^+\mid X^+)+\delta]
\right)^2.
$$

默认 $\delta=0.35$。只有当 Student 对原答案的 NLL 没有足够上升时，这项才产生惩罚。它是对读出行为的最低要求，不是“遗忘已完成”的证书。

### 8.2 保留集约束

在 retain 问答上继续优化正常答案 NLL：

$$
\mathcal L_R=\mathrm{NLL}_S(y_R\mid X_R).
$$

它像一根锚，防止为了遗忘 Alice 而把整体语言能力或其他人的知识一起破坏。

### 8.3 不向 donor 靠近

代码计算 Student 状态相对原状态的移动方向，并惩罚它沿 $A^--A^+$ 的 donor 方向移动。也就是说，**可以削弱目标分量，但不能把“忘记 Alice”偷换成“变成 Bob”。**

donor 自己的答案 $y^-$ 不进入训练损失；`donor_nll` 只用于训练后评估。

### 8.4 提前结束防护

EOS guard 限制答案跨度中的 EOS logit 无约束地压过其他 token，降低模型通过“立刻结束回答”来装作遗忘的风险。

默认总损失的结构为：

$$
\mathcal L
=0.1\,\mathcal L_{\mathrm{BRCD}}
+0.1\,\mathcal L_{\mathrm{neutral}}
+2.0\,\mathcal L_R,
$$

其中 $\mathcal L_{\mathrm{neutral}}$ 内部还包含 fact、donor 和 EOS 项及各自权重。具体数值以 `config.json` 为准。

## 9. 一次训练的完整流程

```text
同一预训练 checkpoint
        │
        ├── 冻结 Teacher：两次前向 (X⁺, y⁺) 与 (X⁻, y⁺)
        │                  │
        │                  ├── 缓存注意力 π、Value、答案隐藏状态
        │                  ├── 精确拆成 route / content
        │                  ├── true/null patch-forward 筛选
        │                  └── 固定目标 T = A⁺ − αC*
        │
        └── 可训练 Student：只输入原问题与原答案
                           │
                           ├── 激活 MSE：靠近 T
                           ├── 原答案 NLL hinge：降低原事实可读性
                           ├── retain NLL：保护其他知识
                           ├── donor guard：避免替换成另一主体
                           └── EOS guard：避免一句话都不说就结束
                                      │
                                      ▼
                              反向传播，仅更新 Student
```

默认关注第 8、10、12、14 个 block 输出位置，Teacher 提供缓存和固定目标，Student 使用 LoRA 更新 `q/k/v/o` 与 `gate/up/down` 等投影。实际配置可通过命令行覆盖。

下面的伪代码对应核心实现，而不是另一套算法：

```python
# Teacher 已冻结；两次前向使用同一个原答案 y_plus
route, content = exact_bilinear_difference(pi_plus, pi_minus, v_plus, v_minus)
selected = unique_component_that_passes_true_and_null_patch_gates()

# 严格档无法唯一选择时停止，而不是硬训练
target = stop_gradient(h_plus - alpha * selected)
method_loss = mse(student_answer_hidden, target)
loss = 0.1 * method_loss + 0.1 * neutral_constraints + 2.0 * retain_nll
loss.backward()  # 只更新 Student 的可训练参数
```

## 10. 部署时还需要 Teacher 和两次前向吗？

**不需要。** Teacher、反事实输入、激活缓存和 patch-forward 都用于训练前的定位与训练时的目标构造。

部署时只保留训练后的 Student。用户输入原问题，Student 正常执行一次前向；没有额外运行时 hook，也不需要再构造 Bob 的反事实问题。

这也是“参数内化”的含义：**训练阶段借用激活目标，部署阶段由参数变化承担效果。**

## 11. 怎么判断它是真遗忘，而不是表面不答？

| 观察 | 希望看到什么 | 能证明什么、不能证明什么 |
|---|---|---|
| forget NLL | 相对 Teacher 上升 | 原答案更难直接读出；不能证明隐藏信息消失 |
| retain/holdout NLL | 尽量稳定 | 其他知识与泛化损害较小 |
| donor NLL / donor 答案 | 不因遗忘训练而更容易 | 降低“换成 Bob 答案”的可能 |
| 生成行为 | 原目标词减少，同时不过度拒答、提前 EOS 或乱码 | 排除部分表面失败模式 |
| 目标作者 probe | 隐藏态中的目标可分性下降 | 辅助观察表示变化；不是事实擦除的充分证据 |
| relearning | 用原答案短程微调后观察恢复速度与幅度 | 检查是否只是容易恢复的行为抑制 |

尤其要警惕：**模型不回答、回答“不知道”、输出另一个答案、生成崩溃，都可能让原答案出现率下降，但它们不是同等意义的遗忘。**

本方法当前的验证范围是：双线性恒等式自测通过，Tiny 合成模型训练链路通过。真实 Llama/TOFU 上的效果仍需 GPU smoke、严格诊断和 pilot 结果支持；不能把代码运行成功写成遗忘有效。

## 12. 当前实现与原方案的差别

当前代码实现的是 **attention 的 $\pi$–Value 双线性差分**。原 proposal 还讨论了 SwiGLU 的 `Swish(g) × u` 双线性分支，但该分支没有混入 CF3-01 当前版本。

原方案用 $\Pi_{\mathcal K}$ 表示向非 donor 可行集投影。当前代码使用的是 fact hinge、donor 方向 guard、EOS guard 和 retain NLL 等软约束组合，**没有实现一个精确凸投影算子**。

因此，更准确的表述是：

> CF3-01 是一种“反事实注意力分量定位 + 受约束的 Student 激活内化”方法。它有精确的代数分解和可证伪的筛选门，但事实语义、删除深度与保留效果都必须由独立实验验证。

## 13. 最后用五句话复述

1. 用原问题与只换主体的反事实问题，测出模型内部哪些计算发生变化。
2. 利用注意力是“权重 × 内容”的结构，把变化精确拆成路由项和内容项。
3. 用冻结 Teacher 的 patch 实验，选择确实影响原答案且优于错误配对的那一个分量。
4. 让 Student 在原问题上的中间状态接近“原状态减去少量该分量”，同时用 retain、donor 和 EOS 约束减少副作用。
5. 训练后只部署 Student；是否真正遗忘，还必须通过保留能力、独立 probe 和 relearning 等证据判断。

## 14. 代码阅读索引

| 想看的部分 | 当前实现位置 |
|---|---|
| 正反事实对与 donor 来源 | `data.py` 的 `build_pairs` |
| 注意力、Value 和答案跨度捕获 | `modeling.py` 的 `forward_model` |
| 精确双线性分解与 `o_proj` 映射 | `method.py` 的 `_raw_components`、`_components` |
| true/null patch 与唯一分量门 | `method.py` 的 `_patch_effect`、`prepare` |
| Student 激活回归目标 | `method.py` 的 `loss` |
| fact、donor、EOS、retain 联合训练 | `runtime.py` 的 `neutral_terms`、`train` |
| probe、relearning 与结果保存 | `runtime.py` 的 `target_fact_probe`、`relearning_recovery` |
| 默认层、$\alpha$、损失权重和训练预算 | `config.json` |

本文解释依据为当前目录的 `proposal.md`、`method.py`、`modeling.py`、`data.py`、`runtime.py`、`config.json`，以及上级 `VALIDATION_REPORT.md`。原始 proposal 属于待验证研究方案；本文不额外宣称文献首创性或真实实验效果。

# Derivation Package

## Target

为 CF3-01 设计两个尚未实现的组件：

1. attention 之外的 SwiGLU `SiLU(g) × u` 精确双线性差分、残差写回与分量筛选。
2. proposal 中 `T = stopgrad[Π_K(A+ − αC*)]` 的可计算代理投影。

目标是得到可证伪、可实现、与当前 Teacher 缓存 / patch / Student MSE 接口兼容的设计，不修改当前默认算法，也不把代理投影声称为精确非线性投影。

## Status

COHERENT AFTER REFRAMING / EXTRA ASSUMPTION

SwiGLU 分支可以作为精确代数恒等式实现。原始 `Π_K` 的事实、donor 和 EOS/长度读出由冻结非线性下游产生，通常不是凸约束，不能直接套用精确 POCS；必须改写为局部仿射约束、信任域和真实读出回验组成的代理投影。

## Invariant Object

保持不变的核心对象是 Student 的固定激活目标：

`z = A+ − αC*` 提供 BRCD 候选编辑；可选代理投影只在与 z 相近的局部区域内修正 z，形成固定目标 T。Student 仍仅通过 `MSE(hS, stopgrad(T))` 内化，梯度不经过 Teacher 或投影过程。

## Assumptions

- Teacher 与 Student 从同一 checkpoint 初始化，Teacher 权重冻结；对隐藏状态求导不等于对 Teacher 权重求导。
- 两次 Teacher 前向仍分别为 `(X+, y+)` 与 `(X-, y+)`，答案保持一致。
- 当前实现默认 attention-only；新增分支必须由显式配置开启，并可独立消融。
- Llama 的 `Swish` 在 β=1 时与 `SiLU` 相同；实际计算为 `m = SiLU(g) ⊙ u`、`y = m W_down^T`。
- 分量名称描述计算角色，不预设事实语义；每个分量仍须经过 true/null patch。
- 局部仿射模型只在信任域内使用；真实约束是否满足由冻结下游重新计算，而非由代理模型自行宣布。
- donor 答案若用于 `s_dj`，只能作为单边“不支持 donor”约束的测量对象，不能作为 Student 的拟合目标或 soft label。

## Notation

- `g,u`：SwiGLU 的 gate/up 预激活，形状为答案跨度 × FFN 维度。
- `a = SiLU(g)`：实际参与逐元素乘法的门控激活，不是原始 g，也不是概率。
- `m = a ⊙ u`：SwiGLU 在 down projection 前的神经元输出。
- `W_down`：当前 Llama 的 `mlp.down_proj.weight`；行向量实现为 `m @ W_down.T`。
- `r_f(h)`：目标答案的长度归一化 span log-likelihood，越大表示越支持原事实。
- `s_dj(h)`：原问题下第 j 个 donor 答案的读出支持度，越大表示越支持 donor。
- `E(h)`：可微 EOS/提前结束代理相对参考状态的漂移；真实生成长度另行验收。
- `φ_i(h) ≤ 0`：统一写法下第 i 个约束。
- `z`：当前局部线性化中心；首轮为 `A+ − αC*`。
- `ρ`：局部信任域半径，限制代理投影离开线性化中心的距离。

## Derivation Strategy

1. 对 SwiGLU 先固定实际双线性对象 `(SiLU(g),u)`，使用与 attention 相同的对称恒等式，再经 down projection 映回残差流。
2. 将 gate/value 与 attention route/content 作为四个独立候选进行诊断，不默认求和。
3. 对 `r_f ≤ c`、`s_dj ≤ d_j` 和 EOS 漂移约束构造局部仿射半空间；加入球形信任域后执行交替投影。
4. 用真实冻结下游检查所有约束和投影后特异性；不满足则缩小信任域、重新线性化或 fail closed。
5. 只有通过验收的目标才 stop-gradient 并进入 Student MSE。

## Derivation Map

1. SwiGLU 输出差依赖 `SiLU(g+)`、`SiLU(g-)`、`u+`、`u-`。
2. 对称双线性恒等式给出 gate/value 两项；down projection 的线性性保持精确重构。
3. 原始 K 的读出约束是非线性的，不能直接得到闭式欧氏投影。
4. 一阶展开把每个标量约束变成半空间；EOS 向量采用分量死区以避免把非线性范数约束误当凸球。
5. POCS 给出局部代理可行点；Dykstra 修正可用于更接近“离 z 最近”的语义。
6. 真实读出回验决定是否接受目标，代理可行不等于原问题可行。

## Main Derivation

### 1. SwiGLU 双线性分支

定义：

```
a+ = SiLU(g+),  a- = SiLU(g-)
m+ = a+ ⊙ u+,   m- = a- ⊙ u-
```

精确分解为：

```
D_gate  = 1/2 (a+ − a-) ⊙ (u+ + u-)
D_value = 1/2 (a+ + a-) ⊙ (u+ − u-)
```

因此：

```
D_gate + D_value = m+ − m-
```

经 down projection 写回：

```
C_gate  = D_gate  @ W_down.T
C_value = D_value @ W_down.T
```

由线性性：

```
C_gate + C_value = (m+ − m-) @ W_down.T
```

解释边界：`D_gate` 表示实际 SiLU 门控振幅变化在平均 up 内容上的作用，`D_value` 表示 up 内容变化在平均门控振幅上的作用。`SiLU(g)` 可负、可大于 1，不应解释为“门打开概率”。若要研究纯 sigmoid 门比例，可另设 `σ(g) × (g⊙u)` 参数化，但那是另一实验臂。

首版 gate 规则：

- attention 内部仍按 route/content 唯一性判断。
- SwiGLU 内部按 gate/value 唯一性判断。
- 只有一个 family 给出唯一候选时才选择；两个 family 都给出候选则记为 cross-family ambiguity。
- 不默认把 attention 与 GLU 分量相加；组合臂须另测二阶交互与等范数对照。

首版沿用 block-output patch，便于复用当前接口。GLU 分量经 down projection 后从该层输出隐藏状态减去；这与真实 MLP 写回位置一致。attention 若改为 post-attention 原生写点，应作为独立消融，不与 block-output 的效应值直接混比。

建议伪代码：

```python
ap = F.silu(gp.float())
am = F.silu(gm.float())
d_gate = 0.5 * (ap - am) * (up + um)
d_value = 0.5 * (ap + am) * (up - um)
c_gate = F.linear(d_gate.to(weight.dtype), down_weight).float()
c_value = F.linear(d_value.to(weight.dtype), down_weight).float()
```

单元测试必须检查 `c_gate + c_value` 与真实 SwiGLU 输出差的相对 Frobenius 误差；不得对 D_gate/D_value 做 ReLU 或符号截断，否则破坏恒等式。大规模实验宜在线计算 float32 分量后释放原始 g/u，避免长期保存半精度大缓存。

### 2. 从原始 K 到可计算约束

使用 log-likelihood 口径：

```
r_f(h)  = −NLL_downstream(h; X+, y+)
s_dj(h) = −NLL_downstream(h; X+, yj−)
```

于是 `r_f(h) ≤ c` 等价于 `NLL ≥ −c`，表示原事实支持足够低。它与当前 runtime 中限制 forget NLL 不要过高的 `L_fact` 不是同一约束，不能直接复用该损失作为投影定义。

donor 约束只限制支持度上限。若还要求相对候选点不上升，可采用：

```
s_dj(h) ≤ min(d_j, s_dj(z) + εj)
```

这会把“绝对中性上界”和“相对不增”同时纳入；donor 答案仍不进入 MSE、KL 或任何正目标。

真实生成长度不可微，首版将 E 改为可微代理，例如答案跨度上的 EOS logit margin 漂移与 EOS 概率漂移，并使用分量死区：

```
|E_k(h)| ≤ e_k,  k = 1,2
```

真实 EOS rate、完整回答率与生成长度在投影后另行检查。若坚持原 Euclidean 范数 `||E|| ≤ e`，应使用二阶锥/QP 求解器，不能假装它只是普通半空间 POCS。

### 3. 局部仿射半空间

把每个约束写成 `φ_i(h) ≤ 0`。在 z 处计算值和关于隐藏状态的梯度：

```
a_i = ∇_h φ_i(z)
φhat_i(h) = φ_i(z) + a_i^T(h − z)
```

代理凸集：

```
C_i(z) = {h : φ_i(z) + a_i^T(h − z) ≤ 0}
Bρ(z)  = {h : ||h − z||₂ ≤ ρ}
```

这里 Teacher 参数保持 `requires_grad=False`，但必须允许对输入隐藏状态 h 求导；现有 no-grad NLL patch 不能直接用于内层 Jacobian。

半空间投影为：

```
P_i(h) = h − [φ_i(z) + a_i^T(h−z)]_+ / (||a_i||² + ε) · a_i
```

若梯度近似为零且约束仍被违反，该局部半空间无法提供修正方向，应判为不可行，而非除以极小值产生巨大跳步。

信任域投影为：

```
P_B(h) = z + ρ(h−z) / ||h−z||₂   当 ||h−z|| > ρ
         h                       否则
```

建议 `ρ = min(ρmax, κ α ||C*||)`，κ 在 discovery 集冻结。这样可诊断投影修正是否压过原始 BRCD 编辑；若持续需要很大 κ 才可行，应视为 BRCD 候选目标不适合该约束系统。

### 4. 交替投影与最近点语义

```text
输入 z = A+ − αC*
若真实约束已满足：T = z，记录 projection_active=false
否则：
  在 z 计算所有 φ_i 和隐藏状态 Jacobian
  h = z
  重复若干 sweep：
      依次投影到 fact、donor、EOS 正负半空间和信任球
  用真实冻结下游重新计算全部约束
  若代理误差过大或真值违约：缩小 ρ / 在当前 h 重新线性化
  超过外层次数仍失败：PROJECTION_INFEASIBLE，严格档停止
T = stopgrad(h)
```

普通 POCS 在凸集交非空时可收敛到某个可行点，但不保证是离 z 最近的点。若要保持 `Π` 的“最近点”含义，应给每个集合加入 Dykstra 校正项；否则文档只能称“POCS 可行点代理”。

现有 donor 方向 guard 可额外保留为精确半空间 `u^T(h−A+) ≤ 0`，无需线性化，但它是新增防护，不应冒充 proposal 原始 `s_dj` 约束。

### 5. 接受投影目标的硬检查

- 所有真实 `r_f/s_dj/E` 约束在容差内，而非仅仿射约束满足。
- `||T−z|| ≤ ρ`，并记录该距离与 `α||C*||` 的比值。
- 投影后重新执行 true/null patch；不能沿用投影前特异性结论。
- donor 支持不升、EOS/长度/完整回答率不恶化，retain 与同实体非目标关系仍在死区。
- 严格档不可行时输出明确状态，不静默退回无约束 z。
- 最终目标 detach；Student 训练只计算 `||hS−T||²`，不反向穿过 POCS。

## Remarks and Interpretation

- SwiGLU 恒等式是精确代数，不是遗忘有效性的证据。gate/value 可能都无效或不可区分。
- POCS 是候选目标的约束修正器，不是主要遗忘方向来源。若 z 已在 K 中，投影应为恒等操作。
- 若投影修正远大于 BRCD 原始编辑，主要效果可能来自通用读出约束，而非双线性分量，方法贡献应判为未成立。
- 首版应采用慢而语义清楚的参考实现：少量 gate pairs、少量层/位置、真实下游 Jacobian。规模化时再考虑固定低秩仿射读出，但必须另测代理误差。

## Boundaries and Non-Claims

- 未声称原始非线性 K 是凸集，未声称有限 POCS sweep 得到精确 `Π_K`。
- 未声称 block-output patch 等于完整内部机制删除，未声称投影后的隐藏状态必然对应自然回答。
- 未声称 donor 单边约束可以排除所有 donor 泄露，未声称 EOS 代理等价于真实生成长度。
- 未声称当前代码已经实现上述分支；本文仅为设计推导，默认 attention-only 实现保持不变。
- 当前 `L_fact` 的 NLL 上界语义与 proposal 的原事实低支持约束不同，正式实现前必须单独裁决，不可混用。

## Open Risks

- 四个候选的效应可能高度相关，导致跨 family 频繁不可识别。
- 全 pair × layer × answer-position 的下游 Jacobian 成本很高；应先测量再决定低秩近似。
- 局部线性化可能在 OOD patch 区域失真，真值回验和信任域不可省略。
- c、d_j、e、ρ 的校准可能决定成败，必须在 discovery/validation 上冻结，不得用测试集调参。
- 投影可能通过拒答、提前 EOS 或一般激活破坏满足表面指标，必须联合生成、retain、probe 与 relearning 验收。
- donor 答案进入单边约束测量改变了当前“仅评估”的工程边界，需显式记录为 `donor_constraint_only`，而非仍写 `donor_answer_in_training=false` 的一刀切声明。

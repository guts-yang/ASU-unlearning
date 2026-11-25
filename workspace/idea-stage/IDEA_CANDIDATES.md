# ΔA 位置创新候选集 — 10 个 ideas（评分均 > 8）

**日期**：2026-11-25　**分支**：Auto/20261122（HEAD d8c3db5）
**基线**：`workspace/idea-stage/方向I_基线_v0.6.1_answer_span.md`（factpair-progressive-alignment-v0.6.1-answer-span，plan-only）
**触发**：`/idea-discovery`，聚焦基线 §2.2.1「ΔA 构造未定稿 / 方法身份张力」开放问题
**模式**：构思与规划，pilot skip，无 GPU、不训练、不部署、不改 configs/、不访问其它线路
**诚信声明**：全部为**机制推断 / plan-level**。每个 idea 的论文级 claim 一律标「待验证」，不写结论。依赖 pilot 的判别实验一律标 `needs manual pilot`，不当成已验证结果。跨模型评审后端不可用，novelty-check / research-review 为 `PROVISIONAL / REVIEW_UNAVAILABLE`，不伪造 accept 回执、不伪造 verdict-id 与 reviewer 模型名。evidence gate = `BLOCKED`。

> 口径变化说明：上一轮把「方向 I / v0.6.1」列为硬禁区（据上一轮指令「用户裁决 2 已归档」）。本轮用户明确 @ 该基线并要求基于它设计 ΔA 创新思路，按「用户当前说法优先」，本轮把 v0.6.1 作为**构思对象**，但只聚焦 ΔA 这一个开放问题，不重提整条路线、不把任何历史负结果（V1 全系 ΔA 正交对齐 / 梯度手术 stage0-14、V2/V3/V4）改写成正向证据。

## 评分口径（来源：跨会话记忆冻结口径，非本轮自创）

`S = 0.30·I + 0.25·M + 0.20·T + 0.15·N + 0.10·C`，各维 0–10，不四舍五入，按 S 降序；相差 ≤0.3 视为同一优先档。

- I 信息价值 30%：对「ΔA 到底是不是事实方向」这一开放问题的信息增益
- M 机制契合度 25%：与 v0.6.1 冻结边界（答案跨度 S_f、非 donor 中性目标、Teacher stopgrad、读出经冻结下游重算、λ_KL^ablation=0）的贴合
- T 最小实验可判别性 20%：无 GPU 下可先纸面预注册、有 GPU 时 1 个轻量判别实验能否定生死
- N 相对近邻差异潜力 15%：与 2026 强近邻（UDS / CFT / CiPO / CUSE / StyliTruth / CAP / RepSelect / ROME / AlphaEdit / Mirage）的可划界空间
- C 简洁性 10%：相对基线的唯一改动点是否单一、不引入新模块

## 评分表（10 个 S 均 > 8）

| # | Idea | I | M | T | N | C | S | 档 |
|---|---|---|---|---|---|---|---|---|
| D1 | ΔA 定义中性约束可行域几何（禁止方向） | 9.5 | 9.5 | 8.5 | 8.5 | 8.0 | **9.000** | 一 |
| D2 | ΔA 深度证书化 / 方法身份张力 (B) | 9.0 | 9.0 | 9.5 | 8.0 | 9.0 | **8.950** | 一 |
| D3 | 因果门控 ΔA（patch 加权的因果分量） | 9.0 | 9.0 | 9.0 | 8.0 | 8.0 | **8.750** | 一 |
| D4 | 事实 / 身份双探针解耦 + 反事实激活潜力 | 9.0 | 8.5 | 9.0 | 8.0 | 8.0 | **8.625** | 二 |
| D5 | 答案跨度轨迹 ΔA(p) 与数据驱动位置权重 | 8.5 | 9.0 | 8.5 | 8.0 | 8.5 | **8.550** | 二 |
| D6 | 多 donor 边际化 / donor 私有成分消去 | 8.5 | 8.5 | 8.5 | 8.0 | 8.5 | **8.425** | 二 |
| D7 | ΔA 纯度驱动渐进 Λ(k) / ρ_ℓ(k) | 8.5 | 8.5 | 8.0 | 7.5 | 8.0 | **8.200** | 二 |
| D8 | 结构化对照族残差化（关系 / 词频 / subword 基线） | 8.0 | 8.5 | 8.5 | 7.5 | 8.0 | **8.150** | 二 |
| D9 | 读出空间重定义 ΔA_readout（旋转不变） | 8.5 | 8.5 | 7.5 | 8.0 | 7.5 | **8.125** | 二 |
| D10 | 零空间 / 抗恢复投影约束 | 8.5 | 8.5 | 8.0 | 7.0 | 8.0 | **8.125** | 二 |

档位：D1–D3（8.750–9.000）为第一优先档；D4–D10（8.125–8.625）为第二优先档。相邻条目差距均 <0.3，按冻结口径整体属同一紧密梯队，不建议据此强行取舍。所有分数为执行者机制推断分，非实测成绩。

## 共同遵守的基线硬边界（每个 idea 均不违反）

- 不拼 `y⁻` / `donor_fact_value`，不算 `logit(y⁺)−logit(y⁻)`（§5 donor 数据边界）
- `L_F` 只作用于答案跨度 `S_f`，prompt 末位不进（§2.3 / §5）
- 冻结 Teacher 一律 stop-gradient，遗忘梯度只经 Student 侧 `L_F` 回传（§2.2 梯度边界）
- 目标必须满足 §4.1 非 donor 中性约束（`r_f,p≤c`、`s_d_j,p≤d_j`、`‖Δr_E,p‖≤e`）
- 读出须经冻结下游重算（N1 ReadoutTarget 式 patch-forward），不得把中间层向量直接过 LM head
- 主结果在 `λ_KL^(ablation)=0` 下成立；统计单位作者 / 实体簇
- §6 六条验收：配对特异性、渐进收益（等预算）、非答案替换、参数内化、非浅层抑制、主结果独立性

---

## D1 — ΔA 定义中性约束可行域几何（禁止方向）· S=9.000【待验证】

- **核心假设**：§2.3 的 `T_{ℓ,p}` 是「满足中性约束且离 `A⁺` 最近」的可行集投影；让 ΔA 重新进入方法身份，但不指定「去哪里」，而是**定义可行域的形状**：允许 Student 激活沿任意方向移动以读不出 `y⁺`，但禁止沿 ΔA 的 donor 语义方向移动，从结构上排除答案替换。这是 §2.2.1 张力 (A) 的中间道路：ΔA 显式进主损失结构（作约束几何 / 边界法向），却严格不教 donor 答案。
- **唯一改动点**：ΔA 从「被降为对照」变为「中性可行域 `K` 的禁止方向算子」，`K = {h : r_f,p(h)≤c, ⟨h−A⁺, ΔÂ⟩≤0, s_d_j,p≤d_j}`；`λ_KL` 仍 0，donor 答案仍不进目标。
- **判别实验**：带 ΔA 边界 vs 无边界可行集，测 donor 答案率、`∇L_F` 与 donor 提升 `g_d` 余弦、§3.5 三臂共谋测试；可行域是否把「改答 donor」排除在解之外。`needs manual pilot`。
- **seed / 预算**：3 seed；单卡 H20；判别为纸面预注册 + 一次 60-step 量级对照；系数、`c/d_j/e` 实验前冻结。
- **失败判据**：ΔA 边界与中性可行域不相交（无可行解）；边界不起作用（退化为纯 `L_F`）；donor 答案率反升。命中即撤回该几何。
- **不与淘汰重复**：既不是 V1「沿 ΔA 正交抑制 / 对齐 A⁻」（ΔA 当删除方向作用于 Student 激活），也不是把配对降为纯诊断；ΔA 是「禁止方向」而非「目标方向」，且落在 v0.6.1 新增的答案跨度 + 非 donor 中性约束框架内。

## D2 — ΔA 深度证书化 / 方法身份张力 (B) · S=8.950【待验证】

- **核心假设**：既然主损失 `L_F` 的中性目标不依赖 ΔA，就把 §2.2.1 张力 (B) 正面转为贡献：ΔA 定位为**训练后、因果式的遗忘深度证书**（patch-forward 恢复测试），训练与证书完全解耦。贡献从「反事实配对驱动遗忘」诚实改写为「反事实配对作为不可被答案替换 / 表征旋转糊弄的深度证书」。
- **唯一改动点**：不改训练；改方法身份与 §6 验收，新增「ΔA-证书分数」作为第 5 条抗恢复证据的一员；`λ_KL` 与 ΔA 都不进主损失。
- **判别实验**：ΔA-证书与 UDS / relearning / 量化恢复的秩相关；证书能否区分「真擦除」与「改答 / 拒答 / 提前 EOS」；无 gold `M_ret` 时与 UDS 无锚点变体的相关性（UDS 报告 0.994 Spearman）。`needs manual pilot`。
- **seed / 预算**：3 seed；证书为纯前向、几乎零训练成本；预注册分数与 5 类恢复通道的对应关系。
- **失败判据**：证书与输出层 FQ / ROUGE 高度共线（无独立信息）；对浅层抑制与真擦除不分化；patch 引入 OOD 伪峰（呼应 Zhang & Nanda ICLR 2024）。
- **不与淘汰重复**：不把任何负结果改写成正向方法，明确是评估 / 协议贡献；与 R3-2（ASU 深度审计基准）的差异是用反事实配对 ΔA 作证书、无需重训 gold 模型、且与 v0.6.1 donor 边界兼容。近邻 UDS(2605.24614)、Mirage(ECCV 2026)。

## D3 — 因果门控 ΔA（patch 加权的因果分量）· S=8.750【待验证】

- **核心假设**：ΔA 里只有一小部分分量对目标事实读出 `r_f,p` 有**因果**贡献（patch 回冻结下游能改变 `y⁺` span log-lik）。用 UDS / ROME 式因果追踪逐层逐位置打分，按因果效应给分量加权，得到 `ΔA_causal`，剔除「只是相关」的身份 / 词频分量。
- **唯一改动点**：ΔA 从「原始差」改为「因果效应加权的差」：`ΔA_causal(ℓ,p) = w_{ℓ,p}·ΔA_{ℓ,p}`，`w` 由 patch 恢复量标定。
- **判别实验**：patch `ΔA_causal` vs patch 等范数随机分量，比较 `r_f,p` 恢复差；与 UDS 的 KE 层定位一致性；random-donor / noise-donor 对照。`needs manual pilot`。
- **seed / 预算**：3 seed；因果扫描为前向 + hook，无训练；`w` 阈值与 patch 层集在 discovery 冻结。
- **失败判据**：因果效应在所有分量近似均匀（无稀疏结构，门控退化为原始差）；加噪 donor 产生 OOD 伪定位峰；门控与 wrong-donor 不可分。
- **不与淘汰重复**：V1 是相关性 SVD（pooled ΔA PCA）后做抑制；这里是因果干预门控，且 ΔA 用于读出 / 门控而非改 Student 激活作删除方向。近邻 UDS、ROME causal tracing、CAP(2026)。

## D4 — 事实 / 身份双探针解耦 + 反事实激活潜力 · S=8.625【待验证】

- **核心假设**：ΔA 混有「主体身份」与「目标事实」，这是 §2.2.1 的头号 killer risk。训练两个冻结探针：事实探针（能否从 ΔA 读出 `y⁺`）、身份探针（能否读出「是哪个主体」）。用 StyliTruth 式正交收缩（orthogonal deflation）把 ΔA 分解为事实子空间与身份子空间，只保留与身份正交的 `ΔA_fact`；再用 CAP 的「被抑制特征」视角补回被身份方向掩盖、但对事实读出有潜力的分量。
- **唯一改动点**：ΔA 作事实 / 身份子空间解耦，输出纯化 `ΔA_fact` + 反事实激活潜力度量。
- **判别实验**：事实 probe AUC vs 身份 probe AUC；wrong-donor ΔA 是否落在身份子空间；解耦后 `ΔA_fact` 是否仍满足 §4.1 中性约束。`needs manual pilot`。
- **seed / 预算**：3 seed；探针为冻结线性 / 轻量读出；probe 划分与冻结下游口径预注册。
- **失败判据**：事实与身份子空间几乎完全重合（不可分，呼应 V2「unpooled ΔA 与 wrong-donor 重叠 0.55」）；解耦后残差无事实信息；probe 命名过度解释。
- **不与淘汰重复**：V1 用 ΔA 主成分直接抑制而未剔除身份分量；这里显式双探针解耦 + 剔除身份，正是针对「ΔA 主要编码主体身份」这一 killer risk。近邻 StyliTruth(ICLR 2026)、CAP(2026)、RepSelect top/bottom-PC。

## D5 — 答案跨度轨迹 ΔA(p) 与数据驱动位置权重 · S=8.550【待验证】

- **核心假设**：基线 §1 说「事实在生成答案时展开」，但 ΔA 仍被当作「某层某点的差」，且 §2.3 对 `S_f` 内位置均匀加权 `1/|S_f|`。把 ΔA 重新定义为沿 `S_f` 的**轨迹** `ΔA_ℓ(p_1…p_T)`，用其时序结构（哪一段答案 token 差异最大、如何传播）定位「事实绑定发生的答案位置」。
- **唯一改动点**：ΔA 从「位置集上的点差」改为「答案跨度上的序列 / 轨迹」，并据此给 `S_f` 内位置分配数据驱动的 `ρ(p)`，替代均匀权重。
- **判别实验**：`ρ(p)` 是否集中到实体 / 属性 token（tokenizer offset mapping 对齐）；轨迹加权 vs 均匀 vs 反向加权对中性约束达标差异；prompt 末位对照（基线 §5 已判其仅作排错）。`needs manual pilot`。
- **seed / 预算**：3 seed；`S_f` offset 定义与 `ρ(p)` 归一化在 discovery 冻结，不随 schedule 变。
- **失败判据**：`ρ(p)` 近似均匀（事实不在特定答案位置展开）；轨迹与 wrong-donor 轨迹不可分；与 stage0「ΔA 逐层递增」混淆（那是层维，此处是答案位置维）。
- **不与淘汰重复**：利用的是 v0.6.1 新增的答案跨度结构（V1 在 prompt 末位或全层）；近邻 CUSE(2610.14763) 的 sensitivity-aware 步权、FDD(ACL 2025) 层间 delta 轨迹，差异在「答案跨度内位置轨迹 + 反事实配对构造 ΔA 权重」而非 KL 步权。

## D6 — 多 donor 边际化 / donor 私有成分消去 · S=8.425【待验证】

- **核心假设**：单个 donor 的 ΔA 混有「该 donor 的身份特异性」（§2.2.1 明列）。用 K 个关系匹配 donor 做边际化，取跨 donor 共有成分（共享低秩子空间 / 均值方向）作「事实相关方向」，把单 donor 私有残差当身份噪声消掉。
- **唯一改动点**：ΔA 从「单配对差」改为 K-donor 集合的共有子空间 `ΔA_marg`，私有成分投影掉；对应 §2.2.1 待选路径 (b)「多 donor 边际化 / SVD 子空间」的定稿版。
- **判别实验**：wrong-donor / shuffled-donor / noise-donor / matched-retain 对照；共有成分 vs 私有成分对目标事实 probe 的 AUC；跨 donor 一致性随 K 的曲线。`needs manual pilot`。
- **seed / 预算**：3 seed；K 与 donor 簇隔离规则、SVD 秩在 discovery 冻结；donor 全部来自 TOFU 内部保留集。
- **失败判据**：共有成分主要编码主体身份；K 增大不改变方向（边际化无信息）；不同 donor 梯度余弦近半为负（呼应 CFT 2605.27083 knowledge conflict μ≈+0.024）。
- **不与淘汰重复**：V1 用单 donor ΔA 主成分；这里在激活差空间做跨 donor 统计分解，且只作纯化 / 定位 / 读出，不作 Student 抑制方向。近邻 CFT no-conf 流水线、StyliTruth 正交收缩。

## D7 — ΔA 纯度驱动渐进 Λ(k) / ρ_ℓ(k) · S=8.200【待验证】

- **核心假设**：§2.3 的渐进 `Λ(k)` / `ρ_ℓ(k)` 仍手设、待冻结。用 ΔA 的逐层「事实纯度」（如 D4 的事实 probe AUC − 身份 probe AUC，或 D3 的因果门控值）数据驱动地决定每阶段加入哪些层、权重多少，把 §2.2.1「ΔA 未净化」与 §2.3「渐进未证」两个开放问题合并解决。
- **唯一改动点**：`Λ(k)` / `ρ_ℓ(k)` 由 ΔA 纯度谱（层 × 答案位置）导出，而非手设；渐进 schedule 变成 ΔA 纯度的函数。
- **判别实验**：ΔA-driven schedule 是否相对**等预算**静态全层 / 反向 / 随机 schedule 有独立收益（§6 第 2 条硬要求）；纯度谱跨 seed 稳定性；避最浅 / 末端 20% 层（LayerSync / FDD 经验）。`needs manual pilot`。
- **seed / 预算**：3 seed；60-step 量级等步数 / 等前后向 / 等 GPU 预算；纯度阈值与 Λ(k) 映射实验前冻结。
- **失败判据**：纯度谱近似平坦（退化为手设全层）；等预算对照无独立收益 → 按 §2.3 自认撤销「渐进」主张，仅保留「多层对齐」表述。
- **不与淘汰重复**：stage4「反事实渐进 schedule」已被证伪，但那是 prompt 末位 + 沿 ΔA 抑制且未做等预算对照；这里是答案跨度 + ΔA 纯度驱动的层选择，并把等预算对照设为强制门（正是 V1 缺的诚实门槛）。近邻 FDD「4 层采样最佳」、LayerSync。

## D8 — 结构化对照族残差化（关系 / 词频 / subword 基线）· S=8.150【待验证】

- **核心假设**：§2.2.1 说 ΔA 混有「关系传播、subword、词频」。构造对照族：relation-swap donor（换关系保主体）、answer-paraphrase、subword-length matched、词频匹配，得到伪配对 `ΔA_null` 作噪声基，从 ΔA 回归掉共享结构成分，得 `ΔA_resid = ΔA − proj_null ΔA`。
- **唯一改动点**：把 §6.1 的 5 类验收对照**前移为 ΔA 的构造算子**（残差化），而非仅事后验收。
- **判别实验**：`ΔA_resid` 对 wrong-relation / matched-retain 对照的区分度；残差化前后 donor 答案率、提前 EOS、完整回答率是否下降（排除输出抑制捷径）。`needs manual pilot`。
- **seed / 预算**：3 seed；对照族构成与 null 基秩在 discovery 冻结；严格不拼 `y⁻`、不算 logit 差。
- **失败判据**：null 基吸收全部事实信号（残差无信息）；对照族本身引入 donor 语义（违反 §5 donor 边界）；与 CFT「反事实语料互相矛盾」同型冲突。
- **不与淘汰重复**：V1 的 5 对照只作验收，这里作构造算子；不教假事实、不改 Student 为 donor 答案。近邻 CFT no-conf 结构化属性抽取、relation-swap probe（基线 §6 已提）。

## D9 — 读出空间重定义 ΔA_readout（旋转不变）· S=8.125【待验证】

- **核心假设**：§4.1 已要求读出经冻结下游重算（N1 patch-forward），但 ΔA 本身仍定义在隐状态空间 `A⁺−A⁻`。UDS / Logit Lens 的教训是隐状态差会被下游非线性旋转掩盖。把 ΔA 定义在**冻结下游读出的泛函**上：`ΔA_readout = r_f,p(A⁺) − r_f,p(A⁻)` 及其局部敏感度方向（Jacobian 一阶方向），而非向量差。
- **唯一改动点**：ΔA 的载体从「隐状态向量差」改为「冻结下游读出的标量差 + 局部敏感度方向」，天然对隐空间正交旋转不变。
- **判别实验**：对隐状态施加正交变换，`ΔA_readout` 应不变而隐空间 ΔA 应漂移；隐空间 ΔA vs 读出空间 ΔA 对 §4.1 中性约束可满足性差异。`needs manual pilot`。
- **seed / 预算**：3 seed；Jacobian 估计的样本数与数值稳定方案预注册；读出全部经冻结下游，不直接过 LM head。
- **失败判据**：读出空间 ΔA 与隐空间 ΔA 高度共线（无新增信息）；Jacobian 估计噪声大于信号；读出维度塌缩为单一标量丢失层 / 位置结构。
- **不与淘汰重复**：完全不涉及 Student 激活抑制，是 ΔA 表示域的重定义；近邻 UDS 指出 Logit Lens 在表征旋转下误判、Mirage 探针恢复度，差异在把「读出」作为 ΔA 的定义域而非评估指标。

## D10 — 零空间 / 抗恢复投影约束 · S=8.125【待验证】

- **核心假设**：让 ΔA 不只是「当前读不出」，而是「投影到抗 relearning / 保留知识零空间」，使沿 ΔA 的更新不易被少量微调或量化恢复（呼应 §6 第 5 条抗恢复证据）。构造 ΔA 时加约束：ΔA 方向避开攻击者会覆盖的高方差子空间（RepSelect 的 top-PC），落到底部 / 零空间。
- **唯一改动点**：ΔA 构造引入零空间 / 抗恢复投影算子（对 retain 梯度 top-PC 正交化），把 ΔA 限制在抗恢复子空间，与 §4.1 中性约束联动。
- **判别实验**：P3 恢复率、INT4/INT8 量化恢复在「零空间 ΔA」vs「原始 ΔA」下的差异；retain NLL 是否受损；与留出的第三类恢复攻击交叉验收。`needs manual pilot`。
- **seed / 预算**：3 seed；投影秩、retain 梯度样本、攻击预算 η 实验前冻结。
- **失败判据**：抗恢复子空间与事实方向不相交（投影后无事实信息）；retain 崩塌；只在带 hook / 带约束时有效、无 hook 崩塌。
- **不与淘汰重复**：V1 的 ΔA 正交抑制是「抹表征」且已证伪；R2-A（RepSelect 式 forget-specific 子空间）已 `REVISE` / 新颖性不足。差异：约束加在**反事实配对差 ΔA 的构造**上、作为 v0.6.1 框架内的一个 gate，而非独立两段式方法，且不训练新模块。近邻 AlphaEdit 零空间、RepSelect bottom-PC、FALCON 正交投影。

---

## Evidence Gate 与评审状态（如实记录）

| 阶段 | 状态 | 说明 |
|---|---|---|
| research-lit | done（composed） | 本轮 WebSearch 3 组 + 本地 PDF；近邻：UDS/Activation-Patching(2605.24614, EMNLP2026)、CFT hidden costs(2605.27083)、CiPO(ACL2026)、CUSE(2610.14763)、StyliTruth(ICLR2026)、CAP(2026)、RepSelect(2606.17168)、ROME causal tracing、AlphaEdit、Mirage(ECCV2026)；gemini / gemini-cli / arxiv helper 不可用，优雅跳过；文献身份未做 D1 三层独立核验，不作可引用 BibTeX |
| idea-creator | done（composed） | 本文件 10 个 ideas，按冻结评分口径打分，全部 S>8 |
| novelty-check | PROVISIONAL | 执行者基于公开文献做机制划界，非跨模型独立评审；无 accept 回执 |
| research-review | REVIEW_UNAVAILABLE | 后端全不可用（codex/copilot/claude/gemini/gemini-cli/knot-cli MISSING，无 .aris/reviewer.json）；执行者自审不构成独立评审 |
| research-refine-pipeline | 未生成 | 本轮用户诉求为 10 个 ideas 清单，未要求 top-1 的 FINAL_PROPOSAL / EXPERIMENT_PLAN；且证据门 BLOCKED |
| evidence gate | BLOCKED | 缺 run_state.py / idea_discovery_gate.py helper（整条解析链未解析到）+ 缺 2 份跨模型回执；不呈现为完成 |

执行者判断未替代 reviewer 或 integrity 裁决；未伪造 verdict-id、reviewer 模型名、accept 回执；未训练、未跑 GPU pilot、未部署、未改 configs/ 与 requirements、未重训 baseline、未读取 .env 值、未打印或写入任何凭据、未创建 credential helper、未写 git 配置、未访问 `../../lrm-unlearning/`。

## 进入 /experiment-bridge 前仍需解除的阻塞项

- 提供 `../INIT_REPORT.md`（上一轮勘验缺失）：Frozen Protocol / active_line_protocol / blockers / heavy_data_root 不得凭空补全
- 解除「全局 init BLOCKED」与「dry-run FAILED」，给出进入实验阶段的明确许可
- 启用至少一个异族 reviewer 后端并产生真实身份回执，把 novelty-check / research-review 从 PROVISIONAL / REVIEW_UNAVAILABLE 升级
- 恢复 RESUMABLE helper（run_state.py / idea_discovery_gate.py）以执行确定性 evidence gate
- 裁决 §2.2.1 方法身份张力 (A)/(B)：D1 走 (A) 中间道路、D2 走 (B) 协议化，二者对 §0/§7 故事影响不同，须用户选定后再定稿
- 冻结 `S_f` offset 定义、ΔA 净化方式、`Λ(k)`/`ρ_ℓ(k)`、`c/d_j/e` 阈值（基线 §2.2.1 列为待冻结量）
- 落实 frozen baseline 数据 / 模型软链与 heavy_data_root；确认 10 epochs / 125 steps / batch4×gradaccum4 / 单卡 H20
- 确认两套指标口径隔离：官方 TOFU 8 指标（eval/tofu.yaml）与 ACTIVE 线 U1–U7 / C1–C11 gate 不混用、不互相换算
- 所有依赖 pilot 的判别实验标 `needs manual pilot`；新结果须过 /experiment-audit + /result-to-claim

## 来源

- 基线：`workspace/idea-stage/方向I_基线_v0.6.1_answer_span.md`（§2.2.1 开放问题、§4.1 中性约束、§5 边界、§6 验收、§7 讲稿）
- 评分口径：跨会话记忆卡（方向 I /idea-discovery 冻结口径 S=0.30I+0.25M+0.20T+0.15N+0.10C）
- 反重复记忆：`research-wiki/log.md`、V1 全系 ΔA 正交对齐 / 梯度手术 stage0-14 已证伪、V2/V3/V4 负结果、R2-A/ADU/ALMPU 划界、Zhang & Nanda ICLR 2024 加噪 donor 为 OOD 伪定位峰
- 文献（WebSearch 摘要，身份未独立核验，不作 BibTeX）：arXiv 2605.24614、2605.27083、2610.14763、2606.17168、2604.15847、2608.23020；StyliTruth ICLR2026；CAP 2026；ROME 2022；AlphaEdit ICLR2025；Mirage ECCV2026

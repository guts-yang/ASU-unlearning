# 方向 I · 激活差异引导 Student 遗忘损失：Idea Discovery Report

> 日期：2026-11-23。协议：`factpair-progressive-alignment-v0.5`。
> **优化基线（baseline under optimization）**：`I/方向I_事实配对路线_重新设计Proposal_20261121.md`。
> **状态声明：本文是 plan-only 的 idea 设计，不含任何新实验结果。所有方法均未实现、未预注册、未训练。**

## 0. 执行状态与诚实声明

用户指定 `/idea-discovery` + `/idea-discovery-robot`。`idea-discovery-robot` 是机器人/具身方向适配（embodiment、sim2real、真机审批），与本任务（LLM 机器遗忘）不匹配，故**不套用其机器人专用内容**，仅沿用它对 idea 质量的硬纪律：每个 idea 必须可证伪、有最小 pilot、有失败模式、有必测指标。

流水线各阶段的真实执行情况：

| 阶段 | 状态 | 说明 |
|---|---|---|
| Phase 0 研究简报 | 已完成 | 无 `RESEARCH_BRIEF.md`；以用户指定的基线 proposal 为优化对象，浓缩为第 1 节问题锚点 |
| Phase 1 文献调研 | 已完成 | 本地证据库（30 篇已核文献）+ 联网查新，见第 2 节 |
| Phase 2 构思与过滤 | 已完成 | 产出 4 个候选，见第 4 节；与基线的对接见第 4.5 节 |
| Phase 3 深度查新 | **部分** | 联网初判完成；**跨模型评审器不可用**（无 codex/gpt 类 MCP、无 `.aris/reviewer.json`），未做对抗式 novelty 确认 |
| Phase 4 外部批判评审 | **未执行** | 评审器不可用，未伪造任何 reviewer 结论 |
| Phase 4.5 refine / pilot | **未执行** | 当前卡点是 G0–G4 激活可行性预检（见 `REMOTE_VALIDATION_TODO.md`），且本机无可用 GPU 训练资源；只给出 pilot 方案，不冒称已跑 |

因此本文是**「构思 + 本地/联网查新」阶段的产物**，不是完整流水线通过报告。第 6 节的 novelty 判断是初判，需外部评审确认；任何"推荐"都不等于"已验证可行"。

## 1. 问题锚点（Problem Anchor）

**优化基线。** 用户明确指定 `I/方向I_事实配对路线_重新设计Proposal_20261121.md` 为当前方向 I 的待优化基线。该基线已定稿的部分**保持不变**：

- §2.1 反事实输入构造 $X^+=X(s_f,r),\ X^-=X(s_r,r)$（同模板、token 等长、槽外不变、实体级隔离）；
- §3 DKT 同前缀 forward KL $\mathcal L_{KL}=(1-\eta)\mathcal L_T+\eta\mathcal L_S$；
- §4 总损失框架 $\mathcal L=\lambda_F\mathcal L_F+\lambda_{KL}\mathcal L_{KL}+\beta\mathcal L_R$，部署只留 Student。

**唯一被优化的部分**是基线 §2.2 末尾与 §2.3 的 $\mathcal L_F$：

> 基线原文自述：「后续需要调研应直接对齐 $A^-$、利用 $\Delta A_\ell$ 引导更新，还是采用其他目标构造；当前不预设 $\Delta A_\ell$ 已是纯净的事实差异」，并把「直接向 $A^-$ 对齐」标为**参考方案而非定稿**。

这正是本文要填的空：给出一个自洽、可证伪、能真正产生遗忘梯度、且不变成答案替换的 $\mathcal L_F$ 定义，替换基线的参考方案。

**必须同时满足的六条硬约束（来自证据调研 G0–G6）：**

1. 有遗忘梯度：去 $\tau$ 后原 ASU 的 forget KL 在初始化处梯度为 0，新信号必须来自明示的 Student-side loss。
2. 不是答案替换：$s_f$ 答案下降不能靠答成 donor $s_r$ 实现。
3. 不是一般扰动/低置信：不能只靠范数匹配噪声或随机目标制造"胡言乱语"。
4. 参数内化：部署只保留 Student，不依赖运行时 hook。
5. 可证伪的"渐进"：必须与等预算静态全层、反向/随机 schedule 比较才有独立收益。
6. 实体级隔离：forget / donor / retain-train / eval 按作者与别名簇互斥。

**非目标。** 不主张"永久删除"或数学意义上的不可恢复；不使用旧 pilot 的正向证据；不引入注意力温度；不在本阶段训练 Teacher/Student。

**关键资产现状。** 10 对 prompt-only 激活已落盘（`a_plus_a_minus.jsonl`，每对 `[33,4096]`，最后 token 位置，复现差为 0，差值多在后部状态索引 29–32）。这是 idea 1/2/4 的最小 pilot 可直接复用的现成输入，**无需训练即可先做证伪**。

## 2. 研究 Landscape（查新结果）

据不完全证据显示，以下方法已占据相邻生态位；逐项来源见文末。

| 方法 | 表示目标 / 机制 | 与本任务的关系 | 它没解决什么 |
|---|---|---|---|
| RMU（Li et al. ICML 2024） | forget 激活推向**随机向量** + retain 锚定 | 激活层遗忘的基线 | 混淆低置信、深层失效、4-bit 量化可恢复 83%；目标无语义 |
| Adaptive RMU（Dang et al. AAAI 2025） | 随机目标系数按冻结表示范数自适应 | 说明"范数匹配"是必要工程细节 | 目标仍是随机，非另一真实主体 |
| LUNAR（Shen et al. NeurIPS 2025） | 激活重定向到 **"I don't know" 拒绝区**，单层 MLP down-proj，闭式解 | 显示"语义化目标"优于随机 | 单点瓶颈，2026 年工作用 GRPO/pass@k 恢复；拒绝≠遗忘 |
| FALCON（Hu et al. NeurIPS 2025） | MI 选层 + SVD 主方向 + 对比 unalignment（InfoNCE）+ 梯度正交投影 | **最接近**：也用对比目标与正交梯度 | 推离的是"错误类/主方向"，没有"另一真实 donor 主体"这一负样本；不防 donor 替换 |
| GU / OEU（2026） | 梯度分解为 retain 正交分量；熵最大化 | 提供梯度手术工具 | 用梯度几何，不定义"事实绑定子空间"从何而来 |
| SAE-guided unlearning（Yamashita et al. 2025） | 在 SAE latent 空间把 target 从 known 推向 **unknown 实体** | 与"对齐到另一实体"最接近 | 依赖 SAE；用"未知实体中心"而非同关系反事实 donor 对 |
| Concept subspace editing（2025） | 从**反事实对**的激活差估计低维概念子空间，约束编辑 | 提供"ΔA→子空间"的直接方法论 | 面向编辑/steering，非遗忘；无 donor 防替换 |
| CSD（Zhao et al. Findings EMNLP 2026） | type-consistent 实体替换构造 counterfactual teacher，**输出分布**层 | 方向 I 的直接先例 | 在分布层，不在逐层隐状态；不防替代事实 |
| PGB-CT（Yan et al. ACL 2026，本地已有） | 可微掩码搜索联合影响某指标的组件集合 | 本地现成资产，可做 ΔA 因果净化 | 找运行时组件，不直接训练 Student 参数 |

**据此得到的核心缺口：** 现有工作要么把 forget 表示推向无语义目标（RMU）、拒绝区（LUNAR）、错类/主方向（FALCON）、未知实体中心（SAE-guided），要么在输出分布层做反事实（CSD）。**没有一条把"同关系、同模板的真实 donor 主体激活差 ΔA"直接转成 Student 的遗忘损失，更没有把"推离 donor、防止答案替换"显式写进目标函数。** 这就是下面 4 个 idea 要填的空。

## 3. 核心洞察：把"ΔA 怎么用"拆成三个正交设计轴

基线卡点之所以难，是因为三个问题被混成了一个"直接对齐 $A^-$"。拆开后它们可独立选择、独立消融：

- **轴 1 · ΔA 的净化（ΔA 从哪来）**：裸 $\Delta A$（混有实体身份、subword、词频、关系传播）→ 多 donor 边际化 → 因果 patching 门控。
- **轴 2 · 目标状态（往哪移）**：$A^-$（会替换）→ 沿 $\Delta A$ 部分插值 → unknown/unknown-实体锚。
- **轴 3 · 损失形式（怎么优化）**：逐层点对齐 → InfoNCE 三重对比 → 梯度投影（ΔA 只当子空间，不当 target）。

4 个 idea 是这三个轴的不同自洽组合，而非四个孤立点子。

## 4. 候选 Ideas

统一记号（每层、subject-last 位置 $P$）：$A^+_\ell=h_{\theta_0,\ell}(X^+;P)$，$A^-_{\ell,k}=h_{\theta_0,\ell}(X^-_k;P)$；$\Delta A_{\ell,k}=A^+_\ell-A^-_{\ell,k}$；$h_{S,\ell}=h_{\theta,\ell}(X^+;P)$；$\hat u_{\ell,k}=\Delta A_{\ell,k}/\|\Delta A_{\ell,k}\|$。Teacher 全冻结、stop-gradient，梯度只经 Student 隐状态与 logits。

### Idea A · TriCon：推离原事实 + 推离全部 donor + 拉向 unknown 锚（首选，novelty 最强）

**机制。** 把"防止答案替换"直接编码为对比损失的负样本集合：以 $h_{S,\ell}$ 为 anchor，$A^+_\ell$（原事实）与**全部** donor $\{A^-_{\ell,k}\}_{k=1}^K$ 同为负样本，目标正样本是"unknown 锚" $U_\ell$（见下）。这是与 RMU/LUNAR/FALCON 的关键分叉：它们从不把 donor 当负样本，因为它们不用 donor。

$$
\mathcal L_{\text{TriCon},\ell}
=-\log\frac{\exp(\operatorname{sim}(h_{S,\ell},U_\ell)/\tau_c)}
{\exp(\operatorname{sim}(h_{S,\ell},U_\ell)/\tau_c)+\exp(\operatorname{sim}(h_{S,\ell},A^+_\ell)/\tau_c)+\sum_{k=1}^{K}\exp(\operatorname{sim}(h_{S,\ell},A^-_{\ell,k})/\tau_c)}
$$

$\operatorname{sim}$ 取余弦，$\tau_c$ 是 InfoNCE 温度（**不是 ASU 的注意力温度**），预冻结。$U_\ell$ 的构造必须预注册：候选为"同关系下同实体簇但标为 unknown 的实体激活均值"或"受控 null prompt 激活"，且须独立验证 $U_\ell$ 确实对应"不知道"而非另一条事实。

**为什么能产生遗忘。** 优化同时压低 $h_S$ 与 $A^+$ 的相似度（移除目标绑定）、压低与所有 donor 的相似度（排除替换）、抬高与 $U$ 的相似度（给出有语义的落点）。三者共同给出一个"既不像知道 $s_f$、也不像知道任何 donor、而像不知道"的明确梯度方向。

**排除的替代解释。** donor 负样本项直接排除 answer replacement；unknown 正样本（而非随机噪声）排除一般置信度崩塌；多 donor 排除单一配对偶然性。

**失败模式。** 完全由 $U_\ell$ 质量决定：若 $U_\ell$ 实际靠近某条真实事实或落在分布外，损失会把模型推向无意义区。这是首要 kill 点。

**最小 pilot（无需训练）。** 直接在激活层求 $\min_{z}\ \mathcal L_{\text{TriCon}}$（把 $h_{S,\ell}$ 当可优化向量、Teacher 激活冻结），验证几何可实现性：是否存在 $z$ 同时远离 $A^+$ 与全部 $A^-$ 而靠近 $U$；再把这个 $z$ patch 回 $q_+$，用预冻结的 $y_+$ span score 检查目标事实支持度下降、donor span score 不升。

**必测指标。** $y_+$ span score 变化；donor/其他实体 span score；matched-retain probe；拒答率；$U$ 到 $A^+/A^-$ 的余弦间隔。

### Idea B · DeltaSubspace：多 donor 边际化 + 沿共享区分方向的部分插值消融（当前阶段最易证伪）

**机制。** 不直接把 $h_S$ 拉到 $A^-$，而是对**每个关系 $r$ 收集 $K$ 个 donor 的 $\Delta A_{\ell,k}$**，对其做 SVD 取 top 主成分张成 $U_r$（"把 $s_f$ 与任意 retain 主体区分开的方向"）。由于各 donor 的特异身份在多个 $\Delta A_{\ell,k}$ 间不一致而被平均掉，$U_r$ 近似只保留共享的"主体-属性可区分性"分量。再让 Student 沿该子空间**部分**移动，插值系数 $\alpha\in(0,1)$ 预冻结（**刻意不取 $\alpha=1$，即不走到 $A^-$**）：

$$
T_\ell=\operatorname{stopgrad}\!\left(A^+_\ell-\alpha\,\operatorname{Proj}_{U_r}(\Delta A_{\ell,1})\right),\qquad
\mathcal L_F=\sum_{\ell}w_\ell\,\big\|h_{S,\ell}-T_\ell\big\|_2^2
$$

**为什么能产生遗忘。** $h_S(X^+)$ 在"可区分方向"上的投影被压缩 → $s_f$ 与 retain 主体的表示更接近 → 下游难以把 $s_f$ 与关系 $r$ 的属性绑定 → 目标事实支持度下降。因为移动方向不含任何具体 donor 身份，不会偏向某个 $s_r$。

**排除的替代解释。** 多 donor 边际化排除"身份噪声"与单一 donor 偶然性；部分插值排除完全替换；保留正交分量减少 utility 损伤。

**失败模式。** 若 $U_r$ 捕获的是"实体身份"而非"属性绑定"，移动会让模型认不出 $s_f$ 但仍能答出属性 → 变成 entity confusion 而非遗忘。这是必须预注册的区分检验。

**最小 pilot（无需训练，最贴合当前资产）。** 现有 10 对 + safetensors 已是天然输入：对多个 donor 求 $U_r$，构造 $T_\ell$，把 $A^+$ patch 成 $T_\ell$（不同 $\alpha$ 扫描），看 $y_+$ span score 是否随 $\alpha$ 单调下降、donor span score 是否不升。这与 `REMOTE_VALIDATION_TODO.md` 的 shuffle/no-op/matched-null 对照完全兼容。

**必测指标。** $y_+$ span score vs $\alpha$ 曲线；donor 答案出现率；实体身份 probe（区分 identity vs attribute 效应）；范数/逆范数对照（排除一般扰动）。

### Idea C · CausalDelta：因果门控 ΔA + 可证伪的逐层渐进 schedule（增强模块，可叠 A 或 B）

**机制。** 用本地已有的 PGB-CT / activation patching 先给 $\Delta A_\ell$ 的每个维度打"因果贡献分" $c_{\ell,j}$（把 $A^+_{\ell,j}$ 替换为 $A^-_{\ell,j}$ 后 $y_+$ span score 的下降量），再做逐元素门控 $\Delta A^{\text{causal}}_\ell=c_\ell\odot\Delta A_\ell$，把 Idea B（或 A）里的 $\Delta A$ 换成净化版本。渐进性做成显式变量 $B(t)=$ 已启用层数/$N$，按 Geva 两阶段假说设计"浅层先实体、深层后属性"的正向 schedule，并与静态全层、反向、随机 schedule 在等预算下比较。

**为什么能产生遗忘。** 只在因果上真正承载"$s_f\!\to$属性"绑定的维度上移动，避开无关表示；schedule 提供可独立检验的"渐进"贡献。

**排除的替代解释。** 因果门控排除"裸差值里混的词频/subword/关系传播"；schedule 消融排除"逐层取激活被误称渐进"。

**失败模式。** localization/editing mismatch（Hase et al. 2023）：因果定位到的层不等于最佳编辑层。缓解：用 patching 只提供**维度 mask**，不用于选层，层选择另按 G0/G3 冻结。

**最小 pilot。** 在现有 10 对的部分层上先跑单维度 patching，验证 $c_\ell$ 的非平凡性（是否显著非零、是否集中于后部索引 29–32）；确认后再谈训练 schedule。

**必测指标。** $c_\ell$ 分布与稳定性；四 schedule 等预算对比；逐阶段 retain 非劣效。

### Idea D · DeltaGuide：ΔA 定义"事实绑定子空间"，梯度手术而非 target（稳健备选）

**机制。** ΔA **不进 loss 当 target**，只用来定义事实绑定子空间 $\mathcal B_\ell=\text{top-}k$ 主成分 of $\{\Delta A_{\ell,k}\}$。forget 梯度 $g_F$ 仍来自 $\mathcal L_{KL}$（或 NPO），retain 梯度 $g_R$ 来自 $\mathcal L_R$；将更新限制在"$\mathcal B$ 内且与 $g_R$ 正交"：

$$
g_{\text{upd}}=\alpha\,\operatorname{Proj}_{\mathcal B_\ell}(g_F)+\beta\,\operatorname{Proj}_{g_R^\perp}(g_F)
$$

借鉴 GU/OEU 的正交思想，但用反事实 donor 对来定义"该动哪里"，而非假设规则化的 retain 梯度空间。

**为什么能产生遗忘。** $g_F$ 本身携带遗忘信号；限制到 $\mathcal B$ 确保信号作用于主体-属性绑定而非一般表示；$g_R^\perp$ 保证 retain 一阶不变。因为不向 $A^-$ 移动，天生不会 answer replacement。

**排除的替代解释。** 子空间限制排除一般扰动；正交投影排除保留知识损伤。

**失败模式。** $\mathcal B$ 的估计质量（同 Idea B/C）；且 $g_F$ 若本身弱（去 $\tau$ 后需靠 $\mathcal L_{KL}$ 提供），投影后可能无有效更新。

**最小 pilot。** 先在激活层检查 $g_F$ 在 $\mathcal B$ 内的投影能量是否显著非零，再检查投影后与 $g_R$ 的余弦是否≈0。

**必测指标。** $g_F$ 子空间能量、更新前后 retain loss 一阶变化、forget/readout 指标。

## 4.5 与优化基线的对接（每个 idea 改哪里）

用户指定的基线 proposal 是优化对象。四个 idea 都是**对基线同一处的替换式优化**，基线其余部分原样保留：

| 基线位置 | 基线现状 | Idea A | Idea B | Idea C | Idea D |
|---|---|---|---|---|---|
| §2.2 ΔA 构造 | 裸 $\Delta A=A^+-A^-$，自述"未预设纯净" | 用 $A^+$/全部 $A^-$ 作对比集 | 多 donor SVD 边际化为 $U_r$ | 因果门控 $c_\ell\odot\Delta A$ | ΔA 只定子空间 $\mathcal B$ |
| §2.3 目标 $T_\ell$ | 直接对齐 $A^-_T$（参考方案） | unknown 锚 $U_\ell$ | $A^+-\alpha\operatorname{Proj}_{U_r}\Delta A$ | 叠 A/B 的净化目标 | 无 target，改梯度投影 |
| §2.3 $\mathcal L_F$ 形式 | 逐层加权 MSE/距离 | InfoNCE 三重对比 | 部分插值 MSE | 同 A/B + 维度 mask | 梯度手术 $g_{\text{upd}}$ |
| §2.3 "渐进" schedule | 预留 $\Lambda(k),\rho_\ell(k)$ | 未涉及 | 未涉及 | 显式 $B(t)$ 四 schedule 消融 | 未涉及 |
| §3 $\mathcal L_{KL}$ | DKT 同前缀 forward KL | 不动 | 不动 | 不动 | 不动（兼容 NPO） |
| §4 总损失 | $\lambda_F\mathcal L_F+\lambda_{KL}\mathcal L_{KL}+\beta\mathcal L_R$ | 替换 $\mathcal L_F$ | 替换 $\mathcal L_F$ | 替换 $\mathcal L_F$+schedule | 替换更新规则 |

**一句话对接：** A/B/C/D 均只改基线 §2.2–2.3 的 $A^-$-对齐参考方案；§2.1 输入构造、§3 KL、§4 框架与系数口径不动。A 与 D 是 $\mathcal L_F$ 的两种替代定义，B 是最小可证伪版本，C 是可叠加的净化与 schedule 模块。

## 5. Ranking 与推荐

| 排名 | Idea | novelty | 当前可执行性 | 推荐角色 |
|---|---|---|---|---|
| 1 | A · TriCon | 高（显式 donor 负样本，未见先例） | 中（依赖 $U$ 构造） | **主推方法**，但先验证 $U$ |
| 2 | B · DeltaSubspace | 中高（多 donor 边际化 + 不取 $\alpha=1$） | **高**（现成 10 对即可 patch smoke，无需训练） | **第一步证伪载体**，最先跑 |
| 3 | C · CausalDelta | 高（因果净化 + 可证伪 schedule） | 低（需先 patching 预计算） | 增强模块，叠 A/B |
| 4 | D · DeltaGuide | 中（正交投影已有 GU/FALCON） | 中 | 稳健备选 / baseline |

**执行建议（不分叉）：** 第一步**用 Idea B 的 activation-level patch smoke 做证伪**——它不需要训练、不需要过 G5、直接吃现有 10 对激活，正好卡在当前 G0–G4 的劫点上。只有当"沿多 donor 共享方向部分移动能让 $y_+$ 降而 donor 不升"这一信号成立，才值得投入 Idea A 的对比损失（并优先解决 $U_\ell$ 构造），再叠 Idea C 的因果门控与 schedule。若 B 的信号不成立，按 kill-switch **停止整条 target design**，不加层、不调指标、不回看确认集。

## 6. Novelty 初判（非跨模型评审，需确认）

- 对整个组合（真实 donor 逐层激活差 × subject-last × 防替换对比目标 × 逐层 schedule）的检索结果：**据不完全证据显示，未找到直接工作**。但这是检索结论，不是穷尽性 novelty 证明。
- 单组件均有近邻：多 donor 边际化≈concept subspace 估计；$U$ 锚≈SAE-guided unknown 对齐；梯度正交≈GU/FALCON/OEU。因此**新颖性必须落在"组合 + 防 donor 替换目标"，不能落在任何单组件**。
- 最大重叠风险是 FALCON（对比 unalignment + 正交梯度）与 SAE-guided（对齐到未知实体）。需在正式查新中逐项划界：FALCON 无真实 donor 负样本；SAE-guided 用 SAE latent 与未知实体中心而非同关系反事实对。
- **以上为初步判断，未经跨模型评审器确认（环境不可用）。**

## 7. 下一步与门控

1. 过 G0–G4：冻结 checkpoint/层口径、实体级隔离、激活坐标映射、缓存复现、matched-null 对照（见 `REMOTE_VALIDATION_TODO.md`）。**过门前不训练。**
2. 跑 Idea B 的最小 patch smoke（无需训练），按 kill-switch 判定是否继续。
3. 若通过：解决 Idea A 的 $U_\ell$ 构造并做激活层 InfoNCE 可行性检查。
4. 叠 Idea C 的因果门控与 schedule，做等预算四 schedule 消融（G5）。
5. G6 独立验收：行为分类 + retain/shared + 因果恢复 + 有限预算攻击，全程用受限"未恢复"措辞。
6. 补齐本流水线缺失的 Phase 3/4：在有跨模型评审器的环境重跑 novelty-check 与 research-review，确认第 6 节初判。

## 参考

- **优化基线**：`I/方向I_事实配对路线_重新设计Proposal_20261121.md`（protocol `factpair-progressive-alignment-v0.5`）。
- 本地：`I/archive/方向I_反事实主体替换SubjectLast对齐去τ_证据调研.md`（30 篇已核文献 [1]–[30]）；`I/archive/2026-11-21_references/PGB-CT_数学原理与通俗讲稿.md`；`I/data/external_donor_prompt_activation_v0.3_activation/summary.md`；`I/data/external_donor_prompt_activation_v0.3/REMOTE_VALIDATION_TODO.md`。
- 联网查新（初判级，未逐篇核验全文）：FALCON（Hu et al., NeurIPS 2025, arXiv:2502.01472）；RMU（Li et al., ICML 2024）；Adaptive RMU（Dang et al., AAAI 2025）；LUNAR（Shen et al., NeurIPS 2025, arXiv:2502.07218）；LUNAR red-team / pass@k（2026）；GU Geometric-disentanglement Unlearning（Zhang et al., 2026）；OEU（arXiv:2602.00567, ICML 2026）；Sparse-Autoencoder-Guided Internal Representation Unlearning（Yamashita et al., arXiv:2509.15631）；Concept subspaces for targeted model editing（2025）；GUARD-IT（2026）；CSD（Zhao et al., Findings EMNLP 2026）。

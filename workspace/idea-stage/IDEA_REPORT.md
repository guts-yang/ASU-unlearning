# Idea Discovery 整合报告 —— 方向 I 基线与 idea-stage 规整

**日期**：2026-11-25
**分支**：`Auto/20261122`（基线提交 `a2ce148`）
**触发**：用户指令「把当前分支上的方向 I 基线按 L7 源文件修改 + 规整 idea-stage（淘汰归档、候选独立维护 proposal）+ 推送远程」+ `/idea-discovery`
**权威基线**：`方向I_基线_v0.6.1_answer_span.md`（源：`/Users/chenyliao/MuRAG/Papers/L7_ASU创新/I/方向I_事实配对路线_重新设计Proposal_20261121.md`）
**裁决口径**：以当前分支 `IDEA_REPORT_V4.md` + `docs/research_contract.md` + `research-wiki/log.md` 为准，执行者不自行改判、不编造评审回执

> **诚信声明**：本次是**结构规整与文档整合**，不是新一轮文献调研 / pilot / 跨模型评审。所有方法均为 `plan-only / provisional`；本机无 GPU、无跨模型评审后端（Codex/Gemini/Grok/Claude-review 均不可用，无 `.aris/reviewer.json`），不伪造 `accept` 回执，不把构思分当实验成绩。所有关键数字标注来源，未核实处用「据不完全证据显示」。

## 1. 执行摘要

1. **方向 I 基线统一**：把当前分支上两份过时/截断的基线副本归档，用 L7 的 `v0.6.1-answer-span` 完整版替换为唯一权威基线（正文逐字保留）。
2. **idea-stage 结构规整**：顶层只留规范报告、当前有效决策报告、叙事与研究契约；被迭代取代的历史报告与 HTML 衍生视图进 `archive/`；6 个未淘汰候选各维护一份独立 `candidates/*.md` proposal。
3. **淘汰/候选判定**：沿用 V4 成文裁决——3 `PROMOTE` + 3 `REVISE` 为候选，`ELIMINATED` 全系归档并保留反重复记忆。**2026-11-25 更新**：R2-B 的 3-seed 识别门 FAIL（等均值 τ=2.25 常数 3/3 优于 `low2high`），与已标 `archived/negative` 的 R2-A、R3-1 一并移入 `archive/candidates/`；现役候选为 `R2-C`、`R3-2`（PROMOTE）与 `R3-3`（REVISE，仅作消融），见 §4。
4. **口径张力如实标注**：方向 I 基线所属的反事实配对路线在 V4 中被整批 `ELIMINATED`，而用户本次要求把该基线更新为 v0.6.1。二者关系见 §5，由用户裁决，本报告不自行判定基线有效性、不删除基线。（该裁决已于 2026-11-25 完成：选 2 · 归档，见 §5.1。）

## 2. 目录结构（规整后）

```text
workspace/idea-stage/
├── IDEA_REPORT.md                          ← 本整合报告（规范交付物 / 唯一真实来源）
├── 方向I_基线_v0.6.1_answer_span.md        ← 当前分支唯一权威的方向 I 基线
├── IDEA_REPORT_V4.md                       ← 当前有效决策报告（脚本硬编码引用，保留顶层）
├── NARRATIVE_ASU_TEMPERATURE_DEPTH.md      ← 当前有效叙事（Markdown 主源）
├── candidates/                             ← 3 个未归档候选，每个一份独立 proposal
│   ├── R2-C_抗恢复即目标_relearning_aware_meta.md (PROMOTE)
│   ├── R3-2_ASU遗忘深度审计基准.md               (PROMOTE)
│   └── R3-3_浅深解耦双阶段遗忘.md                (REVISE，仅作消融)
├── docs/
│   └── research_contract.md                ← W1→W1.5 会话恢复契约（已指向新结构）
└── archive/                                ← 被取代/淘汰内容，逐字节保留
    ├── README.md                           ← 归档口径与淘汰清单
    ├── baseline_versions/
    │   ├── proposal_v0.5_20261122.md
    │   └── proposal_v0.6_truncated_20261124.md
    ├── candidates/                         ← 2026-11-25 归档：R2-B(3-seed 判负)、R2-A、R3-1
    │   ├── R2-B_ASU温度课程_tau_k.md
    │   ├── R2-A_ASU_Selective_表示选择性.md
    │   └── R3-1_SensiTemp_层头自适应温度场.md
    └── reports/
        ├── IDEA_REPORT_V1.md / .html
        ├── IDEA_REPORT_V2.md / .html
        ├── IDEA_REPORT_V3.md / .html
        └── NARRATIVE_ASU_TEMPERATURE_DEPTH.html / .html.review.json
```

`workspace/ideas_archive/`（既有 299 文件 `copy2` 快照归档）保持不变，是跨目录的历史汇总，与本次 `idea-stage/archive/` 分工不同：后者只管 idea-stage 自身被迭代取代的内容。

## 3. 方向 I 基线变更（用户第一项要求）

| 处置 | 原路径 | 新位置 / 状态 |
|---|---|---|
| 归档 | `workspace/proposal/方向I_事实配对路线_重新设计Proposal_20261121.md` | `archive/baseline_versions/proposal_v0.5_20261122.md` |
| 归档 | `workspace/docs/方向 I：反事实主体配对驱动的逐层渐进遗忘 Proposal.md`（HEAD `a2ce148` 新增） | `archive/baseline_versions/proposal_v0.6_truncated_20261124.md` |
| **新增权威基线** | —— | `方向I_基线_v0.6.1_answer_span.md` |

**为什么替换**：v0.5 的 $\mathcal L_{KL}$ 被 v0.6 §1 判定为「donor 答案替换/迁移目标，不是删除目标」；HEAD 提交里的 v0.6 文件只有 93 行、截断在 §3.3 中途，且未包含 v0.6.1 的关键裁决。v0.6.1 相对 v0.5 的三项实质修改：

1. Teacher 只作「另一主体」对照轨迹，**不**把 $p_{\theta_0}(y|X^-)$ 当 forget 目标；
2. 遗忘方向改由 **答案跨度 $S_f$ 上的逐层激活损失 $\mathcal L_F$** + 「原事实读不出」中性约束给出（$X^+\oplus y^+$ teacher-force 前向，prompt 末位不进 $\mathcal L_F$）；
3. DKT 式同前缀 KL **降为对照/消融臂**（$\lambda_{KL}^{\text{(ablation)}}=0$ 默认），主结果不得依赖它，并强制 $\mathcal L_F$-only / $\mathcal L_{KL}$-only / 组合三臂证伪共谋。

新基线以 L7 源文件为**上游真源**：初版（commit `b0761f5`）正文与源文件逐字一致、仅加文件头来源说明；2026-11-25 按用户「先修改显性的 bug，然后 ΔA 构造在 proposal 里标注一下」再做了一次**本地修订**，公式的数学含义、结论、状态声明（`plan-only`、无新实验）、§6 六条验收、§7 讲稿与 §0 故事均未改动：

- 修订 1（§2.2，显性 bug）：Teacher 前向输入由「纯 $X^+/X^-$」改为「teacher-force 拼入答案后的完整序列」，激活/ΔA 记号统一为带答案跨度位置索引 $p\in S_f$。原写法与同段「$P$ 指 $S_f$」及 §4.1/§5「冻结 Teacher 在 $X^+\oplus y^+/X^-\oplus y^+$ 上前向得到」自相矛盾，系 v0.6.1 改 $L_F$ 作用域时 §2.2 未同步更新的补丁残留。
- 核查（§2.3）：源文件本就自洽（$T_{\ell,p}=A^-_{T,\ell}(p)$），未改动。
- 新增标注（§2.2.1）：把 **ΔA 构造未定稿**及「配对被移出主梯度路径、方法身份张力」显式标为**待裁决开放问题**（两条处置 (A)/(B) 并列、未选定、未改故事），未自行拍板机制。

L7 源文件本身未改动；本分支基线是它的「修订版」，文件头已记录修订边界与上游真源路径。

## 4. 候选 ideas（现役 3 + 归档 3）

判定来源：`IDEA_REPORT_V4.md` 决策表。按 `research_contract.md`，`PROMOTE`=低成本/确定性主线，`REVISE`=带重启门槛的条件候选，两者均属「未淘汰」。**2026-11-25 更新**：依 3-seed 实测与 research-wiki 归档状态，R2-B、R2-A、R3-1 已移入 `archive/candidates/`（见 4.3），现役候选见 4.1/4.2。每份 proposal 统一覆盖：领域空白与思路 → 可证伪假设 → 公式化/代码级具体实现 → 30 秒故事 → 与近邻工作的差异化（novelty 落点）→ 当前证据与 provisional 状态 → 判别实验/失败模式/必测指标 → 与方向 I 基线的关系 → 来源与诚信声明。

### 4.1 PROMOTE（2，主线候选）

| ID | 一句话思路 | 关键证据（来源 V4 / contract / NARRATIVE） | 未完成义务 |
|---|---|---|---|
| **R2-C · 抗恢复即目标** | 把 P3 恢复攻击前移进 minimax 训练目标，直接优化"深度"而非事后验收 | ASU-u P2 恢复 $0.524$、P3 $0.371$、$\text{mean\_recovery}=0.448$，确认浅层遗忘；与 R3-2 共享公开攻击阶梯 | E3：实现内步攻击 + 恢复曲线，不能只当审计替代 |
| **R3-2 · ASU 遗忘深度审计基准** | 首个针对 ASU 自身的 P1/P2/P3/INT8/Probe 统一恢复阶梯 + 恢复曲线下面积 + factual/function 分层 | forget $0.96\to$ P2 $0.261\to$ P3 $0.185$；**双向可发表**（稳健=强正面，回退=高价值负面） | E1 已完成；补跨模型复核 |

### 4.2 REVISE（1，条件性候选，非主线）

| ID | 降级原因（据 V4） | 重启 / 保留门槛 |
|---|---|---|
| **R3-3 · 浅-深解耦双阶段** | gate PASS 但仅 L12 边缘（$dF=+0.133$、$dR=+0.099$）；严格阈值 $dF\ge0.15$ 且 $dR\le0.05$ 下通过数 $=0$，V2 抑制失败模式重现 | 仅保留为消融；kill switch：压缩深层 ΔA 若同时伤 retain 即否决 |

### 4.3 已归档（3，2026-11-25）

| ID | 原状态 | 归档原因 |
|---|---|---|
| **R2-B · ASU 温度课程 τ(k)** | PROMOTE | 补跑等均值 $\tau=2.25$ 常数对照后，后者 3/3 seed 均优于 `low2high`（均值 $dF$ 1.857 vs 1.449，配对差 $+0.407$）；按预注册识别门 FAIL，收益由平均平滑水平解释，时间课程无净收益 |
| **R2-A · ASU-Selective** | REVISE | research-wiki 已标 `archived/negative`；RepSelect/ADU/ALMPU 已占据该空间，E4 机制划界未做 |
| **R3-1 · SensiTemp** | REVISE | 等平均 $\tau$ 常数对照 $+1.792$ 高于 sens 场 $+1.747$；收益来自更强总平滑而非结构化分配，与 R2-B 失败模式同源 |

归档文件在 `archive/candidates/`，正文逐字节保留，仅更新 front-matter 的 `status`/`archived_*` 字段。

**novelty 纪律**：6 份 proposal（现役 3 + 归档 3）均明确「novelty 不能落在组件本身」（温度调度 / minimax / 恢复攻击 / SVD 子空间 / trust region 都有先例），只能落在与 ASU forget-teacher 或用户独有资产（stage0-14 实测、反事实配对 pooled ΔA、官方 ASU 代码库）的结合方式。差异化不足的（R2-A、R3-1）如实标注，不包装成新贡献；R2-B 的 3-seed 识别门负结果同样不包装。

## 5. 淘汰 ideas（ELIMINATED，已归档 + 反重复记忆）

V4 决策表唯一 `ELIMINATED` 行 + research-wiki `anti_duplication`（`research-wiki/log.md`，2026-11-23）：

- **R1-I1/I2/I3/I4/I5 反事实配对 + ΔA 正交/对齐/差值引导（全系）**：stage0-14 H20 实测，所有变体 forget/retain 均弱于原 ASU ref（ref 0.555/0.798；idea1 1.068/1.52）；stage8/9/10/11/13/14 失败、stage5 mask 无收益；`Gate=KILL`。**不再投入算力，转为反重复记忆，禁止后续轮次复现。**
- 关联淘汰（V2/V3 报告已写入）：直接对齐 $A^-$（donor 替换）、PCGrad/EWC/GradNorm/GA-orth 梯度手术、单纯 mask / $\lambda\beta$ 调参 / 反事实渐进 schedule、ASU-Selective 与 RepSelect 式子空间（差异化不足，R3 查新）。
- 历史报告 V1/V2/V3 连同 HTML 衍生视图已移入 `archive/reports/`，逐字节保留，不删除负结果证据。

### 5.1 口径张力（重要，需用户裁决）

**方向 I 基线所属的反事实配对路线，在 V4 中被整批 `ELIMINATED`；而用户本次要求把该基线更新为 v0.6.1 完整版。** 这不是执行错误，而是两条历史线的并存：

- V1–V4 的实测与裁决发生在「ASU 温度自蒸馏 + 反事实配对 stage0-14 H20」这条线，结论是反事实配对构造 forget 目标 + 表示层正交/对齐 + 梯度手术系统性失败。
- L7 的 v0.6.1 基线是对**反事实配对方案本身**的重新设计（去 donor-KL、改答案跨度中性约束、`plan-only`、明确声明"历史 pilot confirmation 未通过、旧结果不作正向证据"），属于另一条仍在构思阶段的线。

本报告的处理：**按用户指令更新并保留基线**，但不在整合报告里把基线写成"已验证有效"，也不因 V4 的 ELIMINATED 删除基线。两份旧副本归档、负结果证据保留、6 个候选 proposal 的 §7 都标注了与基线的关系。后续若要合并两条线，建议由用户决定：是把 v0.6.1 当作"反重复记忆之外的新假设重新过 G0–G4 预检门"，还是正式归档整条反事实配对路线。

**2026-11-25 追加（历史记录，写于 (A) 裁决当时；口径张力已由同日裁决 2 解决，见下一段）**：用户已就 §2.2.1 方法身份裁决 (A)（配对重回主梯度来源，裁决记录见基线 §2.2.2）。(A) 使配对重回主梯度，与 V4 `ELIMINATED` 及 `research-wiki/log.md` 的「禁止复现」更直接冲突；当时口径张力尚未裁决，故规定在决定是否重新过 G0–G4 预检门之前不得投入算力，也不得把历史 stage0–14 结果改写为正向证据。裁决 2（归档）后该禁令不仅保留，且升格为「不投入算力」。

**2026-11-25 裁决（2 · 归档，用户裁决）**：口径张力已裁决完毕——**正式归档整条反事实配对路线**。理由：即便 §2.2.1 已选 (A)，配对重新作为主梯度来源仍落在 V4 `ELIMINATED` 与 `research-wiki/log.md:10` 反重复记忆所禁止的「反事实配对构造 forget 目标」机制族内；去 donor-KL、答案跨度与中性约束不足以构成解除禁令的「新机制」。

处置：
- `方向I_基线_v0.6.1_answer_span.md`（含 §2.2.2 的 (A) 裁决记录）**保留但不删除**，转为**仅供存档的设计文档**：不作为实现规格、不进入实验队列、不预注册、不投入算力；§2.2.2 的 6 条强制设计义务**停止执行**，仅作历史设计资料。
- §2.2.1 的 (A) 裁决记录保留为历史；其「§0/§7 不重写」结论不再适用，按归档决定在 §0/§7 补写归档说明。
- stage0–14 结果、V1–V4 负结果证据与两份旧基线备份**全部保留**，不得改写为正向证据，也不得作为 confirmation 复用。
- 后续 ACTIVE 主线**仅**为本报告 §4 / `docs/research_contract.md` 的 6 个 ASU 温度线候选（R2-B、R2-C、R3-2、R2-A、R3-1、R3-3）；会话恢复时不得从方向 I 基线读取 ACTIVE 候选。
- 反重复记忆继续有效、不解除、不例外化；本裁决不改变编排器状态（`init` 仍为 `BLOCKED`），不构成 `accepted`、评审回执或 claim 证据。

**2026-11-26 用户裁决（重新开放方向 I，覆盖裁决 2 的执行含义）**：用户明确要求「重新开放方向 I」，对 `workspace/idea-stage/IDEA_CANDIDATES.md` 的 D1–D10 依次进行实验验证与复现。

处置：
- 裁决 2 的「不投入算力」与 `research-wiki` 的 `anti_duplication` 不再禁止**基于新机制**的方向 I 实验；但「stage0–14 历史结果不得复用为正向证据」继续有效。
- 新建 `workspace/experiments/direction_i_v061/` 独立实现线，与已证伪的 stage0–14 物理隔离；只实现 v0.6.1 的最小**代理**引擎（answer-span `S_f`、teacher-force `X⁺⊕y⁺`、逐层激活、patch-forward、中性 hinge），不声称实现了 §4.2 的完整校准可行集。
- 10 个 idea 的证据等级：`proxy-v061`、pilot skip；D2–D6/D8/D9 为纯前向诊断，D1/D7/D10 为 60-step 训练变体。均不构成 `accepted` 或 claim 证据。

## 6. 证据门与评审状态（如实记录）

| 项 | 状态 |
|---|---|
| 跨模型 novelty-check / research-review | `REVIEW_UNAVAILABLE`（无 `.aris/reviewer.json`，Codex/Gemini/Grok/Claude-review CLI 均不可用）；不伪造 PASS、不记 `accept` 回执 |
| V1 报告自带 Evidence Gate | `BLOCKED`（novelty-check / research-review receipt 缺失），已在 `archive/reports/IDEA_REPORT_V1.md` 内如实保留 |
| 既有 `workspace/.aris/runs/*.json` | 历史运行态证据，未改动 |
| 本次规整 | 结构变更 + 文档整合，不新增实验、不新增评审裁决 |
| 所有候选 | `PROVISIONAL / plan-level`；新结果须过 `/experiment-audit` + `/result-to-claim` |

**不符合技能完整流水线之处（透明披露）**：标准 `/idea-discovery` 为 `research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline` 并带 pilot 与证据门。本次用户指令是**对既有 idea-stage 做归档/候选规整与基线更新**，不是启动新一轮从零发现；因此未重跑文献检索、未跑 pilot（本机无 GPU）、未伪造跨模型评审。6 份候选 proposal 与本报告是对既有成文产物的**忠实整合**，而非新评审结论。若需把某候选升级为 submission-ready，请单独启用评审后端与 `/experiment-bridge`。

## 7. 下一步（checkbox，防止静默跳过）

- [x] 用 v0.6.1 源文件替换方向 I 基线，两份旧副本归档
- [x] 历史 V1–V3 报告与 HTML 衍生视图归档，逐字节保留
- [x] 6 个未淘汰候选各写一份独立 proposal（思路 / 实现 / 故事 / 差异化 / 证据 / 失败模式 / 基线关系）
- [x] 淘汰 ideas 归档并保留反重复记忆（本次规整时未改动 research-wiki；2026-11-25 归档裁决已向其 log.md 追加一条）
- [x] 写整合报告 `IDEA_REPORT.md` 与 `archive/README.md`
- [x] 更新 `docs/research_contract.md` 指向新结构（会话恢复契约）
- [x] （2026-11-25 用户裁决）§5.1 口径张力：**已选 2 · 归档**；整条反事实配对路线正式归档，v0.6.1 基线仅作存档文档；裁决说明见 §5.1，后续 ACTIVE 主线仅 6 个 ASU 温度线候选
- [ ] （需环境）启用跨模型评审后端，把 6 个候选从 `provisional` 升级；补 E2/E3/E4 义务
- [ ] （需 GPU）按候选 proposal 的判别实验与失败模式跑 pilot，结果写新 `runs/<run_id>/`，不覆盖源数据
- [x] （2026-11-25）R2-B 3-seed 识别门 FAIL：等均值 τ=2.25 常数 3/3 seed 优于 `low2high`（1.857 vs 1.449）；与 R2-A、R3-1 一并归档至 `archive/candidates/`，`research-wiki/log.md` 已追加记录
- [x] 提交并推送远程 `origin/Auto/20261122`（commit `b0761f5`，2026-11-25 初版规整）
- [x] （2026-11-25 追加）修基线显性 bug：§2.2 前向输入与激活/ΔA 记号统一为答案跨度口径；文件头与整合报告同步更正「逐字一致」声明
- [x] （2026-11-25 追加）在基线新增 §2.2.1，把 ΔA 构造未定稿与方法身份张力标注为待裁决开放问题（未选机制、未改故事）
- [x] （2026-11-25 用户裁决）§2.2.1 方法身份张力：**已选 (A)** 让 ΔA/A⁻ 重回主梯度来源；裁决记录见基线 §2.2.2；其「§0/§7 不重写」与后续义务已被裁决 2（归档）覆盖，§0/§7 已补归档说明
- [x] （2026-11-25 裁决 2 处置）(A) 的后续设计义务**按归档决定停止执行、不再作为未完成项**；记录保留于基线 §2.2.2 作为历史设计资料
- [ ] （2026-11-25 修订）提交并推送本次基线修订到 `origin/Auto/20261122`

## 8. 来源清单

- 用户指定源文件：`/Users/chenyliao/MuRAG/Papers/L7_ASU创新/I/方向I_事实配对路线_重新设计Proposal_20261121.md`
- 当前分支成文裁决：`workspace/idea-stage/IDEA_REPORT_V4.md`、`docs/research_contract.md`、`NARRATIVE_ASU_TEMPERATURE_DEPTH.md`
- 历史轮次：`archive/reports/IDEA_REPORT_V1.md`、`_V2.md`、`_V3.md`
- 反重复记忆：`research-wiki/log.md`、`research-wiki/ideas/*.md`、`workspace/ideas_archive/README.md`
- 脚本依赖（决定 V4 保留顶层）：`workspace/verification/prepare.py`、`workspace/auto_experiments/run_all.py`、`workspace/verification/program.md`
- 文献标识沿用各源报告（ASU arXiv:2603.01285、RepSelect arXiv:2606.17168、PrivUn arXiv:2604.22076、ADU arXiv:2608.23020、ALMPU DOI 10.1609/aaai.v40i23.39017、FALCON NeurIPS 2025、CiPO arXiv:2604.15847、PACT arXiv:2609.38909 等）；本报告未做自动 citation-audit，标识与引用语境以源报告为准。

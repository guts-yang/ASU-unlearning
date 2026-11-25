# 归档说明（idea-stage/archive）

归档时间：2026-11-25。遵循 `ideas_archive/README.md` 的 `copy2` 精神：**原内容逐字节保留，不删除、不改写结论**。本报告区归档的是"被迭代取代或已判定淘汰"的方向 I 文档，供反重复记忆与审计追溯，不再是当前基线或候选。

## 口径来源

淘汰 / 候选判定以当前分支成文裁决为准，执行者不自行改判：

- `IDEA_REPORT_V4.md` 决策表（PROMOTE / REVISE / ELIMINATED）
- `docs/research_contract.md`（E1–E5 义务、核心主张）
- `research-wiki/log.md`（`anti_duplication` 反重复记忆，2026-11-23）

## baseline_versions/（方向 I 基线的历史版本，已被取代）

| 文件 | 原路径 | 版本 | 被取代原因 |
|---|---|---|---|
| `proposal_v0.5_20261122.md` | `workspace/proposal/方向I_事实配对路线_重新设计Proposal_20261121.md` | `v0.5-reference-draft` | v0.6 §1 判定其 $\mathcal L_{KL}$ 是 donor 答案替换/迁移目标而非删除目标，与 §6 冲突 |
| `proposal_v0.6_truncated_20261124.md` | `workspace/docs/方向 I：反事实主体配对驱动的逐层渐进遗忘 Proposal.md` | `v0.6-corrected` | 文件仅 93 行、截断在 §3.3 中途，且未含 v0.6.1 的 answer-span 裁决 |

当前唯一权威基线：`../方向I_基线_v0.6.1_answer_span.md`（源自 L7 `I/方向I_事实配对路线_重新设计Proposal_20261121.md`，`factpair-progressive-alignment-v0.6.1-answer-span`，正文逐字保留）。

## reports/（被迭代取代的历史报告与 HTML 衍生视图）

| 文件 | 原路径 | 处置原因 |
|---|---|---|
| `IDEA_REPORT_V1.md` + `.html` | `idea-stage/IDEA_REPORT.md` / `.html` | 第 1 轮（Δ-Gated 正交对齐）；V2 实测证伪其推荐 Idea 1 |
| `IDEA_REPORT_V2.md` + `.html` | `idea-stage/IDEA_REPORT_V2.md` / `.html` | 第 2 轮（转向 ASU-Selective）；结论已被 V3/V4 吸收与改判 |
| `IDEA_REPORT_V3.md` + `.html` | `idea-stage/IDEA_REPORT_V3.md` / `.html` | 第 3 轮（SensiTemp 等）；V4 实测把 SensiTemp 降为诊断 |
| `NARRATIVE_ASU_TEMPERATURE_DEPTH.html` + `.review.json` | 同名顶层文件 | HTML 仅为衍生阅读视图；MD 主源 `../NARRATIVE_ASU_TEMPERATURE_DEPTH.md` 仍为当前有效叙事，保留顶层 |

**保留在顶层、不归档**：

- `IDEA_REPORT_V4.md`：当前有效决策报告；被 `workspace/verification/prepare.py`、`workspace/auto_experiments/run_all.py`、`workspace/verification/program.md` 硬编码引用，移动会破坏脚本。
- `NARRATIVE_ASU_TEMPERATURE_DEPTH.md`：当前有效叙事（Markdown 唯一真实来源）。
- `docs/research_contract.md`：W1→W1.5 会话恢复契约。

## candidates/（2026-11-25 归档的候选 proposal）

按 2026-11-25 用户要求与 3-seed 实测，把已降级候选从 `../candidates/` 移入此处。正文逐字节保留，仅更新 front-matter 的 `status` / `archived_at` / `archived_reason`。

| 文件 | 原状态 | 归档原因 |
|---|---|---|
| `R2-B_ASU温度课程_tau_k.md` | PROMOTE | 3-seed 识别门 FAIL：等均值 τ=2.25 常数 3/3 seed 优于 `low2high`（均值 dF 1.857 vs 1.449，配对差 +0.407）；收益由平均平滑水平解释 |
| `R2-A_ASU_Selective_表示选择性.md` | REVISE | research-wiki 已标 `archived/negative`；RepSelect/ADU/ALMPU 占据该空间，E4 未做 |
| `R3-1_SensiTemp_层头自适应温度场.md` | REVISE | 等平均 τ 常数对照 1.792 > sens 场 1.747；收益来自更强总平滑而非结构化分配 |

保留在 `../candidates/` 的现役候选：`R2-C`、`R3-2`（PROMOTE）、`R3-3`（REVISE，仅作消融）。

## 被淘汰 ideas（ELIMINATED，禁止复现）

V4 决策表唯一一条 `ELIMINATED` + research-wiki 反重复记忆：

- **R1-I1/I2/I3/I4/I5 反事实配对 + ΔA 正交/对齐/差值引导（全系）**：stage0-14 H20 实测，所有变体 forget/retain 均弱于原 ASU ref（ref 0.555/0.798；idea1 1.068/1.52）；stage8/9/10/11/13/14 失败，stage5 mask 无收益；`KILL`。转为反重复记忆，禁止后续轮次复现。
- 关联淘汰（V2/V3 已明确写入报告）：直接对齐 $A^-$（donor 替换）、PCGrad/EWC/GradNorm/GA-orth 梯度手术、单纯 mask / $\lambda\beta$ 调参 / 反事实渐进 schedule、ASU-Selective 与 RepSelect 式子空间（差异化不足）。

> 归档不代表删除研究价值：上述负结果是 `NARRATIVE_ASU_TEMPERATURE_DEPTH.md` 与 research-wiki `anti_duplication` 记忆的核心证据。详情见整合报告 `../IDEA_REPORT.md` 的"淘汰 ideas"一节。

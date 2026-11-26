---
id: IDEA-DISCOVERY-20261126
title: 可解释反事实激活差驱动的 LRM 机器遗忘 —— Idea Discovery + /idea-evaluator 评估
date: 2026-11-26
trigger: "/idea-discovery 调研 + /idea-evaluator 评估", 用户指定基线 方向I_基线_v0.6.1_answer_span.md
baseline: workspace/idea-stage/方向I_基线_v0.6.1_answer_span.md (factpair-progressive-alignment-v0.6.1-answer-span, plan-only)
mode: 构思与规划, pilot skip, 无 GPU, 不训练, 不部署, 不改 configs/, 不读取 .env
verdict_type: 执行者机制推断分(PROVISIONAL), 非跨模型独立评审回执, 非实测成绩
---

# 可解释反事实激活差驱动的 LRM 机器遗忘 —— Idea Discovery + /idea-evaluator 评估

> **诚信声明**：本轮为文献调研 + 构思 + 机制评估。所有 idea 的论文级 claim 一律「待验证」；所有分数为 `/idea-evaluator` 的**机制推断分**（标注 `mechanism-based, not yet confirmed by data`），并各附**决定性验证实验**；不编造任何实测数字、BibTeX、arXiv 身份或评审回执。跨模型评审后端本机不可用（`REVIEW_UNAVAILABLE`），`novelty-check = PROVISIONAL`，`evidence gate = BLOCKED`。执行者判断不替代 reviewer 或 integrity 裁决。

## 0. 执行摘要

用户诉求：把 ASU（ICLR 2026）的「注意力温度 forget-teacher」替换成**可解释**机制——Teacher 吃**正反事实对** $(X^+,X^-)$，Student 依据二者在**中间层激活向量的差异** $\Delta A$ 渐进抹去差异，实现「既遗忘又保通用效用」。动机是温度 $\tau$ 不可解释、需 binary search、无「删了什么」的语义。用户点明：**激活向量的处理是主要构思方向**。

本轮按 `/idea-discovery`（research-lit → idea-creator → novelty-check → research-review → refine）编排，用 `/idea-evaluator`（八步 + 五维 + 三档判决）做达标门控：**生成 13 个 → 淘汰 3 个 → 10 个达标候选**（五维平均分均 $\ge 7.2 > 7$，裁决全部 `Strong Accept` 或 `Accept with Revisions`），给出排名，并在 `candidates/` 下为 10 个各写一份带 LaTeX 公式的专属 proposal。

三条被淘汰的 idea 恰好落在项目历史已证伪/近邻已占据的机制族，体现「只有分数达标才作为候选」：朴素 $\Delta A$ 趋势外推（`data-refuted` CRITICAL）、直接对齐 $A^-$ / donor-KL（基线 §1 判死）、SAE 单特征 clamp 持久化（F1 近邻占据）。

## 1. Literature Landscape（真实检索，2026-11-26）

检索 6 组、覆盖 arXiv/Scholar/会议页。**以下文献仅用于近邻划界，身份未做 `verify_papers.py` D1 三层核验，不作可引用 BibTeX**；关键论点标注来源，未核实处按「据不完全证据显示」处理。

**A. 基线与其不可解释缺陷**
- **ASU / Attention Smoothing Is All You Need For Unlearning**（ICLR 2026, arXiv:2603.01285, Zade et al.）：$\alpha^{(\tau)}_{ij}=\mathrm{softmax}(q_ik_j^\top/(\tau\sqrt d))$，用升温后的冻结模型作 forget-teacher 自蒸馏。论文自承温度选择脆弱、需 binary search，并把 dynamic/adaptive temperature 列为 future work（`arxivlens` 摘要、GitHub `Salehzz/ASU-unlearning`、ICLR virtual poster 10007045）。**这正是用户要替换的对象**：$\tau$ 只是标量旋钮，没有「删了哪条事实」的可解释语义。

**B. 反事实激活差的直系鼻祖（含关键风险证据）**
- **CAA / ActAdd / RepE**：$v=\mathrm{mean}(h^+)-\mathrm{mean}(h^-)$，对比提示求差向量、推理时加到残差流（`ossaihub`、`mbrenndoerfer`、`lacuna`）。**用户「正反事实对激活差」的数学原型**，但它是 inference-time、非持久、不内化到参数。
- **Spherical Steering**（ICML 2026, arXiv:2602.08169）：沿超球面测地线旋转而非线性加，指出加法 steering 扭曲 $\lVert h\rVert$、破坏 RMSNorm 几何先验。
- **Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences**（ICLR 2026, arXiv:2510.13900）：ADL 发现微调前后激活差 $\delta_{\ell}=h^{ft}_\ell-h^{base}_\ell$ 编码微调目标语义（可解释 agent 识别成功率 91%）。**直接支持「激活差编码目标事实」这一前提**。
- **Field & Yampolskiy, SAE Feature Unlearning is Shallow**（ICML 2026 MechInterp Workshop）：抑制单特征只改「说什么」不改「知道什么」，理解力/MMLU 不变。**头号 killer risk：浅层抑制 ≠ 知识擦除**。
- **Zhang & Nanda (ICLR 2024) / ROME causal tracing (Meng et al. 2022)**：activation patching 定位事实到中层 MLP；但噪声扰动/评分口径会制造 **OOD 伪定位峰**（`aimap`、`zffjzz`）。

**C. 可解释 / SAE 特征级遗忘（最拥挤近邻）**
- **CRISP**（ACL 2026, arXiv:2508.13650）：target/retain 对比选 SAE 特征 + LoRA 把特征固定到 0，持久概念遗忘。
- **SCALPEL / Neuralyzing the Trace**（arXiv:2609.31056）：anchor-contrastive SAE，指出重构提取器有 energy bias、偏向背景结构，选 forget-specific 特征后减去其贡献（TOFU, Qwen/Llama/Gemma）。
- **ARIA / Test-Time Unlearning via SAE**（arXiv:2609.16229）：SAE 稀疏线性头做 on-policy 门控，含「泄漏/效用由检测器错误率上界」定理，覆盖 **DeepSeek-R1-1.5B（LRM）**。
- **CLReg / Contrastive Representation Shaping**（NeurIPS 2026, arXiv:2601.22028）：同例双视图正对、retain 负对，显式声明「表示分离本身不等于知识删除」。
- **UDS / Measuring Depth via Activation Patching**（EMNLP 2026）：跨模型 patch 定位知识编码层 KE、遗忘深度评分（评估/证书侧）。
- **MUtE**（2026）：概念擦除 + 反事实平移偏差的最优界（连续表示层、非 LLM 参数内化）。

**D. LRM 推理轨迹遗忘 / 反事实目标（输出或文本层）**
- **CPSD / Contrastive On-Policy Self-Distillation**（arXiv:2611.31932）：冻结参考模型对比 plain vs privileged context 的分布差，反转成反事实目标分布，沿 Student 自身轨迹蒸馏。**极近邻：反事实 + 对比 + 冻结 Teacher + on-policy，但作用在输出 token 分布层，非中间层激活差**。
- **CiPO**（ACL 2026, arXiv:2604.15847）：LRM 反事实 CoT + 迭代偏好优化；**STaR**（AAAI 2026, arXiv:2601.09281）、**REDaCT**（arXiv:2610.12769）、**R2MU**、**Selective Forgetting for LRMs**（arXiv:2604.03571）：推理轨迹/答案层，检测-替换-占位符或偏好优化。
- **D2F / Decouple to Forget**（arXiv:2610.17654）：semantic mediator 抽事实级表示，删最小敏感事实集生成反事实答案 + span-masked DPO。

## 2. 空白与差异化轴（一句话）

据现有资料显示，2026 年可解释遗忘分别落在 **SAE 命名特征空间**（CRISP/SCALPEL/ARIA）、**输出分布层**（ASU/CPSD）、**CoT 文本层**（CiPO/STaR/D2F）或**推理时非持久 steering**（CAA/ActAdd）；而「**反事实配对 $(X^+,X^-)$ 在中间层答案/推理跨度上的激活差 $\Delta A$ → 净化/反演成遗忘目标 → Student 参数渐进内化，并强制因果验证 + 效用锚定 + 非 donor 中性约束 + 抗恢复证据**」这一轴少有人系统执行。基线 v0.6.1 已把舞台搭好（答案跨度 $S_f$、teacher-force、Teacher stopgrad、中性约束、$\lambda_{KL}^{\text{ablation}}=0$、§6 六条验收），但 §2.2.1 明确「$\Delta A$ 构造未定稿」——**这正是用户要填的空白，也是本轮 10 个 idea 的统一落点**。

## 3. 评估方法（/idea-evaluator 冻结口径）

按本机 `~/.bg-agent/config-with-app/skills/idea-evaluator/SKILL.md` 及其 references：

- **八步**：①定位+一句话故事 ②致命缺陷审计(F1–F10，先于打分) ③生命周期/能力匹配 ④五维打分 ⑤范式转移四问 ⑥可行性四风险 ⑦诚实性闸门(7 查) ⑧三档判决。
- **五维**：Higher / Faster / Stronger / Cheaper / Broader，**各维 1–10、起评 5、逐分给证据**；高分可为 `mechanism-based, not yet confirmed by data`，但必须命名决定性验证实验（five-dimensions.md §10/§11）。
- **平均分**：本报告取五维算术平均 $\bar S=\tfrac15(H+F+S+C+B)$；门槛 $\bar S>7$（用户要求「平均分 7 分以上」，本轮全部 $\ge 7.2$）。
- **三档判决**：`Strong Accept`（≥2 维 8+、零 CRITICAL、能力匹配绿）、`Accept with Revisions`（维度偏弱/可修复缺陷/≥2 MAJOR）、`Reject and Pivot`（CRITICAL 或 >1 不可修复缺陷）。用户门槛：**裁决 ≥ `Accept with Revisions`**。
- **致命缺陷短路**：`data-refuted` 核心机制被自有数据/基线/简单对照驳倒 = 自动 CRITICAL，锁死 Reject and Pivot，不打分、不写 defense（fatal-flaws.md §6）。项目 `research-wiki/log.md` 的反重复记忆（V1 全系 $\Delta A$ 抑制 stage0–14 证伪、2026-11-26 重新开放但历史结果不得复用为正向证据）据此执行。
- **诚实性闸门**：每维有证据、可行性引用用户资源、新颖性标注 `unverified / literature check required`、命名 ≥2 失败模式、裁决与分数一致；全部分数标 PROVISIONAL。

## 4. 生成 → 筛选（13 → 3 淘汰 → 10 达标）

| 被淘汰 | 致命缺陷 / 依据 | 裁决 |
|---|---|---|
| X1 朴素 $\Delta A$ 趋势外推直接抑制 Student 激活 | `data-refuted` CRITICAL：等价 V1 全系「沿 $\Delta A$ 抑制/对齐 $A^-$」，stage0–14 H20 实测 forget/retain 均弱于原 ASU ref（ref 0.555/0.798 vs idea1 1.068/1.52），`Gate=KILL`，`research-wiki/log.md` 反重复记忆禁止复现 | **Reject and Pivot**（短路，不打分） |
| X2 直接向 $A^-$ 对齐 / donor-KL soft target | CRITICAL：基线 §1 + v0.5 错误链条判定为 donor 答案替换（非删除），fuxian 复现提前 EOS $0.02\to0.46$、完整回答率 $0.98\to0.54$、G4 FAIL | **Reject and Pivot** |
| X3 SAE 单特征 clamp 持久化 | F1 近邻占据：CRISP(ACL2026)/SCALPEL 已占 SAE 特征持久遗忘；Field&Yampolskiy 证明 shallow；五维 $H5/F6/S4/C6/B5=5.2<7$ | **Reject and Pivot** |

淘汰项印证了致命缺陷审计的约束力：本轮 10 个达标 idea **全部带新机制**（因果验证、效用锚定、约束反演、表示域重定义、抗恢复投影、协议化等），严格避开「沿 $\Delta A$ 抑制 / 对齐 $A^-$ / 教 donor 答案 / 梯度手术 / 单纯 schedule-mask」这一已证伪族。

## 5. 排名表（10 个达标候选）

分数为执行者**机制推断分**（PROVISIONAL）；平均分均 $>7$；裁决全部 ≥ `Accept with Revisions`；并列按「信息价值 / 最小实验可判别性」tie-break。每份 proposal 内附完整五维雷达、MAJOR/MINOR 缺陷、决定性验证实验。

| 名次 | ID | Idea（唯一改动点） | H | F | S | C | B | 平均 $\bar S$ | 裁决 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CF-01 | 约束反演目标 $\Delta^*$（因果+效用+身份三约束逆问题） | 9 | 7 | 9 | 7 | 8 | **8.0** | Strong Accept |
| 2 | CF-03 | 因果门控 $\Delta A$（patch 加权因果分量） | 8 | 8 | 8 | 7 | 8 | **7.8** | Strong Accept |
| 3 | CF-02 | 三前向角色分解（删什么/别变成什么/保什么） | 8 | 7 | 8 | 7 | 8 | **7.6** | Strong Accept |
| 4 | CF-04 | 事实/身份双探针正交解耦 $\Delta A_{\text{fact}}$ | 8 | 7 | 8 | 7 | 7 | **7.4** | Strong Accept |
| 5 | CF-06 | LRM 推理轨迹跨度 $\Delta A_{\text{trace}}$（CoT 逐 step） | 8 | 6 | 8 | 6 | 9 | **7.4** | Strong Accept |
| 6 | CF-05 | 多 donor 边际化共有子空间 $\Delta A_{\text{marg}}$ | 7 | 7 | 8 | 7 | 8 | **7.4** | Accept with Revisions |
| 7 | CF-08 | 零空间 / 抗恢复投影 $\Delta A_{\text{null}}$ | 7 | 7 | 9 | 6 | 8 | **7.4** | Accept with Revisions |
| 8 | CF-07 | 读出空间 $\Delta A_{\text{readout}}$（旋转不变） | 7 | 7 | 8 | 7 | 7 | **7.2** | Accept with Revisions |
| 9 | CF-09 | 结构化对照族残差化 $\Delta A_{\text{resid}}$ | 7 | 8 | 7 | 7 | 7 | **7.2** | Accept with Revisions |
| 10 | CF-10 | $\Delta A$ 纯度驱动渐进 $\Lambda(k)/\rho_\ell(k)$ | 7 | 7 | 7 | 7 | 8 | **7.2** | Accept with Revisions |

> 机制分高分统一附 `worth pursuing, pending the validation experiment`；Strong Accept 的 5 个各带 1 个 MAJOR + 若干 MINOR，Accept with Revisions 的 5 个各带 2 个 MAJOR（详见各 proposal §5）。

## 6. 10 个候选共同遵守的基线硬边界（每个 idea 均不违反）

- 反事实配对：$X^+=X(s_f,r),\ X^-=X(s_r,r)$，槽外逐字一致、token 等长；$X^-$ 只作对照/定位，**不**把 $p_{\theta_0}(y|X^-)$ 当 forget 目标。
- Teacher 冻结 $\theta_0$，前向 teacher-force 完整序列 $X^+\oplus y^+$ 与 $X^-\oplus y^+$，答案跨度位置 $p\in S_f$ 取 $A^+_{T,\ell}(p),A^-_{T,\ell}(p)$；$\Delta A_{\ell,p}=A^+_{T,\ell}(p)-A^-_{T,\ell}(p)$ 一律 stop-gradient。
- $\mathcal L_F$ 只作用于 $S_f$；prompt 末位不进；遗忘梯度只经 Student 侧 $\mathcal L_F$ 回传。
- 目标 $T_{\ell,p}$ 满足 §4.1 非 donor 中性约束：$r_{f,p}\le c,\ s_{d_j,p}\le d_j,\ \lVert\Delta r_{E,p}\rVert\le e$；不拼 $y^-$ / `donor_fact_value`，不算 $\mathrm{logit}(y^+)-\mathrm{logit}(y^-)$。
- 读出经冻结下游 patch-forward 重算（N1 ReadoutTarget 式），不把中间层向量直接过 LM head。
- 主结果在 $\lambda_{KL}^{\text{(ablation)}}=0$ 下成立；统计单位作者/实体簇；§6 六条验收（配对特异性、等预算渐进收益、非答案替换、参数内化、非浅层抑制、主结果独立性）。
- 历史 stage0–14 结果不得复用为正向证据；新结果须过 `/experiment-audit` + `/result-to-claim`。

## 7. Proposal 索引（candidates/）

`CF-01_约束反演目标.md` · `CF-03_因果门控_activation_gate.md` · `CF-02_三前向角色分解.md` · `CF-04_事实身份双探针解耦.md` · `CF-06_LRM推理轨迹_deltaA.md` · `CF-05_多donor边际化.md` · `CF-08_零空间抗恢复投影.md` · `CF-07_读出空间_deltaA_readout.md` · `CF-09_结构化对照族残差化.md` · `CF-10_纯度驱动渐进schedule.md`

每份统一结构：0 故事 → 1 问题空白 → 2 假设 → 3 实现（含 LaTeX 公式）→ 4 差异化/近邻划界 → 5 `/idea-evaluator` 评估（五维表 / 平均分 / 裁决 / 缺陷 / 决定性验证实验）→ 6 判别实验·失败模式·必测指标 → 7 与基线关系 → 8 诚信声明与来源。

## 8. Evidence Gate 与评审状态（如实记录）

| 阶段 | 状态 | 说明 |
|---|---|---|
| research-lit | done（composed） | 本文件 §1，6 组 WebSearch；文献身份未做 D1 三层核验，不作 BibTeX |
| idea-creator | done（composed） | 13 生成 → 3 淘汰 → 10 达标，评分表见 §5 |
| novelty-check | PROVISIONAL | 执行者基于公开文献做机制划界，非跨模型独立评审；无 accept 回执 |
| research-review | REVIEW_UNAVAILABLE | codex/copilot/claude/gemini/grok/knot-cli 与 `.aris/reviewer.json` 均不可用；执行者自审不构成独立评审 |
| refine-pipeline | 未生成 | 用户诉求为「10 个候选 + 排名 + 各自 proposal」，非 top-1 FINAL_PROPOSAL；且证据门 BLOCKED |
| evidence gate | BLOCKED | 缺 `run_state.py` / `idea_discovery_gate.py` 解析链 + 缺 2 份跨模型回执；不呈现为完成 |

未训练、未跑 GPU pilot、未部署、未改 configs/ 与 requirements、未重训 baseline、未读取 .env 值、未伪造 verdict-id / reviewer 模型名 / accept 回执、未访问其它线路。执行者判断未替代 reviewer 或 integrity 裁决。

## 9. 进入 /experiment-bridge 前仍需解除的阻塞项

- 启用至少一个异族 reviewer 后端并取得真实身份回执，把 novelty-check / research-review 从 PROVISIONAL / REVIEW_UNAVAILABLE 升级；
- 恢复 RESUMABLE helper（`run_state.py` / `idea_discovery_gate.py`）以执行确定性 evidence gate；
- 冻结 $S_f$ / 推理轨迹 offset 定义、$\Delta A$ 净化方式、$\Lambda(k)/\rho_\ell(k)$、$c,d_j,e,\delta,\varepsilon$、投影秩、patch 层集、donor 簇隔离规则；
- 落实 frozen baseline 数据/模型软链与 heavy_data_root，确认两套指标口径隔离（官方 TOFU 8 指标 vs U1–U7 / C1–C11 gate 不混用不换算）；
- 所有依赖 pilot 的判别实验标 `needs manual pilot`；新结果须过 `/experiment-audit` + `/result-to-claim`；
- 对抗「浅层抑制」头号风险：必须给出至少两类独立抗恢复证据（relearning / 量化 / probe / patch）方可使用受限的「深度遗忘」表述（呼应 Field & Yampolskiy 2026）。

## 10. 来源

- 基线：`workspace/idea-stage/方向I_基线_v0.6.1_answer_span.md`（§1/§2.2/§2.2.1/§2.3/§4.1/§5/§6）
- 项目裁决/反重复记忆：`IDEA_REPORT.md` §5.1、`docs/research_contract.md`、`research-wiki/log.md`（2026-11-26 重新开放方向 I）、`IDEA_CANDIDATES.md`(D1–D10)、`IDEA_INVERSE_TARGET.md`(R1–R6)
- 评估框架：本机 `~/.bg-agent/config-with-app/skills/idea-evaluator/SKILL.md` + references/{fatal-flaws, five-dimensions, lifecycle-capability-matching, worked-examples}.md
- 冻结评分口径记忆卡 `mem_d3899b_55b1365`（S=0.30I+...，本轮按用户指定改用 /idea-evaluator 五维平均）
- 文献（WebSearch 摘要，身份未独立核验，不作 BibTeX）：arXiv 2603.01285 / 2602.08169 / 2510.13900 / 2508.13650 / 2609.31056 / 2609.16229 / 2601.22028 / 2611.31932 / 2604.15847 / 2601.09281 / 2610.12769 / 2610.17654 / 2604.03571；ROME (Meng 2022)、Zhang & Nanda ICLR 2024、CAA/ActAdd/RepE、UDS (EMNLP 2026)、Field & Yampolskiy (ICML 2026 Workshop)

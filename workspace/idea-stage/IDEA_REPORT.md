# Idea Discovery Report — 反事实主体配对的逐层渐进遗忘

**Direction**: 以 ASU-unlearning（ICLR 2026《Attention Smoothing Is All You Need》）为基座，调研"反事实主体配对 + 逐层渐进激活对齐"的遗忘新方向，重点解决 workspace 方向 I proposal 的核心待调研问题：**配对激活差异 ΔA 应如何构造遗忘目标**。
**Date**: 2026-11-23
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline
**Output root**: `workspace/`（用户指定）
**Effort / Assurance**: balanced / draft（评审承载阶段因环境缺评审 MCP，见 §External Critical Review）

> 执行环境声明：本机 `torch.cuda.is_available()=False`、无 transformers/safetensors、checkpoint 不在本机（据不完全证据，参见项目记忆卡），**无法运行任何 Llama 前向 pilot**。本轮所有 "pilot signal" 均为**纸面机制推断 + 文献证据**，不是实测数字；按技能 PILOT 规则标注 `needs manual pilot`。跨模型评审后端（Codex/Gemini/Grok/Claude-review MCP）在当前宿主均未启用，故两个评审承载阶段标记 `REVIEW_UNAVAILABLE`，最终证据闸预期 `BLOCKED`——这是如实记录，不是流程失败隐藏。

## Executive Summary

基于 workspace 已有的方向 I proposal（反事实主体替换 Teacher + 逐层渐进激活对齐 + DKT 同前缀 KL），本轮调研产出 3 个候选 idea 并做了查新与逻辑审查。**推荐方案是 Idea 1（Δ-Gated 正交渐进对齐）**：不直接把 Student 激活对齐到 donor 激活 A⁻（该方案有"donor 答案替换 / context blindness"的逻辑缺陷），而是用配对差 ΔA 的主成分定义"事实子空间"，把 Student 在 X⁺ 上的激活沿该子空间做正交抑制，逐层渐进内化。最近的近邻是 FALCON（NeurIPS 2025，对比正交激活操纵）与 CiPO（ACL 2026，反事实偏好优化），但二者分别用随机对比锚、在输出/偏好层操作；本方向以"关系匹配的真实 donor"为对比锚、在表示层做不教假事实的正交解耦，差异化成立但窗口偏紧。核心 kill risk：ΔA 是否真正编码"目标事实"而非"主体身份"——需最小 probe 实验证实。**下一步（需 GPU）**：先做 ΔA 方向的 probe 判别实验 + 静态全层/渐进 schedule 等预算对照，二者任一不过即收缩或否决。

## Literature Landscape

按主题聚类（检索渠道：web_search 多轮，覆盖 arXiv/ACL/NeurIPS/AlphaXiv/机构库；检索词见附录 A）。Gemini CLI 未安装，按技能规则优雅跳过。

**聚类 1 — 反事实/替换式遗忘（与方向 I 最直接相关）**
- **CiPO: Counterfactual Unlearning for LRM through Iterative Preference Optimization**（arXiv:2604.15847, ACL 2026）：让 LRM 生成"逻辑自洽但内容相反"的反事实 CoT 作为偏好正样本，迭代 DPO/SimPO。证据：在 R-TOFU/RETURN 上同时清除 CoT 与答案、保住 GSM8K/MMLU； compares favorably to GA 与 R²MU。差异：它在**输出/偏好层**教模型生成反事实文本；方向 I 在**表示层**对齐，不教假事实。
- **On the Hidden Costs of Counterfactual Knowledge Training**（arXiv:2605.27083, Ye/Wang/Zhang/Wu, 2026-05）：**关键负面证据**。反事实调优（CFT）有两大隐性代价：(1) knowledge conflict——不一致的反事实语料诱导冲突梯度（自然数据梯度余弦 μ≈+0.136，无约束反事实 μ≈+0.024，近半样本对负相似）；(2) hallucination spillover——拟合假目标植入持续捏造偏见，抬高无关域幻觉率。引入 RWKU+ 诊断基准与梯度余弦诊断。**启示：方向 I 必须避开"教模型说假话"，且 donor 必须关系匹配、簇内一致，否则重蹈 CFT 覆辙。**

**聚类 2 — 逐层/表示级操纵与蒸馏（激活目标构造的方法源）**
- **FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment**（NeurIPS 2025, Hu et al.）：信息论引导参数选择 + 对比机制增强表示分离 + 冲突梯度正交投影。证据：抗恢复鲁棒。差异：对比锚为噪声/随机，方向 I 用真实 donor；FALCON 是全局重要性选择，方向 I 强调逐层渐进 schedule。
- **FDD / Beyond Logits: Aligning Feature Dynamics**（ACL 2025, Gong et al., DOI 10.18653/v1/2025.acl-long.1125）：把残差 Transformer 视为离散 ODE，监督"层间预测轨迹 KL + 相邻层 delta 余弦"，用 LM head 投影到共享词表空间。启示：为 Idea 3（delta 匹配）提供现成损失构造；另注意其消融显示"4 层采样最佳，过多层反而下降"——**直接质疑"全层对齐"的必要性，支持渐进/稀疏层选择**。
- **Layer-wise Distillation / LayerSync**（ICLR 2026）：stop-gradient 固定教师层、只更新学生层；层对选择应避开太浅与末端 20% block。启示：方向 I 的 A⁻ 必须 stopgrad（proposal 已做），层集合 Λ(k) 不应取全 16 层。
- **AlphaEdit**（ICLR 2025）/ **One Mask to Rule Them All**（2026-09）：把扰动投影到 preserve 知识的零空间 / 用稀疏 mask 恢复被 ROME-MEMIT 抑制的原事实——证明"抑制≠擦除"，**支持方向 I 必须做抗恢复审计**。

**聚类 3 — Teacher-Student 自蒸馏与鲁棒性（ASU 的同源方法）**
- **ASU**（Zare Zade et al., ICLR 2026, arXiv:2603.01285）：注意力温度 τ>1 平滑 → forget-teacher → 自蒸馏。方向 I 的出发点：去 τ 后 Teacher=Student 导致初始化零遗忘梯度。
- **MALTO / Dual-Teacher**（SemEval-2025 Task 4, SOTA）：Competent Teacher（保 utility）+ Incompetent Teacher（随机 logit，诱导遗忘），KL 自适应混合；非 ordered 训练更优。
- **Distillation Robustifies Unlearning / UNDO**（NeurIPS 2025, Lee et al.）：unlearn→distill 到随机初始化同构学生，可移除潜在能力、显著抗 relearning。**启示：可作为方向 I 部署前的抗恢复增强步骤。**
- **PACT**（arXiv:2609.38909, 2026-09）：**关键反面证据**——target-based 目标（把模型压到"无上下文生成的中性回答"）虽移除行为，但造成 context blindness（模型学会不读上下文），侵蚀系统提示遵循与秘密保持。启示：直接向 A⁻ 对齐有同类风险。

**Landscape 结论**：结构性空白 = "用关系匹配的真实 donor 作为对比锚、在表示层而非输出层做逐层正交解耦、且不教模型假事实的遗忘目标构造"。该空白与 CFT 的两大代价、PACT 的 context blindness 形成清晰对照，但 FALCON/FDD/CiPO 已是强近邻，窗口偏紧（估计 6-12 个月）。

## Ranked Ideas

**Idea 1 — Δ-Gated Progressive Orthogonal Alignment（推荐）**
- 假设：配对差 ΔA_ℓ = h_T(X⁺)−h_T(X⁻) 的主成分定义了一个"事实相关方向"；遗忘应把 Student 在 X⁺ 的层激活沿该方向正交抑制（保留正交分量），而非对齐到 A⁻。损失：对每层抽点，L_F = Σ_ℓ ρ_ℓ(k)·‖(I−P_Δ) h_S(X⁺;P) − stopgrad(h_T(X⁺;P))‖²，其中 P_Δ 为 ΔA 的低秩投影；或等价地用梯度屏蔽 g⊥=g−proj_Δ g。逐层渐进扩展 Λ(k)。
- 机制依据：FALCON 的正交投影 + AlphaEdit 零空间 + FDD 的分层稀疏；避开 CFT 幻觉溢出（不教假事实）与 PACT context blindness（保留正交/主体表征）。
- Pilot 信号：**needs manual pilot**（无 GPU）。纸面证据：强（三项独立文献支撑机制）。风险：ΔA 主成分是否 = 事实方向未证实（见 Novelty § killer risk）。

**Idea 2 — Counterfactual Difference Matching（backup）**
- 假设：不抑制、也不对齐 A⁻，而是约束 Student 的"差异结构"与 Teacher 一致：h_S(X⁺)−h_S(X⁻) ≈ stopgrad(A⁺−A⁻)，即匹配差值而非绝对位置。这样主体表征本身不动，只移除"事实相关差异方向"。
- 风险：X⁻ 的表示含 donor 事实，约束差值可能把 donor 信息引入 X⁺ 路径；需要 wrong-donor 对照。
- Pilot 信号：needs manual pilot；纸面证据：中（无直接先例，但与 FDD delta 匹配同源）。

**Idea 3 — Layer-wise Prediction-Dynamics Alignment（备选/可并入 Idea 1）**
- 假设：用 FDD 的 LM-head 投影，让 Student 在 X⁺ 的"层间预测 delta 方向"匹配 Teacher 在 X⁻ 的 delta 方向（余弦），而非对齐绝对激活。规避绝对对齐对表示的破坏。
- 风险：依赖 LM-head 投影到词表，架构受限；token 等长假设需验证。
- Pilot 信号：needs manual pilot；纸面证据：中高（FDD 直接方法源）。

**淘汰（见 Eliminated Ideas）**：Idea 4 = 原参考方案"直接向 A⁻ 对齐"；Idea 5 = "纯 ΔA 差值引导更新"。

**排序理由**：Idea 1 差异化最强且正面回应 CFT/PACT 两大负面证据；Idea 2 保守但需验 donor 泄漏；Idea 3 更像可并入 Idea 1 的损失变体。AUTO_PROCEED：选定 Idea 1 进入查新与评审。

## Novelty Verification

**状态：PROVISIONAL（基于公开文献的机制查新，非跨模型独立评审）**。未获得 `REVIEW_UNAVAILABLE` 之外的独立评审身份，故不能记为 cross-family accepted。

| 近邻 | 年份/ venue | 重叠 | 差异点 |
|---|---|---|---|
| CiPO | 2026/ACL | 反事实 + 遗忘 | 输出/偏好层、教假 CoT；本方向表示层、不教假事实 |
| FALCON | 2025/NeurIPS | 对比 + 正交激活操纵 | 随机对比锚、全局参数选择；本方向真实关系 donor、逐层渐进 |
| FDD | 2025/ACL | 逐层 + delta 蒸馏 | 用于压缩、教师更强；本方向用于遗忘、教师是同权重反事实视图 |
| CFT hidden costs | 2026/arXiv | 反事实训练代价 | 其结论是"反事实训练有代价"，本方向正是规避该代价的表示层方案 |
| PACT | 2026/arXiv | target-based 遗忘的 context blindness | 其 target 是无上下文中性回答；本方向目标保留主体/正交表征 |

**不同工作重合度评估**：Idea 1 与 FALCON 的"正交投影"字面重合，但对比锚语义与逐层 schedule 不同；需在论文中显式划界，否则 reviewer 会以"FALCON 换了个对比锚"拒绝。**查新结论 = PROCEED WITH CAUTION**（技能定义：caution 属正面裁决，可带条件推进）。

**Killer risk（决定存亡）**：若 probe 显示 ΔA 主成分主要编码"主体身份"而非"目标事实"，则 Idea 1 的正交抑制会变成"抹掉主体表征"，方案塌缩为 Idea 4 的变体并失败。这是必须先用最小实验排除的假设，不能用论证代替。

## External Critical Review

**状态：REVIEW_UNAVAILABLE** — 当前宿主未启用任何跨模型评审传输（Codex MCP / Antigravity / Grok / Claude-review 均不可用），无 `.aris/reviewer.json`，无 gemini/codex CLI。按 reviewer-backends 契约：不可用评审**绝不能伪造 PASS**，也不得记录 accept 回执。因此 `novelty-check` 与 `research-review` 两阶段保持 `done` 但无 cross-family receipt，最终 gate 预期 BLOCKED。

为不丢失"自行检查逻辑"的价值，以下为**执行者自审（非独立评审，只作 provisional 修改输入）**，逐条检查方向 I proposal 的逻辑链：

- **L1 初始化零梯度**：proposal 称去 τ 后 Teacher=Student、原 forget KL 在初始化无梯度，故用逐层 L_F 提供梯度。检查：成立——X⁻ 换了主体，h_T(X⁻)≠h_S(X⁺) 初始化，L_F 有非零梯度。✅
- **L2 "渐进"的必要性**：proposal 自认必须相对静态全层/反向/随机 schedule 等预算证明独立收益，否则只能叫"多层对齐"。检查：这是最硬的杀手条件，当前 proposal 无证据。⚠️ BLOCKER-for-claim（不是方案否决，是表述降级风险）。
- **L3 donor 答案替换**：直接对齐 A⁻ 会让 Student 在 X⁺ 上产出 donor 事实——不是遗忘是替换。检查：proposal §6 要求报告 donor 答案，但参考机制本身没有解决，属逻辑缺口。⚠️ 构成 Idea 4 淘汰依据。
- **L4 ΔA 纯度**：主体替换差异 ≠ 事实差异，ΔA 可能主要是"主体是谁"。检查：未被 proposal 论证，是 killer risk。⚠️
- **L5 共享前缀 KL**：proposal §3.1 强制两个分布共享同一 y_{<t}，两条前缀分支独立归一化，不逐 token 比较两条不同序列。检查：DKT 分解正确，causal shift（§5）口径正确。✅
- **L6 参数内化**：主训练直接优化 Student 参数、运行时 patch 仅作对照。检查：设计正确，避免"hook 依赖"伪遗忘。✅
- **L7 抗恢复**：proposal §6 要求 ≥2 类独立抗恢复证据。检查：与"Distillation Robustifies Unlearning"一致；可补 unlearn-and-distill 作为增强步骤。✅ 建议增强。
- **L8 过度泛化/语义漂移**：Student 学会"把 X⁺ 当 X⁻ 处理"后，对共享主体但不同关系的 prompt 也可能偏移。检查：proposal 用 retain/shared facts 覆盖，但对 shared-subject 的语义漂移无显式探针。⚠️ 建议加 relation-swap probe。

**最低判别实验（供后续，不在本轮执行）**：(a) ΔA probe：训练线性 probe 判断 ΔA 是否编码目标事实（vs 主体身份）；(b) 等预算 schedule 对照（渐进 vs 静态全层 vs 反向 vs 随机）；(c) relation-swap probe 测语义漂移。任一失败即触发 kill switch。

## Eliminated Ideas

- **Idea 4 — 直接向 A⁻ 对齐（原参考方案）**：淘汰。PACT（2609.38909）证明 target-based 直接对齐造成 context blindness；CFT hidden costs（2605.27083）证明拟合替代目标引入幻觉溢出与梯度冲突；且无法区分"遗忘"与"donor 答案替换"（proposal §6 自认）。
- **Idea 5 — 纯 ΔA 差值引导更新**：淘汰（暂缓）。ΔA 不纯净，主要为"主体替换差异"，直接以其引导更新缺乏事实特异性保证；降级为 Idea 1 中的方向估计器，不独立成方案。

## Refined Proposal

- Proposal: `workspace/refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `workspace/refine-logs/EXPERIMENT_PLAN.md`
- Tracker: `workspace/refine-logs/EXPERIMENT_TRACKER.md`
- Research contract: `workspace/idea-stage/docs/research_contract.md`

## Next Steps

- [ ] 在有 GPU 的 A800 环境跑 ΔA probe + schedule 对照（本轮无法执行）
- [ ] 补跨模型评审（需启用 Codex/Gemini/Grok/Claude-review 之一），把 novelty-check / research-review 从 provisional 升级为 accepted
- [ ] 若 probe 通过：用 `/experiment-bridge` 实现 Idea 1 → `/run-experiment`
- [ ] 若 probe 失败：执行 kill switch，主攻线让位 Idea 2/FDD delta 变体
- [ ] 部署前用 `/ablation-planner` 安排 schedule 消融，`/kill-argument` 做投稿前对抗审查

## 附录 A — 检索披露

检索渠道：web_search（多轮，6 组查询词），命中 arXiv/ACL Anthology/NeurIPS/AlphaXiv/机构库/二次摘要站。关键 ID：arXiv 2603.01285（ASU）、2604.15847（CiPO）、2605.27083（CFT hidden costs）、2509.38909/2609.38909（PACT，日期 2026-09-30）、FALCON（NeurIPS 2025）、FDD（ACL 2025, DOI 10.18653/v1/2025.acl-long.1125）、UNDO（NeurIPS 2025）、MALTO（SemEval-2025）、AlphaEdit（ICLR 2025）。Gemini CLI 缺失，未做 Gemini 广域检索；所有数字/结论均标注来源，未核实处用"据不完全证据显示"。

<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:START -->
## Evidence Gate
**Status:** BLOCKED

The workflow is not complete. Required stage evidence is missing:
- BLOCKED: novelty-check review evidence missing (status=done)
- BLOCKED: research-review review evidence missing (status=done)
<!-- ARIS_IDEA_DISCOVERY_EVIDENCE_GATE:END -->

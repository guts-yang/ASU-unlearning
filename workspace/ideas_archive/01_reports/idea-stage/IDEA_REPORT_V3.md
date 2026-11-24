# Idea Discovery — 第三轮：基于 ASU 自身缺陷 + stage0 矛盾的新方向

**Direction**: 第三轮构思，聚焦"重新设计 ASU 的 forget-teacher 本身"，而非二次组合已被占据的 head-level/子空间方法。
**Date**: 2026-11-23
**触发**: `/idea-discovery` 第三轮。证据 = 远程实测 stage0-14 + 本轮新查新。
**Output**: `workspace/idea-stage/IDEA_REPORT_V3.md`（本文件）

> 诚信声明：本轮无新实验（本机无 GPU）；打分均为**机制推断分**，每个高分附决定性验证实验。跨模型评审后端仍不可用，查新为 PROVISIONAL，不伪造评审回执。Lacuna 的 "Budgeted Attention Teachers" 是**非同行评审的 hypothesis 展示页**，仅作思路参考，不作为可引用 prior art。

## 一句话结论（先看排名）

| 排名 | Idea | 一句话 | 五维 (机制分) | Verdict |
|---|---|---|---|---|
| 🥇 1 | **SensiTemp：层-头自适应温度场** | 用 factual/function token 敏感度给每层每 head 分配 τ，替代全局标量 τ | Higher8/Faster8/Cheaper9/Stronger7/Broader7 | **Strong Accept（待 probe 验证）** |
| 🥈 2 | **ASU 遗忘深度审计基准** | 系统测 ASU 及各变体的抗恢复深度（P1/P2/P3+量化+probe），补用户全部实验缺失的维度 | Higher7/Faster9/Cheaper9/Stronger6/Broader8 | **Accept（确定性最高）** |
| 🥉 3 | **浅-深解耦双阶段遗忘** | 浅层 ASU 阻断信息流 + 深层用 pooled ΔA 向 retain 对齐，解释 stage0 矛盾 | Higher8/Faster6/Cheaper6/Stronger6/Broader6 | **Accept with Revisions（有证伪风险）** |

淘汰：第二轮 Idea A（ASU-Selective 与 RepSelect 高度重合，且本轮 ADU/ALMPU 已占据 head-level+抗恢复空间）；反事实配对全系（实测证伪，见 V2）。

## 为什么这轮换方向（第三轮新证据）

**查新发现 1：head-level + 抗恢复已是红海。** 第二轮推荐的 Idea A（ASU-Selective，attention K/V 子空间选择性）在本轮查新中被三个近邻挤压：
- **ADU: Attention Decoupling Unlearning**（arXiv 2608.23020, 2026-08）：区分 local/global heads，定位 preplan-anchor 路径，训练 attention-projection adapters 抑制路径质量；TOFU Forget Quality 0.93，保 87-98% utility。**已占据"attention 路径级遗忘"。**
- **ALMPU**（AAAI 2026, DOI 10.1609/aaai.v40i23.39017）：给每个 attention head 加可学习 scaling factor 选 top-K 敏感 head，在这些位置做 memory perturbation 抗 fine-tuning 攻击；WMDP 上抗恢复 SOTA。**已占据"head 敏感定位 + 抗恢复扰动"。**
- **ATTENTION SWITCH**（arXiv 2611.08752, jailbreak defense）：head 选择 + scaling 联合学习。跨领域但机制重叠。

结论：再做"attention head 级新模块"会被 reviewer 判为 ADU/ALMPU 变体。**真正的空白不在"加新模块"，而在 ASU 自身的温度机制。**

**查新发现 2：stage0 实测与 ASU 核心论断存在张力（本轮最重要支点）。**
- ASU 论文：事实锚定浅层（Llama-2-7B 平滑 6-8 层即可），factual token 对 τ 远敏感于 function token（Fig.2b），故全局单一 τ 能"只伤事实不伤语法"。论文自承**温度选择脆弱、需 binary search**，并把 "dynamic temperature adjustment / layer-specific interventions / selective unlearning" 列为 future work。
- 用户 stage0 实测（TOFU-1B，32 pairs，last-token ΔA 均值）：layer1 **0.051** → layer8 **0.171** → layer16 **0.380**，单调递增；all-token ΔA 较平（0.18→0.24）。
- 张力：若事实差异主要在深层显现，为何平滑浅层有效？两种解释：(a) 浅层平滑截断信息流，深层差异根本形不成；(b) 深层 ΔA 主要编码**主体身份**而非事实（呼应 V2 中"unpooled ΔA 与 wrong-donor 重叠 0.55"）。
- 这个张力是**用户独有数据**提出的科学问题，现有工作（ADU/ALMPU/RepSelect）都没回答。它直接支撑 Idea 1 与 Idea 3。

**查新发现 3：抗恢复维度是用户全部实验的系统性盲区。** PrivUn（Findings EMNLP 2026, arXiv 2604.22076）、SoK dememorization（2026-09）、RepSelect（arXiv 2606.17168）、ALMPU 都指出：只看 forget/retain NLL 会漏掉 shallow forgetting。用户 stage0-14 全部只测 forget/retain NLL，**从未测 relearning / few-shot / 量化 / probe 恢复**。这既是风险（结论可能只是浅层）也是最低成本的贡献机会（Idea 2）。

## Ranked Ideas（含五维打分与决定性验证实验）

### 🥇 Idea 1 — SensiTemp：层-头自适应温度场（推荐）

**问题**：ASU 用一个标量 τ 平滑所有层所有 head，必然同时压掉与事实无关的语法/局部注意力，这是其温度脆弱与 utility 损失的根因（Lacuna hypothesis 也指出这点，但基于位置的 core/cap 划分非正式且未验证）。

**假设**：按"该层该 head 的 factual-vs-function token 敏感度"分配温度 τ_{ℓ,h}，能让 forget-teacher 只在真正承载事实的层/head 平滑，语法 head 保持 τ=1。敏感度定义沿用 ASU 自己的 Fig.2b 方法：对每个 (ℓ,h)，测 NLL_factual(τ)−NLL_factual(1) 与 NLL_function(τ)−NLL_function(1) 的差，差值大的 head 给高 τ。

**方法**：① 一次性诊断：在 forget set 上用 GPT-4o 标注的 factual/function token（ASU Appendix N 已有协议）扫 τ∈{1,1.5,2,2.5,3}，得敏感度矩阵 S[ℓ,h]；② 构造 temperature field τ_{ℓ,h}=1+α·norm(S[ℓ,h])，只在 S 高的层/head 平滑；③ 冻结 teacher 做标准 ASU 自蒸馏；④ 对照：全局 τ（原 ASU）、随机层/head 场、逐层场、逐 head 场、位置 core/cap 场。

**差异化**：ADU/ALMPU/ATTENTION SWITCH 都**训练新模块**（adapter / scaling+perturbation / selection mask）；SensiTemp **不训练任何新模块**，只重设计 teacher 的温度分配，保持 ASU 单机制简洁性，且首用 factual/function 敏感度（功能视角）而非位置（sink/local）视角分配。

**五维（机制分）**：Higher 8（结构化温度应优于标量）/ Faster 8（一次性诊断+标准训练）/ Cheaper 9（无新参数、无对抗训练）/ Stronger 7 / Broader 7（可迁移 MUSE/WMDP）。
**决定性验证实验（kill switch）**：若各 head 的 factual/function 敏感度矩阵近似均匀（方差小于阈值），τ 场退化为全局 τ → Idea 1 直接否决，收缩为"深度审计"（Idea 2）。这是 1-2 GPU·h 的诊断，先验尸检再训练。

### 🥈 Idea 2 — ASU 遗忘深度审计基准（确定性最高）

**问题**：用户 stage0-14 所有结论（包括"ref 最强"）都只基于 forget/retain NLL，无法排除 ASU 也只是 shallow forgetting。

**假设**：ASU 在强恢复攻击（P3 微调、量化恢复、probe）下会显著回退，尤其对深层分布的事实（stage0 ΔA 深层累积的事实支持的假设）。

**方法**：对 ref / idea1-4 / 强 GA / 现成基线统一跑恢复阶梯：P1 直接查询 → P2 few-shot 恢复 → P3 用少量同域数据微调恢复 → INT4/INT8 量化恢复（DurableUn 路线）→ 线性 probe 从中间层激活解码目标事实。给出"遗忘深度"综合指标（恢复曲线下面积），并建立 token 级分层报告（factual vs function）。

**差异化**：PrivUn 聚焦 PII/Enron、SoK 是 SoK、RepSelect/ALMPU 只把抗恢复当作所提方法的附加评估；**没有人系统审计 ASU 这一 ICLR 2026 方法自身的遗忘深度**，而用户拥有 ASU 官方代码库与全部变体，是天然主场。

**五维（机制分）**：Higher 7 / Faster 9 / Cheaper 9 / Stronger 6 / Broader 8。
**决定性验证实验**：若 ASU 在 P3/量化下完全稳健，则"ASU 是深度遗忘"成为强正面结论；若回退严重，则产生"ASU 浅层遗忘"的负面但高价值结论，并直接给 Idea 1/3 提供动机。两种结果都可发表/可指导后续。
**协同**：与用户 SpecGap/D3 白盒审计资产天然契合（审计侧延伸）。

### 🥉 Idea 3 — 浅-深解耦双阶段遗忘（科学问题最有趣，但风险最高）

**问题**：解释 stage0"事实差异在深层累积"与 ASU"平滑浅层有效"的张力。

**假设**：ASU 浅层自蒸馏只切断了浅层的精确检索路径，深层仍残留事实表示（浅层遗忘）；分层处理——浅层继续用 ASU 平滑阻断信息流，深层用 pooled ΔA basis（stage7b 已生成，且证明与 retain 正交 0.23）做**向 retain 基线方向对齐**（注意：不是 V2 已证伪的"沿 ΔA 抑制"）。

**方法**：阶段1 标准 ASU；阶段2 在层 10-16（ΔA 显著区）对 forget 输入 X+ 的激活施加表示约束，把事实相关方向拉向冻结基线在 retain 上的对应分布；阶段3 接 Idea 2 的深度审计。

**差异化**：RepSelect 用 forget 梯度 SVD 的 bottom-PC、ALMPU 用 head 扰动；本方案用**反事实配对 pooled ΔA basis**这一用户独有资产，并明确切分浅/深两阶段。

**五维（机制分）**：Higher 8 / Faster 6 / Cheaper 6 / Stronger 6 / Broader 6。
**决定性验证实验（kill switch）**：V2 已证伪"沿 ΔA 抑制"，本方案必须换成"向 retain 对齐"且先做一个 probe：深层 ΔA 方向是否能在不伤 retain 的前提下被单独压缩（若压缩同时伤 retain → 与 retain 不可分 → 否决）。
**风险声明**：这是三轮中证伪风险最高的（V2 的失败模式可能重现），因此排第三而非第一。

## 淘汰清单（反重复记忆，已写入 wiki）
- 反事实主体配对 + ΔA 正交抑制/对齐（R1 提出，stage8/9/13 实测证伪）
- ASU-Selective 与 RepSelect 式 forget-specific 子空间（R2 提出；本轮 ADU/ALMPU/RepSelect 已密集占据，差异化不足）
- 梯度手术 PCGrad/EWC/GradNorm/GA-orth（stage13/14 全失败）
- 单纯 mask / λβ 调参 / 反事实渐进 schedule（stage3/4/5 无收益）

## 执行建议（下一步，不在本轮执行）
1. 先跑 Idea 1 的 τ 敏感度诊断（gate，1-2 GPU·h）——决定 Idea 1 生死，成本极低。
2. 并行跑 Idea 2 深度审计（复用全部已有 checkpoint，几乎零训练成本）——无论结果都产出可发表结论。
3. 只有 1 和 2 都正面时，才投入 Idea 3 的双阶段训练。
4. 全部新结果必须过 `/experiment-audit` + `/result-to-claim`；发表前补异族评审复核差异化。

## 附录 — 来源披露
用户实测：`stage0_profile_tofu1b.json`（last-token ΔA 逐层）、`ALL_RESULTS_SUMMARY.json`（H20，TOFU-1B，3 seed）。
文献：ASU arXiv 2603.01285 (ICLR 2026)、ADU arXiv 2608.23020、ALMPU DOI 10.1609/aaai.v40i23.39017 (AAAI 2026)、ATTENTION SWITCH arXiv 2611.08752、PrivUn arXiv 2604.22076 (Findings EMNLP 2026)、RepSelect arXiv 2606.17168、SoK dememorization 2026-09、Budgeted Attention Teachers（Lacuna hypothesis 页，非同行评审）、FALCON/CiPO/CFT/PACT（见 V1）。
检索渠道 web_search 多轮；所有关键数字标注来源；未核实处用"据不完全证据显示"。

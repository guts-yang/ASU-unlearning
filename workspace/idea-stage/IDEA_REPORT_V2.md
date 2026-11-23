# New Idea Round 2 — 基于方向 I 实测结果的转向

**Direction**: 承接远程同步的方向 I stage0-14 实测（H20 + TOFU-1B + 3 seed），重新构思新一代 idea。
**Date**: 2026-11-23
**触发**: `同步远程内容 + /idea-discovery`。远程新增 43 文件实验结果，推翻上一轮纸面判断。
**证据基线**: `workspace/experiments/direction_i/ALL_RESULTS_SUMMARY.json`
**Output root**: `workspace/idea-stage/IDEA_REPORT_V2.md`

> 诚信声明：本轮所有"实测"数字均来自远程同步的 `ALL_RESULTS_SUMMARY.json`（H20 真机运行，非本机）；本机仍无 GPU，不新增实验。新 idea 的 pilot 信号 = 文献机制推断 + 已有实测的间接支撑，标注 `needs pilot`。跨模型评审后端仍不可用，查新为 PROVISIONAL，不伪造回执。

## Executive Summary（本轮最重要结论）

**方向 I（反事实主体配对）已被实测证伪，应战略转向。** H20 上 stage0-14 的硬数据显示：所有反事实配对变体（idea1 正交抑制、idea2/3、idea4 直接对齐 A⁻）都远弱于原 ASU 温度自蒸馏基线 `ref`（forget 0.555 / retain 0.798）；idea1 forget 1.068 / retain 1.52 几乎不遗忘；PCGrad/EWC/GradNorm/GA-orth 等梯度手术全部失败。**上一轮我推荐的 Idea 1 被自己的实验数据推翻。**

新查新（RepSelect 2606.17168、SAGO ACL2026、PrivUn/EMNLP2026）给出转向依据：真正决定"深度遗忘"的是**表示子空间的选择性**——RepSelect 证明 top-PC 是与 retain 共享的高方差方向（攻击可恢复、且伤通用能力），**bottom-PC 才是 forget-specific**；这与"ΔA 主成分=事实方向"的旧假设方向相反。据此提出 3 个新 idea，推荐 **Idea A：ASU-Selective（注意力平滑 × 表示选择性）**：保留已验证有效的 ASU 温度自蒸馏作为输出层遗忘，叠加 RepSelect 式的"仅压缩 forget-specific 低方差子空间"表示约束，把浅层遗忘变深，同时不碰共享 top-PC。这是对 ASU 的增量增强，而非已被证伪的反事实配对。

## 为什么方向 I 失败（实测复盘，全部有据）

| 阶段 | 实测结果 | 结论 |
|---|---|---|
| stage0 | ΔA 逐层 last-token 范数：浅层 0.05 → 深层 0.38，逐层递增 | 事实差异主要在深层 last-token，但不足以指示可用方向 |
| stage4 | ref 0.555/0.798 vs progressive 0.571/0.838 vs static3 0.590/0.890 vs reverse 0.612/0.935 | **"渐进"反而略弱于 ref，且不如静态；渐进主张证伪** |
| stage5 | masked/unmasked 各档 forget 上升微弱(≤0.05)、retain 也上升 | 单纯 mask 无有效遗忘 |
| stage7/7b | ΔA probe：未 pooled KILL（与 wrong-donor 重叠 0.55）；pooled PASS（top5 解释 0.45-0.50，与 retain 正交 0.23） | pooled ΔA 有一定事实特异性，但"正交"未转化为遗忘效果 |
| stage8 | idea1 forget 1.068/retain 1.52；idea4 0.651/1.01；gate `retain_damage_le_half=false` | **核心方案失败**：正交抑制没产生遗忘梯度效益 |
| stage9 | suppress/orthalign × 5 个 λ，pass_gate 全 false | 调参救不回来，不是超参问题 |
| stage10/11 | idea2/idea3/out_idea4/collapse 与 ref 几乎相同 | 变体均无增益 |
| stage12/14 | 强 GA(λ≥6) 能忘(forget 1.47-2.06)但 retain 崩(1.6-1.9)；GA-orth forget 1.75 | "靠强度遗忘"必然牺牲 retain，正交化 GA 仍崩 retain |
| stage13 | PCGrad/EWC/GradNorm forget≈1.06-1.16, retain≈1.52，无改善 | 梯度手术在 1B/TOFU 上无效 |

**类级结论（可作为反重复记忆）**：在 TOFU-1B 这个 setting 下，"反事实配对构造 forget 目标"+"表示层正交/对齐"+"梯度手术"三条路线均系统性失败，且都弱于原 ASU。**继续沿这些方向微调是红海死胡同。**

## 查新（本轮新增，PROVISIONAL）

| 近邻 | 年份/venue | 与本轮关系 |
|---|---|---|
| **RepSelect: Robust LLM Unlearning via Representation Selectivity**（arXiv 2606.17168, 2026-06） | 直接方法源 | SVD 分解 forget 梯度：**top-PC 是 retain/攻击共享方向，bottom-PC 才是 forget-specific**；只压缩 bottom-PC，抗 relearning 提升 4-50×，跨 Llama3/Qwen3.5/Gemma4/DeepSeek-V2。局限：仅 MLP 模块、未测 attention K/V。**这是本轮 Idea A 的核心依据与最需划界的近邻。** |
| **SAGO: Modeling LLM Unlearning as Asymmetric Two-Task**（arXiv 2604.14808, ACL 2026） | 备选依据 | retention 为主、forgetting 为辅，element-wise sign gating 保证更新不与 retain 梯度对立；MMLU 恢复 44.6%→96%。**解释了我方 stage13 PCGrad 失败：PCGrad 是 module-wise，SAGO 是 element-wise sign gating，粒度不同。** |
| **PrivUn / Old Memories Die Hard**（arXiv 2604.22076, Findings EMNLP 2026） | 评估动机 | 浅层遗忘+深层存储：label-based 方法只改最后几层，抗直接查询但抗不过微调恢复；提出 P1/P2/P3 三级攻击与梯度型涟漪效应。**说明我方实验只看 forget/retain NLL 不够，必须加抗恢复维度。** |
| **SoK: Unlearnability and Unlearning for Dememorization**（2026-09） | 理论背景 | 浅层 dememorization 的权重扰动恢复攻击 + 深度认证理论；GA/IF 在 η=0.4 微扰下几乎完全恢复，FT 更深。 |
| FALCON / CiPO / CFT costs / PACT（上一轮已查） | 背景 | 反事实配对与正交激活的红海，本轮实测已证伪我方对应变体。 |

**查新裁决**：Idea A 与 RepSelect 字面高度重合（都用"forget-specific 低方差子空间"）。差异化必须落在两点：**(1) 与 ASU 注意力平滑机制耦合**（RepSelect 只作用于权重梯度 SVD，不动注意力分布）；**(2) 作用在 attention K/V 子空间**（RepSelect 自承局限"仅 MLP、未测 K/V"）。若这两点做不出差异，应诚实标为"RepSelect 在 ASU 上的复现/迁移"，不构成新贡献。裁决 = **PROCEED WITH CAUTION（强条件）**。

## Ranked Ideas（全新，不与上一轮重复）

### Idea A — ASU-Selective：注意力平滑 × forget-specific 子空间选择性（推荐）
- **假设**：ASU 的温度 τ 自蒸馏已在输出层实现有效遗忘（实测 ref 最优），但属浅层；真正的深度遗忘需在表示层只压缩 forget-specific 低方差子空间（RepSelect bottom-PC），保留与 retain 共享的 top-PC。两者正交互补：ASU 管"输出行为遗忘"，子空间选择性管"内部表示深度"。
- **机制**：阶段1 用现成 ASU 得到 unlearned 模型 M_u；阶段2 在 M_u 上对 forget set 做 SVD（分别在 MLP 与 attention K/V），把权重梯度/激活更新限制在 bottom-PC 子空间（或等价地：对 top-PC 方向施加 retain-anchor 约束），得到 M_u'。评估 P1/P2/P3 三级抗恢复 + MMLU/retain。
- **依据**：RepSelect（4-50× 抗 relearning）+ ASU（实测最强基线）+ PrivUn（浅层遗忘需深层干预）。
- **差异化**：RepSelect 是"纯表示选择性、无 forget-teacher"；ASU-Selective 是"输出分布遗忘 + 表示选择性"两段式，且首次检验 attention K/V 上的选择性（RepSelect 明确留白）。
- **Pilot**：needs pilot（复现成本中等）。**kill risk**：若 M_u 的 forget 梯度 SVD 中 bottom-PC 与 ASU 平滑后的注意力变化正交（即两者作用于不相交子空间），叠加可能无收益 → 需先做子空间重叠诊断（类似已有 stage7b，但对象是 forget 梯度而非 ΔA）。

### Idea B — ASU 温度的课程/自适应调度（低成本增量）
- **假设**：stage4 显示反事实"渐进"无效，但 ASU 自身的 τ 从未被调度。ASU 固定 τ>1；是否"先高 τ 快速压制输出、后低 τ 精细保 retain"能在保持 ref 遗忘强度下改善 retain？
- **机制**：τ(k) 课程（高→低 / 低→高 / 随机对照），等预算；只动 forget-teacher 的 softmax 温度，不改架构。
- **依据**：FDD/temperature scheduling 经验 + 我方 stage2/3 显示 β/λ 权重对 forget-retain 权衡敏感（λ=0 时 forget 0.764 但 retain 也降，提示调度空间存在）。
- **Pilot**：needs pilot（成本最低，可直接复用 ASU trainer）。**kill risk**：λ/β sweep 已显示单纯权重调整收益有限，τ 调度可能同样只是权衡曲线移动而非帕累托改进。

### Idea C — 抗恢复即目标：ASU + relearning-aware meta 目标
- **假设**：我方实验只测 forget/retain NLL，未测抗恢复。把"抗微调恢复"直接纳入训练目标（对抗式：内步模拟攻击者用少量同域数据微调，外步最大化恢复后的 forget 概率），与 ASU 自蒸馏叠加。
- **机制**：TAR 式 tamper-resistant 目标 + ASU forget-teacher，双环训练。
- **依据**：SoK（GA/IF 易恢复、FT 更深）、PrivUn（三层攻击）、TAR（tamper-resistant）、UNDO/Distillation Robustifies（蒸馏过滤潜在知识）。
- **Pilot**：needs pilot（成本最高）。**kill risk**：1B 规模上对抗训练不稳定；且"抗恢复"与"retain 不受损"可能直接冲突（攻击者微调用的同域数据也是 retain 分布）。

### 排序理由
A 差异化最强且有明确可检验的 kill risk，复用已有 ASU 与实验基建；B 成本最低适合作为快速 sanity check；C 最有理论价值但成本/风险最高。AUTO_PROCEED：推荐先跑 A 的子空间重叠诊断（gate），B 作为并行低成本对照。

## 淘汰 / 不再尝试（反重复）
- 反事实主体配对 + ΔA 正交抑制/对齐（Idea 1 上一轮 + 本轮证伪）
- 直接对齐 A⁻（idea4，donor 替换 + retain 1.01）
- PCGrad/EWC/GradNorm/GA-orth 梯度手术（stage13/14 全失败）
- 单纯 mask / 调 λβ / 反事实渐进 schedule（stage3/4/5 无独立收益）

## 查新与评审诚实声明
跨模型评审后端仍不可用（Codex/Gemini/Grok/llm-chat 均未启用，knot-cli 未安装），本轮 PROVISIONAL 查新为执行者基于公开文献与实测的判断，不记 cross-family accepted，不伪造 PASS。若启用任一异族评审通道，应对 Idea A 的"与 RepSelect 划界"做独立复核。

## 下一步
- [ ] （gate）复现 RepSelect，在其 SVD 框架下测 ASU-M_u 的 forget 梯度 top/bottom-PC 与 retain 的重叠 → 判断 Idea A 是否值得做
- [ ] （低成本）并行跑 Idea B 的 τ 课程 sanity check（复用 ASU trainer）
- [ ] Idea A 若 gate 通过 → 实现两段式并在 TOFU + WMDP 上做 P1/P2/P3 抗恢复评估
- [ ] 启用异族评审 → 复核 Idea A 与 RepSelect 的差异化
- [ ] 用 `/experiment-audit` 审计新实验，`/result-to-claim` 判定 C1/C2

## 附录 — 本轮证据
实测：`workspace/experiments/direction_i/ALL_RESULTS_SUMMARY.json`（H20，TOFU-1B，seed 0/1/2）。
文献：RepSelect arXiv 2606.17168、SAGO arXiv 2604.14808 (ACL2026)、PrivUn arXiv 2604.22076 (Findings EMNLP2026)、SoK dememorization 2026-09、FALCON NeurIPS2025、CiPO arXiv 2604.15847、CFT costs arXiv 2605.27083、PACT arXiv 2609.38909。
检索渠道 web_search 多轮；所有数字标注来源；未核实处用"据不完全证据显示"。

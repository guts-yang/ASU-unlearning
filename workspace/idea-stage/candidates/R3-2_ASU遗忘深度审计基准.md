---
id: R3-2
title: ASU 遗忘深度审计基准 —— 行为遗忘 ≠ 知识擦除
status: CANDIDATE · PROMOTE（确定性最高的贡献线）
decision_source: workspace/idea-stage/IDEA_REPORT_V4.md（PROMOTE）、workspace/idea-stage/docs/research_contract.md
domain: LLM Unlearning / 评估基准 / 遗忘深度度量
evidence_level: provisional（P1/P2/P3/INT8 实测已完成；无跨模型独立评审）
baseline_relation: 为方向 I 基线 v0.6.1 提供可复用的 G5/G6 抗恢复验收口径（见 §7）
---

# R3-2 · ASU 遗忘深度审计基准

## 0. 故事（30 秒讲稿）

> 用户拥有 ASU（ICLR 2026）官方代码库与其全部变体，却从没有人问过：**ASU 忘掉的，是真没了，还是只是暂时读不出？** 我们系统审计这一 ICLR 2026 方法自身的遗忘深度：对 base / ASU-u / GradAscent 统一跑恢复阶梯 P1 直接查询 → P2 few-shot → P3 少量同域微调 → INT8 量化 → 线性 probe 从中间层解码目标事实，给出"遗忘深度"综合指标（恢复曲线下面积）和 factual/function token 级分层报告。实测结论：ASU-u forget $0.96$ → P2 恢复 $0.261$ → P3 恢复 $0.185$，平均恢复 $\approx0.448$——**近 $45\%$ 的遗忘增益可被找回**。这既是一个负面但高价值的结论，也给所有后续方法（温度课程、抗恢复目标、方向 I 中性约束）提供了统一的、不可被"改答/拒答/提前 EOS"糊弄过去的验收尺度。无论审计结果是正是反，都能发表、都能指导后续。

## 1. 领域问题与空白

LLM unlearning 的评估默认只看 forget/retain NLL，无法排除 **shallow forgetting**：模型可能只学会不读上下文（PACT 的 context blindness，arXiv:2609.38909）、提前停答、或把答案换成别的实体，而知识仍留在参数里，可被微调/量化/probe 找回。PrivUn（Findings EMNLP 2026, arXiv:2604.22076）提出 P1/P2/P3 三级攻击与梯度型涟漪效应；SoK dememorization（2026-09）指出 GA/IF 在 $\eta=0.4$ 微扰下几乎完全恢复；RepSelect / ALMPU 把抗恢复当作附加评估（来源：`IDEA_REPORT_V2.md`、`IDEA_REPORT_V3.md` §查新发现 3）。

**空白（主场优势）**：没有人系统审计 **ASU 这一 ICLR 2026 方法自身**的遗忘深度。用户所有 stage0-14 实验只测 forget/retain NLL，从未测 relearning / few-shot / 量化 / probe 恢复——这既是既有结论的可信度风险，也是最低成本、确定性最高的贡献机会：复用全部已有 checkpoint，几乎零训练成本。

## 2. 核心假设（双向可发表）

**H**：ASU 及其变体在强恢复攻击下会显著回退，尤其对深层分布的事实（呼应 stage0 ΔA 逐层递增、深层累积事实支持的观测）。

**关键特性**：结论**双向都有价值**——若 ASU 在 P3/量化下完全稳健，则"ASU 是深度遗忘"成为强正面结论；若回退严重，则产生"ASU 浅层遗忘"的负面但高价值结论，并直接给 R2-C（抗恢复目标）与 R2-B（课程）提供动机。这是该候选被 V4 判为"确定性最高"的原因。

## 3. 具体实现

**统一恢复阶梯**（对 base / ASU-u / 各 idea 变体 / 强 GA / 现成基线一视同仁）：

| 攻击 | 操作 | 目的 |
|---|---|---|
| P1 | 直接查询 forget QA | 行为遗忘基线 |
| P2 | few-shot / in-context 恢复 | 测上下文可唤醒的浅层抑制 |
| P3 | 少量同域数据微调（固定预算 $\eta$、公开协议） | 测参数内残留知识 |
| INT4/INT8 | 量化恢复（DurableUn 路线） | 测数值扰动是否找回 |
| Probe | 线性 probe 从中间层激活解码目标事实 | 测表示层残留 |

**"遗忘深度"综合指标**：恢复曲线下面积（以 forget-NLL / span score 为纵轴、攻击强度为横轴），并给 token 级分层报告（factual vs function token，沿用 ASU Appendix N 的标注协议）。

**验收硬口径**（不可被捷径糊弄）：

- 联合报告原目标事实、donor 答案、其他错答、拒答、提前 EOS、完整回答率、长度分布、同实体非删除关系、独立 retain 效用；
- 禁止把"正确率/NLL 下降、改答 donor、拒答、提前 EOS、回答截断"单独当作选择性遗忘；
- 禁止 `logit(y^+)-logit(y^-)` 作为读出；
- 至少**两类独立抗恢复证据**改善，才允许受限的"深度遗忘"表述；
- 统计单位是作者/实体簇，多模板/多 donor/多 seed 为簇内重复测量。

**代码落点**：复用 `workspace/auto_experiments/` 与 `workspace/verification/` 已复现的恢复阶梯与 `train.py` 汇总链；新增统一的 `eval_probe.py` / 恢复曲线报告生成器，结果写新目录，不覆盖源评测。

## 4. 差异化（novelty 落点）

| 近邻 | 聚焦 | 本候选差异 |
|---|---|---|
| PrivUn (arXiv:2604.22076) | PII / Enron、提出 P1/P2/P3 | 本候选审计 **ASU 自身**，并加量化/probe 与恢复曲线下面积 |
| SoK dememorization (2026-09) | 理论 SoK、权重扰动恢复 | 本候选是实证基准 + 可复用协议，非综述 |
| RepSelect / ALMPU | 抗恢复当作所提方法的附加评估 | 本候选把抗恢复做成**独立、统一、token 分层的基准与度量** |
| V4 既有 stage0-14 | 只看 forget/retain NLL | 本候选补上全部缺失的恢复维度 |

**novelty 落点**：不是"恢复攻击"组件（有先例），而是"首个针对 ASU 的、覆盖 5 类恢复通道、带 token 分层与曲线下面积的公开基准 + 由此得出的『行为遗忘≠知识擦除』实证"。与用户 SpecGap/D3 白盒审计资产天然协同。

## 5. 当前证据与状态（provisional）

- P1/P2/P3/INT8 已完成（`research_contract.md` E1：本轮完成）：ASU-u forget $0.96$ → P2 恢复 $0.261$ → P3 恢复 $0.185$，$\text{mean\_recovery}\approx0.448$。
- V4 决策 `PROMOTE`："无论裁决均保留——本次为 PASS（ASU 存在浅层遗忘），给出统一恢复阶梯与深度指标，作为独立基准贡献"。
- 与 R2-C 共享攻击阶梯；为 R2-C、R2-B 及方向 I 提供验收尺度。
- 跨模型评审不可用，裁决 provisional，不声称 accepted。

## 6. 判别实验、失败模式、必测指标

**决定性判别**：恢复阶梯在 base / ASU-u / GA / 各变体间是否产生**显著分化**；分化越大，基准越有判别力（也是 R2-C/H 类对抗目标是否有优化空间的前提）。

**失败模式 / 口径风险**

- 若所有方法恢复曲线几乎重合 → 基准判别力不足，须诚实报告；
- 恢复攻击预算 $\eta$ 未在训练前冻结并公开 → 结论不可复现；
- 把"改答/拒答/提前 EOS"误记为遗忘 → 违反硬口径，判 G4 FAIL；
- 用参与训练闭环的 probe 自证深度（与 R2-C 的留出攻击混淆）。

**必测指标**：P1/P2/P3/INT8/Probe 各通道 forget-NLL 与 span score、恢复曲线下面积、factual/function token 分层、retain NLL、长度/EOS/完整回答率、3 seed std。

## 7. 与方向 I 基线 v0.6.1 的关系

方向 I 基线 §6 第 5 条要求"结合输出访问变体、独立 probe/patching、量化和受控 relearning，至少两类独立抗恢复证据改善，才允许受限『深度遗忘』表述"，第 6 条要求主结果在 $\lambda_{KL}^{\text{(ablation)}}=0$ 下成立。R3-2 的**统一恢复阶梯 + 硬读出口径**正是把这些要求落成可执行协议的产物，可直接作为方向 I 的 G5/G6 验收框架复用。但两者证据链必须分开登记：方向 I 的历史 pilot 不复用为 confirmation，旧 pilot 的正向结论已在 V4 被整批 ELIMINATED（见整合报告"口径张力"一节）。

## 8. 诚信声明与来源

plan-level 候选，整合自成文报告，不新增实验；数字标注来源；未核实处用"据不完全证据显示"。跨模型评审不可用，不伪造回执；所有新结果须过 `/experiment-audit` + `/result-to-claim`。

- `workspace/idea-stage/IDEA_REPORT_V4.md`、`IDEA_REPORT_V2.md` §Idea C 依据、`IDEA_REPORT_V3.md` §Idea 2
- `workspace/idea-stage/docs/research_contract.md`（E1、C1、共享攻击阶梯）
- `workspace/idea-stage/NARRATIVE_ASU_TEMPERATURE_DEPTH.md`（$44.8\%$ 恢复、诚实标注）
- `workspace/idea-stage/方向I_基线_v0.6.1_answer_span.md` §6.5、§6.6
- 文献：ASU arXiv:2603.01285、PrivUn arXiv:2604.22076、SoK dememorization 2026-09、RepSelect arXiv:2606.17168、ALMPU DOI 10.1609/aaai.v40i23.39017、PACT arXiv:2609.38909、DurableUn。

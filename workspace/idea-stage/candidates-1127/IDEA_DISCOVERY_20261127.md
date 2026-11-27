---
id: IDEA-DISCOVERY-20261127
title: 反事实激活差驱动的可解释 LLM 机器遗忘——第二轮差异化候选
date: 2026-11-27
mode: 文献调研 + 构思 + 机制评估；无 GPU、无训练、无部署
baseline: workspace/idea-stage/方向I_基线_v0.6.1_answer_span.md（历史问题锚点，非 ACTIVE 实现规格）
verdict_type: 执行者按本机 idea-evaluator rubric 给出的机制推断分（PROVISIONAL）
evidence_gate: BLOCKED（缺跨模型回执与 RESUMABLE helpers）
---

# 反事实激活差驱动的可解释 LLM 机器遗忘——第二轮候选

## 0. 执行摘要

本轮把 Teacher 固定为冻结基座，以正反事实对的中间层激活差为唯一问题载体，围绕“如何处理激活差”生成 16 个机制候选，淘汰 6 个重复、已被近邻占据或与历史证伪机制同族的方案，保留 10 个。所有入选者五维平均分均 **$\ge7.2>7$**，裁决均为 **Strong Accept 或 Accept with Revisions**，并在本目录下各有独立 proposal。

首推 **CF2-01 跨层反事实通量**：用 $\Phi_\ell=\Delta A_{\ell+1}-\Delta A_\ell$ 消去静态主体偏移，集中处理层间新增事实写入。其次是 **CF2-02 因子绑定交互**：用 2×2 因子设计扣除主体和关系主效应，只处理非加性绑定项。

> **状态边界**：分数为机制推断，不是实验成绩，也不是异族独立 reviewer 的接受回执。当前 `research-review=REVIEW_UNAVAILABLE`、`novelty-check=PROVISIONAL`、`evidence gate=BLOCKED`。基线文件已归档，本轮只重新开放构思，不复用其历史 pilot 为正向证据，不启动实验。

## 1. 基线与反重复约束

保留的硬边界：Teacher 不使用注意力温度；正反事实输入槽外一致；Teacher 在 $X^\pm\oplus y^+$ 上 teacher-force，答案跨度 $p\in S_f$ 取激活；Teacher 目标 stop-gradient；不拼 donor 答案 $y^-$；主结果不依赖 donor KL；Student 参数内化，部署无 hook；原事实读出、donor 支持度、EOS/长度与 retain 效用联合验收。

本轮明确避开 `candidates/` 已有机制：约束反演、三前向角色分解、单点因果门控、事实/身份双探针、多 donor 均值/SVD、LRM 轨迹作用域、下游读出 Jacobian、retain 零空间、对照族残差回归、纯度 schedule，以及 relearning minimax 和浅深双阶段方案。

## 2. Literature Landscape

**检索覆盖**：16 组实时 Web 查询，包含 arXiv 原文、会议页、OpenReview 和项目资料。`papers/`、`literature/` 不存在，本地 PDF 未贡献内容；arXiv helper、Gemini、Zotero/Obsidian 和 D1 批量核验 helper 不可用。**Sources contributed: web**。以下来源用于近邻划界，统一标记 `⚠ UNVERIFIED (D1 helper unavailable)`，不据此生成 BibTeX。

### 2.1 反事实与表示遗忘的强基线

- **PINT / Forget What Comes Next**（Shkabrii, Plant & Pawelczyk，2026，arXiv:2611.03955）：直接用 HSIC 降低最终层前缀表示与下一 token 的依赖。因此朴素“HSIC + 激活遗忘”被淘汰。
- **PARS / Do LLMs Really Forget? Hidden-State Leakage...**（2026）：说明输出抑制与隐层信息残留可分离，并以 probe-adversarial 目标处理隐层泄露。所有新方案必须用独立 probe/relearning，不得只报输出变化。
- **CIR**（Sondej & Yang，2025，arXiv:2509.11816）与 **LEACE**（Belrose et al., NeurIPS 2023）：覆盖 PCA 通用表示去除和闭式白化线性擦除，限制 CF2-08 的新颖性。
- **Erase or Hide?**（2025，arXiv:2509.22263）与 **FDCU**（2026，arXiv:2609.39279）：强调原知识通路削弱与新增抑制壳不同，强化“不是浅层抑制”的验收义务。

### 2.2 反事实差、因果交互与动力系统

- **Factorized Causal Activation Patching**：用 2×2 因子干预测量证据协作的非加性交互。CF2-02 只把“训练内化绑定交互”作为增量，不能宣称发明因子 patching。
- **Transformer Dynamics**（Fernando & Guitchounts，2025）：把残差流视为跨层动力系统并研究表示速度。CF2-01 的新对象是“反事实差的层间差分”，不是一般动力学分析。
- **Transfer Entropy as a Measure of Information Flow in LLMs and VLMs**（Liang & Shi，2026）：有向信息流诊断与剪枝。CF2-03 必须证明反事实条件差和遗忘训练的独立价值。
- **Shapley / Circuit Fingerprints / Neuron Shapley**：覆盖组件合作归因，CF2-10 因成本与近邻排在末位。

### 2.3 分布传输与几何干预

- **AcT**（ICLR 2025，arXiv:2410.23054）、**CHaRS**、**Opt-Out**（ACL 2025）：分别覆盖推理时激活传输、异质性聚类 OT 和参数空间 Wasserstein 正则。
- **UOT-Unlearn**（2026，arXiv:2603.16489）：在一步生成模型中以 UOT 重分配生成分布质量。CF2-04 只保留“LLM 反事实中间激活的未配对源质量删除”这一差异。
- **Householder Pseudo-Rotation**（EMNLP 2024）、**Angular Steering**（NeurIPS 2025）、**Selective Steering**（2026）：使单纯保范数旋转候选失去足够新颖性。

### 2.4 多尺度与高阶表示

- **Conditional Scale Entropy**（2026，arXiv:2605.21391）：小波用于跨层隐喻计算解释。CF2-05 的贡献只能是反事实尺度掩码训练，不能声称首次小波分析。
- **The Structure of Relation Decoding Linear Operators**（2025，arXiv:2510.26543）：关系解码算子可由三阶张量网络压缩。CF2-07 研究的是正反事实三阶共同累积量，不是关系解码器压缩。
- **Robust PCA / activation-difference-monitoring**：已有低秩 + 稀疏异常检测原型。CF2-09 必须先证明稀疏项的目标语义，不能从分解形式直接命名“事实”。

### 2.5 反事实不变性与训练代价

- **Aligning Language Models Towards Counterfactual Invariance**（2026，arXiv:2610.22942）：输出/奖励层的反事实不变性已存在，CF2-06 仅限中间层归一化割线和选择性遗忘。
- **On the Hidden Costs of Counterfactual Knowledge Training**（2026，arXiv:2605.27083）：反事实答案替换会产生知识冲突与幻觉外溢，支持本轮“不使用 donor 答案作目标”的共同边界。

**核心空白**：据当前检索，尚未发现系统比较“因子交互、层间通量、条件跨位置流、未配对质量、多尺度、三阶累积量”这些**反事实激活差算子**并逐项训练 Student 内化的工作。该结论是检索范围内的暂定判断，不是完备性证明。

## 3. 评分与裁决口径

按本机 `~/.bg-agent/config-with-app/skills/idea-evaluator/SKILL.md` 及其 references 执行兼容评估：先做 F1–F10 致命缺陷审计，再做生命周期/资源匹配、Higher/Faster/Stronger/Cheaper/Broader 五维评分、范式探测与可行性检查。

$\bar S=(H+F+S+C+B)/5$，用户门槛为 $\bar S>7$ 且裁决至少 Accept with Revisions。高分均为 `mechanism-based, not yet confirmed by data`，每项都附决定性验证实验。`data-refuted` 的旧机制直接 Reject and Pivot，不用高分掩盖。

## 4. 最终排名

| 排名 | ID | 核心激活处理 | H | F | S | C | B | 平均 | 裁决（PROVISIONAL） |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | CF2-01 | 跨层通量 $\Phi_\ell=\Delta A_{\ell+1}-\Delta A_\ell$ | 9 | 8 | 8 | 8 | 8 | **8.2** | Strong Accept |
| 2 | CF2-02 | 因子绑定 $\Delta^2A$ 扣除主效应 | 9 | 7 | 8 | 8 | 8 | **8.0** | Strong Accept |
| 3 | CF2-03 | 主体→答案条件交叉算子差 | 8 | 6 | 9 | 7 | 9 | **7.8** | Accept with Revisions |
| 4 | CF2-04 | UOT 未配对源质量销毁 | 8 | 6 | 9 | 7 | 9 | **7.8** | Accept with Revisions |
| 5 | CF2-05 | 层×答案位置小波尺度掩码 | 8 | 7 | 8 | 7 | 8 | **7.6** | Accept with Revisions |
| 6 | CF2-06 | 主体替换归一化割线敏感度 | 8 | 7 | 8 | 8 | 7 | **7.6** | Accept with Revisions |
| 7 | CF2-07 | 主体×关系×答案三阶累积量差 | 8 | 6 | 8 | 7 | 8 | **7.4** | Accept with Revisions |
| 8 | CF2-08 | 配对散射/retain 协方差广义谱 | 7 | 8 | 8 | 8 | 6 | **7.4** | Accept with Revisions |
| 9 | CF2-09 | 稳健低秩背景 + 稀疏绑定事件 | 7 | 7 | 8 | 7 | 7 | **7.2** | Accept with Revisions |
| 10 | CF2-10 | 组件联盟的非加性 Shapley 交互 | 8 | 5 | 9 | 6 | 8 | **7.2** | Accept with Revisions |

并列时按“最小实验的信息价值、闭式程度、近邻强度、首验成本”排序。Strong Accept 也仍需验证实验，不能理解为论文或投稿接受。

## 5. 逐项 Novelty Verification

| ID | 最强近邻 | 可验证差异 | 执行者查新判断 |
|---|---|---|---|
| CF2-01 | Transformer Dynamics | 对反事实差再做层间差分并训练内化 | PROCEED |
| CF2-02 | Factorized Causal Activation Patching | 因子绑定项成为遗忘目标而非解释终点 | PROCEED |
| CF2-03 | Transfer Entropy / PINT | 反事实条件差 + 跨位置主体→答案 + 边缘死区 | PROCEED WITH CAUTION |
| CF2-04 | AcT / UOT-Unlearn | 未配对源残差定义删除，不向 donor 全量映射 | PROCEED WITH CAUTION |
| CF2-05 | Conditional Scale Entropy | 反事实尺度一致性掩码 + Student 训练 | PROCEED WITH CAUTION |
| CF2-06 | Counterfactual Invariance | 中间层归一化割线、目标关系选择性遗忘 | PROCEED WITH CAUTION |
| CF2-07 | LRE 三阶张量网络 | 三阶共同累积量的正反事实差，非关系解码压缩 | PROCEED WITH CAUTION |
| CF2-08 | LEACE / CIR | 配对差分散射 + 独立 retain 尺度 + 参数内化 | PROCEED WITH CAUTION |
| CF2-09 | CIR / activation monitoring | $L+S+E$ 后做语义归属门，再训练遗忘 | PROCEED WITH CAUTION |
| CF2-10 | Circuit Fingerprints / Shapley | 组件联盟非加性交互成为遗忘目标 | PROCEED WITH CAUTION |

未调用到异族独立 reviewer，故本表不是 novelty accept receipt；`ABANDON` 只用于有明确论文或项目历史证据的淘汰项。

## 6. 淘汰的 6 个方案

| 淘汰项 | 原因 | 裁决 |
|---|---|---|
| 朴素 HSIC/MMD 独立性遗忘 | PINT 已直接占据最终层表示—下一 token HSIC；泛化 MMD 不足以形成新贡献 | Reject and Pivot（F1） |
| balanced OT 全量映射到反事实分布 | AcT/CHaRS 近邻强，且重新引入 donor 语义迁移 | Reject and Pivot（F1 + donor 风险） |
| Householder/Angular 保范数旋转 | HPR、Angular、Selective/Spherical Steering 已形成密集近邻 | Reject and Pivot（F1） |
| 多 donor 坐标中位数/符号投票 | 与 CF-05 多 donor 边际化同族，只是稳健统计替换 | Reject and Pivot（反重复） |
| donor 维 Tucker/SVD 共有子空间 | 同时接近 CF-05 与 LRE 张量压缩，贡献主要为分解器更换 | Reject and Pivot（F1） |
| 直接抑制原始 $\Delta A$ 或对齐 $A^-$ | 项目历史已证伪/答案替换风险，触发 data-refuted 边界 | Reject and Pivot（CRITICAL） |

## 7. 共同验证门与建议顺序

所有候选先做缓存上的可分性/稳定性，再进入 Student 训练；等删除能量、等步数、等前后向和等 GPU 预算比较；真实 donor 必须优于 wrong、shuffle、noise、matched-retain 和随机算子对照。

联合报告 $r_f$、独立生成式 probe、relearning、量化恢复、retain NLL、同实体非目标关系、donor 答案率、其他错答、拒答、提前 EOS、完整回答率和长度分布。至少两类独立抗恢复证据改善后，才允许使用受限“深度遗忘”措辞。

建议顺序：先 CF2-01 与 CF2-02 的闭式机制判别；再 CF2-03 的线性低秩估计；CF2-04 仅在 UOT 未配对质量跨 split 稳定后继续。末两名先过语义归属/交互稳定性，未过门则不进入训练。本轮所有 pilot 均标 `needs manual pilot`，未消耗 GPU。

## 8. 外部评审与证据状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| research-lit | done（composed） | 16 组 Web 查询；无本地 PDF；D1 批量核验不可用 |
| idea-creator | done（sequential lenses） | 16 生成 → 6 淘汰 → 10 入选；未使用子代理 |
| novelty-check | PROVISIONAL | 有逐项近邻与差异，但无异族 reviewer verdict-id |
| research-review | REVIEW_UNAVAILABLE | 当前无 Codex/manual/Claude/Grok 等注册评审工具，也无 `.aris/reviewer.json` |
| idea-evaluator | rubric applied，PROVISIONAL | 使用本机真实 SKILL.md 与 references；分数不是独立回执 |
| research-refine-pipeline | 未运行 | 用户要求 10 个候选及各自 proposal，不是 top-1 实验规格 |
| research-wiki ingest | SKIPPED | helper/scanner 不可用；未读取未扫描 query_pack，未写 wiki |
| evidence gate | **BLOCKED** | 缺 `run_state.py`、`idea_discovery_gate.py` 及两份跨模型回执 |

因此，本轮交付是“完整候选调研包”，不是通过 ARIS 五层投稿审计或完整 W1 evidence gate 的 Final Report。

## 9. Proposal 索引

1. [CF2-01 跨层反事实通量](CF2-01_跨层反事实通量/CF2-01_proposal.md)
2. [CF2-02 因子绑定交互](CF2-02_因子绑定交互/CF2-02_proposal.md)
3. [CF2-03 条件有向信息流](CF2-03_条件有向信息流/CF2-03_proposal.md)
4. [CF2-04 不平衡传输质量销毁](CF2-04_不平衡传输质量销毁/CF2-04_proposal.md)
5. [CF2-05 跨层答案小波谱](CF2-05_跨层答案小波谱/CF2-05_proposal.md)
6. [CF2-06 反事实割线敏感度](CF2-06_反事实割线敏感度/CF2-06_proposal.md)
7. [CF2-07 三阶累积量绑定](CF2-07_三阶累积量绑定/CF2-07_proposal.md)
8. [CF2-08 保留协方差白化谱](CF2-08_保留协方差白化谱/CF2-08_proposal.md)
9. [CF2-09 稳健低秩稀疏事件](CF2-09_稳健低秩稀疏事件/CF2-09_proposal.md)
10. [CF2-10 联盟交互激活擦除](CF2-10_联盟交互激活擦除/CF2-10_proposal.md)

## 10. 核心来源

- PINT：https://arxiv.org/html/2611.03955v1
- Transfer Entropy for LLMs/VLMs：https://openreview.net/forum?id=rFAQRmjVMf
- AcT：https://arxiv.org/abs/2410.23054
- UOT-Unlearn：https://arxiv.org/pdf/2603.16489
- Conditional Scale Entropy：https://arxiv.org/abs/2605.21391
- Relation Decoding Tensor Networks：https://arxiv.org/pdf/2510.26543
- CIR：https://arxiv.org/html/2509.11816
- Counterfactual Invariance：https://arxiv.org/html/2610.22942
- 基线与旧候选：`workspace/idea-stage/方向I_基线_v0.6.1_answer_span.md`、`workspace/idea-stage/candidates/`。

深度阅读优先建议：PINT（统计独立性强基线）、Transfer Entropy（有向激活流）、Factorized Causal Activation Patching（因子交互）。其余文献按各 proposal 的具体机制选读。

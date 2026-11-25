---
id: R2-B
title: ASU 温度课程 τ(k) —— 把"何时给温度"变成免费自由度
status: CANDIDATE · PROMOTE
decision_source: workspace/idea-stage/IDEA_REPORT_V4.md（PROMOTE）、workspace/idea-stage/docs/research_contract.md
domain: LLM Unlearning / forget-teacher 自蒸馏 / ASU 机制再设计
evidence_level: provisional（H20 单 seed 实测；无跨模型独立评审回执）
baseline_relation: 与方向 I 基线 v0.6.1 正交（见 §7）
---

# R2-B · ASU 温度课程 τ(k)

## 0. 故事（30 秒讲稿）

> ASU（ICLR 2026）用**一个固定标量** $\tau>1$ 平滑注意力，让冻结 teacher 的分布变钝，学生用 KL 逼近它来"忘掉"事实。所有人都把 $\tau$ 当作训练前要 binary search 的静态超参。我们追问一个从没人问的问题：**温度该在训练的什么时候给？** 实测答案是——沿训练步数从低到高给 $\tau$（`low2high`，$1.5\to3$），在等步数、零新增参数下比固定 $\tau=2$ 多拿到 $+0.149$ 的 forget-NLL 增益，且 retain 不退化。故事不是"我们加了一个新模块"，而是"我们发现时间维度是一个被忽略的、免费的自由度；结构化地分配温度到层/头反而被等平均对照否决"。这与 `NARRATIVE_ASU_TEMPERATURE_DEPTH.md` 的主线相互呼应：创新从"提出新机制"收缩为"发现更有效的调度 + 给出诚实的度量"。

## 1. 领域问题与空白（思路来源）

LLM unlearning 要在不重训的前提下从模型中移除特定知识，同时保住通用能力。ASU 的做法是自蒸馏：teacher 与 student 同权重，但 teacher 的注意力 logits 除以 $\tau>1$ 使 softmax 变平（事实 token 间关联被削弱），student 用输出分布 KL 逼近这个钝化 teacher。ASU 论文自承**温度选择脆弱、需 binary search**，并把 "dynamic temperature adjustment / layer-specific interventions" 明确列为 future work（来源：`IDEA_REPORT_V3.md` §查新发现 2、`NARRATIVE_ASU_TEMPERATURE_DEPTH.md` §2）。

据此可拆出三个正交自由度，本方向只取第三个：

- **给谁**（结构维度：层/头分配 $\tau_{\ell,h}$）→ 这是 R3-1 / SensiTemp，V4 实测被等平均 $\tau$ 全局对照否决。
- **给多大**（标量 $\tau$ 的取值）→ ASU 原论文，已做。
- **何时给**（训练时间维度 $\tau(k)$）→ **本候选，此前为空白**。

关键空白：所有已有工作把 $\tau$ 当训练前冻结的标量；$\tau$ 作为训练进度函数从未被系统研究。而 stage2/3 的实测（$\lambda/\beta$ 权重对 forget-retain 权衡敏感，$\lambda=0$ 时 forget 0.764 但 retain 也降）提示存在可调度的空间（来源：`IDEA_REPORT_V2.md` Idea B）。

## 2. 核心假设（可证伪）

**H**：forget-teacher 的温度存在最优时间课程——训练早期/后期对"压制事实检索"与"恢复 retain 表达"的需求不同；沿时间调度 $\tau$ 能在等预算下**帕累托改进** forget-retain 权衡，而不只是在固定权衡曲线上移动一个点。

**必须同时排除的替代解释**：收益不得来自"平均平滑总量更强"。因此对照里强制包含**等平均 $\tau$ 全局**——若等平均 $\tau$ 追平 `low2high`，则结构化课程无净收益，本方向降级（这正是 R3-1 SensiTemp 被推翻的同一教训，见 `IDEA_REPORT_V4.md` R3-1 行）。

## 3. 具体实现

**基座与复现配置**（来自 `NARRATIVE_ASU_TEMPERATURE_DEPTH.md` §1，H20 / TOFU / `tofu_Llama-3.2-1B-Instruct_full`）：

- ASU 自蒸馏，forget = TOFU `forget10`，retain = `retain90`；answer-token NLL $F(\theta),R(\theta)$（$F$ 越大越答不出被忘事实，$R$ 越小通用能力越好）。
- 训练 $60$ steps、$\alpha=0.1$、$\mathrm{lr}=5\times10^{-6}$、seed $0$。**诚实标注**：ASU 原实现默认 $\alpha=1.0$；本复现把 $\alpha$ 降到 $0.1$，因为 $\alpha=1$ 时初始 KL $\approx8.3$ 会在前两步推坏 retain（实测 $R:0.465\to1.016$）。这是复现配置偏离，不是 ASU 原结论。

**温度注入点**（注意力 logits，不是输出 logits）：

$$A_{\ell,h}(x;\tau(k))=\operatorname{softmax}\!\left(\frac{\mathbf Q_{\ell,h}(x)\mathbf K_{\ell,h}(x)^{\top}}{\tau(k)\sqrt{d_h}}+M_{\text{causal}}\right)$$

**课程定义**（$k$ = 训练步，$K=60$）：

| 臂 | $\tau(k)$ | 角色 |
|---|---|---|
| `low2high` | $1.5\to3$ 线性 | **默认主线** |
| `high2low` | $3\to1.5$ 线性 | 消融 |
| `global` | 恒 $2$ | 原 ASU 对照 |
| `equal-mean-global` | 恒 $\overline{\tau(k)}$ | **关键对照**：排除"只是更强平滑" |
| `random` | 每步随机重排课程 | 排除"任何随时间变化都有效" |

训练循环：每步用当前 $\tau(k)$ 构造冻结 forget-teacher → student 前向 → 标准 ASU KL 更新 student；teacher/课程表均 stop-gradient，无新增参数、无对抗训练。

**统计口径**：统计单位为作者/实体簇，多 prompt/多 seed 为簇内重复测量；必须报 3 seed 的 std（当前证据仅 seed 0，见 §5/E2）。

## 4. 与近邻工作的差异化（投稿划界）

| 近邻 | 机制 | 本候选差异 |
|---|---|---|
| ASU（Zare Zade et al., ICLR 2026, arXiv:2603.01285） | 固定标量 $\tau$ | 首次把 $\tau$ 作为训练进度函数；不改架构、不加参数 |
| R3-1 / SensiTemp（本仓库 V3） | 结构维度给层/头分配 $\tau_{\ell,h}$ | 本候选是**时间维度**；且 SensiTemp 已被等平均对照否决，本候选把该对照设为强制门 |
| RepSelect (arXiv:2606.17168) / ADU (arXiv:2608.23020) / ALMPU (AAAI 2026) | head 级/子空间新模块 | 本候选不训练任何新模块，只调度已有 teacher 的超参 |
| FDD / temperature scheduling 经验（ACL 2025） | 用于蒸馏压缩、教师更强 | 本候选用于遗忘、teacher 是同权重钝化视图 |

**差异化落点**：贡献不在"温度调度"这一组件（调度本身有先例），而在"ASU forget-teacher 的时间课程 + 等平均对照证伪"这一组合，以及与方向 I"去 $\tau$"路线构成的正反对照。若 `equal-mean-global` 追平，须诚实标为"权衡曲线移动"，不构成贡献。

## 5. 当前证据与状态（provisional，非跨模型确认）

来自 `IDEA_REPORT_V4.md`、`research_contract.md`、`NARRATIVE_ASU_TEMPERATURE_DEPTH.md`（H20 / TOFU-1B / 60-step 等预算 / seed 0）：

- 60-step 等预算 $dF$：`global=0.800`、`high2low=0.562`、`low2high=0.949`；最佳 `low2high` $dF=+0.949$、$dR=-0.256$。
- `low2high` 相对 `global` 的 forget-NLL 增益 $=+0.149$，retain 未受损。
- 决策：`PROMOTE`（将 `low2high` 设为默认 ASU forget-teacher 调度；`high2low` 仅作消融）。
- **未完成义务 E2**：3 seed 重复确认 `low2high > global`（本轮为 seed 0）。跨模型评审后端不可用，novelty/research-review 为 `PROVISIONAL / REVIEW_UNAVAILABLE`，不声称 accepted。

## 6. 判别实验、失败模式、必测指标

**决定性判别（先于写入结论）**

1. 3 seed 复现 `low2high > global`（E2）。
2. `equal-mean-global` 对照：若追平 `low2high`，收益归因为"更强平滑"而非"课程"，本方向降级。
3. `random` 课程对照：若同样有效，则"时间结构"无信息。
4. 等算力/等 token/等前后向/等参数/等 GPU 预算。

**失败模式（命中即登记 NEGATIVE，不放宽阈值凑指标）**

- 3 seed 下增益消失或 std 过大；
- `equal-mean-global` 或 `random` 追平；
- 增益伴随提前 EOS / 完整回答率下降 / 拒答上升（输出抑制捷径，非选择性遗忘）；
- 正确率/NLL 下降单靠改答或截断实现。

**必测指标**：$F(\theta),R(\theta)$ 曲线、3-seed std、长度分布、提前 EOS 率、完整回答率、拒答率、独立 retain 效用；统计单位作者簇。

## 7. 与方向 I 基线 v0.6.1 的关系（诚实标注）

两条**机制轴相反、并行**的线，不可混为一谈：

- 方向 I 基线 `workspace/idea-stage/方向I_基线_v0.6.1_answer_span.md` 的前提是**去掉 $\tau$**：去温度后 Teacher=Student 导致初始化零遗忘梯度，于是改用"反事实对照输入 $X^-/X^+$ + 答案跨度 $S_f$ 上的中性约束 $\mathcal L_F$"给出遗忘方向，DKT-KL 降为消融臂。
- 本候选的前提是**保留 $\tau$ 并调度它**，完全不依赖 donor / 反事实配对 / 答案跨度。

因此本候选可作为方向 I 的 **stripped-down ASU 对照基线**（最简已验证机制），但不能为方向 I 的 $\mathcal L_F$ 提供正向证据。同时须注意：方向 I 基线所属的反事实配对路线在 V4 中被整批 `ELIMINATED`（见整合报告 `IDEA_REPORT.md` 的"口径张力"一节），两者关系需由用户裁决，本文件不自行判定基线有效性。

## 8. 诚信声明与来源

本文件为**候选 proposal（plan-level）**，整合自本仓库成文报告，不含新实验；所有数字标注来源，未核实处用"据不完全证据显示/暂定"。跨模型评审不可用，不伪造回执。所有新结果须过 `/experiment-audit` 与 `/result-to-claim`。

- `workspace/idea-stage/IDEA_REPORT_V4.md`（决策表与 $dF$ 数据）
- `workspace/idea-stage/IDEA_REPORT_V2.md` §Idea B（$\tau$ 课程动机）、`IDEA_REPORT_V3.md` §查新发现 2（ASU future work）
- `workspace/idea-stage/docs/research_contract.md`（E2 义务、默认调度）
- `workspace/idea-stage/NARRATIVE_ASU_TEMPERATURE_DEPTH.md`（ASU 数学、$\alpha=0.1$ 偏离、$+0.149$、$44.8\%$ 恢复）
- 文献：ASU arXiv:2603.01285 (ICLR 2026)、RepSelect arXiv:2606.17168、ADU arXiv:2608.23020、ALMPU DOI 10.1609/aaai.v40i23.39017、FDD ACL 2025 DOI 10.18653/v1/2025.acl-long.1125。

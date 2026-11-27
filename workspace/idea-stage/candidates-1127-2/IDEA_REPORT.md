---
id: IDEA-DISCOVERY-20261127-CF3
title: 反事实激活差驱动的可解释 LLM 机器遗忘——第三轮差异化候选
date: 2026-11-27
mode: 文献调研 + 构思 + idea-evaluator 原 rubric 兼容评估；无 GPU、无训练、无部署
baseline: workspace/idea-stage/方向I_基线_v0.6.1_answer_span.md（按本轮用户指令重新开放构思）
verdict_type: 执行者机制评估（PROVISIONAL），不是独立跨模型接受回执
evidence_gate: BLOCKED（完整 W1 所需跨模型回执与确定性证据门未完成）
---

# 反事实激活差驱动的可解释 LLM 机器遗忘——第三轮候选

## 0. 执行摘要

本轮围绕“Teacher 不使用注意力温度，而从正反事实对的中间层激活差构造遗忘机制”重新开放构思。以 `方向I_基线_v0.6.1_answer_span.md` 为问题锚点，读取并对照既有 `candidates/`、`candidates-1127/` 的机制与复评结论，生成 18 个新草案；经定向检索、致命缺陷审计和五维评分，淘汰 8 个，保留 10 个。

最终 10 项平均分均严格大于 7，范围为 **7.2–8.0**，裁决全部为 **Accept with Revisions（PROVISIONAL）**。没有为达标而降低评分标准：NTK、离散码本、通用因果中介、Fisher 几何等看似可包装的方向均因直接近邻或语义不可证伪而被淘汰；保留项也没有被拔高为 Strong Accept。

首推 **CF3-01 双线性路由—内容差分**：用精确恒等式区分“看哪里”与“写入什么”，低成本且直接针对原始 $\Delta A$ 的角色混淆。其次是 **CF3-02 自校正持续子空间**：只处理经下游 replay 后仍持续的差异，避免把会被网络自然校正的瞬态变化误当事实。

> **状态边界**：全部分数为 mechanism-based、plan-only，不是实验成绩，也不是独立 reviewer 的论文接受。`idea-evaluator` 未注册为当前可调用技能，本轮读取并执行其本机原始 `SKILL.md` 与 references；未伪造技能调用或跨模型 verdict-id。所有 pilot 均为 `needs manual pilot`。

## 1. 基线、历史负结果与反重复边界

共同保留的硬边界：Teacher 为冻结基座且不使用注意力温度；正反事实输入槽外一致；默认在 $X^\pm\oplus y^+$ 的答案跨度 $p\in S_f$ 读取激活；Teacher 与目标 stop-gradient；不拼 donor 答案 $y^-$；主结果不依赖 donor KL；Student 参数内化，部署无 Teacher、patch 或运行时 hook。

本轮将基线中的历史归档视为风险与反重复资料，而本轮用户明确要求重新探索该方向，故只开放构思，不把历史 stage0–14 负结果改写成正向证据。直接对齐 $A^-$、原始 $\Delta A$ 正交抑制、donor soft-target KL、单纯 mask/梯度手术均不进入新主线。

明确避开两轮既有对象：约束反演、三前向角色分解、单点因果门控、事实/身份双探针、多 donor 边际化、LRM 轨迹、读出 Jacobian、retain 零空间、结构化残差、纯度 schedule、跨层通量、因子双差分、条件信息流、不平衡传输、小波、割线、三阶累积量、retain 白化、稳健低秩稀疏和 Shapley 联盟。

## 2. Literature Landscape

本轮使用多组实时 Web 查询覆盖 arXiv、OpenReview、ACL Anthology、CVF 与项目资料；本地 `papers/`、`literature/` 未形成可用的专题论文库。文献身份未完成 D1 批量核验，因此以下来源只用于元数据级近邻划界，不生成 BibTeX，不引用未经核对的性能数字。

### 2.1 遗忘不再等于输出改变

- **PARS / Do LLMs Really Forget?**（2026，arXiv:2609.36612）区分输出抑制与隐藏态信息残留，并以生成式 probe 和对抗目标处理泄露。本轮所有方案必须经独立 probe/relearning，不能只报输出变化。
- **Erase or Hide?**（ICLR 2026，arXiv:2509.22263）指出新增负向抑制神经元可形成浅层抑制壳。所有删除目标必须同时监测原始正向通路削弱与新增抑制。
- **Unlearning Isn't Deletion**（arXiv:2505.16831）及 **UDS** 强调 relearning、表示几何与因果 patch 的必要性，限制“向量差变小即遗忘”的叙事。

### 2.2 表征遗忘已有强近邻

- **CIR**（2025，arXiv:2509.11816）、**LEACE**（NeurIPS 2023）、**ReGLU**（2026，arXiv:2604.17396）和 **PTP-U**（2026）已覆盖线性擦除、retain 子空间与投影保护。因此单纯 Fisher/零空间/白化候选不再具备足够增量。
- **LUNAR**（2025，arXiv:2502.07218）覆盖向“无法回答”激活区域重定向；本轮任何中性目标都不能直接复制该故事。
- **SCALPEL** 与多种 SAE 遗忘工作占据稀疏特征定位；本轮不将 SAE 换成反事实输入后重新包装。

### 2.3 反事实、路径与动态机制

- **D2F** 与 **CAS-BiPO** 已覆盖语义中介遗忘和稀疏中介激活控制；通用 causal mediation 草案被淘汰。
- **IG-Lens**（2026，arXiv:2606.29693）已做跨层完备概率归因；CF3-07 只能落在“主体反事实路径的激活差分配稳定性 + Student 遗忘”，不能声称首次 IG。
- **Transformer Dynamics**（2025，arXiv:2502.12131）和 **LLM Layers Immediately Correct Each Other**（2026）提供自校正依据；CF3-02 的新对象是受控 replay 后的持续差异子空间。
- **Trajectory Signatures of Deception**、**High-dimensional layer-state trajectories**（2026，arXiv:2610.26866）与签名序列模型说明路径/轨迹近邻密集；CF3-10 只有在非交换面积超出端点和通量时才成立。

### 2.4 离散状态、尾部与信息流

- **CodeUnlearn**（2024，arXiv:2410.10866）直接覆盖离散概念码删除，因此码本草案被拒绝；CF3-04 不增加码本，而处理 SwiGLU 预激活的四状态转移。
- **GLUScope**（2026，arXiv:2602.23826）和 **The Sign Code**（2026，arXiv:2610.23616）支持门控状态作为可检验对象，但不构成遗忘证据。
- **SuperActivators**（2026，arXiv:2512.05038）支持尾部概念信号；**PINT**（2026，arXiv:2611.03955）占据统计独立性遗忘。CF3-08 必须限定为剥离边缘后的跨位置 copula 尾依赖差。

**暂定空白**：据当前检索，尚未发现系统比较“精确双线性差分、下游生存子空间、访问测度加权、SwiGLU 状态转移、答案跨度 Grassmann 投影差、配对置换 FDR、多路径稳定归因、跨位置 copula 尾依赖、双端 Hankel 模态、非交换 log-signature”并逐项内化到 Student 的工作。这不是首创性证明；未检索到不等于不存在。

## 3. 评分与裁决口径

按本机 `~/.bg-agent/config-with-app/skills/idea-evaluator/SKILL.md` 及其 `fatal-flaws.md`、`five-dimensions.md`、`lifecycle-capability-matching.md`、`paradigm-shift-probe.md` 执行兼容评估：先审计 F1–F10，再检查生命周期/资源，随后按 Higher / Faster / Stronger / Cheaper / Broader 从 5 起评分，并给出范式探测和决定性实验。

$$
\bar S=(H+F+S+C+B)/5.
$$

用户门槛为 $\bar S>7$ 且裁决至少 Accept with Revisions。所有高于 5 的分值均在各 proposal 中给出具体机制依据，统一标注 `mechanism-based, not yet confirmed by data`。由于每项仍至少有一个 MAJOR 风险，本轮不使用 Strong Accept。

## 4. 最终排名

| 排名 | ID | 核心激活处理 | H | F | S | C | B | 平均 | 裁决（PROVISIONAL） |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | CF3-01 | 双线性路由/内容精确差分 | 8 | 8 | 8 | 8 | 8 | **8.0** | Accept with Revisions |
| 2 | CF3-02 | 下游 replay 持续子空间 | 8 | 7 | 9 | 7 | 8 | **7.8** | Accept with Revisions |
| 3 | CF3-03 | Student 状态访问测度加权 | 8 | 7 | 8 | 7 | 8 | **7.6** | Accept with Revisions |
| 4 | CF3-04 | SwiGLU 四状态转移图 | 8 | 7 | 8 | 8 | 7 | **7.6** | Accept with Revisions |
| 5 | CF3-05 | 答案跨度 Grassmann 投影差 | 7 | 8 | 8 | 8 | 7 | **7.6** | Accept with Revisions |
| 6 | CF3-06 | 配对符号置换 + BY-FDR | 7 | 8 | 8 | 8 | 7 | **7.6** | Accept with Revisions |
| 7 | CF3-07 | 多路径完备归因稳定分量 | 8 | 6 | 8 | 7 | 8 | **7.4** | Accept with Revisions |
| 8 | CF3-08 | retain 秩坐标 copula 尾依赖差 | 8 | 6 | 9 | 6 | 8 | **7.4** | Accept with Revisions |
| 9 | CF3-09 | 双端可控/可读 Hankel 模态 | 8 | 6 | 8 | 6 | 8 | **7.2** | Accept with Revisions |
| 10 | CF3-10 | 非交换二阶 log-signature 面积 | 8 | 5 | 8 | 6 | 9 | **7.2** | Accept with Revisions |

并列时按零训练判别价值、闭式程度、近邻强度、首验成本与反重复距离排序。CF3-09/10 虽过线，但估计成本和近邻风险明显更高，不建议先于前五项投入训练。

## 5. 逐项 Novelty Verification

| ID | 最强近邻 | 可验证差异轴 | 执行者查新判断 |
|---|---|---|---|
| CF3-01 | Contribution Weights；QK/V causal decomposition | 双线性恒等式直接成为非 donor 删除对象，并统一 attention/GLU | PROCEED WITH CAUTION |
| CF3-02 | Transformer Dynamics；TLCM；CF2-01 | 多个受控下游 replay 深度形成生存子空间，不是相邻层差分 | PROCEED WITH CAUTION |
| CF3-03 | DKT；CF-06；CF-05 | 当前 Student 的状态访问测度决定共享前缀激活差权重 | PROCEED WITH CAUTION |
| CF3-04 | GLUScope；Sign Code；CodeUnlearn | 无码本的 SwiGLU 源状态迁出与参数内化 | PROCEED WITH CAUTION |
| CF3-05 | CIR/LEACE；Grassmann readout；CF2-08 | 答案位置张成的投影差，不是 donor 维 SVD 或 retain 白化 | PROCEED WITH CAUTION |
| CF3-06 | 神经影像置换/FDR；旧 null/纯度门 | 可交换配对零假设 + 任意依赖 FDR + 独立 patch 门 | PROCEED WITH CAUTION |
| CF3-07 | IG-Lens；多路径 IG；CF2-06 | 主体替换路径对中间激活差的完备分配稳定性 | PROCEED WITH CAUTION |
| CF3-08 | SuperActivators；PINT；CF2-03 | 剥离边缘后的正反事实跨位置尾依赖，而非均值/协方差或全局独立 | PROCEED WITH CAUTION |
| CF3-09 | Control-Theoretic Interpretability；CF-07；Fast-NTK | 主体输入与答案读出的双端平衡模态，不使用参数 NTK | PROCEED WITH CAUTION |
| CF3-10 | 路径签名 Transformer；CF2-01/05/07 | 非交换有序面积作为遗忘对象，并有反转/重参数化代数检验 | PROCEED WITH CAUTION |

这些判断不是跨模型 novelty accept receipt；只有检索到直接重叠论文或明确历史数据时才使用 ABANDON/Reject。

## 6. 淘汰的 8 个新草案

| 淘汰草案 | 原因 | 裁决 |
|---|---|---|
| 反事实 NTK 参数核擦除 | Fast-NTK 与 NTK scrubbing 是直接强近邻，仅替换为配对激活差的增量不足；机制复评均值约 6.6 | Reject and Pivot |
| 离散码本转移删除 | CodeUnlearn 已覆盖离散概念码定位与删除，核心对象重合 | Reject and Pivot（F1） |
| 通用自然直接/间接中介遗忘 | D2F、CAS-BiPO、GCM 已形成密集近邻，无法提出独立单一贡献 | Reject and Pivot（F1） |
| Fisher-retain 反事实测地线 | FDCU、ReGLU、PTP-U、CF2-08 已覆盖保留几何；机制复评均值约 6.8 | Reject and Pivot |
| 层×位置 Hodge 旋度遗忘 | 从工具出发寻找问题，旋度与事实语义之间缺少可判别链条 | Reject and Pivot（F9/F6） |
| 反事实 SVM 支持边界擦除 | 仍是线性概念方向，主体身份与事实边界不可分；机制复评均值约 6.6 | Reject and Pivot |
| 可逆 Procrustes 传输残差 | 与 OT/Householder 近邻密集，且同一 Teacher 下“坐标规范”动机不足 | Reject and Pivot |
| 主体插值 Hessian 曲率冲击 | 与 CF2-01/05/06 及路径签名重叠，易沦为高阶算子替换；机制复评均值约 6.8 | Reject and Pivot |

## 7. 共同验证门与建议顺序

所有候选先过零训练缓存判别，再进入 Student 训练。真实 donor 必须优于 wrong、shuffle、noise、matched-retain 和随机等能/等秩对照；比较必须等删除能量、等步数、等前后向与等 GPU 预算。

联合报告原事实 $r_f$、独立生成式 probe、relearning、量化恢复、retain NLL、同实体非目标关系、donor 答案率、其他错答、拒答、提前 EOS、完整回答率和长度分布。统计单位是作者/实体簇，多 prompt、多 donor 和 seed 只作为簇内重复测量。至少两类独立抗恢复证据改善后，才允许使用受限“深度遗忘”措辞。

建议顺序：先做 CF3-01 的精确分量 patch 与 CF3-02 的少量 replay 生存谱；再比较 CF3-03 的等预算 on-policy 收益；CF3-04/05/06 作为低成本备选；CF3-07/08 需先证明估计稳定；CF3-09/10 只有通过独立信息门才进入训练。

## 8. 评审与证据状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| research-lit | done（composed 到本报告） | 多组 Web 查询与本地项目资料；文献身份未完成 D1 批量核验 |
| idea-creator | sequential lenses fallback | 18 个草案 → 8 个淘汰 → 10 个入选；未使用子代理或外部生成模型 |
| novelty-check | PROVISIONAL | 有逐项近邻和差异轴，无独立 reviewer verdict-id |
| idea-evaluator | 原 rubric 已读取并兼容执行，PROVISIONAL | 当前未注册为可调用技能；执行者评分不构成独立评审 |
| research-review | REVIEW_UNAVAILABLE | 本轮没有获得异族独立 reviewer 回执 |
| research-refine-pipeline | 未运行 | 用户要求是 10 个候选与独立 proposal，不是 top-1 实验规格 |
| GPU pilot | NOT_RUN | 无训练、无部署、无新实验结果；全部 `needs manual pilot` |
| evidence gate | **BLOCKED** | 不将本候选包呈现为完整 W1 PASS 或投稿审计通过 |

## 9. Proposal 索引

1. [CF3-01 双线性路由—内容差分](CF3-01_双线性路由内容差分/CF3-01_proposal.md)
2. [CF3-02 自校正持续子空间](CF3-02_自校正持续子空间/CF3-02_proposal.md)
3. [CF3-03 访问分布加权反事实激活差](CF3-03_访问分布加权激活差/CF3-03_proposal.md)
4. [CF3-04 门控状态转移擦除](CF3-04_门控状态转移擦除/CF3-04_proposal.md)
5. [CF3-05 答案跨度 Grassmann 子空间擦除](CF3-05_答案跨度Grassmann子空间/CF3-05_proposal.md)
6. [CF3-06 配对置换—FDR 激活门](CF3-06_配对置换FDR激活门/CF3-06_proposal.md)
7. [CF3-07 多路径积分稳定归因](CF3-07_多路径积分稳定归因/CF3-07_proposal.md)
8. [CF3-08 反事实 Copula 尾依赖擦除](CF3-08_反事实Copula尾依赖/CF3-08_proposal.md)
9. [CF3-09 双端可控—可读 Hankel 激活模态](CF3-09_双端Hankel激活模态/CF3-09_proposal.md)
10. [CF3-10 非交换路径签名遗忘](CF3-10_非交换路径签名遗忘/CF3-10_proposal.md)

## 10. 核心来源

- PARS：https://arxiv.org/abs/2609.36612
- Erase or Hide?：https://arxiv.org/abs/2509.22263
- CIR：https://arxiv.org/abs/2509.11816
- Transformer Dynamics：https://arxiv.org/abs/2502.12131
- IG-Lens：https://arxiv.org/abs/2606.29693
- SuperActivators：https://arxiv.org/abs/2512.05038
- CodeUnlearn：https://arxiv.org/abs/2410.10866
- Fast-NTK：https://arxiv.org/abs/2312.14923
- Control-Theoretic Interpretability：https://arxiv.org/abs/2511.12852
- High-dimensional layer-state trajectories：https://arxiv.org/abs/2610.26866
- 基线与旧候选：`workspace/idea-stage/方向I_基线_v0.6.1_answer_span.md`、`workspace/idea-stage/candidates/`、`workspace/idea-stage/candidates-1127/`。

深度阅读优先建议：`PARS`（隐藏态泄露与恢复验收）、`LLM Layers Immediately Correct Each Other`（自校正机制）、`Contribution Weights`（注意力路由与 value 几何）。其余文献按具体 proposal 的近邻选读。

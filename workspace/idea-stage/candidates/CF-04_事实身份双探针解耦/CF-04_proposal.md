---
id: CF-04
rank: 4
title: 事实 / 身份双探针正交解耦 ΔA_fact
avg_score: 7.4
verdict: Strong Accept (PROVISIONAL, mechanism-based)
decision_source: IDEA_DISCOVERY_20261126.md §5
domain: LLM/LRM Unlearning / 激活差子空间解耦
evidence_level: provisional（机制推断，无实测，无跨模型回执）
baseline_relation: 把 §2.2.1「ΔA 混有实体身份」的 killer risk 定稿为双探针正交收缩
---

# CF-04 · 事实 / 身份双探针正交解耦 $\Delta A_{\text{fact}}$

## 0. 故事（30 秒讲稿）

> $\Delta A$ 最大的隐患是：它到底编码「目标事实」还是「换了哪个主体」？基线 §2.2.1 把这个列为头号 killer risk，V2 里 unpooled $\Delta A$ 与 wrong-donor 重叠达 $0.55$ 也印证了。我们训练两个冻结探针：事实探针读「$y^+$ 是什么」，身份探针读「主体是谁」。用正交收缩把 $\Delta A$ 拆成事实子空间与身份子空间，只保留与身份正交的 $\Delta A_{\text{fact}}$，再用「反事实激活潜力」补回被身份方向掩盖但对事实读出有贡献的分量。可解释性来自：事实 probe AUC 与身份 probe AUC 明明白白摆出来，读者能判断我们抹的到底是知识还是名字。

## 1. 领域问题与空白

- 基线 §2.2.1 第 1 点：原始配对差混有**实体身份、subword、词频、关系传播**，未证明编码目标事实而非主体身份；历史证据调研明确判定 $A^+-A^-$ 不能读成事实差异。
- V2 负结果：unpooled $\Delta A$ 与 wrong-donor 重叠 $0.55$（来源：`IDEA_REPORT_V3.md`），说明「换主体」信号可能压过「删事实」信号。
- **近邻空白**：StyliTruth(ICLR 2026) 用正交收缩解耦风格/事实、RepSelect 用 forget 梯度 top/bottom-PC、SCALPEL 用 contrastive SAE 选 forget 特征、MUtE(2026) 做连续表示概念擦除；**没有人把「事实 probe vs 身份 probe」双探针显式作为 $\Delta A$ 的解耦判据，并强制 wrong-donor 落在身份子空间这一可证伪检验**。

## 2. 核心假设（可证伪）

**H**：事实与身份在 $\Delta A$ 张成的子空间上可分；正交收缩后 $\Delta A_{\text{fact}}$ 仍满足 §4.1 中性约束且事实 probe AUC 显著高于身份 probe AUC；wrong-donor 的 $\Delta A$ 主要落在身份子空间。若两子空间几乎完全重合，或解耦残差无事实信息，则本思路被证伪。

## 3. 具体实现（LaTeX）

**双冻结探针**（轻量线性/读出，经冻结下游校准，不直接过 LM head）：

$$
z_{\text{fact}} = \phi_{\text{fact}}(\Delta A_{\ell,p})\in\mathbb R^{d_f},\qquad
z_{\text{id}} = \phi_{\text{id}}(\Delta A_{\ell,p})\in\mathbb R^{d_i},
$$

分别在「能否读出 $y^+$」与「能否读出主体 $s_f$ vs $s_r$」上训练，输出 AUC 记为 $\mathrm{AUC}_f,\ \mathrm{AUC}_{\text{id}}$。

**正交收缩（deflation）**：设身份方向基 $V_{\text{id}}$（身份 probe 的 top 奇异方向），把 $\Delta A$ 投影掉身份成分：

$$
\Delta A_{\text{fact}} = \big(I - V_{\text{id}}V_{\text{id}}^{\top}\big)\,\Delta A,\qquad
\Delta A_{\text{id}} = V_{\text{id}}V_{\text{id}}^{\top}\,\Delta A .
$$

**反事实激活潜力**（补回被身份掩盖但对 $r_f$ 有因果潜力的分量，呼应 CAP 2026「被抑制特征」视角）：

$$
\Delta A_{\text{pot}} = \Delta A_{\text{fact}} + \sum_{j}\lambda_j\, u_j,\qquad
u_j \perp V_{\text{id}},\ \ \big|\partial r_f/\partial u_j\big|>\tau_{\text{pot}},
$$

$$
\mathcal P = \mathrm{AUC}_f(\Delta A_{\text{pot}}) - \mathrm{AUC}_{\text{id}}(\Delta A_{\text{pot}}) \quad\text{（纯度证书）}.
$$

**目标与内化**（$\Delta A_{\text{pot}}$ 只作定位/读出/负约束，不作删除方向）：

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}\big[\,h^{+}_{\ell}(p)-\alpha\,\widehat{\Delta A_{\text{pot},\ell,p}}\,\big]\right),
$$

$$
\mathcal K=\{h:\ r_{f,p}(h)\le c,\ s_{d_j,p}(h)\le d_j,\ \lVert\Delta r_{E,p}(h)\rVert\le e\},
\qquad
\boxed{\ \mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R\ } .
$$

判别硬约束：wrong-donor $\Delta A^{\text{wrong}}$ 应满足 $\lVert V_{\text{id}}V_{\text{id}}^{\top}\Delta A^{\text{wrong}}\rVert / \lVert\Delta A^{\text{wrong}}\rVert \ge \pi_{\text{id}}$。

## 4. 差异化（novelty 落点）

| 近邻 | 本方案差异 |
|---|---|
| StyliTruth (ICLR 2026) | 后者解风格/事实用于生成；本方案解**事实/身份**并把「wrong-donor 落身份子空间」设为可证伪硬检验 |
| RepSelect (2606.17168) / V1 pooled $\Delta A$ PCA | 后者用相关性主成分且 V1 已证伪；本方案是双探针**监督解耦** + 纯度证书，不作相关性抑制 |
| SCALPEL (2609.31056) | 后者 contrastive SAE 选特征；本方案是显式双探针 + 正交收缩 + 反事实潜力，不训练 SAE |
| MUtE (2026) | 后者连续表示概念擦除、非 LLM 参数内化；本方案在 Transformer 中间层且接 Student 训练内化 |

**novelty 落点**：把「$\Delta A$ 主要编码主体身份」这一 killer risk 直接转成可证伪的双探针判别门，并给出纯度证书 $\mathcal P$ —— 这是 V1 缺的诚实门槛。

## 5. /idea-evaluator 评估（PROVISIONAL）

| 维度 | 分 | 证据（机制推断，未实测） | Lift |
|---|---|---|---|
| Higher | 8 | 显式剔除身份分量，应提升事实方向纯度与 forget-retain 权衡（`mechanism-based`） | 与 CF-03 因果门控交叉 |
| Faster | 7 | 两个冻结线性探针 + 正交投影，几乎零训练成本 | 复用缓存 $\Delta A$ |
| Stronger | 8 | 双探针 + wrong-donor 硬检验，抗「只换名字」捷径（`mechanism-based`） | 加 multi-paraphrase 对照 |
| Cheaper | 7 | 无 SAE、无标注、无外部 oracle | 复用 stage0 资产 |
| Broader | 7 | 可迁移到其它配对差纯化（`mechanism-based`） | 与 CF-06 联动 |

**平均分 $\bar S=37/5=7.4>7$。**

- **定位**：Novel Method；一句话故事见 §0。
- **致命缺陷审计**：1 MAJOR（F6：事实 probe AUC vs 身份 probe AUC + wrong-donor 硬检验须预注册）+ 2 MINOR（F1 近邻划界、F10 失败模式已命名）。零 CRITICAL。
- **生命周期**：Frontier exploration 3–9 个月，能力匹配绿。
- **范式探测**：First Principles=Yes、Elephant=Yes、Technology Cycle=Partial、Hamming=Partial。
- **可行性**：Compute 低、Data 低、Engineering 中、Timeline 中。
- **裁决**：**Strong Accept**（H/S 两维 8+、零 CRITICAL），`worth pursuing, pending the validation experiment`。
- **决定性验证实验**：事实/身份 probe AUC；wrong-donor $\Delta A$ 是否落身份子空间；解耦后 $\Delta A_{\text{fact}}$ 是否仍满足 §4.1 中性约束；`needs manual pilot`。
- 分数为执行者机制推断分；`novelty-check = PROVISIONAL`，非跨模型回执。

## 6. 判别实验、失败模式、必测指标

- **判别**：$\mathrm{AUC}_f-\mathrm{AUC}_{\text{id}}$ 是否显著为正；wrong-donor 身份投影占比是否 $\ge\pi_{\text{id}}$；解耦残差是否仍有事实信息。
- **失败模式**：事实与身份子空间几乎完全重合（呼应 V2 重叠 $0.55$）；解耦后残差无事实信息；probe 命名过度解释；$\Delta A_{\text{pot}}$ 把 donor 语义 reintroduce。
- **必测指标**：$\mathrm{AUC}_f,\mathrm{AUC}_{\text{id}},\mathcal P$、wrong/shuffled/noise-donor 对照、$r_f$、retain NLL、donor 答案率、长度/EOS、3 seed std。

## 7. 与方向 I 基线 v0.6.1 的关系

只把 §2.2.1 第 1 点（$\Delta A$ 未净化、混有身份）与第 3 点方法身份张力下的「判别实验」定稿；$S_f$ 作用域、Teacher stopgrad、§4.1 中性约束（$\Pi_\mathcal K$）、读出经冻结下游重算、$\lambda_{KL}^{\text{ablation}}=0$、§6 配对特异性全部沿用。不沿 $\Delta A$ 直接抑制（避 V1）。

## 8. 诚信声明与来源

plan-level，不新增实验；claim 待验证；分数机制推断、PROVISIONAL、`REVIEW_UNAVAILABLE`；新结果须过 `/experiment-audit` + `/result-to-claim`。

- 基线 §2.2.1/§4.1/§6.1；`IDEA_CANDIDATES.md` D4；`IDEA_REPORT_V3.md`（V2 重叠 0.55）
- 近邻（身份未 D1 核验，不作 BibTeX）：StyliTruth ICLR 2026、RepSelect arXiv:2606.17168、SCALPEL arXiv:2609.31056、CAP 2026、MUtE 2026

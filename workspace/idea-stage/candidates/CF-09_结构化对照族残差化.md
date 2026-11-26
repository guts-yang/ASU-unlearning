---
id: CF-09
rank: 9
title: 结构化对照族残差化 ΔA_resid
avg_score: 7.2
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
decision_source: IDEA_DISCOVERY_20261126.md §5
domain: LLM/LRM Unlearning / 激活差噪声基残差化
evidence_level: provisional（机制推断，无实测，无跨模型回执）
baseline_relation: 把 §6.1 的 5 类验收对照前移为 ΔA 的构造算子（残差化），而非仅事后验收
---

# CF-09 · 结构化对照族残差化 $\Delta A_{\text{resid}}$

## 0. 故事（30 秒讲稿）

> 基线 §2.2.1 说 $\Delta A$ 混着「关系传播、subword、词频」，§6.1 又列了 5 类验收对照（噪声、乱序 donor、wrong-donor、matched-retain、随机层/位置）。我们不让这些对照只当「事后打分」——把它们前移成 $\Delta A$ 的**构造算子**：构造一组伪配对（relation-swap、answer-paraphrase、subword-length matched、词频匹配）得到噪声基 $\Delta A_{\text{null}}$，从真实 $\Delta A$ 里回归掉共享结构成分，剩下 $\Delta A_{\text{resid}}$ 才进目标。可解释性来自：被减掉的每一类噪声都有名字，残差就是「扣除已知混淆后仍剩下的差异」。

## 1. 领域问题与空白

- 基线 §2.2.1 明列 $\Delta A$ 混有实体身份、subword、词频、关系传播；§6.1 把 5 类对照列为「配对特异性」验收，但没用于构造。
- CFT hidden costs (arXiv:2605.27083) 指出反事实语料的结构化属性抽取与互相矛盾问题；D8 思路（来源：`IDEA_CANDIDATES.md`）提出把对照族前移为残差化算子但当时未定稿。
- **近邻空白**：CLReg(2601.22028) 用 retain 负样本做对比分离但属正则；SCALPEL 用 contrastive SAE 选特征；**没有人把「结构化对照族」显式构造成噪声基，并在激活差空间做残差回归，且把「残差化前后输出抑制捷径是否下降」设为判别**。

## 2. 核心假设（可证伪）

**H**：由 relation-swap / paraphrase / subword-matched / frequency-matched 构成的伪配对张成噪声子空间 $N$，真实 $\Delta A$ 投影掉 $N$ 后所得 $\Delta A_{\text{resid}}$ 对 wrong-relation / matched-retain 对照仍有显著区分度，且残差化后 donor 答案率、提前 EOS、完整回答率不升（排除输出抑制捷径）。若 null 基吸收全部事实信号（残差无信息），或对照族本身引入 donor 语义（违反 §5 边界），则本思路被证伪。

## 3. 具体实现（LaTeX）

**对照族（伪配对）**：对同一 $s_f$ 构造不引入 $y^-$ 的对照输入：

$$
\mathcal C=\{\ X^{\text{rel-swap}},\ X^{\text{para}},\ X^{\text{subword}},\ X^{\text{freq}}\ \},\qquad
\Delta A^{(c)}_{\ell,p}=A^+_{T,\ell}(p)-A^{(c)}_{T,\ell}(p),\ c\in\mathcal C .
$$

**噪声基**（对照族张成的子空间，秩 $r_{\text{null}}$ 在 discovery 冻结）：

$$
N_{\ell,p}=\mathrm{span}\{\Delta A^{(c)}_{\ell,p}: c\in\mathcal C\},\qquad
\Pi_{N}=B_{\text{null}}B_{\text{null}}^{\top}\ \ (B_{\text{null}}=\mathrm{orth}(\mathcal C)).
$$

**残差化**：

$$
\Delta A_{\text{resid},\ell,p}=\big(I-\Pi_{N}\big)\,\Delta A_{\ell,p},\qquad
\Delta A_{\ell,p}=A^+_{T,\ell}(p)-A^-_{T,\ell}(p).
$$

**目标与内化**：

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}\big[\,h^{+}_{\ell}(p)-\alpha\,\widehat{\Delta A_{\text{resid},\ell,p}}\,\big]\right),
$$

$$
\mathcal K=\{h:\ r_{f,p}(h)\le c,\ s_{d_j,p}(h)\le d_j,\ \lVert\Delta r_{E,p}(h)\rVert\le e\},\qquad
\boxed{\ \mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R\ } .
$$

硬约束：对照族不拼 $y^-$ / `donor_fact_value`、不算 $\mathrm{logit}(y^+)-\mathrm{logit}(y^-)$（§5 donor 数据边界）；wrong-relation / matched-retain 对照同时作判别。

## 4. 差异化（novelty 落点）

| 近邻 | 本方案差异 |
|---|---|
| CLReg (2601.22028) | 后者 retain 负样本做对比正则；本方案是**结构化对照族噪声基 + 残差回归**，对象与算子不同 |
| CFT hidden costs (2605.27083) | 后者报告反事实语料冲突；本方案把「结构化属性」构造成可减掉的噪声基，并设残差信息判别 |
| SCALPEL (2609.31056) | 后者 contrastive SAE；本方案无 SAE、无训练提取器 |
| 基线 §6.1 五类对照 | **唯一改动点**：把「事后验收对照」前移为 $\Delta A$ 的**构造算子**（残差化） |
| V1 五对照只作验收 | 本方案对照族参与构造，且残差化前后测输出抑制捷径，不与 V1 同族 |

**novelty 落点**：把「配对特异性」从一句实验结论变成 $\Delta A$ 构造里的可解释减法（扣掉具名混淆项），并给出「残差是否仍有事实信息」的可证伪门。

## 5. /idea-evaluator 评估（PROVISIONAL）

| 维度 | 分 | 证据（机制推断，未实测） | Lift |
|---|---|---|---|
| Higher | 7 | 残差化降已知混淆，增益依赖「null 基不吸收事实」（`mechanism-based`） | 与 CF-03 因果门控叠加 |
| Faster | 8 | 对照族为若干次冻结前向 + 一次正交投影，纯前向（`mechanism-based`） | 批量缓存对照激活 |
| Stronger | 7 | 具名噪声基 + 输出抑制捷径判别，抗「减掉信号还不知道」（`mechanism-based`） | 加 wrong-relation 对照 |
| Cheaper | 7 | 无 SAE、无标注、无外部模型 | 复用 §6.1 对照协议 |
| Broader | 7 | 残差化算子可迁移其它配对差（`mechanism-based`） | 统一对照族接口 |

**平均分 $\bar S=36/5=7.2>7$。**

- **定位**：Novel Method；一句话故事见 §0。
- **致命缺陷审计**：**2 MAJOR** → (1) F6 残差信息 / 输出抑制捷径判别须预注册；(2) F1 与 CLReg / CFT / §6.1 五对照的划界须核验（易被判「把验收当方法」）。零 CRITICAL。故 `Accept with Revisions`。
- **生命周期**：Frontier exploration 3–9 个月，能力匹配绿。
- **范式探测**：First Principles=Partial、Elephant=Partial、Technology Cycle=No、Hamming=Partial → possible。
- **可行性**：Compute 中（多次前向）、Data 低、Engineering 中、Timeline 中。
- **裁决**：**Accept with Revisions**，`worth pursuing, pending the validation experiment`。**Revisions**：先写清「对照族前移 ≠ 把验收当贡献」的划界，再开跑。
- **决定性验证实验**：$\Delta A_{\text{resid}}$ 对 wrong-relation / matched-retain 对照的区分度；残差化前后 donor 答案率、提前 EOS、完整回答率；`needs manual pilot`。
- 分数为执行者机制推断分；`novelty-check = PROVISIONAL`，非跨模型回执。

## 6. 判别实验、失败模式、必测指标

- **判别**：残差是否仍有事实信息；对照族是否真吸收了 subword/词频/关系传播；输出抑制捷径是否下降。
- **失败模式**：null 基吸收全部事实信号（残差无信息）；对照族本身引入 donor 语义（违反 §5）；与 CFT「反事实语料互相矛盾」同型冲突；残差化退化为纯 $\mathcal L_F$。
- **必测指标**：残差范数占比、wrong-relation/matched-retain 区分度、$r_f$、retain NLL、donor 答案率、提前 EOS、完整回答率、3 seed std。

## 7. 与方向 I 基线 v0.6.1 的关系

把 §6.1 五类对照前移为 §2.2.1 的 $\Delta A$ 构造算子；$S_f$ 作用域、Teacher stopgrad、§4.1 中性约束（$\Pi_\mathcal K$）、§5 donor 数据边界、读出经冻结下游重算、$\lambda_{KL}^{\text{ablation}}=0$、§6 六条验收（五类对照仍全跑）全部沿用。不教假事实、不改 Student 为 donor 答案。

## 8. 诚信声明与来源

plan-level，不新增实验；claim 待验证；分数机制推断、PROVISIONAL、`REVIEW_UNAVAILABLE`；新结果须过 `/experiment-audit` + `/result-to-claim`。

- 基线 §2.2.1/§4.1/§5/§6.1；`IDEA_CANDIDATES.md` D8
- 近邻（身份未 D1 核验，不作 BibTeX）：CLReg arXiv:2601.22028、CFT arXiv:2605.27083、SCALPEL arXiv:2609.31056

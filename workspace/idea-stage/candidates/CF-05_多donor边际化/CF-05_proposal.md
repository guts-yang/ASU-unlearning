---
id: CF-05
rank: 6
title: 多 donor 边际化共有子空间 ΔA_marg
avg_score: 7.4
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
decision_source: IDEA_DISCOVERY_20261126.md §5
domain: LLM/LRM Unlearning / 跨 donor 统计分解
evidence_level: provisional（机制推断，无实测，无跨模型回执）
baseline_relation: 定稿 §2.2.1 待选路径 (b)「多 donor 边际化 / SVD 子空间」，donor 全来自 TOFU 内部保留集
---

# CF-05 · 多 donor 边际化共有子空间 $\Delta A_{\text{marg}}$

## 0. 故事（30 秒讲稿）

> 单个 donor 的 $\Delta A$ 混着「这个 donor 自己的身份特异性」（基线 §2.2.1 明列）。与其赌某一个 donor 干净，不如取 $K$ 个关系匹配的保留主体，做跨 donor 边际化：把「所有配对共有的低秩成分」当作事实相关方向 $\Delta A_{\text{marg}}$，把单 donor 私有残差当身份噪声投影掉。可解释性来自：共有成分跨 donor 稳定、私有成分随 donor 变，读者能直观看到「我们抹的是关系结构，不是某个人名」。

## 1. 领域问题与空白

- 基线 §2.2.1 待选路径 (b) 明列「多 donor 边际化 / SVD 子空间」但未定稿；v0.3 的 10 对缓存 `q_plus/q_minus` 长度不等且为外部真实人物 donor，不满足严格等长反事实配对，只能排错。
- CFT hidden costs (arXiv:2605.27083) 报告反事实语料不同 donor 梯度余弦近半为负、knowledge conflict $\mu\approx+0.024$——**多 donor 可能互相矛盾**，必须验证共有成分是否真存在。
- **近邻空白**：SCALPEL(2609.31056) 用同 anchor 多语义视图 + contrastive SAE；CLReg(2601.22028) 用同例双视图；**没有人在「反事实主体配对差」空间做跨 donor 的共有/私有统计分解，并把共有成分作为遗忘定位方向**。

## 2. 核心假设（可证伪）

**H**：$K$ 个关系匹配 donor 的 $\Delta A^{(k)}$ 存在一个跨 donor 共享的低秩子空间，其特征值显著高于私有残差；$\Delta A_{\text{marg}}$ 对目标事实 probe 的 AUC 显著高于私有成分，且随 $K$ 增大方向收敛。若共有成分主要编码主体身份，或不同 donor 梯度余弦近半为负（CFT 冲突同型），则本思路被证伪。

## 3. 具体实现（LaTeX）

**K-donor 集合**（全来自 TOFU 内部保留集、按作者/别名簇隔离）：

$$
X^{-}_{(k)}=X(s_{r_k},r),\ k=1..K,\qquad
\Delta A^{(k)}_{\ell,p}=A^+_{T,\ell}(p)-A^-_{T,(k),\ell}(p),\quad p\in S_f .
$$

**堆叠与边际化**（跨 donor 均值 = 一阶共有成分）：

$$
\overline{\Delta A}_{\ell,p}=\frac{1}{K}\sum_{k=1}^{K}\Delta A^{(k)}_{\ell,p},\qquad
M_{\ell,p}=\big[\Delta A^{(1)}_{\ell,p}\ \cdots\ \Delta A^{(K)}_{\ell,p}\big] .
$$

**共有子空间（SVD）**：取 $M_{\ell,p}$ 的左奇异向量中前 $r$ 个（跨 donor 一致方向）作共有基 $U^{(r)}_{\ell,p}$：

$$
M_{\ell,p}=U^{(r)}_{\ell,p}\Sigma^{(r)}_{\ell,p}V^{(r)\top}_{\ell,p}+E_{\ell,p},\qquad
\Delta A_{\text{marg},\ell,p}=U^{(r)}_{\ell,p}U^{(r)\top}_{\ell,p}\,\overline{\Delta A}_{\ell,p}.
$$

私有残差 $\Delta A_{\text{priv}}^{(k)}=\Delta A^{(k)}-\Delta A_{\text{marg}}$ 被投影掉；$K,r$ 在 discovery 冻结。

**目标与内化**：

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left(\Pi_{\mathcal K}\big[\,h^{+}_{\ell}(p)-\alpha\,\widehat{\Delta A_{\text{marg},\ell,p}}\,\big]\right),\qquad
\boxed{\ \mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R\ } .
$$

判别硬约束：$\mathrm{AUC}_f(\Delta A_{\text{marg}}) > \mathrm{AUC}_f(\Delta A_{\text{priv}})+\Delta_{\text{marg}}$；wrong/shuffled/noise/matched-retain 五类对照（§6.1）全跑。

## 4. 差异化（novelty 落点）

| 近邻 | 本方案差异 |
|---|---|
| CFT hidden costs (2605.27083) | 后者报告多反事实语料冲突；本方案把「冲突」转成可证伪门（共有成分须显著），并显式在激活差空间分解 |
| SCALPEL (2609.31056) / CLReg (2601.22028) | 后者是同 anchor 多视图 + SAE / 对比正则；本方案是跨 **donor 主体**的统计分解，对象不同 |
| V1 单 donor pooled $\Delta A$ PCA（已证伪） | 后者单 donor 相关性主成分直接抑制；本方案多 donor 边际化且只作定位/读出，不作删除方向 |
| 基线 §2.2.1 路径 (b) | **唯一改动点**：把「待选」定稿为 K-donor 共有子空间 $\Delta A_{\text{marg}}$ |

**novelty 落点**：把 donor 身份特异性从「单个配对的污染」变成「跨 donor 可统计分离的私有残差」，并给出随 $K$ 收敛的纯度曲线。

## 5. /idea-evaluator 评估（PROVISIONAL）

| 维度 | 分 | 证据（机制推断，未实测） | Lift |
|---|---|---|---|
| Higher | 7 | 边际化降方差、提纯度，但依赖「共有成分真存在」假设（`mechanism-based`） | 与 CF-03 因果门控叠加 |
| Faster | 7 | K 次冻结前向 + 一次 SVD，成本线性可控 | 批量前向缓存 |
| Stronger | 8 | 跨 donor 一致性 + 五类对照，抗单 donor 偶然/身份污染（`mechanism-based`） | 加 wrong-relation 对照 |
| Cheaper | 7 | 无 SAE、无标注，donor 全内部集 | 复用 TOFU 保留集 |
| Broader | 8 | 分解框架可迁移其它配对/其它关系（`mechanism-based`） | 统一分解接口 |

**平均分 $\bar S=37/5=7.4>7$。**

- **定位**：Novel Method；一句话故事见 §0。
- **致命缺陷审计**：**2 MAJOR** → (1) F6 共有成分显著性 / 随 K 收敛须预注册；(2) F1 与 CFT 多反事实冲突 + SCALPEL 多视图的划界须核验（否则易被判变体）。零 CRITICAL。故按 SKILL.md 判 `Accept with Revisions`（2+ MAJOR，defend all flaws before starting）。
- **生命周期**：Frontier exploration 3–9 个月，能力匹配绿。
- **范式探测**：First Principles=Partial、Elephant=Partial、Technology Cycle=No、Hamming=Partial → possible。
- **可行性**：Compute 中（K 次前向）、Data 低、Engineering 中、Timeline 中。
- **裁决**：**Accept with Revisions**，`worth pursuing, pending the validation experiment`。**Revisions**：先把 K-donor 冲突诊断（CFT 余弦）与 SCALPEL 划界写进实验计划，再开跑。
- **决定性验证实验**：共有成分 vs 私有成分对目标事实 probe 的 AUC；跨 donor 一致性随 K 曲线；不同 donor 梯度余弦分布；`needs manual pilot`。
- 分数为执行者机制推断分；`novelty-check = PROVISIONAL`，非跨模型回执。

## 6. 判别实验、失败模式、必测指标

- **判别**：$\mathrm{AUC}_f(\text{marg})$ 是否显著高于私有成分；$K$ 增大方向是否收敛；wrong-donor 是否落在私有残差。
- **失败模式**：共有成分主要编码主体身份；$K$ 增大不改变方向（边际化无信息）；不同 donor 梯度余弦近半为负（CFT 同型冲突）；共有子空间与事实方向不相交。
- **必测指标**：奇异值谱、跨 donor 余弦、$\mathrm{AUC}_f$/$\mathrm{AUC}_{\text{id}}$、五类对照、$r_f$、retain NLL、donor 答案率、3 seed std。

## 7. 与方向 I 基线 v0.6.1 的关系

定稿 §2.2.1 待选路径 (b)；严格满足 §2.1「token 等长、槽外 ID 不变、donor 来自 TOFU 内部保留集、按作者/别名簇隔离」；$S_f$ 作用域、Teacher stopgrad、§4.1 中性约束、读出经冻结下游重算、$\lambda_{KL}^{\text{ablation}}=0$、§6 五类对照全部沿用。不拼 $y^-$、不算 logit 差。

## 8. 诚信声明与来源

plan-level，不新增实验；claim 待验证；分数机制推断、PROVISIONAL、`REVIEW_UNAVAILABLE`；新结果须过 `/experiment-audit` + `/result-to-claim`。

- 基线 §2.1/§2.2.1/§4.1/§6.1；`IDEA_CANDIDATES.md` D6
- 近邻（身份未 D1 核验，不作 BibTeX）：CFT arXiv:2605.27083、SCALPEL arXiv:2609.31056、CLReg arXiv:2601.22028、StyliTruth ICLR 2026

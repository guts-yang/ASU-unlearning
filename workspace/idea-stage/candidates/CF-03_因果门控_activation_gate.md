---
id: CF-03
rank: 2
title: 因果门控 ΔA —— patch 加权的因果分量
avg_score: 7.8
verdict: Strong Accept (PROVISIONAL, mechanism-based)
decision_source: IDEA_DISCOVERY_20261126.md §5
domain: LLM/LRM Unlearning / 因果激活差纯化
evidence_level: provisional（机制推断，无实测，无跨模型回执）
baseline_relation: 把 §2.2.1「ΔA 未净化」定稿为因果效应加权的 ΔA_causal；唯一改动点是 ΔA 的构造算子
---

# CF-03 · 因果门控 $\Delta A$（patch 加权因果分量）

## 0. 故事（30 秒讲稿）

> 原始 $\Delta A_{\ell,p}=A^+ - A^-$ 里混着实体身份、subword、词频、关系传播（基线 §2.2.1 自己写明）。「相关」不等于「因果」：只有把某个分量 patch 回冻结下游、真的能让目标答案 $y^+$ 读不出的那些分量，才是可解释的「事实绑定方向」。我们用 UDS / ROME 式因果追踪，给每个 $(\ell,p)$ 分量打一个 patch 恢复分数，按因果效应加权得到 $\Delta A_{\text{causal}}$，把只是「一起变」的身份/词频分量门控掉。可解释性来自：每个保留下来的分量都附一条「我 patch 过，它确实改变了 $r_f$」的证据。

## 1. 领域问题与空白

- 基线 §2.2.1 把 $\Delta A$ 判为「原始配对差，未净化，混有身份/subword/词频/关系传播，未证明编码目标事实而非主体身份」，并点名 stage0–14 中 token-level $\Delta A$ 低秩性不足（`stage7_da_probe` KILL）与 GA / 梯度手术失败模式。
- **观察型手段会被表征旋转糊弄**：UDS(EMNLP 2026) 指出 CKA / Fisher / Logit Lens / 线性探针在隐空间被旋转或扭曲时会误判「知识已消除」；必须从被动观测转向**因果干预**。Zhang & Nanda (ICLR 2024) 与记忆卡 `mem_b42a20_246e24f` 警告加噪 donor 是 **OOD 伪定位峰**，只靠观察会选中假峰。
- **近邻空白**：ROME / activation patching 用于**定位事实**（不是构造遗忘目标）；UDS 用于**评估遗忘深度**（不是训练信号）；SCALPEL 用 contrastive SAE 选特征但基于重构 energy；**没有人把「逐层逐位置 patch 因果效应」作为 $\Delta A$ 的构造算子加权，再喂进 Student 的 $\mathcal L_F$**。

## 2. 核心假设（可证伪）

**H**：$\Delta A$ 中只有稀疏的一小部分 $(\ell,p)$ 分量对 $r_{f,p}$ 有显著因果贡献；按 patch 恢复量加权后，$\Delta A_{\text{causal}}$ 相对等范数随机分量 / wrong-donor 方向能显著区分，且加噪 donor 不产生伪峰。若因果效应在所有分量近似均匀（门控退化为原始差），或与 wrong-donor 不可分，则本思路被证伪。

## 3. 具体实现（LaTeX）

**逐分量因果效应标定**（冻结 $\theta_0$，patch-forward，无训练）：把第 $(\ell,p)$ 个分量替换/注入后测冻结下游读出的变化：

$$
\mathrm{Eff}_{r_f}(\ell,p)=
r_{f,p}\!\left(\mathrm{PatchFwd}\big(\theta_0;\,h\mapsto h+\Delta A_{\ell,p}\big)\right)-r_{f,p}\!\left(\mathrm{Fwd}(\theta_0)\right),
$$

$$
\mathrm{Eff}_{U}(\ell,p)=
U\!\left(\mathrm{PatchFwd}\big(\theta_0;\,h\mapsto h+\Delta A_{\ell,p}\big)\right)-U\!\left(\mathrm{Fwd}(\theta_0)\right).
$$

**因果门控权重**（只保留「降事实、不动效用」的分量，杀 OOD 伪峰）：

$$
w_{\ell,p}=\sigma\!\left(\frac{-\mathrm{Eff}_{r_f}(\ell,p)-\eta_f}{t_f}\right)\cdot
\sigma\!\left(\frac{\eta_U-|\mathrm{Eff}_{U}(\ell,p)|}{t_U}\right),\qquad
\Delta A^{\text{causal}}_{\ell,p}=w_{\ell,p}\,\Delta A_{\ell,p}.
$$

$\sigma$ 为 sigmoid，$t_f,t_U,\eta_f,\eta_U$ 在 discovery 冻结；对加噪 donor / noise-donor / wrong-donor 各跑一遍，要求 $w$ 对真实配对显著、对伪配对塌缩。

**目标与内化**：把净化后的差用作**方向引导**（而非直接对齐 $A^-$），目标取「沿 $-\widehat{\Delta A^{\text{causal}}}$ 走、满足中性约束」的可行投影：

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left(
\Pi_{\mathcal K}\Big[\,h^{+}_{\ell}(p)-\alpha\,\widehat{\Delta A^{\text{causal}}_{\ell,p}}\,\Big]\right),
$$

$$
\mathcal K=\{h:\ r_{f,p}(h)\le c,\ s_{d_j,p}(h)\le d_j,\ \lVert\Delta r_{E,p}(h)\rVert\le e\},
$$

$$
\mathcal L_F(k)=\frac{1}{|\Lambda(k)||S_f|}\sum_{\ell\in\Lambda(k)}\sum_{p\in S_f}\rho_\ell(k)\,
\operatorname{dist}\!\left(h_{S,\ell}(X^+\oplus y^+;p),\,T_{\ell,p}\right),\qquad
\boxed{\ \mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R\ } .
$$

$\Pi_\mathcal K$ 为到中性可行集的投影；所有读出经冻结下游重算；$\lambda_{KL}^{\text{(ablation)}}=0$。

## 4. 差异化（novelty 落点）

| 近邻 | 作用 | 本方案差异 |
|---|---|---|
| ROME / causal tracing (Meng 2022) | 定位事实存储层 | 本方案把 patch 因果效应当**加权算子**构造 $\Delta A$，而非只画热力图 |
| UDS (EMNLP 2026) | 遗忘深度评分 / KE 层定位（评估侧） | 本方案把因果追踪放进**训练目标构造**，是方法不是证书 |
| SCALPEL (2609.31056) | contrastive SAE 选 forget 特征 | 本方案在原始 $(\ell,p)$ 激活差上做因果门控，不训练 SAE |
| V1 全系（已证伪） | 相关性 SVD(pooled $\Delta A$ PCA) 后抑制 | **关键差异**：V1 用相关性主成分直接抑制且已证伪；本方案是**因果干预门控**，$\Delta A$ 用于读出/门控而非删除方向，且设 anti-OOD 伪峰对照 |

**novelty 落点**：把「因果验证」从评估工具升级为 $\Delta A$ 的构造算子，并以 anti-OOD 伪峰对照（Zhang & Nanda 2024）作为预注册门——这恰是 V1 缺失的诚实门槛。

## 5. /idea-evaluator 评估（PROVISIONAL）

| 维度 | 分 | 证据（机制推断，未实测） | Lift |
|---|---|---|---|
| Higher | 8 | 因果筛选剔除身份/词频噪声，应提升 $\Delta A$ 作为事实方向的纯度（`mechanism-based`） | 与 CF-04 双探针交叉 |
| Faster | 8 | 纯前向 + hook 扫描，零训练成本得到 $w$（`mechanism-based`） | 稀疏化 $(\ell,p)$ 扫描 |
| Stronger | 8 | anti-OOD 伪峰 + wrong-donor 对照，直接抗「假定位峰」（`mechanism-based`） | 加 relearning 恢复 |
| Cheaper | 7 | 无 SAE、无标注、无外部模型 | 复用 stage0/diffing 资产 |
| Broader | 8 | 因果门控可迁移到任意激活差构造（含 CF-06 轨迹）（`mechanism-based`） | 统一门控接口 |

**平均分 $\bar S=39/5=7.8>7$。**

- **定位**：Novel Method；一句话故事见 §0。
- **致命缺陷审计**：1 MAJOR（F6：patch 因果门控 vs 等范数随机 / wrong-donor 的判别实验须预注册）+ 2 MINOR（F1 近邻划界、F3 SOTA 基线）。零 CRITICAL。
- **生命周期**：Frontier exploration 3–9 个月，能力匹配绿。
- **范式探测**：First Principles=Yes（挑战「激活差相关即因果」）、Elephant=Partial、Technology Cycle=Partial、Hamming=Yes。
- **可行性**：Compute 低（前向 + hook）、Data 低、Engineering 中、Timeline 中。
- **裁决**：**Strong Accept**（4 维 8+、零 CRITICAL），`worth pursuing, pending the validation experiment`。
- **决定性验证实验**：patch $\Delta A_{\text{causal}}$ vs patch 等范数随机分量 vs noise/wrong-donor，比较 $r_f$ 恢复差；与 UDS KE 层定位一致性；`needs manual pilot`。
- 分数为执行者机制推断分；`novelty-check = PROVISIONAL`，非跨模型回执。

## 6. 判别实验、失败模式、必测指标

- **判别**：因果效应是否在 $(\ell,p)$ 上稀疏；门控后是否与 wrong-donor 可分；加噪 donor 是否产生伪峰（命中即撤回）。
- **失败模式**：因果效应近似均匀（门控退化为原始差）；patch 引入 OOD 伪定位峰；门控与 wrong-donor 不可分；$w$ 与随机方向共线。
- **必测指标**：$\mathrm{Eff}_{r_f}/\mathrm{Eff}_U$ 分布、稀疏度、noise/wrong/shuffled-donor 对照 AUC、$r_f$、retain NLL、donor 答案率、P2/P3/INT8 恢复、3 seed std。

## 7. 与方向 I 基线 v0.6.1 的关系

只把 §2.2.1 待选路径「因果门控纯化」定稿；$S_f$ 作用域、Teacher stopgrad、§4.1 中性约束（$\Pi_\mathcal K$ 就是其集合式实现）、读出经冻结下游重算、$\lambda_{KL}^{\text{ablation}}=0$、§6 配对特异性（真实 donor 优于 5 类对照）全部沿用，且 §6.1 的 5 类对照在此前移为门控的判别实验。不沿 $\Delta A$ 直接抑制（避 V1）。

## 8. 诚信声明与来源

plan-level，不新增实验；claim 待验证；分数机制推断、PROVISIONAL、`REVIEW_UNAVAILABLE`；新结果须过 `/experiment-audit` + `/result-to-claim`。

- 基线 §2.2.1/§2.3/§4.1/§6.1；`IDEA_CANDIDATES.md` D3；`IDEA_INVERSE_TARGET.md` R3
- 近邻（身份未 D1 核验，不作 BibTeX）：ROME (Meng 2022)、Zhang & Nanda ICLR 2024、UDS EMNLP 2026、SCALPEL arXiv:2609.31056、Narrow Finetuning arXiv:2510.13900
- 记忆卡 `mem_b42a20_246e24f`（OOD 伪定位峰）

---
id: CF-06
rank: 5
title: LRM 推理轨迹跨度 ΔA_trace —— 逐 CoT step 的激活差
avg_score: 7.4
verdict: Strong Accept (PROVISIONAL, mechanism-based)
decision_source: IDEA_DISCOVERY_20261126.md §5
domain: LRM Unlearning / 推理链激活差 / 可解释轨迹
evidence_level: provisional（机制推断，无实测，无跨模型回执）
baseline_relation: 把基线「答案跨度 S_f」在 LRM 上扩展为「推理轨迹跨度 S_trace = 事实相关 CoT step 集」
---

# CF-06 · LRM 推理轨迹跨度 $\Delta A_{\text{trace}}$（逐 CoT step）

## 0. 故事（30 秒讲稿）

> 用户研究的是 **LRM 机器遗忘**。LRM 的事实不只藏在最终答案里，更在长思维链 CoT 的中间 step 反复出现（CiPO/STaR/R2MU/R-TOFU 都证明 CoT 是泄露载体）。基线的答案跨度 $S_f$ 是给普通 QA 设计的；在 LRM 上，我们把作用域扩成「推理轨迹跨度」$S_{\text{trace}}$——在反事实对 $(X^+,X^-)$ 各自生成的 CoT 上，逐 step 取中间层激活差 $\Delta A_{\text{trace}}$，让 Student 渐进抹去「事实在推理链中被绑定」的那些 step 的差异。可解释性最强：我们能指着第几个推理步说「事实是在这里被用起来的，我们抹的是这一步」，而不是黑盒地说「改了输出」。

## 1. 领域问题与空白

- **LRM 特异性**：CiPO(ACL 2026, arXiv:2604.15847)、STaR(AAAI 2026, arXiv:2601.09281)、REDaCT(arXiv:2610.12769)、R2MU、Selective Forgetting for LRMs(arXiv:2604.03571)、R-TOFU 都指出：LCM 敏感信息深嵌 CoT 轨迹，只改最终答案不够；ARIA(arXiv:2609.16229) 已覆盖 DeepSeek-R1-1.5B 但作用在 SAE 检测门控。
- **基线局限**：v0.6.1 的 $S_f$ 由 tokenizer 对 $y^+$ 的 offset 确定，是「答案跨度」；对 LRM 的长 CoT 没有对应结构，§1 说「事实在生成答案时展开」但未覆盖推理 step 维度。
- **近邻空白**：所有上述 LRM 遗忘近邻都在**文本/偏好/检测-替换/CoT 改写层**（用反事实 CoT 文本、占位符、soft prompt、检测器门控）；**没有人把「反事实配对在 CoT 各 step 的中间层激活差」作为遗忘目标构造，并逐 step 渐进内化**。CPSD(2611.31932) 是 on-policy 但作用在输出 token 分布。

## 2. 核心假设（可证伪）

**H**：事实在 LRM 的 CoT 轨迹上有稀疏的「绑定 step」；$\Delta A_{\text{trace}}$ 在这些 step 上显著、在无关推理 step 上近零；按 step 数据驱动加权 $\rho(t)$ 的轨迹内化，相对「只抹答案跨度」能显著降低 CoT 泄露而保住 GSM8K 类通用推理。若 $\rho(t)$ 近似均匀或与 wrong-donor 轨迹不可分，则本思路被证伪。

## 3. 具体实现（LaTeX）

**反事实轨迹对**（冻结 $\theta_0$，teacher-force 各自 CoT）：

$$
\tau^+ = (c^+_1,\dots,c^+_{T}),\quad \tau^- = (c^-_1,\dots,c^-_{T}),\qquad
S_{\text{trace}}=\{t:\ \text{step }t\text{ 含目标事实绑定}\}.
$$

$S_{\text{trace}}$ 由「事实 probe 在 step $t$ 的激活上是否显著」+ tokenizer offset 联合冻结，不随 schedule 变；无关推理 step（纯计算/格式）不进 $\mathcal L_F$。

**逐 step 轨迹激活差**（第 $\ell$ 层、第 $t$ 个 CoT step 的抽点）：

$$
\Delta A_{\text{trace},\ell,t}=h_{T,\ell}(X^+\oplus\tau^+;p_t)-h_{T,\ell}(X^-\oplus\tau^+;p_t),\qquad t\in S_{\text{trace}} .
$$

**数据驱动 step 权重**（替代基线 §2.3 对 $S_f$ 的均匀 $1/|S_f|$）：

$$
\rho(t)=\frac{\exp\big(\mathrm{Eff}_{r_f}(t)/t_0\big)}{\sum_{t'\in S_{\text{trace}}}\exp\big(\mathrm{Eff}_{r_f}(t')/t_0\big)},\qquad
\mathrm{Eff}_{r_f}(t)=\text{patch-forward 在 step }t\text{ 的因果读出变化}.
$$

**轨迹内化损失**（把基线 $\sum_{p\in S_f}$ 换成 $\sum_{t\in S_{\text{trace}}}\rho(t)$）：

$$
\mathcal L_F^{\text{trace}}(k)=\frac{1}{|\Lambda(k)|}\sum_{\ell\in\Lambda(k)}\sum_{t\in S_{\text{trace}}}\rho_\ell(k)\,\rho(t)\,
\operatorname{dist}\!\left(h_{S,\ell}(X^+\oplus\tau^+;p_t),\,\operatorname{stopgrad}(T_{\ell,t})\right),
$$

$$
T_{\ell,t}=\Pi_{\mathcal K}\big[\,h^{+}_{\ell}(p_t)-\alpha\,\widehat{\Delta A_{\text{trace},\ell,t}}\,\big],\qquad
\boxed{\ \mathcal L=\lambda_F\mathcal L_F^{\text{trace}}+\beta\mathcal L_R\ } .
$$

$\mathcal K$ 为 §4.1 中性可行集；通用推理保能力另加 $\mathcal L_{\text{reason}}$（在 GSM8K 类非 forget 推理上的 NLL）作为 $\mathcal L_R$ 的一部分。$\lambda_{KL}^{\text{ablation}}=0$。

## 4. 差异化（novelty 落点）

| 近邻 | 作用层 | 本方案差异 |
|---|---|---|
| CiPO (2604.15847) / D2F (2610.17654) | 反事实 CoT 文本 + 偏好优化 | 本方案反事实对用于**取中间层激活差**，不生成/改写 CoT 文本，不教反事实答案 |
| STaR (2601.09281) / REDaCT (2610.12769) | 推理时轨迹抑制 / corrective embedding | 本方案是**参数内化**的逐层激活损失，非推理时 hook、非 embedding 注入 |
| ARIA (2609.16229) | SAE 稀疏检测头门控 | 本方案在原始 $(\ell,t)$ 激活差，无 SAE、无检测头 |
| CPSD (2611.31932) | 输出 token 分布反转 | 本方案在 CoT 中间层激活差，作用域是推理 step 而非 vocab 分布 |
| 基线 v0.6.1 §2.3 | 答案跨度 $S_f$、均匀位置权重 | **唯一改动点**：作用域扩为 $S_{\text{trace}}$、权重 $\rho(t)$ 数据驱动、加 $\mathcal L_{\text{reason}}$ |

**novelty 落点**：首个把「反事实配对 + 中间层激活差 + 渐进内化」搬到 **LRM 推理轨迹 step 粒度**的方法；贡献落在「推理链哪一步绑定了事实」这一可解释科学问题，而非再加一个 CoT 改写模块。

## 5. /idea-evaluator 评估（PROVISIONAL）

| 维度 | 分 | 证据（机制推断，未实测） | Lift |
|---|---|---|---|
| Higher | 8 | 直接针对 LRM CoT 泄露，覆盖近邻盲区（`mechanism-based`） | 报告 AFE/CFE 双指标 |
| Faster | 6 | 长 CoT teacher-force 前向成本高于短答案，是主要代价 | 只前向事实绑定子集 step |
| Stronger | 8 | 逐 step 因果权重 + 中性约束，抗「只改最终答案」浅层捷径（`mechanism-based`） | 加多解码一致性 MCS |
| Cheaper | 6 | 需标注/定位事实绑定 step，比纯 QA 成本高 | 复用 ASU Appendix N 标注协议 |
| Broader | 9 | 直接命中 LRM 这一 2026 前沿形态，可迁移 R-TOFU/WMDP 推理模型（`mechanism-based`） | 与 ARIA 的 R1 实验对照 |

**平均分 $\bar S=37/5=7.4>7$。**

- **定位**：Novel Method / New Setting（LRM 轨迹粒度）；一句话故事见 §0。
- **致命缺陷审计**：1 MAJOR（F6：$\rho(t)$ 是否集中到事实绑定 step、轨迹加权 vs 均匀 vs 反向的判别须预注册）+ 2 MINOR（F1 与 CiPO/STaR/ARIA 划界、F5 长 CoT 算力预算）。零 CRITICAL。
- **生命周期**：Frontier exploration 3–9 个月；长 CoT 使算力偏紧，能力匹配 Yellow→收窄 step 子集后绿。
- **范式探测**：First Principles=Yes、Elephant=Yes（CoT 泄露是公认难题）、Technology Cycle=Yes（LRM 兴起使此轴可行）、Hamming=Yes → strong disruptive seeds。
- **可行性**：Compute 中高（长 CoT）、Data 中（step 定位）、Engineering 中、Timeline 中。
- **裁决**：**Strong Accept**（H/S/B 三维 8+、零 CRITICAL），`worth pursuing, pending the validation experiment`。
- **决定性验证实验**：$\rho(t)$ 是否集中到事实绑定 step；轨迹加权 vs 仅答案跨度两臂，比较 CoT 泄露（CFE）与 GSM8K 效用；`needs manual pilot`。
- 分数为执行者机制推断分；`novelty-check = PROVISIONAL`，非跨模型回执。

## 6. 判别实验、失败模式、必测指标

- **判别**：$S_{\text{trace}}$ 是否与事实 probe/tokenizer offset 对齐；轨迹加权是否优于均匀/反向；CoT 泄露降而 GSM8K/MU 不崩。
- **失败模式**：$\rho(t)$ 近似均匀（事实不在特定 step 展开）；轨迹与 wrong-donor 轨迹不可分；长 CoT 成本超预算；只改答案未改链（浅层）；$\mathcal L_{\text{reason}}$ 把事实重新引入 CoT。
- **必测指标**：AFE（答案遗忘）、CFE（推理链遗忘）、MCS（多解码一致性）、$r_f$ / step 读出、retain NLL、GSM8K/MMLU、donor 答案率、提前 EOS/完整回答率、P2/P3/INT8 恢复、3 seed std。

## 7. 与方向 I 基线 v0.6.1 的关系

是基线在 LRM 上的**作用域扩展**：把 §2.3 的 $S_f$（答案跨度）与均匀位置权重，替换为 $S_{\text{trace}}$（推理轨迹事实绑定 step）与数据驱动 $\rho(t)$；保留 teacher-force、逐层 $\Lambda(k)/\rho_\ell(k)$ 渐进、Teacher stopgrad、§4.1 中性约束、读出经冻结下游重算、$\lambda_{KL}^{\text{ablation}}=0$、§6 六条验收，并新增「通用推理保能力 $\mathcal L_{\text{reason}}$」与 LRM 专属的 CFE/MCS 指标。不把反事实 CoT 文本当目标（避 CiPO/D2F 型答案替换）。

## 8. 诚信声明与来源

plan-level，不新增实验；claim 待验证；分数机制推断、PROVISIONAL、`REVIEW_UNAVAILABLE`；新结果须过 `/experiment-audit` + `/result-to-claim`。

- 基线 §1/§2.2/§2.3/§4.1/§5/§6；`IDEA_CANDIDATES.md` D5（答案跨度轨迹）
- 近邻（身份未 D1 核验，不作 BibTeX）：CiPO arXiv:2604.15847、STaR arXiv:2601.09281、REDaCT arXiv:2610.12769、R2MU、R-TOFU、Selective Forgetting for LRMs arXiv:2604.03571、ARIA arXiv:2609.16229、CPSD arXiv:2611.31932、ASU arXiv:2603.01285 Appendix N

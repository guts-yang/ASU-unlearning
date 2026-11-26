---
id: CF-10
rank: 10
title: ΔA 纯度驱动渐进 Λ(k) / ρ_ℓ(k)
avg_score: 7.2
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
decision_source: IDEA_DISCOVERY_20261126.md §5
domain: LLM/LRM Unlearning / 渐进 schedule 数据驱动
evidence_level: provisional（机制推断，无实测，无跨模型回执）
baseline_relation: 把 §2.3 手设 Λ(k)/ρ_ℓ(k) 定稿为 ΔA 纯度谱的函数，合并解决「ΔA 未净化」+「渐进未证」两开放问题
---

# CF-10 · $\Delta A$ 纯度驱动渐进 $\Lambda(k)/\rho_\ell(k)$

## 0. 故事（30 秒讲稿）

> 基线既要「逐层渐进抹去差异」又把 $\Lambda(k)$（每阶段参与对齐的层集合）和 $\rho_\ell(k)$（层权重）留成手设、待冻结（§2.3 / §2.2.1）。我们让 schedule 由 $\Delta A$ 自己的「事实纯度谱」决定：哪一层哪一答案位置的 $\Delta A$ 更像事实（事实 probe AUC 减身份 probe AUC，或因果门控值），就先加入、给更高权重。这样「渐进」不再是一个拍脑袋的课程，而是「从最干净的事实信号往最脏的层推进」——可解释性直接写在训练进度里。

## 1. 领域问题与空白

- 基线 §2.3 明确「训练时逐步扩展 $\Lambda(k)$ 或调整 $\rho_\ell(k)$」，但「渐进必须相对等预算的静态全层、反向和随机 schedule 证明独立收益，否则只保留『多层对齐』表述」；§2.2.1 把 $\Lambda(k)/\rho_\ell(k)$ 列为待冻结量。
- 项目历史负结果：stage4「反事实渐进 schedule」已被证伪（来源：`IDEA_REPORT_V3.md` 淘汰清单、`IDEA_REPORT_V4.md`），但那是 **prompt 末位 + 沿 $\Delta A$ 抑制 + 未做等预算对照**；R2-B 的 τ(k) 课程也因「等均值常数对照 3/3 seed 更优」判负（`research-wiki/log.md` 2026-11-25）。教训：**任何 schedule 主张必须过等预算对照门**。
- **近邻空白**：FDD(ACL 2025) 报告「4 层采样最佳」、LayerSync 避最浅/末端 20% 层、CUD(ACL 2026) 用电路长度做 pre-unlearning 难度指标、SensiTemp/R3-1 用敏感度分配温度但已归档；**没有人用「反事实配对差 $\Delta A$ 的逐层事实纯度」来驱动遗忘的渐进 layer schedule，并把等预算对照设为强制门**。

## 2. 核心假设（可证伪）

**H**：$\Delta A$ 的事实纯度在（层 × 答案位置）上非平坦，纯度谱跨 seed 稳定；以纯度为 $\Lambda(k)/\rho_\ell(k)$ 的驱动，相对**等预算**静态全层 / 反向 / 随机 schedule 有独立收益。若纯度谱近似平坦（退化为手设全层），或等预算对照无独立收益，则按 §2.3 自认撤销「渐进」主张，仅保留「多层对齐」表述。

## 3. 具体实现（LaTeX）

**纯度谱**（层 $\ell$ × 答案位置 $p$，来自 CF-03 因果门控或 CF-04 双探针）：

$$
\mathrm{pur}_{\ell,p}=\underbrace{\mathrm{AUC}_f(\ell,p)-\mathrm{AUC}_{\text{id}}(\ell,p)}_{\text{双探针纯度}}\ \ \text{或}\ \ \underbrace{w_{\ell,p}}_{\text{因果门控值（CF-03）}},
$$

$$
\mathrm{pur}_{\ell}=\frac{1}{|S_f|}\sum_{p\in S_f}\mathrm{pur}_{\ell,p},\qquad
\text{稳定性判据：}\ \mathrm{std}_{\text{seed}}(\mathrm{pur}_{\ell})<\sigma_{\text{pur}} .
$$

**纯度驱动的 $\Lambda(k)/\rho_\ell(k)$**（按纯度从高到低逐阶段加入层）：

$$
\Lambda(k)=\Big\{\ \ell:\ \mathrm{rank}\big(\mathrm{pur}_{\ell}\big)\le n_k\ \Big\},\qquad
n_k=\Big\lceil |\mathcal L|\cdot\frac{k}{K}\Big\rceil,
$$

$$
\rho_\ell(k)=\frac{\exp\big(\mathrm{pur}_{\ell}/t_\rho\big)}{\sum_{\ell'\in\Lambda(k)}\exp\big(\mathrm{pur}_{\ell'}/t_\rho\big)} .
$$

$n_k, t_\rho, \sigma_{\text{pur}}, K$ 在 discovery 冻结；避最浅/末端 20% 层（LayerSync / FDD 经验）作为软先验。

**主损失**（沿用基线 §2.3 形式，只替换 schedule 来源）：

$$
\mathcal L_F(k)=\frac{1}{|\Lambda(k)|\,|S_f|}\sum_{\ell\in\Lambda(k)}\sum_{p\in S_f}\rho_\ell(k)\,
\operatorname{dist}\!\left(h_{S,\ell}(X^+\oplus y^+;p),\,\operatorname{stopgrad}(T_{\ell,p})\right),
$$

$$
T_{\ell,p}=\Pi_{\mathcal K}\big[\,h^{+}_{\ell}(p)-\alpha\,\widehat{\Delta A^{\text{pure}}_{\ell,p}}\,\big],\qquad
\boxed{\ \mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R\ } .
$$

**等预算对照门（强制，呼应 §6 第 2 条）**：纯度驱动 schedule vs 静态全层 / 反向 / 随机 / 等均值常数 schedule，在**等步数、等前后向、等 GPU 预算**下比较 $r_f$ / $U$ / 跨 seed std；无独立收益即撤销「渐进」主张。

## 4. 差异化（novelty 落点）

| 近邻 | 本方案差异 |
|---|---|
| FDD (ACL 2025) / LayerSync | 后者经验性「4 层/避末端」；本方案由 $\Delta A$ 纯度谱数据驱动，且设跨 seed 稳定性 + 等预算门 |
| CUD (ACL 2026, 2601.09624) | 后者用电路长度做 pre-unlearning 难度画像；本方案用反事实配对差的逐层事实纯度驱动 layer schedule |
| R3-1 SensiTemp / R2-B τ(k)（均归档） | 后者用敏感度/时间课程且被等均值常数对照判负；本方案把「等预算对照」设为强制门（正是它们缺的诚实门槛） |
| stage4 反事实渐进 schedule（已证伪） | 后者 prompt 末位 + 沿 $\Delta A$ 抑制；本方案答案跨度 $S_f$ + 纯度驱动层选择 + 等预算对照 |
| 基线 §2.3 手设 $\Lambda(k)/\rho_\ell(k)$ | **唯一改动点**：schedule 变成 $\Delta A$ 纯度谱的函数 |

**novelty 落点**：把「渐进」从超参课程升级为「沿事实纯度推进」的可解释训练进度，并主动设等预算对照门防止重蹈 R2-B / stage4 覆辙。

## 5. /idea-evaluator 评估（PROVISIONAL）

| 维度 | 分 | 证据（机制推断，未实测） | Lift |
|---|---|---|---|
| Higher | 7 | 纯度驱动应减少在「脏层」上的无效/有害更新，但历史 schedule 多次判负，增益保守（`mechanism-based`） | 与 CF-01 逆问题叠加 |
| Faster | 7 | 不增模块，schedule 只是重排层顺序 | 复用纯度谱缓存 |
| Stronger | 7 | 等预算对照 + 跨 seed 稳定性 + 避末端层，抗课程过拟合（`mechanism-based`） | 加反向/随机对照 |
| Cheaper | 7 | 无 SAE、无标注、无额外前向 | 复用 CF-03/CF-04 谱 |
| Broader | 8 | 纯度驱动 schedule 可迁移其它激活差方法 / LRM 轨迹（`mechanism-based`） | 与 CF-06 联动 |

**平均分 $\bar S=36/5=7.2>7$。**

- **定位**：Novel Method；一句话故事见 §0。
- **致命缺陷审计**：**2 MAJOR** → (1) F6 等预算对照（含等均值常数，针对 R2-B 失败模式）须预注册；(2) F1 与 FDD/LayerSync/CUD/stage4 划界须核验。零 CRITICAL。故 `Accept with Revisions`。
- **生命周期**：Frontier exploration 3–9 个月，能力匹配绿。
- **范式探测**：First Principles=Partial、Elephant=Yes（schedule 收益常被平均值解释）、Technology Cycle=No、Hamming=Partial → possible。
- **可行性**：Compute 中（等预算多臂）、Data 低、Engineering 中、Timeline 中。
- **裁决**：**Accept with Revisions**，`worth pursuing, pending the validation experiment`。**Revisions**：先冻结等预算对照协议与纯度谱稳定性阈值，再开跑；若无独立收益则按 §2.3 降级为「多层对齐」。
- **决定性验证实验**：纯度谱跨 seed 稳定性；纯度驱动 schedule 是否相对等预算静态全层/反向/随机/等均值常数有独立收益；`needs manual pilot`。
- 分数为执行者机制推断分；`novelty-check = PROVISIONAL`，非跨模型回执。

## 6. 判别实验、失败模式、必测指标

- **判别**：纯度谱非平坦且跨 seed 稳定；等预算对照下纯度驱动是否有独立收益；避末端层是否必要。
- **失败模式**：纯度谱近似平坦（退化为手设全层）；等预算对照无独立收益（撤销「渐进」）；收益来自更强总更新而非结构化分配（R2-B / R3-1 失败模式同源）；纯度谱随 answer-span 位置漂移。
- **必测指标**：$\mathrm{pur}_\ell$ 谱与跨 seed std、等预算下 $r_f$ / $U$ / 跨 seed std、各 schedule 的 $dF,dR$、retain NLL、donor 答案率、3 seed std。

## 7. 与方向 I 基线 v0.6.1 的关系

定稿 §2.2.1 / §2.3 的待冻结量 $\Lambda(k)/\rho_\ell(k)$；$S_f$ 作用域固定不参与 schedule、Teacher stopgrad、§4.1 中性约束（$\Pi_\mathcal K$）、读出经冻结下游重算、$\lambda_{KL}^{\text{ablation}}=0$ 全部沿用；把 §6 第 2 条「等预算渐进收益」前移为构造期的强制门。历史 stage4 / R2-B / R3-1 负结果不复用为正向证据，仅作为「必须设等预算对照」的设计义务来源。

## 8. 诚信声明与来源

plan-level，不新增实验；claim 待验证；分数机制推断、PROVISIONAL、`REVIEW_UNAVAILABLE`；新结果须过 `/experiment-audit` + `/result-to-claim`。

- 基线 §2.2.1/§2.3/§6.2；`IDEA_CANDIDATES.md` D7；`IDEA_REPORT_V3.md`（stage4 淘汰）、`IDEA_REPORT_V4.md`、`research-wiki/log.md`（R2-B 3-seed 判负）
- 近邻（身份未 D1 核验，不作 BibTeX）：FDD ACL 2025、LayerSync、CUD arXiv:2601.09624、SensiTemp/R3-1 归档报告

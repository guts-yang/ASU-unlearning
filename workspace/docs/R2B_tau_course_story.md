# R2-B ASU 温度课程 τ(k)：当前实验结果与遗忘底层原理

> 📝 **一句话结论（当前状态）**：R2-B 在 V4 阶段因"比 `global τ=2` 多 +0.149 的 ΔF、retain 未受损"被判 `PASS / PROMOTE`；但随后更严格的**等预算收缩复核**补上缺失的对照 `global τ̄=2.25` 后，`ΔF=1.1416 > 0.9491`——所谓"时间调度有效"完全由**平均温度更高**解释。等预算下时间调度不是增益而是损失，`low2high` 及全部候选被 `discard`。当前口径：R2-B 已从"可发表增量"收缩为"增益可由平均 τ 解释"的负结果。

> 💡 **一针见血**：ASU 的"遗忘"从来不是删除知识，而是用温度 $\tau$ 把冻结 teacher 的注意力搅浑、输出分布摊平，再逼 student 用 forward KL 去模仿这摊浑水，把正确答案的概率稀释掉。整个方法**唯一的力**就是 $\tau$ 造成的分布差 $\mathrm{KL}(p_T^\tau\|p_S)$：\tau=1 时 teacher=student、梯度恒为零。$\tau(k)$ 课程想调的只是"什么时候把水搅得更浑"，而等预算对照证明连"什么时候"都不重要——决定 $\Delta F$ 的只有**平均 $\tau$**。

**范围**：H20 / `Llama-3.2-1B-Instruct` / TOFU `forget10`+`retain90` / 60 steps / $\alpha=0.1$ / $\text{lr}=5\times10^{-6}$ / 单 seed。**裁决为执行者 provisional，跨模型评审后端不可用（`REVIEW_UNAVAILABLE`）**。

---

## 1. R2-B 是什么

ASU（ICLR 2026）用**一个标量** $\tau>1$ 平滑注意力做自蒸馏遗忘。问题锚定：既然 $\tau$ 是唯一杠杆，那"何时给多大的 $\tau$"就是唯一无需新模块、无需新参数的可调自由度。

三种等步数 schedule（$K=60$）：

$$
\tau_{\text{global}}(k)=2,\qquad
\tau_{\text{low2high}}(k)=1.5+1.5\frac{k}{K-1},\qquad
\tau_{\text{high2low}}(k)=3.0-1.5\frac{k}{K-1}
$$

设想的机制（**推断，非证明**）：$\mathrm{KL}(p_T^\tau\|p_S)$ 关于 $\tau>1$ 单调递增；把大 $\tau$ 放在后期＝把大遗忘梯度放在后期，早期低 $\tau$ 时 teacher 仍"认得"事实、梯度温和，先稳住 $L_R$。零新参数、零新模块。

---

## 2. 当前实验结果：两个版本，后者推翻前者

### 2.1 V4 阶段（`PROMOTE`，对比 `global τ=2`）

来源：`workspace/idea-stage/IDEA_REPORT_V4.md`、`05_results_ledgers/auto_experiments/train_*.json`。

| schedule | $\Delta F$（forget NLL↑） | $\Delta R$（retain NLL↓） | 相对 global |
|---|---|---|---|
| `global τ=2` | $+0.7998$ | $-0.2957$ | 基线 |
| `high2low 3→1.5` | $+0.5621$ | $-0.3200$ | 更差 |
| **`low2high 1.5→3`** | **$+0.9491$** | $-0.2563$ | **$+0.1493$，$\Delta R\le0.05$** |

据此 V4 判 `PASS / PROMOTE`：「将 low2high 设为默认 τ 调度，high2low 仅作消融」。

> ⚠️ **陷阱就在这里**：`low2high` 的平均温度 $\bar\tau=2.25$，却与 `global τ=2` 比较；而 $\tau=2.607$ 时 $\Delta F$ 高达 $1.7914$，说明 $\Delta F$ 对 $\bar\tau$ 极敏感。「+0.149 增益」完全可能是"平均温度更高"，而非"时间形状更聪明"。

### 2.2 等预算收缩复核（证伪 / 降级）

来源：`03_design_contracts/shrink-story/STORY_DESIGN.md` §7、`05_results_ledgers/shrink-story/loop/results.tsv`（Round 1，8/8 轮实跑）。单一指标为**等预算收缩遗憾**：

$$
r(c)=\bigl[\Delta F_{\mathrm{ctrl}(\bar\tau(c))}-\Delta F_c\bigr]+2\max(0,\Delta R_c-\Delta R_{\mathrm{ctrl}})+0.02\,\kappa(c),\qquad r(c)<0 \text{ 才存活}
$$

$\kappa$：0=常数标量、1=时间调度、2=空间温度场、3=新模块；复杂度项保证"同等收益下更简单者胜出"。第一项**不取 $\max(0,\cdot)$**，因为负缺口正是"比等预算对照更差"的证据。

补上的 anchor：`global τ̄=2.25` → $\Delta F=1.1416,\ \Delta R=-0.3013$；`global τ̄=2.607` → $\Delta F=1.7914,\ \Delta R=-0.3042$（与既有结果一致，交叉校验）。

| 轮 | 配置 | $\bar\tau$ | $\Delta F$ | $\Delta R$ | $r(c)$ | 判定 |
|---|---|---|---|---|---|---|
| C1 | `linear_up 1.5→3`（low2high） | 2.25 | $+0.9491$ | $-0.2563$ | $+0.3026$ | **discard** |
| C2 | `linear_down 3→1.5` | 2.25 | $+0.5621$ | $-0.3200$ | $+0.5995$ | discard |
| C3 | `random τ_k∼U(1.5,3)`（时序打乱） | 2.25 | $+0.9137$ | $-0.2997$ | $+0.2511$ | discard |
| C4 | `step 30/30` | 2.25 | $+0.5339$ | $-0.2067$ | $+0.8169$ | discard |
| C5 | SensiTemp 空间场 | 2.607 | $+1.7470$ | $-0.2997$ | $+0.0934$ | discard |
| C6 | 打乱空间场 | 2.607 | $+1.5166$ | $-0.3053$ | $+0.3149$ | discard |

三条硬事实：

- $\Delta F_{\mathrm{ctrl}(2.25)}=1.1416 > \Delta F_{\text{low2high}}=0.9491$ → 等预算下时间调度是**损失**，"+0.149 增益"纯属平均温度不对等。
- C3 随机时序的 $r=+0.2511$ **低于** C1 的 $+0.3026$ → 连"渐进/课程形状"都无可辨识收益，只剩"平均 $\tau$ 增大"这一强度效应。
- 6 个候选 $r$ 全为正，`keep` 数为 0。

> ❌ **收缩后结论**：第 3 幕「时间维度比空间维度更有效」**不能成立**。$\tau$ 的空间分配（SensiTemp）、时间调度（τ 课程）、forget-gradient 子空间投影在等预算下全线无效；此前观察到的所有 $\Delta F$ 增益都可由平均温度解释。`NARRATIVE_ASU_TEMPERATURE_DEPTH.md` §8 仍写「幕3 PROMOTE」，是**较早的旧口径**，应以 `STORY_DESIGN.md` §7 的改写为准。

### 2.3 演化时间线

| 时间 | 工件 | R2-B 状态 |
|---|---|---|
| 2026-11-23 | `IDEA_REPORT_V4.md` | `PASS / PROMOTE`（对比 `global τ=2`） |
| 2026-11-23 稍后 | `shrink-story/STORY_DESIGN.md` §7 + `loop/results.tsv` | 补 `ctrl(2.25)` 后收缩为「增益由平均温度解释」负结果，全 `discard` |
| 2026-11-24 | `auto_train_loop_v5/AUTORESEARCH_SUMMARY.md` | 主线转向深度/抗恢复稳定化（EMA、hinge），R2-B 不再作为贡献线 |

---

## 3. 遗忘底层原理（一针见血）

### 3.1 温度加在哪

标准注意力（$\ell$ 层 $h$ 头）：$A_{\ell,h}=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M_{\text{causal}}\right)$。ASU 把温度放在**注意力 logits**（不是输出 logits）上：

$$
A^{\tau}_{\ell,h}=\mathrm{softmax}\!\left(\frac{QK^\top}{\tau\sqrt{d_h}}+M_{\text{causal}}\right),\qquad \tau>1
$$

$\tau\uparrow$ ⇒ 注意力熵增 ⇒ 事实检索依赖的 sharp 关联路径被抹平。**关键**：teacher 的输出层仍是标准 softmax，平滑发生在**内部信息路由**上，不是常见"输出软标签蒸馏"。

### 3.2 为什么 τ>1 才有梯度（零梯度论证）

冻结 teacher 与可训练 student 初始化同权重 $\theta_S=\theta_T$：

$$
p_T^{\tau}=\mathrm{softmax}(f_{\theta_T}(x;A^\tau)),\qquad p_S=\mathrm{softmax}(f_{\theta_S}(x;A^{1}))
$$

$\tau=1 \Rightarrow p_T^1\equiv p_S \Rightarrow \mathrm{KL}\equiv0 \Rightarrow \nabla_\theta L_F=0$ —— **零遗忘梯度**。$\tau>1$ 使 teacher 分布变钝，二者出现分布差，梯度才出现。**ASU 全部的"力"都来自这一个量**，这也是 $AUTO\_REVIEW$/方向 I proposal 里"去 τ 后必须靠显式 loss 补梯度"的根因。

### 3.3 forward KL 的方向是关键（概率稀释）

源码用 `F.kl_div(input=log p_S, target=log p_T, log_target=True)`，展开为 $\mathrm{KL}(p\|q)=\sum_v p(v)\log\frac{p(v)}{q(v)}$：

$$
L_F=\frac{1}{\sum_t m_t}\sum_t m_t\,\mathrm{KL}\!\left(p_T^\tau(\cdot\mid x_{<t})\,\middle\|\,p_S(\cdot\mid x_{<t})\right),\qquad L=\alpha L_F+L_R
$$

方向是 $\mathrm{KL}(p_T^\tau\|p_S)$（**mode-covering**）：被抹平、把概率摊到多个词上的 teacher 在前，student 被迫在这些词上都放质量 ⇒ 原本集中在正确答案上的概率被**稀释** ⇒ $F(\theta)$ 上升（答不出被忘的事实）。若写成反向 $\mathrm{KL}(p_S\|p_T^\tau)$ 则是 mode-seeking，行为完全不同。梯度只经 student 回传，teacher 全程 `stopgrad`；两个分布共享同一条 $y_{<t}$ 前缀。

### 3.4 实现技巧：缩放 q_proj 等价于 logits 温度

避免改 HF forward：把 `self_attn.q_proj` 输出缩放 $1/\tau$。因 RoPE 是正交变换 $R$，$R(q/\tau)=R(q)/\tau$，故

$$
\frac{(Q/\tau)K^\top}{\sqrt{d_h}}=\frac{QK^\top}{\tau\sqrt{d_h}}
$$

只需挂 forward hook 即可实现**逐层、逐头**温度场，零新增参数。这是 SensiTemp 诊断能成立的技术前提。

### 3.5 本质：行为抑制 ≠ 知识擦除

$L_R$ 优化 retain NLL、$L_F$ 只压低当前参数下"答出被忘事实"的概率，二者都作用在**输出行为**上，没有任何机制删除权重里的知识。恢复阶梯给出铁证：

$$
\rho_{\mathrm{P2}}=\frac{g_{\mathrm{P2}}}{F_1-F(M_0)},\quad \rho_{\mathrm{P3}}=\frac{g_{\mathrm{P3}}}{F_1-F(M_0)},\quad \mathrm{depth}=\tfrac12(\rho_{\mathrm{P2}}+\rho_{\mathrm{P3}})
$$

ASU-u 实测 $\rho_{\mathrm{P2}}=0.524,\ \rho_{\mathrm{P3}}=0.371,\ \mathbf{depth=0.448}$ —— 约 **44.8% 的遗忘增益能被 few-shot 或 20 步微调恢复**。这把前两幕拼命优化的 $F(\theta)$ 定性为"当前参数下的行为"，而非"知识被擦除"。

> ✅ **一针见血的一句话**：ASU 不是擦掉记忆，是把 teacher 的输出搅浑再让 student 照抄这摊浑水，让正确答案的概率被摊薄；$\tau$ 是唯一的开关，$\tau(k)$ 只是想调"什么时候把水搅得更浑"——而等预算对照证明，连"什么时候"都不重要，唯一重要的是"平均搅得多浑"。

---

## 4. 课程在这个原理里的位置，以及为什么被击穿

把 §3 的结论串起来：$\tau$ 是唯一的力 ⇒ 唯一能调的自由度为「给谁（空间）、何时给（时间）、给多大（平均）」。`R3-1` 试「给谁」、`R2-B` 试「何时给」，二者都被**等平均 $\tau$ 对照 + 打乱对照**击穿：`1.747<1.791`（空间）、`0.9491<1.1416`（时间）、随机时序还不比精心设计的课程差。

于是整条叙事从"提出新调度"收缩为**"负结果 + 评估基准"**：在固定平滑预算下，$\tau$ 的空间分配、时间调度、forget-gradient 子空间投影都未能优于最简单的全局标量 $\tau$；可辩护的正面贡献是等预算对照协议、P1/P2/P3/INT8 恢复深度指标，以及"简单标量 $\tau$ 已是该 setting 下的强基线"。这比原故事短，但每一句都有等预算对照支撑，不会被"换成同平均 $\tau$ 呢"一击击穿。

---

## 5. 浓缩故事

主角是 ASU 的**一个标量 $\tau$**。它用温度把 teacher 的注意力搅平，制造出 $\mathrm{KL}(p_T^\tau\|p_S)$ 这一条唯一的遗忘梯度，让 student 把正确答案的概率摊薄。我们替它追问：温度该给谁、该何时给、忘掉之后还剩多少。

给谁（SensiTemp 层-头温度场）被等平均对照否决；何时给（R2-B 的 $\tau$ 课程 low2high）看似赢了 $+0.149$，补上 `ctrl(2.25)` 后发现 `1.1416>0.9491`，随机时序还更不差——时间维度也被击穿；最后忘掉之后还剩多少（深度审计）给出转折：$\mathrm{depth}=0.448$，近一半增益可被微调恢复。故事从"提出新方法"一路收缩为"我们其实只证明了：把水平均搅浑就够了，而 ASU 只是把知识藏起来、没擦掉，并给出了度量这件事的尺子"。

---

## 6. 边界与诚实声明

- 单 seed、60 steps、仅 TOFU-1B / `forget10`；未测 3-seed、MMLU/GSM8K、生成质量、WMDP/MUSE。$\alpha=0.1$ 与 ASU 默认 $1.0$ 不同（复现配置偏离）；factual/function 为停用词启发式、INT8 为伪量化。
- 「早期低 τ 稳住 $L_R$、后期高 τ 释放梯度」是**机制推断，非证明**；等预算复核已否定其可主张的收益。
- `R2-C` 双环 meta 目标 `未实现`；事后审计不等于已实现抗训练。
- 跨模型评审后端全部不可用：`review-stage/AUTO_REVIEW.md` 与 `REVIEW_STATE.json` 记为 `REVIEW_UNAVAILABLE`，取证 verdict 亦为 `REVIEW_UNAVAILABLE`。**不得据此声称"论文诚实/通过"**。本 HTML 为阅读视图，未经独立评审。
- Markdown 为唯一真实来源，HTML 为衍生视图；漂移以 `<meta>` 中 SHA256 与生成时间戳为准。

## 7. 工件索引（来源）

| 工件 | 作用 |
|---|---|
| `workspace/idea-stage/IDEA_REPORT_V4.md` | R2-B `PROMOTE` 裁决与 60-step 数字 |
| `05_results_ledgers/auto_experiments/train_low2high.json` 等 | 原始 $\Delta F/\Delta R$ 与训练 history |
| `03_design_contracts/shrink-story/STORY_DESIGN.md` §7 | 等预算收缩复核、收缩遗憾、故事改写 |
| `05_results_ledgers/shrink-story/loop/results.tsv` | Round 1 八轮 `val_bpb` 与 keep/discard |
| `workspace/idea-stage/NARRATIVE_ASU_TEMPERATURE_DEPTH.md` | ASU 数学、恢复阶梯、（旧口径）六幕故事 |
| `workspace/ideas_archive/05_results_ledgers/auto_train_loop_v5/AUTORESEARCH_SUMMARY.md` | 2026-11-24 主线转向深度/抗恢复 |
| `review-stage/AUTO_REVIEW.md` | 评审后端 `REVIEW_UNAVAILABLE` 证据 |

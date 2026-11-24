# 一个标量温度，和遗忘到底有多深

**副标题**：ASU 注意力温度自蒸馏的复现、证伪与再叙事
**日期**：2026-11-23　**范围**：`workspace/auto_experiments/` 与 `workspace/verification/` 已复现结果
**状态**：所有数字来自实跑（H20 / TOFU-1B / `Llama-3.2-1B-Instruct`）；跨模型评审后端不可用，判断为执行者 provisional

> **本文目的**：把"当前已经复现的创新方向"讲成一个可投稿的故事，并把每一步的数学写清楚。Markdown 是唯一真实来源，HTML 只是阅读视图。

## 0. 一句话故事

ASU（ICLR 2026）用**一个标量** $\tau>1$ 平滑注意力，让冻结的 teacher 分布变钝，学生通过 KL 逼近它来"忘掉"事实。我们追问三件事：温度**该给谁**、**该何时给**、**忘掉之后还剩多少**。实测答案是：给谁（结构化温度场）被等平均对照否决；何时给（$\tau$ 的时间课程）在等步数下赢了 $+0.149$ 的 forget-NLL 增益且零新增参数；但深度审计显示 ASU 的遗忘增益有 $44.8\%$ 能被 few-shot / 少量微调恢复——**行为遗忘不等于知识擦除**。于是故事从"提出新模块"收缩为"发现时间维度更有效 + 给出遗忘深度的度量与基准"。

## 1. 符号与指标

| 符号 | 含义 |
|---|---|
| $M_\theta$ | 自回归 LM，参数 $\theta$；基座 $M_0$ = `tofu_Llama-3.2-1B-Instruct_full` |
| $x=(q,a)$ | 一条 QA 样本；$a=(a_1,\dots,a_T)$ 为答案 token |
| $m_t\in\{0,1\}$ | answer-token mask（prompt 部分为 0） |
| $\mathcal{D}_f,\mathcal{D}_r$ | forget set = TOFU `forget10`；retain set = `retain90` |
| $p_\theta(v\mid x_{<t})$ | 输出层 softmax 的词表分布 |
| $F(\theta),R(\theta)$ | forget / retain 上的 answer NLL |

$$\mathcal{L}_{\text{NLL}}(\theta;\mathcal{D})=\frac{1}{|\mathcal{D}|}\sum_{x\in\mathcal{D}}\frac{1}{\sum_t m_t}\sum_{t} m_t\bigl[-\log p_\theta(a_t\mid q,a_{<t})\bigr]$$

$$F(\theta)=\mathcal{L}_{\text{NLL}}(\theta;\mathcal{D}_f),\qquad R(\theta)=\mathcal{L}_{\text{NLL}}(\theta;\mathcal{D}_r)$$

口径：$F$ **越大**表示越答不出被忘的事实（忘得干净），$R$ **越小**表示通用能力保持越好。所有训练用 $60$ steps、$\alpha=0.1$、$\text{lr}=5\times10^{-6}$、seed $0$；除 V1 既有 3-seed 结果外，本轮均为单 seed。

> **诚实标注**：ASU 原实现的默认 $\alpha=1.0$；我们的轻量复现把 $\alpha$ 降到 $0.1$，因为 $\alpha=1$ 时初始 KL 约 $8.3$ 会在前两步就把 retain 推坏（实测 $R:0.465\to1.016$）。这是**复现配置的偏离**，不是 ASU 原结论。

## 2. ASU 的数学：温度为什么能产生遗忘梯度

### 2.1 注意力温度加在哪里

第 $\ell$ 层第 $h$ 个头的标准注意力（$d_h$ 为 head dim，$M_{\text{causal}}$ 为因果 mask）：

$$A_{\ell,h}(x)=\operatorname{softmax}\!\left(\frac{\mathbf{Q}_{\ell,h}(x)\,\mathbf{K}_{\ell,h}(x)^{\top}}{\sqrt{d_h}}+M_{\text{causal}}\right)$$

ASU 把温度放在**注意力 logits** 上，而不是输出 logits 上：

$$A^{\tau}_{\ell,h}(x)=\operatorname{softmax}\!\left(\frac{\mathbf{Q}_{\ell,h}\mathbf{K}_{\ell,h}^{\top}}{\tau\sqrt{d_h}}+M_{\text{causal}}\right),\qquad \tau>1$$

$\tau\uparrow$ 使注意力分布熵增大，事实检索所依赖的 sharp 路径被抹平。**关键区别**：这不是常见的"输出软标签蒸馏"，teacher 的输出层仍是标准 softmax；平滑发生在内部信息路由上。

**我们复现时用的等价技巧**（避免改动 HF forward）：把 $q\_proj$ 的输出缩放 $1/\tau$。因为 RoPE 是正交变换 $R$，满足 $R(q/\tau)=R(q)/\tau$，于是

$$\frac{(Q/\tau)K^{\top}}{\sqrt{d_h}}=\frac{QK^{\top}}{\tau\sqrt{d_h}}$$

即只需在 `self_attn.q_proj` 上挂 forward hook，**逐层、逐头**的温度场都能实现，且不新增任何参数。这是后面 SensiTemp 诊断能成立的技术前提。

### 2.2 为什么 $\tau>1$ 才有梯度

冻结 teacher 与可训练 student 在初始化时权重相同 $\theta_S=\theta_T$。定义

$$p_T^{\tau}(\cdot\mid x_{<t})=\operatorname{softmax}\!\bigl(f_{\theta_T}(x;A^{\tau})\bigr),\qquad p_S(\cdot\mid x_{<t})=\operatorname{softmax}\!\bigl(f_{\theta_S}(x;A^{1})\bigr)$$

若 $\tau=1$，则 $p_T^{1}\equiv p_S$，KL 恒为 0，$\nabla_\theta L_F=0$——**零遗忘梯度**。$\tau>1$ 使 teacher 分布变钝，二者出现分布差，梯度才出现。ASU 全部的"力"都来自这一个量。

### 2.3 损失函数（方向要写对）

ASU 源码用 `F.kl_div(input=log p_S, target=log p_T, log_target=True)`，展开为

$$\mathrm{KL}(p\|q)=\sum_{v\in\mathcal{V}} p(v)\log\frac{p(v)}{q(v)}$$

$$L_F(\theta)=\frac{1}{\sum_t m_t}\sum_t m_t\,\mathrm{KL}\!\left(p_T^{\tau}(\cdot\mid x_{<t})\,\middle\|\,p_S(\cdot\mid x_{<t})\right)$$

$$L_R(\theta)=\mathcal{L}_{\text{NLL}}(\theta;\mathcal{D}_r),\qquad L(\theta)=\alpha\,L_F(\theta)+L_R(\theta)$$

注意方向是 $\mathrm{KL}(p_T^\tau\|p_S)$（**mode-covering**）：被抹平、把概率摊到多个词上的 teacher 在前，学生被迫在这些词上都放质量，从而把原本集中在正确答案上的概率稀释掉，$F(\theta)$ 上升。若写成反向 $\mathrm{KL}(p_S\|p_T^\tau)$ 则是 mode-seeking，行为完全不同。梯度只经 student 回传，teacher 全程 `stopgrad`；两个分布共享同一条 $y_{<t}$ 前缀，不逐 token 比较两条不同序列。

### 2.4 复现基线

| 配置 | $F$：$0.5025\to$ | $\Delta F$ | $R$：$0.4651\to$ | $\Delta R$ |
|---|---|---|---|---|
| global $\tau=2$ | $1.3023$ | $+0.7998$ | $0.1695$ | $-0.2957$ |

$R$ 同时下降，是因为 $L_R$ 本身在优化 retain NLL，并非"遗忘不伤 utility"的独立证据。

## 3. 第一问：温度该给谁（SensiTemp，被证伪）

### 3.1 敏感度的定义

对位置 $(\ell,h)$ 施加 $\tau$ 干预，分别测 factual / function token 的 NLL 变化：

$$\Delta_f(\ell,h)=\mathcal{L}_{\text{fact}}(M;\tau_{\ell,h})-\mathcal{L}_{\text{fact}}(M;1),\qquad \Delta_u(\ell,h)=\mathcal{L}_{\text{func}}(M;\tau_{\ell,h})-\mathcal{L}_{\text{func}}(M;1)$$

$$S_{\ell,h}=\Delta_f(\ell,h)-\Delta_u(\ell,h),\qquad S_{\ell}=\text{整层同时干预时的同量}$$

直觉：$S$ 大表示该位置被平滑时主要伤事实、少伤语法，是"值得平滑"的位置。factual/function 划分我们用了停用词/标点启发式（ASU Appendix N 原用 GPT-4o 标注，这是近似，属已知局限）。

### 3.2 温度场构造

记 $[z]_+=\max(0,z)$，$c_\ell=\max_\ell|S_\ell|$、$c_h=\max_{\ell,h}|S_{\ell,h}|$（实测 $c_\ell=0.28794$，$c_h=0.03478$）：

$$\tau_{\ell,h}=2+2\,\frac{[S_\ell]_+}{c_\ell}+\frac{[S_{\ell,h}]_+}{c_h}$$

即"全局基线 $2$ + 层调制 + 头调制"。该场 $16\times32=512$ 个取值，$\tau\in[2.0,4.0]$，**平均** $\bar\tau=2.6070$。

### 3.3 非均匀，但没有净收益

gate 判定：layer CV $=1.115$、head CV $=9.220$、最敏感层 $[3,5,2]$ → 敏感度矩阵确实非均匀（`PASS`）。但把"更强的总平滑"与"更聪明的分配"分离开后：

| 配置 | $\Delta F$ | $\Delta R$ |
|---|---|---|
| global $\tau=2$ | $+0.7998$ | $-0.2957$ |
| global $\bar\tau=2.607$（等平均对照） | $+1.7914$ | $-0.3042$ |
| SensiTemp 场（$\bar\tau$ 相同） | $+1.7470$ | $-0.2997$ |
| 打乱场（同 $\tau$ 多重集，层/头重排） | $+1.5166$ | $-0.3053$ |

$$\Delta F_{\text{field}}=1.7470 < \Delta F_{\text{equal}}=1.7914$$

**数学结论**：在当前 setting 下，把 $\tau$ 从标量放宽到张量场 $\tau_{\ell,h}$ 没有产生可辨识的额外自由度收益；增益主要由 $\bar\tau$ 解释。这是一个干净的**负结果**：它把可主张的贡献从"新温度场/新模块"收缩为"调度"。讲故事时这一幕必须保留，否则整篇会被等预算对照一击即溃。

## 4. 第二问：温度该何时给（$\tau$ 课程，真正赢的增量）

不改空间结构，只让 $\tau$ 成为训练步 $k$ 的函数（$K=60$）：

$$\tau_{\text{low2high}}(k)=1.5+1.5\,\frac{k}{K-1},\qquad \tau_{\text{high2low}}(k)=3.0-1.5\,\frac{k}{K-1},\qquad \tau_{\text{global}}(k)=2$$

| 配置 | $\Delta F$ | $\Delta R$ |
|---|---|---|
| global $\tau=2$ | $+0.7998$ | $-0.2957$ |
| high2low $3\to1.5$ | $+0.5621$ | $-0.3200$ |
| **low2high $1.5\to3$** | **$+0.9491$** | $-0.2563$ |

$$\Delta F_{\text{low2high}}-\Delta F_{\text{global}}=+0.1493,\qquad \Delta R_{\text{low2high}}=-0.2563\le 0.05$$

机制解释（**推断，非证明**）：$\mathrm{KL}(p_T^\tau\|p_S)$ 关于 $\tau>1$ 单调递增，把大 $\tau$ 放在后期等于把大遗忘梯度放在后期；早期低 $\tau$ 时 teacher 仍"认得"事实，梯度温和，先稳住 $L_R$。这是**同一标量**的时间函数：零新参数、零新模块、零架构改动，符合简单性优先。

## 5. 第三问：忘掉了吗（深度审计，最有价值的转折）

### 5.1 恢复阶梯的定义

$$F_1=F(M),\qquad F_2=F(M\mid \text{3 个 forget QA 作 few-shot 前缀}),\qquad F_3=F\bigl(\text{SGD}_{K=20}(M;\mathcal{D}_f^{\text{sub}},\eta=10^{-5})\bigr)$$

INT8 伪量化：$s=\max_{i,j}|W_{ij}|/127$，$\;W^{q}=s\cdot\operatorname{clip}(\operatorname{round}(W/s),-127,127)$，得 $F_Q$。

恢复增益与归一化恢复率（相对 $M_0$ 的遗忘增益）：

$$g_{\text{P2}}=F_1-F_2,\quad g_{\text{P3}}=F_1-F_3,\qquad \rho_{\text{P2}}=\frac{g_{\text{P2}}}{F_1-F(M_0)},\quad \rho_{\text{P3}}=\frac{g_{\text{P3}}}{F_1-F(M_0)}$$

$$\mathrm{depth}(M)=\tfrac{1}{2}\bigl(\rho_{\text{P2}}+\rho_{\text{P3}}\bigr)\quad(\text{越大} = \text{越浅})$$

### 5.2 实测

| 模型 | $F_1$ | $R$ | $g_{\text{P2}}$ | $\rho_{\text{P2}}$ | $g_{\text{P3}}$ | $\rho_{\text{P3}}$ | $\mathrm{depth}$ | $F_Q-F_1$ |
|---|---|---|---|---|---|---|---|---|
| $M_0$ base | $0.4621$ | $0.4700$ | $-0.0158$ | — | $+0.0709$ | — | — | $+0.1946$ |
| ASU-u | $0.9604$ | $0.4222$ | $+0.2613$ | $0.524$ | $+0.1850$ | $0.371$ | $\mathbf{0.448}$ | $+0.2568$ |
| GradAscent | $0.9335$ | $0.8012$ | $+0.2607$ | $0.553$ | $+0.0970$ | $0.206$ | $0.379$ | $+0.1614$ |

ASU-u 有 $44.8\%$ 的遗忘增益可被 few-shot 或 20 步微调恢复。这说明前两幕优化的 $F(\theta)$ 只是"当前参数下的行为"，不是"知识被擦除"。

### 5.3 这如何改变目标函数

审计把约束显式化：

$$\min_\theta\ \alpha L_F(\theta)+L_R(\theta)\quad \text{s.t.}\quad \rho_{\text{P3}}(\theta)\le\varepsilon$$

更激进的做法是把抗恢复写进目标（`R2-C`，**尚未实现**）：内步模拟攻击者在外步眼里最大化恢复后的 forget NLL，

$$\theta'=\theta-\eta\nabla_\theta\mathcal{L}_{\text{NLL}}(\theta;\mathcal{D}_f^{\text{atk}}),\qquad \min_\theta\ \alpha L_F+L_R-\beta\,\mathcal{L}_{\text{NLL}}(\theta';\mathcal{D}_f)$$

这是双环 meta 目标，与本文的恢复阶梯共享攻击协议。**必须标注为下一步，不能把事后审计当作已经实现的抗训练。**

## 6. 往哪深挖：子空间选择性（条件性方向）

$$G_f=\nabla_W\mathcal{L}_{\text{NLL}}(\theta;\mathcal{D}_f),\qquad G_r=\nabla_W\mathcal{L}_{\text{NLL}}(\theta;\mathcal{D}_r)$$

对每个模块权重 $W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ 做 SVD：$G_f=U\Sigma V^{\top}$，取 $r=8$，

$$V_{\text{top}}=V[:, :r],\qquad V_{\text{bot}}=V[:, -r:],\qquad \operatorname{ov}(g,B)=\frac{\|gB\|_F^{2}}{\|g\|_F^{2}}$$

在 ASU-u 的 `v_proj` / `down_proj`（层 2/8/14）上实测：

| 模块 | $\operatorname{ov}(G_r,V_{\text{top}})$ | $\operatorname{ov}(G_r,V_{\text{bot}})$ |
|---|---|---|
| L2 v / down | $0.887$ / $0.129$ | $0.0004$ / $0.0006$ |
| L8 v / down | $0.735$ / $0.314$ | $0.0007$ / $0.0004$ |
| L14 v / down | $0.800$ / $0.325$ | $0.0007$ / $0.0005$ |

$$\overline{\operatorname{ov}(G_r,V_{\text{top}})-\operatorname{ov}(G_r,V_{\text{bot}})}=0.531$$

含义：高方差方向与 retain 共享，低方差方向才是 forget-specific——只压缩 $V_{\text{bot}}$ 有希望在不必伤 $R$ 的前提下加深遗忘。**但** RepSelect / ADU / ALMPU 已密集占据该空间，新颖性冲突，故 `REVISE`；只在能证明"ASU 输出遗忘 $\times$ attention K/V 子空间"的耦合区别于上述工作时才重启。

## 7. 被证伪的分支：深层 $\Delta A$ 方向

反事实主体配对（token-equal donor）：$\Delta A_\ell=h_\ell(X^{+})-h_\ell(X^{-})$；对 pooled $\Delta A$ 做 PCA 取前 $r$ 个右奇异向量 $\mathbf{B}\in\mathbb{R}^{d\times r}$：

$$h'_{\text{suppress}}=h-(h\mathbf{B})\mathbf{B}^{\top},\qquad h'_{\text{align}}=h-(h\mathbf{B})\mathbf{B}^{\top}+\mu_r\mathbf{B}^{\top},\quad \mu_r=\mathbb{E}_{x\in\mathcal{D}_r}\!\left[h_r\mathbf{B}\right]$$

gate：$\Delta F\ge0.15$ 且 $\Delta R\le0.05$。实测（baseline $F=0.4635,R=0.4712$）：

| 层 | suppress $(\Delta F,\Delta R)$ | align $(\Delta F,\Delta R)$ |
|---|---|---|
| 10 | $(+0.083,+0.068)$ | $(+0.775,+0.734)$ |
| 12 | $(+0.133,+0.099)$ | $(+1.448,+1.501)$ |
| 14 | $(+0.066,+0.055)$ | $(+1.018,+1.044)$ |
| 15 | $(+0.022,+0.016)$ | $(+0.699,+0.707)$ |

align 各层 $\Delta F/\Delta R\approx1$，说明 hook 等价于给整个残差流加了一个与事实无关的扰动，而不是"只擦掉事实方向"；严格阈值下 $0$ 项通过。这与 V2 中"沿 $\Delta A$ 抑制"的失败模式一致，故 `REVISE`。

## 8. 完整故事线（建议的论文口径）

| 幕 | 主张 | 证据 | 结果 |
|---|---|---|---|
| 起点 | ASU 的标量 $\tau$ 是唯一杠杆，但"给谁 / 何时给 / 够不够深"未被回答 | 论文自承温度脆弱、需 binary search | 问题锚定 |
| 1 诊断 | 事实/功能敏感度在层与头上高度非均匀 | layer CV $1.115$、head CV $9.220$ | `PASS` |
| 2 证伪 | 结构化温度场不带来净收益 | $1.747<1.791$（等平均）、打乱场 $1.517$ | `REVISE` |
| 3 增量 | 时间维度比空间维度更有效 | low2high $\Delta F=0.9491$ vs $0.7998$，$\Delta R<0$ | `PROMOTE` |
| 4 转折 | 行为遗忘是浅的 | ASU-u $\mathrm{depth}=0.448$ | `PROMOTE` |
| 5 下一步 | 抗恢复目标 / bottom-PC 选择性 | 审计动机 + 子空间 gate $0.531$ | 条件性 |

**口径建议**：不要写"我们提出新的层-头温度场"（已被等平均对照否决）；写"我们发现温度的时间调度比空间分配更有效，并用统一恢复阶梯证明现有 SOTA 的遗忘是浅层的"。前者会被一句"换成同平均 $\tau$ 呢"击穿，后者把负结果变成论证链的一环。

## 9. 局限与可复现性

- 单 seed、$60$ steps、仅 TOFU-1B / `forget10`；未测 3-seed、MMLU/GSM8K、生成质量、WMDP/MUSE。
- factual/function 为停用词启发式，非 GPT-4o 标注；INT8 为伪量化（fake quant），非真实量化 kernel。
- $\alpha=0.1$ 与 ASU 默认 $1.0$ 不同；$\tau$ 干预用 $q\_proj$ 缩放实现，已论证与 logits 温度等价。
- `R2-C` meta 目标、`R2-A` 两段式、`R3-3` 双阶段均**未实现**；本轮只复现 gate / 动机。
- 全部原始 JSON 在 `workspace/auto_experiments/results/`；复现验证在 `workspace/verification/`（独立 $7/7$、串联 $11/11$、`OVERALL: PASS`），逐步记录见 `VERIFICATION_LOG.md`。
- 跨模型评审后端不可用；本文的论证是执行者 provisional，未经独立评审。

## 10. 结论

当前已复现的创新方向可以讲成一个完整的**"收缩型"故事**：从"给 ASU 加一个新温度场"出发，被等预算对照逼退到"只加一个时间调度"，再被恢复阶梯推到"我们其实还没证明知识被擦除"。数学上贯穿始终的是同一个量——$\tau$ 造成的 teacher/student 分布差 $\mathrm{KL}(p_T^\tau\|p_S)$；贡献不在于扩大这个差，而在于**何时扩大它**，以及**用 $\rho_{\text{P2}},\rho_{\text{P3}}$ 度量扩大之后到底剩下了什么**。

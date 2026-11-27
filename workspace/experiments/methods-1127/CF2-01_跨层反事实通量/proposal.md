---
id: CF2-01
rank: 1
title: 跨层反事实通量 Φℓ —— 抹去新增事实写入，而非静态主体差异
avg_score: 8.2
verdict: Strong Accept (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无跨模型独立回执
novelty_status: PROCEED（执行者查新判断，非独立裁决）
baseline_relation: 把端点差 ΔAℓ 改为相邻层差分 Φℓ=ΔAℓ+1−ΔAℓ
---

# CF2-01 · 跨层反事实通量 $\Phi_\ell$

## 0. 一句话故事

> 原始 $\Delta A_\ell$ 同时包含“主体本来就不同”的静态偏移和“事实在某一层被写入”的新增变化。我们对激活差沿层做一次差分，让不随层变化的身份偏移自动抵消，只处理 $\Phi_\ell=\Delta A_{\ell+1}-\Delta A_\ell$ 所刻画的跨层新增写入；Student 逐层内化这些写入事件的反向目标，而不是粗暴压掉整条激活轨迹。

## 1. 问题与近邻空白

残差流把各层写成累积和，因此端点差无法区分“从浅层一路携带的差异”与“当前层新生成的差异”。Fernando 与 Guitchounts（2025，*Transformer Dynamics*）已把残差流视为跨层动力系统并分析表示速度，但该工作属于解释性分析，不是遗忘训练。

本方案的科学问题不是“哪一层数值最大”，而是：**事实绑定在哪个层间转移中新增？** 这与现有候选不同：CF-03 用 patch 做单点因果门控，CF-10 用纯度决定层调度，CF-06 改 CoT step 作用域；本方案不训练探针、不做 patch 扫描、不主张 schedule，唯一改动是激活差的微分算子。

## 2. 可证伪假设

**H**：真实配对的 $\Phi_{\ell,p}$ 在少数事实写入转移上显著大于 wrong-donor / shuffled-donor，而静态主体偏移主要留在 $\Delta A_{0,p}$ 或被差分消除。若 $\Phi$ 对真实与伪配对不可分，或其收益可由等范数原始 $\Delta A$ 解释，则核心机制被证伪。

## 3. 方法

### 3.1 Teacher 反事实轨迹

冻结 Teacher $\theta_0$ 分别前向 $X^+\oplus y^+$ 与 $X^-\oplus y^+$，在答案跨度 $p\in S_f$ 读取：

$$
A^\pm_{\ell,p}=h_{T,\ell}(X^\pm\oplus y^+;p),\qquad
\Delta A_{\ell,p}=A^+_{\ell,p}-A^-_{\ell,p}.
$$

Teacher 激活全部 stop-gradient；不引入注意力温度，不拼入 donor 答案 $y^-$。

### 3.2 跨层通量与累积分解

定义从层 $\ell$ 到 $\ell+1$ 的反事实通量：

$$
\boxed{\Phi_{\ell,p}=\Delta A_{\ell+1,p}-\Delta A_{\ell,p}}
$$

于是：

$$
\Delta A_{L,p}=\Delta A_{0,p}+\sum_{\ell=0}^{L-1}\Phi_{\ell,p}.
$$

若主体身份造成近似常数偏移 $\Delta A_{\ell,p}=c_p+\epsilon_{\ell,p}$，则 $\Phi_{\ell,p}=\epsilon_{\ell+1,p}-\epsilon_{\ell,p}$，静态项 $c_p$ 精确消失。这里“事实写入”是可检验解释，不是由公式自动保证的语义标签。

用一组结构匹配的伪配对估计通量尺度，得到无训练权重：

$$
w_{\ell,p}=\sigma\!\left(
\frac{\|\Phi^{\mathrm{true}}_{\ell,p}\|_2-
\operatorname{median}_{\mathrm{shuffle}}\|\Phi^{\mathrm{shuffle}}_{\ell,p}\|_2}{\tau}
\right).
$$

$w$ 只表示相对配对特异性，不宣称因果效应，避免与 CF-03 的 patch 门控重合。

### 3.3 Student 目标与内化

在层间转移的终点 $\ell+1$ 构造目标：

$$
T_{\ell+1,p}=\operatorname{stopgrad}\!\left[
\Pi_{\mathcal K}\!\left(A^+_{\ell+1,p}-\alpha w_{\ell,p}\Phi_{\ell,p}\right)
\right],
$$

$$
\mathcal K=\{h:r_{f,p}(h)\le c,\ s_{d_j,p}(h)\le d_j,\ \|\Delta r_{E,p}(h)\|\le e\}.
$$

$\Pi_{\mathcal K}$ 保证原事实读出下降、donor 答案不升、EOS/长度不漂移；读出经冻结下游重算。Student 损失为：

$$
\mathcal L_F=\frac1{|S_f|L}\sum_{\ell=0}^{L-1}\sum_{p\in S_f}
\left\|h_{S,\ell+1}(X^+\oplus y^+;p)-T_{\ell+1,p}\right\|_2^2,
\qquad
\boxed{\mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R}.
$$

主结果要求 $\lambda_{KL}^{(\mathrm{ablation})}=0$，部署只保留 Student，无运行时 hook。静态全层训练是唯一默认；若另加 schedule，必须单独等预算验证，不能归功于通量。

## 4. 与既有工作的差异

| 近邻 | 已有对象 | 本方案新增轴 |
|---|---|---|
| *Transformer Dynamics*（2025） | 残差流速度、连续性和动力学解释 | 对“正反事实差”再沿层差分，并转为参数内化目标 |
| CAA / ActAdd | 配对均值差作为推理时 steering 向量 | 不使用端点向量做推理加法，训练 Student 消除新增写入 |
| CF-03 / CF-10 | 单点 patch 因果权重 / 纯度课程 | 无 patch、无课程，贡献只在 $\partial_\ell\Delta A$ |
| V1 沿 $\Delta A$ 抑制（项目已证伪） | 直接压原始差值 | 处理相邻层增量并保留累积守恒分解，必须过伪配对门 |

据当前检索，没有发现把 $\Delta A_{\ell+1}-\Delta A_\ell$ 作为反事实遗忘目标的直接论文；这不是“已证明首创”。

## 5. /idea-evaluator 兼容评估（PROVISIONAL）

**定位**：Novel Method。致命缺陷：1 MAJOR（F6：通量是否对应事实写入需决定性实验）+ 1 MINOR（F1：与残差动力系统近邻划界）；零 CRITICAL。按项目 TOFU-1B、单卡 H20 的首验规模，Frontier Exploration 周期约 3–9 个月，资源匹配绿。

| 维度 | 分数 | 机制依据 |
|---|---:|---|
| Higher | 9 | 分离静态身份偏移与新增写入，直接针对原始 $\Delta A$ 的首要混淆 |
| Faster | 8 | 与基线共享两次 Teacher 前向；差分和权重均为缓存上的闭式运算 |
| Stronger | 8 | shuffle/wrong 配对尺度和累积守恒提供可审计的抗混淆检查 |
| Cheaper | 8 | 无 SAE、无 probe、无 patch 扫描、无额外标注 |
| Broader | 8 | 同一差分可用于答案跨度、LRM 轨迹和其他事实类型 |

**平均分 $41/5=8.2$。裁决：Strong Accept，worth pursuing, pending the validation experiment。** 范式探测：First Principles=Yes（端点差不是写入事件）、Elephant=Yes、Technology Cycle=Partial、Hamming=Yes。

前三项行动：冻结通量定义与伪配对族；先跑真实/伪配对通量可分性；再做通量目标与等范数端点目标的双臂 pilot。

## 6. 决定性实验与失败模式

**最小判别实验**：真实、wrong、shuffled 三类配对各提取 $\Phi$；先检验层间稀疏性与可分性，再比较“通量目标 / 等范数原始差目标 / 随机层间差目标”。主要指标为 $r_f$、retain NLL、donor 答案率、提前 EOS、完整回答率、无 hook probe/relearning 恢复。

**失败即撤回**：真实与伪配对的 $\Phi$ 不可分；$\Phi$ 在所有层近似均匀且等效于原始差；目标主要造成输出抑制；去掉静态偏移后也去掉目标事实信号。历史 stage0–14 不复用为正向证据。

## 7. 来源与诚信状态

- 基线：`方向I_基线_v0.6.1_answer_span.md` §2.2、§2.3、§4.1、§6。
- Fernando & Guitchounts（2025），*Transformer Dynamics: A Neuroscientific Approach to Interpretability of Large Language Models*，检索全文：https://2025.ccneuro.org/abstract_pdf/Fernando_2025_Transformer_Dynamics_A_Neuroscientific_Approach_Interpretability.pdf 。
- Elhage et al.（2021），*A Mathematical Framework for Transformer Circuits*，残差流读写框架。
- 本方案为 plan-only；分数是执行者机制推断，不是实验结果或独立 reviewer verdict。文献未完成 D1 批量身份核验，不能据此生成 BibTeX。

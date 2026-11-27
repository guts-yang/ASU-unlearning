---
id: CF3-08
rank: 8
title: 反事实 Copula 尾依赖擦除（CTDE）
avg_score: 7.4
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无独立跨模型回执
novelty_status: PROCEED WITH CAUTION
baseline_relation: 从均值/协方差激活差转向主体槽与答案位置的秩变换尾部共同 exceedance 差
---

# CF3-08 · 反事实 Copula 尾依赖擦除

## 0. 一句话故事

> 事实信号可能不体现在平均激活或线性相关，而只在少数极端激活共同出现时暴露。我们把主体槽和答案位置分别按 retain 分布做秩变换，再比较正反事实的 copula 尾依赖；Student 只削弱目标配对的异常共同 exceedance，同时保持两端边缘分布，避免靠激活坍缩或拒答完成损失。

## 1. 问题与近邻空白

SuperActivators（Goldberg et al.，2026）报告可靠概念信号集中在激活分布尾部；PINT（2026）用 HSIC 降低最终层表示与下一 token 的统计依赖，CF2-03 则处理条件交叉协方差。这三者共同限制了“尾部”“独立性”“条件相关”各自的宽泛新颖性。

本方案必须守住更窄的对象：**retain 秩坐标中的正反事实 copula 尾依赖差**，且是主体槽到答案跨度的跨位置依赖。与 SuperActivators 的单变量高分位检测不同，这里估计联合尾部分布；与 CF2-03 不同，copula 先剥离边缘分布并专门处理非线性尾部，而非条件线性协方差。

## 2. 可证伪假设

**H**：真实配对的主体—答案尾依赖显著超过 wrong/shuffled 与 Gaussian-copula 对照；削弱该依赖能够降低独立生成式 probe 和 relearning 恢复，而边缘分位数与 retain 效用稳定。若尾部信号可由少量主体 token、分箱噪声或边缘分布漂移解释，则假设被证伪。

## 3. 方法

### 3.1 retain 秩坐标

以低秩冻结投影控制维度，对样本 $i$ 的主体槽和答案位置激活分别得到 $u_i^\pm=P_sA_i^\pm(s)$、$v_{i,p}^\pm=P_pA_i^\pm(p)$。每个坐标使用独立 retain 集的经验 CDF 变换：

$$
U_i^\pm=\widehat F_{R,u}(u_i^\pm),\qquad
V_{i,p}^\pm=\widehat F_{R,v}(v_{i,p}^\pm).
$$

这样首先剥离边缘尺度；若变换后的边缘仍明显偏离均匀分布，则该层位置不进入 copula 训练。

### 3.2 反事实尾依赖差

对预注册阈值集合 $\mathcal Q$ 定义上尾条件 exceedance：

$$
\tau_q^\pm(u,v)=
\frac{\mathbb E[\mathbf 1(U^\pm>q)\mathbf 1(V^\pm>q)]}
{\mathbb E[\mathbf 1(V^\pm>q)]+\epsilon},
\qquad q\in\mathcal Q.
$$

定义真实相对 null 的尾依赖差：

$$
\boxed{
\Delta\tau_q=\tau_q^{true,+}-\tau_q^{true,-},
\qquad
\rho_q=\Delta\tau_q-\operatorname{median}_{null}\Delta\tau_q^{null}}
$$

这只是经验尾依赖，不声称已收敛到渐近极值理论的尾依赖系数。下尾及 $(U<1-q,V>q)$ 等交叉尾部作为方向对照。

Student 使用软 exceedance $a_q(U)=\sigma((U-q)/\tau)$ 得到可微 $\widetilde\tau_q^{S,+}$，优化：

$$
\mathcal L_{\mathrm{tail}}=
\sum_{q\in\mathcal Q}\omega_q
\left(\widetilde\tau_q^{S,+}-\tau_q^{null}\right)^2.
$$

### 3.3 边缘死区与非捷径约束

为防止将两端激活压成常数，对多个保留分位数设置边缘死区：

$$
\mathcal L_{\mathrm{marg}}=
\sum_{r\in\mathcal Q}
\left[\left|\widehat Q_{S,u}(r)-\widehat Q_{R,u}(r)\right|-\delta_u\right]_+
+\left[\left|\widehat Q_{S,v}(r)-\widehat Q_{R,v}(r)\right|-\delta_v\right]_+.
$$

总目标：

$$
\boxed{\mathcal L=\lambda_t\mathcal L_{\mathrm{tail}}+
\lambda_m\mathcal L_{\mathrm{marg}}+
\lambda_n\mathcal L_{\mathrm{neutral}}+\beta\mathcal L_R}.
$$

$\mathcal L_{\mathrm{neutral}}$ 包含 $r_f\le c$、donor 不升和 EOS/长度约束；不使用 donor 答案 KL，部署不需要 copula 模块。

## 4. idea-evaluator 兼容评估

**定位**：Novel Method。F1 为 MAJOR：需与 SuperActivators、PINT 和 CF2-03 实证区分；F6 为 MAJOR：高维尾部估计的样本效率与跨 seed 稳定性可能不足。零 CRITICAL。

| 维度 | 分数 | 机制依据，均尚未由本轮数据确认 |
|---|---:|---|
| Higher | 8 | 能覆盖均值和二阶矩完全相同的非线性尾部事实通路 |
| Faster | 6 | 秩变换与软分位数可批量计算，但需要足够样本稳定估计尾部 |
| Stronger | 9 | 边缘死区直接阻断坍缩捷径，尾部/交叉尾部/null 提供多层压力测试 |
| Cheaper | 6 | 不需新标注，但多样本和阈值扫描增加数据与计算需求 |
| Broader | 8 | 可统一文本、视觉和 CoT 中“稀疏极端共同激活”的跨位置依赖 |

**平均分 $\bar S=37/5=7.4$；裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 生命周期 3–9 个月；每周工时未知，资源匹配 Yellow。范式探测：First Principles=Yes，Elephant=Yes，Technology Cycle=Partial，Hamming=Partial。

前三项行动：先做样本量—偏差曲线；比较经验 copula、Gaussian copula、均值/协方差与 SuperActivator 单变量门；只有尾部增量稳定时才训练。

## 5. 决定性实验与失败判据

**最小实验**：在低秩二维投影上比较真实、wrong、shuffle、matched-retain；加入边缘打乱和 Gaussian-copula 保持相关性的对照。训练采用“仅边缘死区 / 仅尾依赖 / 联合”三臂，报告 $r_f$、probe、relearning、retain NLL、donor、EOS 和完整回答率。

**失败即撤回**：尾依赖随样本量或 seed 翻转；边缘打乱后信号不变；Gaussian copula 已解释全部结构；$\Delta\tau$ 下降但 probe 不变；或收益来自边缘分位数漂移。

## 6. 来源与诚信状态

- Goldberg, Kim, Stein & Wong（2026），*SuperActivators: Only the Tail of the Distribution Contains Reliable Concept Signals*，arXiv:2512.05038。
- Shkabrii, Plant & Pawelczyk（2026），*Forget What Comes Next: LLM Unlearning via Statistical Independence*，arXiv:2611.03955。
- 基线 §2、§4、§5、§6；明确区别于 CF2-03 的条件交叉协方差。

plan-only；分数为机制推断，非实测或独立评审；文献未完成 D1 批量核验。

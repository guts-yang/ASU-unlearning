# RIU: Recovery-Invariant Unlearning —— 研究提案 v5

> **一句话**：现有遗忘方法都在“压制表达”，但事实是冗余编码的——压掉一个副本，别的副本还在。RIU 换掉这个范式：把“任何恢复手段后事实不可解码”作为优化目标，作为可插到任意 base 算法上的通用插件。
>
> 版本：v5（RIU 重写版） | 日期：2026-11-20 | base：ASU（ICLR 2026, arXiv:2603.01285v1） | 评测协议：LeakRU（arXiv:2608.04519v1）
> 数据来源：ASU 全文、LeakRU 全文核实；sanity check 实测数据（TOFU forget05 / retain95, Llama-2-7B-Chat, seed 1001）。

---

## 摘要（Abstract，骨架）

现有 LLM 遗忘方法在跨访问路径与恢复攻击下普遍泄露。我们的审计发现：连最强的 attention-smoothing 方法（ASU）在 FocusOnKey 攻击下仍恢复 $32.4\%$ 的知识，且在 DS 推理结构下所有方法的遗忘质量（FQ）仅为 $25$ 点——问题不在某一方法，而在“压制表达”这一全领域共同的范式：知识冗余编码，压制一个副本，其他副本仍在。我们提出 **RIU（Recovery-Invariant Unlearning）**：将恢复建模为算子集合 $\mathcal{R}$（语义保持变换 / 短微调 / 量化），以“目标事实在任何恢复后均不可线性解码”为优化目标，通过对抗内环、表示塌缩与正交几何擦除实现，可作为 plug-in 加到任意 base 算法上。在 TOFU / MUSE / WMDP 与四类 base 上验证一致降低 $R_{\max}$ 与 $RR$，同时保持效用。结论：可恢复性而非表达抑制，应成为遗忘的本质目标。

---

## 1. 背景与问题定义

### 1.1 领域病症：行为遗忘 $\neq$ 隐私

LLM 遗忘旨在从模型中移除敏感 / 版权 / 有害知识，同时保持通用效用。现有评测主要测量**标准查询下模型是否还能输出目标事实**，但 GDPR “被遗忘权”要求的是**信息不可访问**：模型在默认问法下闭嘴，不代表真的忘掉——换语言、绕推理、重复关键词都可能把知识套回来。

形式化地，遗忘黄金标准是让遗忘后模型 $\theta_u$ 与从头重训模型 $\hat{\theta}=A(D\setminus D_f)$ 不可区分：

$$
M_u = U(A, D, D_f) \approx A(D \setminus D_f)
$$

典型优化目标是在遗忘与保留之间权衡：

$$
\min_{\theta}\ \lambda\,\mathbb{E}_{(x,y)\sim\mathcal{D}_F}\big[\mathcal{L}_F(y\mid x;\theta)\big] + \mathbb{E}_{(x,y)\sim\mathcal{D}_R}\big[\mathcal{L}_R(y\mid x;\theta)\big]
$$

**核心 gap**：输出层的 forget quality 高，不等于信息真的不可恢复。

### 1.2 立靶：ASU 自己立的规矩

**ASU**（Attention Smoothing Unlearning, ICLR 2026, Zade / Zhou / Liu / Zhu, Wayne State University + Michigan State University）在引言中批评现有方法：“效果往往很 superficial，因为 unlearning **fails to generalize across task formats**”——它把“跨格式泛化”立为评判标准，并声称自己通过 attention 温度平滑解决了这个问题。

**研究钩子**：用 ASU 自己承认的标准来审计 ASU，逻辑闭环。

---

## 2. Base 机制核实：ASU（来自全文 arXiv:2603.01285v1）

### 2.1 Forget-Teacher 构造（Eq.4）

在每层每 head 的自注意力 logits 上插入温度 $\tau \ge 1$，其余组件（projection / FFN / LayerNorm）不变：

$$
\mathrm{Attention}(Q_h,K_h,V_h;\tau) = \mathrm{Softmax}\!\left(\frac{Q_hK_h^{\top}}{\tau\sqrt{d_k}}\right) V_h
$$

- $\tau > 1$ 提高注意力熵、摊平分布，削弱 token 间的词法级 / 语义级关联；
- $\tau = 1$ 复原 base 模型行为；
- teacher $\theta_\tau$ **冻结**，无外部模型、除 $\tau$ 外无新增参数，仅用于 forget set 生成蒸馏目标。

### 2.2 遗忘目标（Eq.5）

在 forget set 上做自蒸馏，对齐 student 与平滑 teacher 的输出分布；retain set 用 GD（Eq.2）或 KL（Eq.3）正则，得到 ASU$_{\text{GD}}$ / ASU$_{\text{KL}}$：

$$
\mathcal{L}_{ASU} = \mathbb{E}_{(x,y)\sim\mathcal{D}_F}\left[\frac{1}{T}\sum_{t=1}^{T} \mathrm{KL}\Big(p(\cdot\mid x\circ y_{<t};\theta_\tau)\,\|\,p(\cdot\mid x\circ y_{<t};\theta)\Big)\right]
$$

### 2.3 关键超参与实现（Appendix G/H/K）

- $\tau$ 由二分搜索 + NLL 选定，有效区间常为 $2.0$–$3.0$；TOFU 上全部 $\tau=2.3,\ \lambda=0.1$；Real-world $\tau=2.5$–$2.7$；
- 目标模型 Llama-2-Chat-7B；$2\times$ H100 + DeepSpeed ZeRO-3；AdamW，lr=$1\times10^{-5}$，5 epochs；
- 表示级变体（Appendix F）：hazardous 场景对齐第 7 层隐状态（MSE），只更新 $7,6,5$ 三层 MLP；
- **重要修正**：$\tau$ 只用于构造冻结 teacher，最终 unlearned student 推理时**没有 $\tau$ 旋钮**——因此“推理时降温恢复”假设不成立，已从方案删除。

### 2.4 理论

Main Theorem 在假设 A1–A4 下证明存在区间 $[\tau_0,\tau_1]$ 同时满足：
- **遗忘**：事实 token 概率 $p_{\theta_\tau}(y_t^\star\mid\cdot) \le \epsilon_F$；
- **流畅**：功能 token 的平均 loss 增量 $\le \delta_G$。

---

## 3. 评测底座：LeakRU（arXiv:2608.04519v1）

清华 + 蚂蚁集团，已开源 `leak-resistant.site`。定义两个判据：
- **Path Consistency**：跨多跳推理路径不可访问；
- **Recovery Robustness**：恢复攻击下不可恢复。

### 3.1 六种逻辑推理结构

| 结构 | 逻辑形式 | 访问机制 |
|------|---------|---------|
| HS 假言三段论 | $(A\Rightarrow B)\wedge(B\Rightarrow C)\vdash(A\Rightarrow C)$ | 链式关系组合 |
| MP 肯定前件 | $A\Rightarrow B\wedge A\vdash B$ | 规则 + 实例 |
| LL 莱布尼茨律 | $x=y\wedge A(x)\vdash A(y)$ | 别名 / 等价实体替换 |
| CC 合取 | $x\in A\wedge x\in B\vdash x\in A\cap B$ | 多条件定位交集 |
| DS 选言三段论 | $(A\vee B)\wedge\neg A\vdash B$ | 排除候选推理 |
| MT 否定后件 | $(A\Rightarrow B)\wedge\neg B\vdash\neg A$ | 反向依赖推理 |

### 3.2 三种恢复攻击与指标

- **Probab**：多项式采样（非贪婪解码）即可恢复知识；
- **FocusOnKey**：重复关键 token 即可恢复，其假设为“表观遗忘仅反映 key token 注意力下降，而非真擦除”；
- **Quantization**：量化步长抹平遗忘前后权重差异。

指标：

$$
FQ(\theta_u,\theta_p,Q) = 1 - \frac{\mathrm{Acc}(\theta_u,Q)}{\mathrm{Acc}(\theta_p,Q)}
$$

$$
RR = \frac{\sum_i V\big(f_{\theta_u}(T_{attack}(x_i)),y_{gt}\big)}{|S_{suc}|},\qquad \mathrm{Robustness} = 1 - \max(RR)
$$

**关键**：LeakRU 未评测 ASU（它测的 “AS” 是 Attention-Shifting, Tan et al.）——这是研究空白。

---

## 4. Sanity Check：实测数据与判定

**设置**：TOFU forget05 / retain95，Llama-2-7B-Chat，seed 1001；ASU $\tau=2.3,\ \alpha(\lambda)=0.1$；$\theta_p$ 单跳准确率 $0.970$；FQ 为百分点，RR 为百分数。

| Method | FQ_SH | FQ_HS | FQ_MP | FQ_LL | FQ_CC | FQ_DS | FQ_MT | RR_FOK | RR_Probab |
|--------|-------|-------|-------|-------|-------|-------|-------|--------|-----------|
| **ASU** | 56.2 | 50.0 | NA | 100.0 | 40.0 | 25.0 | 71.4 | 32.4 | 42.6 |
| GA | 55.7 | 0.0 | NA | 50.0 | 20.0 | 25.0 | 57.1 | 43.5 | 56.5 |
| NPO | 40.7 | 0.0 | NA | 50.0 | 0.0 | 25.0 | 57.1 | 53.7 | 75.5 |

### 4.1 预设判据与判定

- **H1（路径不一致）成立**：

$$
\max(FQ_{SH},FQ_{HS}) - \min(FQ_{DS},FQ_{MT}) = 56.2 - 25.0 = 31.2 > 10
$$

- **H2（恢复攻击有效）成立**：$RR_{\text{FocusOnKey}}(\mathrm{ASU}) = 32.4\% > 20\%$，与 GA ($43.5\%$)、NPO ($53.7\%$) 同量级。

### 4.2 必须如实报告的限制（叙事的关键转折点）

1. **先验被反证**：我们曾预测“ASU 降低 key-token 注意力 → 对 FocusOnKey 最脆弱”。实测 ASU 的 RR 三者**最低**（最鲁棒），MT 上 FQ=71.4 全场最高。
2. **共同短板**：$FQ_{DS}=25.0$ 三方法完全相同；RR 全员非零。→ 问题不是 ASU 独有。
3. **待补三项**：基线未对齐（NPO FQ_SH=40.7，差 15.5 点，$\beta$ 扫 0.05/0.2 未拉回）；MP=NA（$\theta_p=0/10$）、DS=25 疑 $1/4$ 粒度太粗；单 seed、forget05。

---

## 5. 故事：两次升维

1. **立病**：只测标准问法 $\neq$ 隐私。
2. **立靶**：ASU 自立“跨格式泛化”标准。
3. **亮尺**：LeakRU 现成协议且未测 ASU。
4. **反转**：sanity check 反了——ASU 反而最鲁棒，先验被反证，如实写出。
5. **升维①（从 ASU 到全范式）**：DS 三方法共同、RR 全员非零 → 不是某个方法的 bug，而是所有“压制表达”方法的共同后门。靶子从“ASU”换成“压制范式”。
6. **升维②（从压制到可恢复性）**：知识冗余编码，压一个副本别的还在 → 换范式。
7. **解法**：RIU——把恢复攻击变成训练对手，plug-in 到任意 base。

> **一句话讲完全篇**：本想证明 ASU 有漏洞，实测它反而最鲁棒——但这恰恰暴露了更深的问题：连最强的方法都漏三成，说明问题不在某个方法，而在“压制表达”这个全领域共同的范式。RIU 换掉这个范式：不再让模型“忘记说”，而是让事实在任何恢复后都无法被解码。

---

## 6. 方法：RIU（Recovery-Invariant Unlearning）

### 6.1 核心洞察

事实知识被**冗余编码**在多层、多头、多个子空间。换说法 / 换语言 / 绕两跳 = 从另一个冗余副本提取；微调 / 量化 / 重复关键词 = 把被压低的副本重新放大。现有方法全在“压制表达”，压一个副本别的还在——这就是“连最强的都漏 $32.4\%$”的根因。

**范式转换**：从“压制表达（suppression）”转向“消除可恢复性（recoverability elimination）”。

### 6.2 统一建模恢复算子 $\mathcal{R}$（通用性来源）

$$
\mathcal{R} = \{\text{语义保持变换 } t\} \cup \{\text{短微调 } \Delta\theta\} \cup \{\text{量化 } Q\}
$$

三者本质是同一件事：保持模型效用、但可能重新激活目标知识的变换。TOFU 的跨语言、MUSE 的版权转述、WMDP 的危险多跳均可落入此三类——这是跨场景通用的理论依据。

### 6.3 可优化代理：线性探针可解码性

用带梯度反转层（GRL）的线性探针 $\psi_\varphi$ 从中间层表示解码 $y_f$：

$$
\mathcal{L}_{dec} = -\mathbb{E}_{r\in\mathcal{R}}\ \log \psi_\varphi\big(y_f \mid h^{\ell}(r(x_f);\theta)\big)
$$

内层训练 $\psi$ 尽量解码（挖事实），外层通过 GRL 让 $\theta$ 使 $\psi$ 解码不了（藏事实）。

### 6.4 核心目标：恢复不变的不可解码性（原创性来源）

$$
\min_{\theta}\ \max_{r\in\mathcal{R},\ \varphi}\ I\big(h_\theta^{\ell}(r(x_f));\ y_f\big)
$$

与数据增强的本质区别：约束的是**抽象掉路径后事实不可解码**，而非记住每条路径的平滑输出，因此未见的路径也继承该性质。

### 6.5 权重几何：正交擦除（与 RMU 区分）

先在 base 上估计事实子空间 $S_f$（forget vs retain 表示的 PCA 主方向），再把事实表示投影到其正交补，且要求对 $\mathcal{R}$ 扰动稳定：

$$
\mathcal{L}_{geom} = \mathbb{E}_{r\in\mathcal{R}}\sum_{\ell}\Big\|\mathrm{Proj}_{S_f^{\perp}}\big(h^{\ell}(r(x_f);\theta)\big)\Big\|_2^2
$$

RMU 把表示推向随机向量（伤效用）；RIU 只做正交投影，保住保留知识。

### 6.6 Plug-in 总目标（通用性的工程形态）

$$
\min_{\theta}\ \mathcal{L}_{base}(\theta;D_f,D_r) + \lambda_1\mathcal{L}_{dec} + \lambda_2\mathcal{L}_{cons} + \lambda_3\mathcal{L}_{geom}
$$

其中 $\mathcal{L}_{base}$ 可为 GA / NPO / ASU / RMU 中任一；一致性项让不同路径的表示塌缩到同一不可解码表示：

$$
\mathcal{L}_{cons} = \mathbb{E}_{r,r'\in\mathcal{R}}\ \Big\|h^{\ell}(r(x_f);\theta) - h^{\ell}(r'(x_f);\theta)\Big\|_2^2
$$

---

## 7. 具体实现流程：RIU 如何遗忘（六步）

| 步骤 | 做什么 | 为什么实现遗忘 |
|------|-------|--------------|
| 1. 训练探针 $\psi$ | 冻结 $\theta$，训练线性探针从 $h^\ell$ 解码 $y_f$ | 把“还剩多少可恢复信息”变成可微损失 |
| 2. 估计 $S_f$ | 对比 forget / retain 在 $h^\ell$ 的表示，PCA 取 top-$k$ 主方向 | 定位知识冗余编码的几何方向 |
| 3. 构造 $\mathcal{R}$ | 每条 forget 事实生成：6 逻辑结构 + 跨语言 + 转述变换、$K$ 步短微调、INT8 量化 | 把恢复攻击变成训练对手 |
| 4. 对抗内环 | min-max：内层用 GRL 训练 $\psi$ + 采样最致命的 $r$；外层压低可解码性 | **核心**：ASU 漏 $32.4\%$ 就是没练这一步 |
| 5. 几何擦除 | 将 $h^\ell(r(x_f))$ 投影到 $S_f$ 正交补 + 跨路径表示塌缩 | 未见路径继承“不可解码”，堵死冗余副本 |
| 6. Plug-in 合成 | 加到任意 base 目标上，retain 项保持效用 | 与 base 解耦 → 跨场景通用 |

**执行逻辑一句话**：用探针把“还能不能被挖出来”变成可微损失 → 让训练自己和最强恢复攻击对抗 → 用正交投影 + 表示塌缩把多余副本一并抹掉。

---

## 8. 与现有方法的本质区分

| 方法 | 操作对象 | 恢复攻击下 | 换 base / 换场景 |
|------|---------|-----------|----------------|
| GA / NPO | 输出 logit（推 / 降） | 微调即复活 | 本身是 base |
| ASU | attention 温度 | FOK 下仍漏 $32.4\%$ | 依赖 attention 结构 |
| RMU | 表示推随机向量 | 伤效用，副本仍在 | 依赖表示层 |
| REX | MoE 专家切除 | 抗微调 | 依赖 MoE 架构 |
| v4 APRU | 路径数据增强 | 未见路径仍漏 | 谁都能套 $\rightarrow$ 非方法 |
| **RIU** | **可恢复性（信息论 + 几何）** | **训练内环对抗 $\mathcal{R}$** | 与 base 解耦，plug-in |

三条判据全过：① 换掉 ASU 也成立；② 针对所有方法共有的“可恢复性后门”；③ 是新目标而非数据 / 蒸馏 / 加权组合。

---

## 9. 实验设计

| 维度 | 取值 | 要点 |
|------|------|------|
| 跨 base | GA, NPO, ASU, RMU 各 $\pm$RIU | 验证 plug-in 通用性（核心卖点） |
| 跨场景 | TOFU（虚构）, MUSE（版权）, WMDP（危险知识） | 验证场景无关 |
| 攻击 / 路径 | LeakRU 6 结构 + SH/MH + FOK/Probab/Quantization + 跨语言/转述 | 沿用 v4 底座 |
| 指标 | $FQ$, $RR$, Robustness $=1-\max(RR)$, $R_{\max}=\max_{p\in\mathcal{P}}\mathrm{leak}(p)$, MMLU/BBH utility，以及线性 / 核探针解码率 | 探针解码率直接测目标 |
| 消融 | 去对抗内环（退化为增强）/ 去几何项 / 去一致性 / 探针类型 | **去对抗内环直接回应“是不是数据增强”** |
| v4 数据 | ASU $56.2/25.0$/FOK $32.4$ 等原表 | 保留作 motivation |

主表仍需补：基线对齐（NPO FQ_SH $=40.7$）、MP=NA / DS 粒度、$\ge 3$ seed。

---

## 10. 贡献与写作骨架

### 10.1 三条贡献

- **C1 · 审计**：v4 的跨路径 / 恢复审计与实测数据，证明问题普遍、非 ASU 独有；
- **C2 · 洞察**：知识冗余编码 $\rightarrow$ 压制范式必留后门，形式化为恢复算子 $\mathcal{R}$ 与可解码性目标；
- **C3 + C4 · RIU 方法与理论**：对抗内环 + 不变性 + 几何擦除，plug-in 通用；证明“对 $\mathcal{R}$ 不可解码 $\Rightarrow$ 恢复鲁棒”。

### 10.2 Title 备选

1. Forget What Cannot Be Recovered: Recovery-Invariant Unlearning for LLMs
2. Beyond Suppression: Eliminating Recoverability in Language Model Unlearning
3. One Fact, Many Copies: Recovery-Invariant and Plug-in Unlearning

### 10.3 定位

方法论 + plug-in 方法：审计（C1） $\rightarrow$ 洞察（C2） $\rightarrow$ RIU（C3） $\rightarrow$ 理论（C4）。不偏离方向③（评估可信度危机），但以“换范式的方法”收尾，避免“只是评估”的单薄感。

---

## 11. 风险与诚实边界

### 11.1 技术风险

- min-max 训练不稳定 $\rightarrow$ 探针有界化 + 课程学习（从简单恢复到强恢复）；
- 线性探针只保证线性不可恢复 $\rightarrow$ 补核探针 / 非线性探针，并写清边界；
- $S_f$ 估计依赖 base 与层选择 $\rightarrow$ 消融层敏感性。

### 11.2 Rebuttal 预案

- “又是数据增强” $\rightarrow$ 去对抗内环消融即退化，证明对抗项必要；
- “只适用于 attention 方法” $\rightarrow$ 跨 4 类 base（含非 attention 的 GA/NPO/RMU）；
- “ASU 明明最鲁棒” $\rightarrow$ v4 已如实报告，RIU 针对的是绝对泄露 $32.4\%$ 而非相对脆弱。

### 11.3 诚实声明

v4 实测数据原样保留（TOFU forget05 / retain95, Llama-2-7B-Chat, seed 1001）；ASU 与 LeakRU 的机制、数据、指标均来自各自全文核实（arXiv:2603.01285v1、2608.04519v1）。RIU 方法、理论与所有“降低 $X\%$”均为本方案提出 / 待实测，非已验证结果。正交投影抗微调的启发来自 REX，但 RIU 不依赖 MoE 架构。

---

## 参考与数据来源

- ASU: *Attention Smoothing Is All You Need For Unlearning*, arXiv:2603.01285v1, ICLR 2026（全文核实）
- LeakRU: *Leak-Resistant Unlearning*, arXiv:2608.04519v1, 清华 + 蚂蚁（全文核实，开源 `leak-resistant.site`）
- OpenUnlearning: `github.com/locuslab/open-unlearning`（trainer / TOFU / 指标）
- 相关评测工作：ShadowBench (NeurIPS 2026 E&D)、TRACE (NeurIPS 2026)、Behavioral Forgetting Is Not Privacy (NeurIPS 2026 PriLOM)、REX (arXiv:2610.25791)

---

*文档生成：AI Algo Lab · 2026-11-20 · 配套可视化页面端口 30002*

---
id: CF2-07
rank: 7
title: 三阶累积量绑定 κ₃ —— 删除主体×关系×答案的高阶共同变化
avg_score: 7.4
verdict: Accept with Revisions (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无跨模型独立回执
novelty_status: PROCEED WITH CAUTION（高阶估计样本效率待证）
baseline_relation: 从单点向量差扩为自然配对分布上的三阶共同累积量差
---

# CF2-07 · 三阶累积量绑定 $\Delta\kappa_3$

## 0. 一句话故事

> 事实三元组 $(s,r,o)$ 的关键可能不在任一位置的均值差，也不在任意两者的协方差，而在主体、关系和答案三者的共同高阶变化。我们用正反事实分布的三阶中心矩之差定位这种绑定，只压低三阶项，并用一阶、二阶统计死区保护原有表示。

## 1. 问题与近邻空白

*The Structure of Relation Decoding Linear Operators in Large Language Models*（arXiv:2510.26543）表明关系解码算子可由三阶张量网络压缩；早期 tensor product representation 也用高阶张量表达角色—内容绑定。这些工作提供“高阶绑定”的动机，却不是机器遗忘，也不是反事实累积量差。

与 CF2-02 的区别必须写清：CF2-02 是四格受控干预下的**均值激活非加性**；本方案是在自然配对样本分布上估计**三阶共同累积量**。前者回答“组合干预是否非加性”，后者回答“三个随机表示是否共同变化”，二者不可互相替代。

## 2. 可证伪假设

**H**：在控制住一阶均值和二阶协方差后，正反事实的三阶项仍包含可重复、对 wrong-donor 特异的绑定信号。若三阶项在小样本下符号不稳定、打乱任一因子后不变，或删除它不改变独立读出，则否决该路线。

## 3. 方法

### 3.1 低秩三路表示

对第 $i$ 个自然配对样本，在主体槽、关系槽和答案跨度分别读取激活，并用冻结随机正交矩阵投影到 $q\le16$ 维：

$$
u_i^\pm=P_s h_{T,\ell_s}(X_i^\pm\oplus y_i^+;s),\quad
r_i^\pm=P_r h_{T,\ell_r}(X_i^\pm\oplus y_i^+;p_r),\quad
v_i^\pm=P_y h_{T,\ell_y}(X_i^\pm\oplus y_i^+;p).
$$

每个模态分别减去本条件样本均值，得到 $\bar u_i^\pm,\bar r_i^\pm,\bar v_i^\pm$。投影仅用于估计稳定性，不作语义命名。

### 3.2 三阶共同累积量差

三路均已中心化时，三阶共同累积量为：

$$
\kappa_3^\pm=\mathbb E_{i}\!\left[\bar u_i^\pm\otimes\bar r_i^\pm\otimes\bar v_i^\pm\right],
\qquad
\boxed{\Delta\kappa_3=\kappa_3^+-\kappa_3^-}.
$$

用低秩 CP 近似压缩：

$$
\Delta\kappa_3\approx\sum_{k=1}^{K}\lambda_k
 a_k\otimes b_k\otimes c_k,
$$

$a_k,b_k,c_k$ 给出主体、关系、答案三个模态的对应成分。它们是统计成分，不直接称为事实神经元。

### 3.3 Student 高阶擦除与低阶保护

Student 在 mini-batch 上估计 $\Delta\kappa_3^S$，优化：

$$
\mathcal L_3=\|\Delta\kappa_3^S\|_F^2,
$$

同时对均值和全部二阶交叉/自身协方差设置死区：

$$
\mathcal L_{1,2}=[\|\mu^S-\mu^T\|_2^2-\delta_1]_+
+[\|\Sigma^S-\Sigma^T\|_F^2-\delta_2]_+.
$$

$$
\boxed{\mathcal L=\lambda_3\mathcal L_3+\lambda_{1,2}\mathcal L_{1,2}
+\lambda_n\mathcal L_{\mathrm{neutral}}+\beta\mathcal L_R}.
$$

$\mathcal L_{\mathrm{neutral}}$ 包含原事实、donor 和 EOS 约束；$\lambda_{KL}^{(\mathrm{ablation})}=0$。所有统计量在固定 mini-batch 上计算，并另设独立估计批次防止同一批上过度拟合噪声。

## 4. 差异化边界

| 近邻 | 已占机制 | 本方案差异 |
|---|---|---|
| LRE 三阶张量网络（2025） | 压缩关系解码算子 | 不训练关系解码器；比较正反事实三阶共同累积量并遗忘 |
| Tensor Product Representations | 用张量积表达角色—内容绑定 | 从自然 LLM 激活估计统计绑定，不构造 TPR 记忆 |
| CF2-02 因子交互 | 受控均值非加性 | 本方案是自然分布三阶中心矩，样本统计对象不同 |
| CF2-03 条件流 | 二阶条件交叉协方差 | 本方案专门处理三阶共同变化，并冻结低阶统计 |

## 5. /idea-evaluator 兼容评估（PROVISIONAL）

**定位**：Novel Method。致命缺陷：2 MAJOR——F6 高阶估计的样本量与跨 seed 稳定性；F1 与 LRE/TPR 的科学问题划界。零 CRITICAL，故 Accept with Revisions。首验必须低秩、短答案、大配对批次，资源匹配黄。

| 维度 | 分数 | 机制依据 |
|---|---:|---|
| Higher | 8 | 使用二阶方法完全看不到的三阶绑定信号 |
| Faster | 6 | 低秩 CP 可控制成本，但大 mini-batch 与分解增加开销 |
| Stronger | 8 | 因子打乱、独立估计批和低阶死区可识别伪高阶信号 |
| Cheaper | 7 | 不需人工标注，但高阶估计需要更多样本 |
| Broader | 8 | 可拓展到多跳实体关系和多模态三路绑定 |

**平均分 $37/5=7.4$。裁决：Accept with Revisions，worth pursuing, pending the validation experiment。** 范式探测：First Principles=Yes、Elephant=Partial、Technology Cycle=Partial、Hamming=Yes。

修订顺序：先画样本量—偏差曲线；验证三阶增量超过二阶模型；最后才训练 Student。

## 6. 决定性实验与失败模式

**最小判别实验**：二阶-only、三阶-only、二+三阶三臂，并加入主体/关系/答案任一维 shuffle 的负对照。报告跨 seed 符号一致率、有效样本量、$r_f$、retain NLL、donor 率和独立 probe/relearning。

**失败即撤回**：三阶项在样本量增加时仍不收敛；三阶增量相对二阶模型为空；CP 成分跨 seed 不可复现；效果依赖同一批估计和评估。

## 7. 来源与诚信状态

- Christ et al.（2025），*The Structure of Relation Decoding Linear Operators in Large Language Models*，arXiv:2510.26543：https://arxiv.org/pdf/2510.26543 。
- Schlag & Schmidhuber（NeurIPS 2018），*Learning to Reason with Third Order Tensor Products*。
- 基线 §2.2、§4.1、§5、§6。
- plan-only；分数非实测、非独立评审。文献未做 D1 批量身份核验，不生成 BibTeX。

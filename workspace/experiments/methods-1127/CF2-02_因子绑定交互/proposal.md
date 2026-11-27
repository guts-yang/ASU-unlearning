---
id: CF2-02
rank: 2
title: 因子绑定交互 Δ²A —— 只擦除主体×关系的非加性绑定项
avg_score: 8.0
verdict: Strong Accept (PROVISIONAL, mechanism-based)
evidence_level: plan-only；无实测、无跨模型独立回执
novelty_status: PROCEED（执行者查新判断，非独立裁决）
baseline_relation: 从单次主体替换扩为 2×2 主体×关系因子设计
---

# CF2-02 · 因子绑定交互 $\Delta^2 A_{\mathrm{bind}}$

## 0. 一句话故事

> “换了主体”与“换了关系”都有主效应，但真正要删的是“这个主体与这个关系组合后才出现的事实绑定”。我们构造主体 $\{s_f,s_r\}$ ×关系 $\{r_f,r_0\}$ 的四格反事实，用双重差分 $\Delta^2A$ 同时扣掉主体主效应和关系主效应，Student 只处理剩下的非加性绑定项。

## 1. 问题与近邻空白

单对 $\Delta A=A^{s_f,r_f}-A^{s_r,r_f}$ 把“主体是谁”和“主体—关系绑定”混在一起。解决它不应只是换一个更复杂的投影器，而应改变实验设计，使不需要的项在代数上抵消。

*Required or Redundant? Tracing Evidence Collaboration with Factorized Causal Activation Patching* 使用 2×2 因子干预测量证据协作的非加性交互；项目外公开的 AARM 材料也讨论四结果交互代数。它们说明因子交互具有解释力，但目的都是机制分析，不是把绑定交互变成持久遗忘目标。本方案的新意若成立，只能落在“因子化绑定项 → Student 参数内化”，不能声称发明了 2×2 patching。

与旧 CF-09 的区别是：CF-09 从伪配对噪声基做回归残差化；本方案没有噪声基回归，使用正交因子对比消去主效应。与 CF-03 的区别是：不做逐组件 patch 权重。

## 2. 可证伪假设

**H**：存在关系匹配的控制 $r_0$，使四格激活满足近似可加主效应 + 小交互残差；真实 $\Delta^2A$ 能区分目标绑定与 wrong-relation / wrong-donor，而单纯两个 $\Delta A$ 不能。若控制关系本身引入明显 OOD，或 $\Delta^2A$ 与主体词形/答案长度高度相关，则否决该构造。

## 3. 方法

### 3.1 四格因子输入

令 $s_f$ 为遗忘主体、$s_r$ 为保留主体，$r_f$ 为目标关系、$r_0$ 为结构匹配但不含目标绑定的控制关系。四格输入共享模板、token 长度和槽外 ID：

$$
X(s,r),\qquad s\in\{s_f,s_r\},\quad r\in\{r_f,r_0\}.
$$

为隔离 prompt 干预，四路 Teacher 均在相同相对答案位置上 teacher-force 同一 $y^+$，读取：

$$
A^{s,r}_{\ell,p}=h_{T,\ell}(X(s,r)\oplus y^+;p),\qquad p\in S_f.
$$

控制关系必须通过预注册 OOD 门：若 $r_0$ 使共享 $y^+$ 的冻结模型 NLL 偏离目标条件超过 $\delta_{\mathrm{OOD}}$，则更换控制关系而不是把异常激活解释为绑定。另用各自自然答案做稳健性复核，防止单一共享答案带造成伪交互。

### 3.2 双重差分

定义：

$$
\boxed{
\Delta^2A_{\ell,p}
=(A^{s_f,r_f}_{\ell,p}-A^{s_r,r_f}_{\ell,p})
-(A^{s_f,r_0}_{\ell,p}-A^{s_r,r_0}_{\ell,p})
}
$$

若 $A^{s,r}=\mu+\alpha_s+\beta_r+\Gamma_{s,r}$，则：

$$
\Delta^2A=\Gamma_{s_f,r_f}-\Gamma_{s_r,r_f}-\Gamma_{s_f,r_0}+\Gamma_{s_r,r_0}.
$$

主体主效应 $\alpha$ 与关系主效应 $\beta$ 均被消去。它表示**非加性绑定对比**，不自动等价于完整因果事实子空间。

### 3.3 Student 内化

$$
T_{\ell,p}=\operatorname{stopgrad}\!\left[
\Pi_{\mathcal K}\!\left(A^{s_f,r_f}_{\ell,p}-\alpha\,\Delta^2A_{\ell,p}\right)
\right],
$$

$$
\mathcal L_F=\frac1{|\Lambda||S_f|}\sum_{\ell\in\Lambda}\sum_{p\in S_f}
\|h_{S,\ell}(X(s_f,r_f)\oplus y^+;p)-T_{\ell,p}\|_2^2,
\qquad
\boxed{\mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R}.
$$

$\mathcal K$ 与基线一致：$r_f\le c$、donor 支持度不升、EOS/长度漂移受限；读出经冻结下游。$\lambda_{KL}^{(\mathrm{ablation})}=0$，不拼 $y^-$，不把控制关系的输出当监督答案。

## 4. 差异化边界

| 近邻 | 重叠 | 本方案差异 |
|---|---|---|
| Factorized Causal Activation Patching | 2×2 因子干预与非加性交互 | 从“解释证据协作”改为“构造遗忘目标并训练内化” |
| CF-09 结构化对照族残差化 | 都使用额外对照 | 因子主效应由设计恒等式消去，不学噪声基、不做残差回归 |
| CF-04 事实/身份双探针 | 都想分开事实与身份 | 无事实/身份 probe，绑定由四格干预定义 |
| 基线单对 $\Delta A$ | 主体替换差 | 新增第二因子与双重差分 $\Delta^2A$ |

据当前检索，未找到把该双重差分用于 LLM 机器遗忘训练目标的直接论文；因子 patching 本身不是新组件。

## 5. /idea-evaluator 兼容评估（PROVISIONAL）

**定位**：Novel Method。致命缺陷：1 MAJOR（F6：控制关系和共享答案带的 OOD 门必须预注册）+ 1 MINOR（F1：因子 patching 近邻）；零 CRITICAL。四路短序列前向 + 闭式运算可纳入项目 TOFU-1B、单卡 H20 首验，能力匹配绿。

| 维度 | 分数 | 机制依据 |
|---|---:|---|
| Higher | 9 | 直接去除主体和关系主效应，目标比单对差更接近“绑定” |
| Faster | 7 | 前向数从 2 增至 4，但无迭代求解、无训练提取器 |
| Stronger | 8 | 控制关系、wrong-donor 和自然答案复核能暴露伪交互 |
| Cheaper | 8 | 无需人工特征标注、SAE 或 patch 扫描 |
| Broader | 8 | 可扩展到多关系、多跳事实及 LRM 步骤因子设计 |

**平均分 $40/5=8.0$。裁决：Strong Accept，worth pursuing, pending the validation experiment。** 范式探测：First Principles=Yes、Elephant=Yes、Technology Cycle=Partial、Hamming=Yes。

前三项行动：构建两个独立控制关系；冻结 OOD 与自然答案复核；比较 $\Delta^2A$、单对 $\Delta A$ 和 CF-09 式残差化。

## 6. 决定性实验与失败模式

**最小判别实验**：在两个控制关系上重复四格构造，检验 $\Delta^2A$ 的跨控制稳定性；随后比较“绑定目标 / 单对差目标 / 仅主体主效应目标 / 仅关系主效应目标”。报告目标事实读出、同实体非目标关系、retain NLL、donor 率、EOS、probe/relearning 恢复。

**失败即撤回**：控制条件 NLL 严重 OOD；不同 $r_0$ 给出互相矛盾的绑定方向；$\Delta^2A$ 主要编码词形或长度；消除交互项同时消除同实体全部关系能力。

## 7. 来源与诚信状态

- 基线：`方向I_基线_v0.6.1_answer_span.md` §2.1–2.3、§4.1、§6。
- *Required or Redundant? Tracing Evidence Collaboration with Factorized Causal Activation Patching*，检索页：https://lacuna.tiptreesystems.com/work/required-or-redundant-tracing-evidence-collaboration 。
- Zhang & Nanda（ICLR 2024），activation patching 的 OOD 与度量风险。
- 本方案为 plan-only；分数为机制推断，非独立评审或实测。文献身份未做 D1 批量核验，不生成 BibTeX。

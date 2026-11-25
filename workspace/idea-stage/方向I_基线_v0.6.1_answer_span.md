<!--
方向 I 基线（当前分支唯一权威版本）
来源：/Users/chenyliao/MuRAG/Papers/L7_ASU创新/I/方向I_事实配对路线_重新设计Proposal_20261121.md
版本：factpair-progressive-alignment-v0.6.1-answer-span（2026-11-24）
处置：按用户指令，用该源文件替换当前分支上两份过时的方向 I 基线副本：
  - workspace/proposal/…20261121.md（v0.5 旧版，KL 被判为 donor 答案替换错误链条）
  - workspace/docs/方向 I：…（修正版）.md（v0.6，但文件被截断在 §3 中途）
  两者已逐字节备份于 archive/baseline_versions/，不再作为基线使用。

2026-11-25 本地修订（仅修显性文本/口径 bug，不改算法设计与结论）：
  - 上游真源仍为上面的 L7 源文件（v0.6.1-answer-span，未改动）；本分支文件是其「修订版」，正文不再与源文件逐字一致。
  - 修订 1（§2.2）：Teacher 前向输入由「纯 X+ / X-」改为「teacher-force 拼入答案后的完整序列」，激活与 ΔA 记号统一为带答案跨度位置索引 p∈S_f 的形式。
    原写法与同段「P 指答案跨度 S_f」自相矛盾（纯问题输入不含答案 token），且与 §4.1 / §5「冻结 Teacher 在拼接序列上前向得到」冲突；系 v0.6.1 把 L_F 作用域改为答案跨度时 §2.2 未同步更新所致。
  - 核查结论（§2.3）：源文件「其中」段本就写作 $T_{\ell,p}=A^-_{T,\ell}(p)$（带位置索引），开头「旧版以 $T_\ell=A^-_{T,\ell}$」是对 v0.6 单抽点旧版的正确叙述，二者均自洽，**本次未改动**（上一轮一度误判为记号残留，核查后据实更正，不做无谓编辑）。
  - 新增标注（§2.2.1）：把 ΔA 构造未定稿及由此产生的方法身份张力显式标为「待裁决的开放问题」，未选定机制、未改故事。
  公式的数学含义、结论、状态声明、§6 验收口径与 §0/§7 故事均未改动；仅统一 §2.2 前向输入口径与记号，并新增开放问题标注小节。
-->

# 方向 I：反事实主体配对驱动的逐层渐进遗忘 Proposal（修正版）

> 协议：`factpair-progressive-alignment-v0.6.1-answer-span`；日期：2026-11-24。
> **状态：plan-only。尚未实现、预注册或训练，没有新的实验结果；激活差异处理是核心待调研问题，当前仅保留参考方案。**
> 本版相对 `v0.5-reference-draft`（2026-11-22）的修正集中在 §0、§1、§3、§4、§7：把原「donor 分布 KL」从遗忘目标降级为对照/消融臂，并补上去 τ 后的梯度来源与答案替换风险说明。v0.6.1 追加修正：**`\mathcal L_F` 的作用域从 prompt 末尾单抽点改为答案跨度（answer span）**，并把 §4.1 中性约束焊进 §2.3 的统一公式。v0.5 与 v0.6 修改前逐字节备份于 `archive/2026-11-24_proposal_v0.5_backup/`。

## 0. 一句话故事

**修正后，Teacher 不再通过注意力温度制造遗忘目标，也不是把 Student 推向 donor 的答案；它只用来提供一条「另一主体」的对照表示轨迹。真正的遗忘方向由 Student 侧的显式激活损失给出：在答案跨度上，用原事实读不出、且不借助 donor 答案、拒答或提前停答的中性约束，逐层、逐位置地把目标事实绑定从参数里移除，最终内化到 Student 参数中。激活差异的具体处理仍是核心待调研问题。**

我们要验证的不是「输出是否变了」，而是：这种变化是否真正移除了目标事实，同时没有变成 donor 答案替换、普遍拒答、回答缩短或仍可恢复的浅层抑制。

---

## 1. 为什么这样改（含旧版错误链条）

原 ASU 用带注意力温度的 Teacher 产生 forget 分布，再用 KL 训练 Student。去掉温度后，若 Teacher 与初始 Student 相同，原 forget KL 在初始化处没有遗忘梯度。历史证据调研（`archive/方向I_反事实主体替换SubjectLast对齐去τ_证据调研.md`，截至 2026-11-16）已判定：反事实主体对齐可能只是答案替换或一般扰动，去 τ 后 forget 信号只能来自显式 loss；冻结 Teacher 的激活没有梯度路径可供「靠齐」。

**旧版 v0.5 的错误链条（本版修正的对象）：** 为了补回梯度，旧版把 Teacher 改成「吃反事实输入 $X^-$」、让 $p_T(y|x)=p_{\theta_0}(y|X^-)$ 与 Student 在 $X^+$ 上的分布做 DKT 式 KL。形式上有梯度了，但那是**用引入 donor 答案监督换来的梯度**：Teacher-prefix 分支采样的 $y^T$ 是 donor 对自己问题的答案（Wellington / Seattle …），却拿来当 $X^+$ 的 soft target；$\mathcal L_{KL}$ 与 $\mathcal L_F$（向 $A^-$ 对齐）又都指向同一 donor，二者共谋把「忘」拧成「换」。forward KL 是 mode-covering，只逼 $q$ 铺满 donor 完整答案分布，既不奖励「原事实读不出」，也不惩罚「原答案仍可读」。因此旧版 §3 的 $\mathcal L_{KL}$ 是**迁移/替换目标，不是删除目标**，与 §6「不能把改答 donor 当遗忘」正面冲突。fuxian 复现（用户整理、未独立复算）里提前 EOS 从 $0.02$ 涨到 $0.46$、完整回答率 $0.98\to0.54$、G4 FAIL，也提示该组合容易走输出抑制捷径。

因此，本方案做三项改动：

1. **输入构造只作对照，不作目标语义**：仅把遗忘主体 $s_f$ 换成关系匹配的保留主体 $s_r$ 得到 $X^-$，冻结 Teacher 用它提供「另一主体」的对照表示/输出轨迹；**不**把 $p_{\theta_0}(y|X^-)$ 当作 forget 目标。
2. **遗忘方向由显式激活损失给出**：Student 侧用**答案跨度上的逐层激活损失** $\mathcal L_F$（配合「原事实读不出」的中性约束）提供遗忘梯度；作用域必须是答案跨度而非 prompt 末尾——事实是在生成答案时展开的，prompt 末 token 上对齐既看不到答案实现，也无法阻止事实从答案阶段重新浮现。donor 相关信息只用于定位/对照，不写入目标答案。
3. **DKT 式 KL 降为对照/消融臂**：保留其同前缀数学口径，用于证伪「答案替换 / donor 共谋」与做分布错配诊断，但**不进入主损失**；主结果不得依赖 $\mathcal L_{KL}$ 成立。

历史 pilot 的 confirmation 未通过，因此旧层集合和旧结果不能作为本方案的正向证据。

---

## 2. 方法主线

### 2.1 构造反事实输入

对同一关系 $r$：

$$
X^+=X(s_f,r),\qquad X^-=X(s_r,r)
$$

两条输入除主体槽外逐字一致，token 等长，槽外 token ID 不变。$X^+$ 来自 forget 集，$X^-$ 的 donor 来自 TOFU 内部保留集。

forget、donor、retain-train 和 evaluation 按作者/别名簇隔离。TOFU 只提供「数据集内真值」，不声称真实世界事实。

> **注意（v0.6）：** $X^-$ 仅用于提供对照轨迹与定位信号。旧版 v0.3 的 10 对缓存 `q_plus/q_minus` 长度不等且为外部真实人物 donor，不满足本节严格等长反事实配对，只能作激活可行性排错，不作确认实验。

### 2.2 核心参考设计：Teacher 提供逐层对照激活

> **核心但待调研：** 如何利用 $A^+$、$A^-$ 及其差异构造遗忘目标，目前尚未定稿。本节保留「直接向 $A^-$ 对齐」作为**待证伪的参考实现**，不代表已选定最终机制，更不代表 $A^-$ 已是纯净的删除目标。

Teacher 是冻结基座 $\theta_0$，Student 参数记为 $\theta$，并从同一 checkpoint 初始化。二者均使用标准 Attention，不引入额外注意力温度。

Teacher 分别前向**已 teacher-force 拼入目标答案的完整序列** $X^+\oplus y^+$ 与 $X^-\oplus y^+$（后者为定位对照轨迹而构造；$y^-$ / `donor_fact_value` 一律不拼入，见 §5 donor 数据边界）。在抽点 $P$ 读取第 $\ell$ 层激活：

$$
A^+_{T,\ell}(p)=h_{T,\ell}(X^+\oplus y^+;p),\qquad
A^-_{T,\ell}(p)=h_{T,\ell}(X^-\oplus y^+;p),\qquad p\in S_f
$$

$P$ 首选每层 DecoderLayer 输出；残差相加位置作为位置消融。**抽点位置的定义随作用域而变：本版 $P$ 指答案跨度位置集 $S_f$ 上的抽点，而非 prompt 末尾 token**；$S_f$ 由 tokenizer 对 $y^+$ 的 offset mapping 确定（含答案首 token 与其 causal prefix 对齐），在 discovery 冻结。prompt 末位可作旧版对照，不得作为主结果的 $L_F$ 作用域。

> **v0.6.1 补丁口径（本次修订）**：原 §2.2 写作 $A^\pm_{T,\ell}=h_{T,\ell}(X^{+/-};P)$，但纯问题输入 $X^+/X^-$ 不含答案 token、不存在答案跨度位置，与同段「$P$ 指 $S_f$」及 §4.1「由冻结 Teacher 在 $X^+\oplus y^+ / X^-\oplus y^+$ 上前向得到」自相矛盾。该写法是 v0.6.1 把 $\mathcal L_F$ 作用域改为答案跨度时 §2.2 未同步更新的补丁残留，本版按 §4.1 与 §5 的统一口径修正；$A^-_{T,\ell}(p)$ 的构造与语义仍属待证伪对照，不预设其已为纯净删除目标。

配对差值在答案跨度位置上逐点定义：

$$
\Delta A_{\ell,p}=A^+_{T,\ell}(p)-A^-_{T,\ell}(p),\qquad p\in S_f
$$

只表示「主体替换引起的表示差异」。后续需要调研应直接对齐 $A^-$、利用 $\Delta A_{\ell,p}$ 引导更新，还是采用其他目标构造；当前不预设 $\Delta A_{\ell,p}$ 已是纯净的事实差异。

> **梯度边界（v0.6）：** 冻结 Teacher 的 $A^+_{T,\ell}(p)$、$A^-_{T,\ell}(p)$ 及离散输出 token 一律 stop-gradient，**没有梯度路径供 Teacher「靠齐」**。所有遗忘梯度只能经 Student 侧 $\mathcal L_F$ 回传。任何「冻结 Teacher 向某目标移动」的表述均为逻辑错误。

### 2.2.1 ΔA 构造：标注为未定稿（待裁决的开放问题，非已解决）

> **本小节为状态标注，不新增机制、不选定方案、不改 §0/§7 故事。** ΔA 在本文中只作为「主体替换引起的表示差异」的记号，不进主损失、不作为已确认的遗忘方向。

1. **未定稿项**：§2.2 自述「如何利用 $A^+$、$A^-$ 及其差异构造遗忘目标，目前尚未定稿」。当前 $\Delta A_{\ell,p}=A^+_{T,\ell}(p)-A^-_{T,\ell}(p)$ 仅为**原始配对差**，未净化（混有实体身份、subword、词频、关系传播），未证明其编码目标事实而非主体身份；历史证据调研（`archive/方向I_反事实主体替换SubjectLast对齐去τ_证据调研.md` §4.2/§5.2、RQ2）明确判定 $A^+-A^-$ 不能直接读成事实差异。
2. **构造方式待选**：直接对齐 $A^-$、多 donor 边际化 / SVD 子空间、用 $\Delta A$ 作方向引导、因果门控纯化、或改用非 donor 的中性目标（§4.1）——五条路径尚未选定；L7《激活目标 12 项审核》与《激活损失模块》报告均把 $T_\ell$ 构造列为「参考设计、尚未定稿、待用户审核」，本版沿用，不自行拍板。
3. **方法身份张力（需用户裁决）**：§1 已把 donor KL 降为消融臂（默认 $\lambda_{KL}=0$）、§2.3 把 $A^-$ 降为待证伪对照，主损失 $\mathcal L=\lambda_F\mathcal L_F+\beta\mathcal L_R$（§4.2）中 $\mathcal L_F$ 的定稿目标 $T_{\ell,p}$ 是「满足 §4.1 中性约束且离 $A^+_{T,\ell}(p)$ 最近」的落点——该目标只需 $X^+\oplus y^+$ 前向即可定义，**不依赖 $X^-/A^-/\Delta A$ 进入主梯度路径**。于是「反事实主体配对」目前仅作 §3 消融臂与 §6 诊断对照，而非产生遗忘梯度的主机制；方法实际退化为「答案跨度上的最小中性扰动 / 概念擦除」家族。这是把 donor 共谋从主损失移除后未同步重定贡献的遗留问题，须由用户选择：
   - (A) 保留配对为**主梯度来源**：则必须重新设计 $T_{\ell,p}$ 使其显式依赖 $\Delta A$ 或 $A^-$，并重新论证 §1 为何又把 donor 信息排除出目标；
   - (B) 承认贡献改为**证伪 / 验证协议**：则 §0、§7 故事与「反事实主体配对驱动」标题需整体重写，配对降为诊断工具。
   本标注不执行 (A) 或 (B)，仅登记为开放问题；在裁决前，不得把「配对特异性」（§6.1）当作方法已生效的证据。
4. **相关待冻结量**：$S_f$ 的 offset 定义、$\Delta A$ 净化方式（若采用）、以及 §2.3 的 $\Lambda(k)$/$\rho_\ell(k)$ 与 §4.1 的 $c,d_j,e$ 阈值，均属 discovery 冻结项，本版未冻结。

### 2.3 Student 逐层渐进对齐（承载遗忘方向的主部件）

Student 接收原始遗忘输入 $X^+$，并把目标答案 $y^+$ teacher-force 拼入得到完整序列 $X^+\oplus y^+$，使答案跨度位置集 $S_f$ 在 forward 中真实出现。令第 $k$ 个训练阶段参与对齐的层集合为 $\Lambda(k)$，层权重为 $\rho_\ell(k)$。设 $T_{\ell,p}$ 为**非 donor 的遗忘目标**（见 §4.1 的中性约束；旧版以 $T_\ell=A^-_{T,\ell}$ 作参考实现，本版仅保留为待证伪对照）。**$\mathcal L_F$ 只作用于答案跨度 $S_f$ 上的抽点，prompt 位置与 $p\notin S_f$ 不进入 $\mathcal L_F$**：

$$
\mathcal L_F(k)=
\frac{1}{|\Lambda(k)|\,|S_f|}
\sum_{\ell\in\Lambda(k)}\sum_{p\in S_f}
\rho_\ell(k)\,
\operatorname{dist}\!\left(
h_{S,\ell}(X^+\oplus y^+;p),
\operatorname{stopgrad}(T_{\ell,p})
\right)
$$

其中 $T_{\ell,p}$ 取满足 §4.1 中性约束（$r_{f,p}\le c$、$s_{d_j,p}\le d_j$、$\|\Delta r_{E,p}\|\le e$）且离 $A^+_{T,\ell}(p)$ 最近的落点；实现可为集合式可行边界，不必是单点。作为**待证伪对照**，可令 $T_{\ell,p}=A^-_{T,\ell}(p)$ 或改用点回归；主结果必须在不使用 donor 目标、且中性约束真正生效的形式下成立。读出 $r_{f,p},s_{d_j,p}$ 由冻结 Teacher 在答案跨度上校准；对 Student 激活施加时须经同一冻结下游重算（N1 ReadoutTarget 式 patch-forward），不得把中间层向量直接过 LM head 当作真实读出。

训练时逐步扩展 $\Lambda(k)$ 或调整 $\rho_\ell(k)$，而不是一次性替换最终激活；$S_f$ 固定不参与 schedule。主训练直接优化 Student 参数；运行时 patch 只作对照。

「渐进」必须相对等预算的静态全层、反向和随机 schedule 证明独立收益，否则只保留「多层对齐」表述。

---

## 3. 长序列 KL：保留 DKT 同前缀形式，但降为对照/消融臂

> **v0.6 定位变更：** 本节全部公式在数学上仍然自洽、实现口径仍然正确，但 $\mathcal L_{KL}$ **不再是遗忘目标、不进入主损失**。它只用作三类用途：(a) 证伪「$\mathcal L_F$ 与 donor 分布共谋」的诊断量；(b) 分布错配（on-policy / off-policy）对照；(c) 与中性目标方案做消融。主结果必须在不依赖 $\mathcal L_{KL}$ 的前提下成立。

### 3.1 反事实 Teacher 策略（仅用于对照臂）

令外部输入 $x=X^+$，输入变换 $\phi(x)=X^-$。定义：

$$
p_T(y\mid x):=p_{\theta_0}(y\mid\phi(x))=p_{\theta_0}(y\mid X^-)
$$

$$
q_\theta(y\mid x):=p_\theta(y\mid X^+)
$$

这里 $p_T$ 是**反事实对照策略**，$q_\theta$ 是当前 Student 策略。第 $t$ 步生成状态为：

$$
s_t=(x,y_{<t})
$$

对任一固定输出前缀 $y_{<t}$，实际前向是：

$$
p_T(a\mid s_t)=p_{\theta_0}(a\mid X^-,y_{<t}),\qquad
q_\theta(a\mid s_t)=p_\theta(a\mid X^+,y_{<t})
$$

其中 $a$ 是下一 token。**每个 KL 内的两个分布必须共享同一个 $y_{<t}$。** 不能直接比较 Teacher 在 $y^T_{<t}$ 下的分布与 Student 在另一前缀 $y^S_{<t}$ 下的分布。

> **语义警示（v0.6）：** $p_T(\cdot\mid s_t)$ 是 donor 主体对自己问题的答案分布。把它作为 $X^+$ 的 soft target 等于 donor 答案替换；forward KL($p_T\|q$) 是 mode-covering，只要求 $q$ 覆盖 donor 答案，既不奖励原事实读不出、也不惩罚原答案仍可读。因此该 KL 不能作删除证据，只能作替换/共谋的对照量。

### 3.2 前缀来源与 KL 方向

分别采样：

$$
y^T\sim p_T(\cdot\mid x),\qquad y^S\sim q_\theta(\cdot\mid x)
$$

由它们诱导的第 $t$ 步前缀分布记为 $d_T^t$ 和 $d_\theta^t$。DKT 的序列 forward KL 分解为：

$$
\operatorname{KL}(p_T\parallel q_\theta)(x)
=
\sum_{t=1}^{L_y}
\mathbb E_{s_t\sim d_T^t}
\left[
\operatorname{KL}\!\left(p_T(\cdot\mid s_t)\parallel q_\theta(\cdot\mid s_t)\right)
\right]
$$

其中 $L_y=|y|$。DKT 的核心是：**前缀来源决定在哪里计算损失，KL 方向决定如何比较同一状态下的两个分布。** reverse KL 使用 Student 诱导的前缀并反转散度方向，但不进入首版对照臂。

### 3.3 序列损失（对照臂专用）

采用两种前缀来源上的 forward KL：

- Teacher-prefix：覆盖反事实（donor）目标轨迹，属于 off-policy soft-label SFT；
- Student-prefix：覆盖 Student 实际访问的状态，属于 DAgger-style on-policy SFT。

令 $z\in\{T,S\}$ 表示前缀来源，$s^z_{b,t}=(x_b,y^z_{b,<t})$，$L_{z,b}$ 为对应输出的有效长度，$m^z_{b,t}$ 为有效位置 mask。统一定义：

$$
\mathcal L_z
=
\frac1B\sum_b
\frac{
\sum_{t=1}^{L_{z,b}}m^z_{b,t}\,
\operatorname{KL}\!\left(
p_T(\cdot\mid s^z_{b,t})
\parallel
q_\theta(\cdot\mid s^z_{b,t})
\right)
}{
\max\!\left(1,\sum_t m^z_{b,t}\right)
}
$$

然后混合两条分支（仅用于对照臂）：

$$
\mathcal L_{KL}
=
(1-\eta)\mathcal L_T
+\eta\mathcal L_S
$$

其中 $\eta\in[0,1]$ 在训练前冻结。两条分支分别计算和归一化，**不要求 $y^T$ 与 $y^S$ 等长，也不逐 token 比较两条不同序列。**

### 3.4 梯度（对照臂实现口径）

固定状态 $s_t$ 并冻结 Teacher 时：

$$
\nabla_\theta
\operatorname{KL}\!\left(p_T(\cdot\mid s_t)\parallel q_\theta(\cdot\mid s_t)\right)
=
-\mathbb E_{a\sim p_T(\cdot\mid s_t)}
\left[\nabla_\theta\log q_\theta(a\mid s_t)\right]
$$

因此对照臂等价于 Teacher soft target 的交叉熵。Teacher 分布、Teacher 激活和离散输出 token 均 stop-gradient；梯度只经 Student 隐状态和 logits 更新 Student。**在对照臂里这恰恰是要被证伪的「答案替换梯度」，不是遗忘梯度。**

### 3.5 强制三臂对照（v0.6 新增）

为拆开「$\mathcal L_F$ 与 donor 分布共谋」，每个进入 pilot 的候选都必须跑：

- $\mathcal L_F$-only（主结果依赖它成立）；
- $\mathcal L_{KL}$-only（donor 替换基线，预期出现答案替换）；
- 组合 $\mathcal L_F+\mathcal L_{KL}$（检验二者是否共谋）。

并联合报告 donor 答案率、$\nabla\mathcal L_F$ 与 donor 提升方向 $g_d=\nabla_\theta\log p_\theta(y_d\mid X^+)$ 的余弦、原事实读出、提前 EOS 与完整回答率。若组合臂 donor 上升而 $\mathcal L_F$-only 不上升，即坐实共谋，$\mathcal L_{KL}$ 必须保持为对照项，不得回到主损失。

---

## 4. 总损失与遗忘目标（v0.6 修正）

### 4.1 遗忘目标：非 donor 的中性约束

设 $r_{f,p}(h)$ 为在**答案跨度位置 $p\in S_f$** 上、冻结读出对目标答案 $y^+$ 的长度归一化 span log-likelihood（越大表示仍支持原事实），$c$ 为用一组已知中性状态（拒答 / EOS / 低置信 / 错答）校准的死区上界，冻结于 discovery。中性约束要求 Student 在**答案跨度上**的激活落进「原事实读不出」的区域，且**不**指定到 donor 或任何具体替代答案。它与 §2.3 的 $\mathcal L_F$ 是同一个作用域（$p\in S_f$），不再是在 prompt 末位上对一个向量做读出：

$$
r_{f,p}\big(h_{S,\ell}(p)\big)\le c,\qquad
s_{d_j,p}\big(h_{S,\ell}(p)\big)\le d_j\ (\forall j),\qquad
\|\Delta r_{E,p}\|\le e,\qquad p\in S_f
$$

其中 $s_{d_j,p}$ 为第 $j$ 个 donor 答案在跨度位置的读出支持度（只用于「不升」约束，不作为目标）、$\Delta r_{E,p}$ 为 EOS/长度漂移。$A^+_{T,\ell}(p)$、$A^-_{T,\ell}(p)$ 同样取自答案跨度位置，由冻结 Teacher 在 $X^+\oplus y^+$ / $X^-\oplus y^+$ 上前向得到。旧版 prompt 末位缓存（如 `[33,4096]` 的 `a_plus/a_minus`，实为 `[/INST]` 尾部 token）不满足本作用域，只能作历史排错，不能当确认数据。

### 4.2 主损失

$$
\boxed{
\mathcal L=
\lambda_F\mathcal L_F
+\beta\mathcal L_R
}
$$

- $\mathcal L_F$：§2.3 的**答案跨度上的逐层激活遗忘损失**，是唯一承载遗忘方向的主项；§4.1 的中性约束已焊入其目标 $T_{\ell,p}$ 定义（不是可选插件）。实现时可用单边 hinge / 可行集投影 / 反 EOS 捷径等具体形式（见候选报告，实验前冻结），但必须保持 $p\in S_f$ 作用域与「非 donor 目标」两条硬约束。
- $\mathcal L_R$：独立 retain-train 上的答案位 NLL，用于保留非目标知识与通用能力。
- $\mathcal L_{KL}$（§3）：**不进入主损失**，仅作对照/消融臂，系数记为 $\lambda_{KL}^{\text{(ablation)}}$，默认 $0$；任何启用它的实验都必须是标注清楚的证伪/诊断实验。

所有系数、schedule、生成配置、中性约束阈值 $c,d_j,e$ 和训练预算在实验前冻结。

### 4.3 部署

部署主结果只保留 Student，不依赖 Teacher、donor、$\mathcal L_{KL}$ 或运行时 hook。主结果必须在 $\lambda_{KL}^{\text{(ablation)}}=0$ 下复现。

---

## 5. 边界情况与实现检查

| 项目 | 明确口径 |
|---|---|
| 首 token | $s_1=(x,\varnothing)$；对照臂中 Teacher 使用 $X^-$、Student 使用 $X^+$，所以从 $t=1$ 起就有反事实（donor）监督；主损失不依赖此项 |
| 长度不同 | 两条前缀分支独立处理，不做共同长度截断；另行报告长度差和提前结束率 |
| EOS | EOS 是最后一个有效目标；EOS 后不计算损失；提前 EOS 上升判为 G4 FAIL，不算遗忘 |
| Padding / mask | padding 的 attention mask、label 和 KL mask 均无效；分母只包含有效位置 |
| Causal shift | 若 prompt 长度为 $L_x$，预测 $y_t$ 应读取位置 $L_x+t-1$ 的 logits，即 $x$ 末尾预测 $y_1$、$y_{t-1}$ 位置预测 $y_t$ |
| **$L_F$ 作用域（v0.6.1 裁决）** | **仅答案跨度 $S_f$ 上的层×位置抽点；必须在 $X^+\oplus y^+$ 上 teacher-force 前向产生答案位置。prompt 末位（含 `[/INST]` 尾 token）不进 $L_F$；旧 `[33,4096]` 缓存仅排错用。$S_f$ 由 tokenizer offset 确定并冻结，不随 schedule 变** |
| Tokenizer | Teacher 与 Student 必须共享 tokenizer、词表顺序和 chat template |
| 显存 | 不长期保存完整 `[B,T,V]` 分布；前向后立即计算 KL，只保留标量轨迹和少量调试分布 |
| 梯度边界 | 冻结 Teacher 无梯度路径；所有目标/激活/采样 token 一律 stop-gradient；遗忘梯度只经 Student 侧 $\mathcal L_F$ 回传 |
| donor 数据边界 | $y^-$ / `donor_fact_value` 不拼入 prompt 或 loss；不算 $\operatorname{logit}(y^+)-\operatorname{logit}(y^-)$ |

---

## 6. 怎样判断方法真的成立

1. **配对特异性**：真实 donor 必须优于噪声、乱序 donor、wrong-donor、matched-retain 和随机层/位置对照；
2. **渐进收益**：渐进 schedule 必须在等预算下优于静态全层对齐；
3. **不是答案替换**：同时报告原目标事实、donor 答案、其他错答、拒答、长度和 EOS，不能把改答 donor 当作遗忘；组合臂 donor 上升即判共谋；
4. **参数内化**：无 hook 的 Student 仍需保持效应，且 retain/shared facts 与通用能力合格；
5. **不是浅层抑制**：结合输出访问变体、独立 probe/patching、量化和受控 relearning。至少两类独立抗恢复证据改善，才允许使用受限的「深度遗忘」表述；
6. **主结果独立性（v0.6 新增）**：主结论必须在 $\lambda_{KL}^{\text{(ablation)}}=0$ 下成立；不得把依赖 donor KL 的结果写成方法有效。

统计单位是作者/实体簇；多 prompt、多 donor 和多 seed 只作为簇内重复测量。旧 pilot 不复用为 confirmation。

---

## 7. 30 秒讲稿

> 我们把 ASU 的「注意力温度 Teacher」改成「反事实对照 Teacher」：只把遗忘主体替换成关系匹配的保留主体，Teacher 由此提供另一条逐层表示和输出轨迹，但**不**把它当作要逼近的答案。去 τ 后原 forget KL 没有遗忘梯度，我们不再用 donor 答案监督去补这个梯度，而是由 Student 侧**答案跨度上的**逐层激活损失 $\mathcal L_F$ 配一个「原事实读不出」的中性约束来给出遗忘方向（不再盯 prompt 末尾那个 token），再用 $\mathcal L_R$ 保住保留知识。DKT 式同前缀 KL 保留为对照/消融臂，专门用来证伪 donor 答案替换和 $\mathcal L_F$ 与 KL 的共谋，主结果不依赖它。激活差异究竟应直接对齐、差值引导还是采用其他中性构造，仍是当前核心待调研问题。训练只更新 Student，部署不需要 hook。

## 参考

- DKT：Zhao et al., [Decoupling KL and Trajectories](papers/【DKT】Decoupling%20KL%20and%20Trajectories-A%20Unified%20Perspective%20for%20SFT,%20DAgger,%20Offline%20RL,%20and%20OPD%20in%20LLM%20Distillation.pdf), 2026，采用其状态、前缀分布、序列 KL 分解和梯度定义（本版仅作对照臂口径）。
- [方向 I 历史证据调研](archive/方向I_反事实主体替换SubjectLast对齐去τ_证据调研.md)：判定去 τ 后 forget 信号只能来自显式 loss、反事实对齐不等于擦除。
- [v0.5 原始备份与修正说明](archive/2026-11-24_proposal_v0.5_backup/README.md)。
- [stage0 数据分区](data/stage0/summary.md)。

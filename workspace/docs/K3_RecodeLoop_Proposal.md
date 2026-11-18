# K3 Proposal：RecodeLoop——恢复引导的非线性重编码闭环遗忘

> **版本：** 2026-11-18
> **状态：** 研究 proposal；未运行实验，不构成 finding、预注册完成或安全批准
> **M0 对象：** P08-M2
> **数据与模型：** 以 TOFU 为主；使用公开 base、target 与相关 checkpoint
> **资源口径：** 2×A800 80GB；6 GPU 小时为软目标；无硬性总量上限
> **负责人：** chenyliao；任一 kill-switch 命中触发复审

## 摘要

[idea] 本方向研究一个恢复引导的审计—定位—修正闭环：在每次局部遗忘干预后，主动训练非线性恢复器寻找最终模型中仍可被利用的目标事实残余；通过因果激活替换定位残余来源；在严格预算配准下实施局部低秩修正；最后由未参与修正的线性、非线性、因果、量化与重学习五类独立审计臂评估恢复优势。

[idea] 与 ASU 的全局注意力温度不同，本方向不把输出沉默当作遗忘的充分条件，也不把因果定位当作表示擦除的替代证据。核心问题是：**在事前声明的读取器族与威胁模型内，目标事实是否仍相对于 never-known 对照表现出恢复优势，以及残余是否会经非线性重编码或替代路径持续迁移？**

[idea] 本方向只主张读取器相对的 L2 访问阻断，不主张 L3 表示不可逆擦除。由于 base、target、旧 checkpoint、权重差分、优化器状态、日志、缓存激活和恢复器均可得，直接读取历史模型获得事实不属于“从最终 student 内部恢复”，但会使任何系统级不可访问主张无效。

---

## 1. 一句话故事

[idea] **ASU 让一个平滑 teacher 指导 student 不再说出事实；RecodeLoop 则持续攻击最终 student，寻找事实还能被恢复的方式，把残余发现转化为可审计的局部干预，并用独立读取器检验事实是否只是换了一种编码。**

[idea] 研究贡献不预设闭环必然收敛：闭环收敛与残余持续迁移分别提供关于可解释遗忘边界的正反两类证据。

---

## 2. 问题来源：ASU 证据链中缺失的一环

[located evidence] ASU 在冻结 teacher 的所有层、所有 head 的 attention logits 中施加温度 $\tau$：

$$
\operatorname{Attention}(Q,K,V;\tau)
=\operatorname{Softmax}\!\left(\frac{QK^\top}{\tau\sqrt{d_k}}\right)V,
\qquad \tau\ge 1.
$$

[located evidence] student 从原始基座出发，通过 forget 集上的 teacher→student KL 学习平滑 teacher 的输出分布；原论文理论主要支持 teacher 侧 factual-token logit margin 下降。

[idea] 从“teacher margin 下降”到“最终 student 不含可恢复事实”之间仍缺少独立证据。至少存在四类替代解释：

1. **行为抑制：**自由生成不再出现目标事实，但内部表示仍可恢复。
2. **读取器隐藏：**线性 probe 失效，非线性读取器仍能重建目标事实。
3. **路径迁移：**原路径被压制后，其他 head、MLP 或残差路径提供等价信息。
4. **数值或优化脆弱性：**量化或有限重学习重新暴露目标事实。

[idea] 因此，本方向把“是否遗忘”拆成两个必须分别成立的问题：

- **遗忘问题：**声明读取器下是否仍有恢复优势？
- **机制问题：**哪一个组件、方向或路径的改变造成了观察到的恢复变化？

---

## 3. 遗忘判据：L1 / L2 / L3

| 层级 | 操作定义 | 本方向的使用方式 |
|---|---|---|
| L1 行为沉默 | 自由生成不再吐出目标事实 | 必要但不足；不能单独支持遗忘机制 |
| L2 读取器相对的访问阻断 | 对事前声明的读取器族、攻击者产物和协议预算，未发现相对 never-known 对照的恢复优势 | 本方向唯一目标判据 |
| L3 表示擦除/不可逆 | 权重与激活中不再含任何可重建目标事实的信息 | 开放参数与公开历史产物下不作为可证明目标 |

[decision] 人类已确认将 focal question 的“不可恢复”锁定为 L2。

[idea] 对读取器 $r$ 和模型 $M$，令 $A_r(M;F)$ 表示在事实集 $F$ 上恢复**具体事实值**的度量，而不是区分 forget/retain 的集合成员准确率。对同一最终模型在 matched never-known 事实集 $F_{\mathrm{NK}}$ 上建立参照，并报告：

$$
\Delta_r
=A_r(M_{\mathrm{student}};F_{\mathrm{forget}})
-A_r(M_{\mathrm{student}};F_{\mathrm{NK}}).
$$

[idea] 不同读取器的量纲不混合为单一总分。线性/非线性 probe、causal patch、量化和重学习分别报告原始恢复结果、对照差与区间。对可表示为恢复优势 $\Delta_r$ 的读取器，应事前给定容忍界限 $\epsilon_r>0$，以区间上界 $\operatorname{UCB}(\Delta_r)<\epsilon_r$ 支持读取器相对的等效性；不得以 $p>0.05$ 或“不显著”替代“未发现恢复优势”。只有全部事前声明的审计臂均满足各自冻结判据，才支持当前设置下的 L2 主张。

---

## 4. 核心机制：恢复引导的闭环

设第 $t$ 轮当前模型为 $M_{t-1}$。

### 4.1 残余发现

[idea] 冻结 $M_{t-1}$，仅在 forget-train 的指定实体上训练小型非线性恢复器 $R_t^{\mathrm{edit}}$，要求其从激活中解码**目标事实值**（如答案实体、属性值或候选答案排序），而非仅分类样本来自 forget 还是 retain/never-known。这样才能寻找线性修正后仍可读取的事实残余，并降低读取器利用集合构造偏差的风险。

[assumption] 非线性重编码并非无限复杂；若主要残余存在，至少部分可由容量受控、结构不同的恢复器暴露。

### 4.2 因果定位

[idea] 对恢复器发现的层与表示视图执行事前定义的单组件及联合激活替换，以估计哪些组件或路径供给残余信息。对组件 $c$，用 matched never-known 激活替换目标事实激活，定义其对该读取器的因果贡献为：

$$
\phi_{c,t}^{r}
=A_r(M_{t-1};F_{\mathrm{forget}})
-A_r\!\left(\operatorname{Patch}_{c\leftarrow\mathrm{NK}}(M_{t-1});F_{\mathrm{forget}}\right).
$$

$\phi_{c,t}^{r}$ 越大，表示该替换越能削弱目标事实恢复；必须与 self-patch、反向 patch 和随机组件 patch 配对解释。

[idea] 激活替换只作为因果定位证据；激活差异、attention 权重或相关性排序均不直接证明事实存储位置。

### 4.3 局部低秩修正

[idea] 对定位组件实施低秩权重残差更新：

$$
\Delta W_{c,t}=B_{c,t}A_{c,t},\qquad
\operatorname{rank}(\Delta W_{c,t})\le k_c,\qquad
M_t=M_{t-1}+\{\Delta W_{c,t}\}_{c\in C_t}.
$$

[idea] 每轮固定恢复器与当前模型，只在事前选定的候选集合 $\mathcal D_t$ 内优化低秩参数。一个可执行的最小目标为：

$$
\min_{\Delta\in\mathcal D_t}
\mathcal L_{\mathrm{rec}}\!\left(R_t^{\mathrm{edit}},M_{t-1}+\Delta\right)
+\lambda_{\mathrm{ret}}\mathcal L_{\mathrm{ret}}\!\left(M_{t-1}+\Delta\right),
$$

$$
\mathcal D_t=\left\{\Delta:
\operatorname{rank}(\Delta W_{c,t})\le k_c,\quad
\lVert\Delta W_{c,t}\rVert_F=g_{c,t},\quad
c\in C_t\right\}.
$$

其中 $\mathcal L_{\mathrm{rec}}$ 是随事实可恢复性下降而减小的残余目标；若恢复器使用交叉熵 $\mathcal L_{\mathrm{CE}}$，则 M0 可取 $\mathcal L_{\mathrm{rec}}=-\mathcal L_{\mathrm{CE}}$。$\mathcal L_{\mathrm{ret}}$ 是近 retain 与通用 retain 的约束项，$g_{c,t}$ 是逐组件能量预算。优化过程中重新训练恢复器会把“攻击—防御共同适应”混入单轮效果，故不作为 M0 默认实现；最终是否只是骗过训练恢复器，必须由独立审计臂判定。

[idea] 真实组件和方向必须与等能量随机低秩修正、shuffle 主体/边以及 matched-retain pseudo-intervention 比较。

### 4.4 独立复核

[idea] 训练恢复器 $R_t^{\mathrm{edit}}$ 不得参与最终审计。独立审计使用不同作者簇、模型结构、初始化与训练种子的读取器，并覆盖 patch、$\mathrm{Q4}/\mathrm{Q8}$ 与限预算重学习。

[idea] 闭环轮数、每轮预算、组件选择规则和停止条件在 calibration 后冻结；不得在 confirmation 上根据结果追加轮数或修改规则。

---

## 5. 研究问题与可证伪假设

### RQ1：残余发现

[idea] 输出沉默或线性不可读后，是否仍存在可重复的非线性恢复信号？

### RQ2：定位特异性

[idea] 恢复器发现的残余能否被激活替换稳定定位到超越随机与 shuffle 对照的组件或路径？

### RQ3：闭环收敛

[idea] 局部低秩修正能否在固定预算下逐步降低独立读取器的恢复优势，而不是使残余持续迁移？

### RQ4：L2 恢复阻断

[idea] 最终 student 在五类审计臂下是否均不显示相对 never-known 对照的恢复优势？

### RQ5：选择性

[idea] 上述变化能否在不越过近 retain、通用 retain 与输出健康门的前提下发生？

### 事前预测

- **P1｜机制预测：**[prediction] 线性修正后应能观察到“线性读取接近对照、训练用非线性恢复器仍成功”的分离；真实组件修正应优于等能量随机修正。
- **P2｜L2 预测：**[prediction] 封存 confirmation 上，五类审计臂均未发现超过冻结容忍界限的恢复优势。
- **P3｜行为—内部一致性预测：**[prediction] 目标事实自由生成下降应伴随内部读取器结果变化；若只发生输出沉默，不支持 L2。
- **P4｜retain 预测：**[prediction] 近 retain 与通用 retain 均不越过预注册阈值，且目标事实下降不能由拒答、截断或实体混淆解释。

---

## 6. M0 数据与实体级隔离

[idea] 使用 TOFU forget05 的实体级派生机制划分；派生结果与 canonical TOFU 结果分开报告。

| 集合 | 草案用途 |
|---|---|
| Forget-train | 4 个作者簇；其中 2 簇供训练恢复器/编辑，2 簇供不同架构审计 probe 训练 |
| Forget-calibration | 2 个作者簇；仅用于冻结秩、轮数、学习率和数值配置 |
| Forget-confirmation | 其余 4 个作者簇；封存，全部冻结后只解封一次 |
| Matched-retain | 非 forget 作者的同关系、同模板事实；按 train/calibration/confirmation 实体级隔离 |
| Never-known | 同模板的新构造虚构实体—属性组合，未进入 base/target 训练；仅作读取器与重学习参照 |
| General retain | TOFU Real Authors、World Facts 与官方 retain 评测 |

隔离规则：

1. 作者 canonical ID、别名与实体簇是隔离单位，不按 QA 行随机拆分。
2. confirmation 不参与恢复器结构、组件、层、秩、轮数或阈值选择。
3. 若可用 confirmation 少于 4 个独立作者簇，结果为 INCONCLUSIVE；不得用多次生成补足独立样本量。
4. 所有 manifest、源行、别名、tokenizer 配置和哈希在 confirmation 解封前冻结。

---

## 7. 对照与主比较

| 对照 | 目的 |
|---|---|
| 无干预 base/target | 测量原始可恢复性与行为基线 |
| No-op / self-patch | 排除 hook、缓存与前向实现伪影 |
| 随机等能量低秩扰动 | **机制主对照**；检验残余方向与组件定位是否必要 |
| 等总预算一次性修正 | **闭环主对照**；将相同组件范围、秩与总能量预算一次性施加，检验多轮恢复—定位是否必要 |
| Shuffle 主体/边 | 检查配对与路径特异性 |
| Matched-retain pseudo-intervention | 检查非目标事实是否受到同类损伤 |
| Never-known | 提供未学习事实时的 probe、量化与重学习参照 |
| 正对照 | 验证读取器在 base/target 上原本具有恢复能力 |

[idea] 两个主比较必须分开回答两个问题：

1. **真实残余引导修正 vs 随机等能量修正：**检验目标特异定位；若二者无差别，不能把效果解释为目标特异机制。
2. **多轮 RecodeLoop vs 等总预算一次性修正：**检验闭环本身的增量价值；若二者无差别，只能主张低秩修正有效，不能主张恢复—定位—再攻击闭环必要。

两个比较均须固定报告，不得只选择显著或更符合叙事的一项。

---

## 8. 五类独立审计臂

| 审计臂 | M0 协议草案 |
|---|---|
| 独立线性 probe | 固定线性分类器；不参与方向或组件选择 |
| 独立非线性 probe | 与训练恢复器使用不同层宽、深度、初始化和种子 |
| 指定 causal patch | 在冻结层、组件和位置做双向替换；patch 恢复不单独证明擦除 |
| $\mathrm{Q4}/\mathrm{Q8}$ 量化 | 固定量化方法、校准数据、种子和配置；比较 FP/Q 的 base 与 student |
| 限预算重学习 | 固定参数高效或全参选择、优化器和数据顺序；草案为 1 个 forget-train epoch、最多 128 更新步并记录实际 token |

[idea] 线性/非线性 probe 是表示读取器；causal patch 是干预审计，量化是变换鲁棒性审计，重学习是自适应攻击。五者统称“审计臂”，不把所有结果强行解释成同一种 probe 分数。

[idea] 训练恢复器与审计读取器必须在数据、结构和用途上分离。任何审计臂未实际执行，结果写“未覆盖”，不得默认通过。

[idea] 重学习的 128 步是该审计臂的协议定义，不是全局硬性资源上限。

---

## 9. 反证与 kill-switch

| Kill-switch | 触发条件 | 停止的主张 |
|---|---|---|
| KS0 正对照失败 | base/target 上读取器无法恢复已知事实，或 never-known 对照异常可恢复 | 读取器无效；不评价科学假设 |
| KS1 非特异定位 | 真实组件不优于等能量随机或 shuffle 组件 | 机制定位主张 |
| KS2 非线性残留 | 独立非线性 probe 在 confirmation 稳定恢复目标事实 | 当前 L2 主张 |
| KS3 跨层/跨路径迁移 | 每轮压制一种编码后出现等强新编码，或未编辑路径仍可恢复 | 闭环机制与当前 L2 主张 |
| KS4 量化恢复 | $\mathrm{Q4}/\mathrm{Q8}$ 相对 FP 出现超过冻结界限的恢复优势 | 当前 L2 主张 |
| KS5 重学习优势 | student 相对 never-known 对照异常快速恢复 | 当前 L2 主张 |
| KS6 retain 失守 | 近 retain 或通用 retain 越过冻结阈值 | 当前选择性遗忘臂 |
| KS7 替代解释 | 目标下降主要由拒答、截断、实体混淆或通用退化解释 | 遗忘解释；只报告行为/损伤 |

[idea] 任一 kill-switch 命中即停止对应主张并由 chenyliao 复审。不得在 confirmation 上改层、改秩、改读取器或重设阈值。

---

## 10. 预算配准与资源记录

- 各训练臂等有效 token、等更新步数、等可训练参数量和等模型架构。
- 主配准为终点：

$$
G_2=\sqrt{\sum_l\lVert\Delta W_l\rVert_F^2},
$$

草案容忍差值为 $\pm 2\%$。

- 敏感性报告：

$$
G_1=\sum_l\lVert\Delta W_l\rVert_F.
$$

- 主对照采用相同闭环轮数和每轮预算，不以额外训练解释真实臂收益。真实与随机修正还须在**同一组件、同一秩、同一逐组件预算 $g_{c,t}$** 下配准；仅匹配全局 $G_2$ 可能掩盖层间尺度差异。
- 敏感性分析报告组件激活位移 $D_{c,t}=\mathbb E_x\lVert h_{c,t}(x)-h_{c,t-1}(x)\rVert_2^2$，检验相同权重能量是否产生相近功能扰动；该项是机制解释辅助，不替代权重能量主配准。
- 恢复器训练、激活提取、量化、生成和重学习成本分别记录；相同 epoch 不自动视为等算力。
- 未配准项包括实际 wall-clock 与不同读取器实现开销，只报告，不据此作公平性宣称。
- 硬件为 2×A800 80GB；6 GPU 小时为软目标，按各卡使用时间求和；无硬性总量上限，软目标超时不自动构成科学 FAIL。

---

## 11. Retain 约束

以下为预注册常数草案，不是实验结果：

| 约束 | 草案阈值 |
|---|---|
| 近 retain：matched-retain、同实体非删除事实 | 目标准确率/出现率绝对下降不超过 1 个百分点 |
| 通用 retain：Real Authors、World Facts | 汇总及关键子任务绝对下降均不超过 1 个百分点 |
| 输出分布 | retain token 平均 KL $\le 0.02$ nat/token |
| 输出健康 | retain 拒答、截断或重复率增幅均 $\le 1$ 个百分点 |

[idea] 近 retain 与通用 retain 分开报告；阈值在 confirmation 解封前冻结。calibration 只用于估计可实现性和方差，不得根据 confirmation 结果调整。

---

## 12. 统计分析草案

1. 独立单位为作者/实体簇。
2. 每条 prompt 草案生成 $K=8$ 条完整回答；$K$ 条回答只作簇内重复测量。
3. 分报目标事实、替代事实、拒答、未提及、截断/不完整及目标—替代同现。
4. 报告每个实体簇的原始估计、簇级配对效应及全部审计臂结果。
5. 区间采用实体簇 bootstrap，草案为 10,000 次重采样并报告 95% 区间；confirmation 仅 4 簇时明确标为探索性、分辨率有限。
6. 主对照另报告实体簇配对 sign-flip permutation 结果。
7. 最低可分析要求为 4 个 confirmation 作者簇且必要对照完整，否则为 INCONCLUSIVE。
8. 不构造跨读取器综合成功分；评分区间不得冒充统计置信区间。
9. 仅 4 个 confirmation 作者簇时，实体簇 bootstrap 与 sign-flip 的分辨率均有限；M0 即使全部门通过也只作探索性支持。未来 confirmatory 版本须增加独立实体簇并重新冻结未触碰的数据，不能在同一 4 簇上反复试验后改称确认。

---

## 13. 逐门停止规则

| 门 | PASS | FAIL | BLOCKED |
|---|---|---|---|
| G0 资产与数据 | checkpoint、manifest、实体/别名隔离和哈希完整 | 数据或隔离定义不成立，停止当前臂 | 来源/环境不可用，不写成科学失败 |
| G1 读取器正对照 | base/target 可恢复，never-known 对照符合预期 | 读取器无效，修复后重跑 | 工具或依赖不可用 |
| G2 机制特异性 | 真实方向/组件优于随机等能量与 shuffle | KS1，停止机制主张 | 激活或 patch 环境受阻 |
| G3 预算与实现 | token/步数/参数/$G_2$ 配准，no-op 正确 | 实现或配准失败，不进入确认 | 外部计算资源不可用 |
| G4 L2 审计 | 五类审计臂均未发现超过界限的恢复优势 | KS2–KS5 任一命中 | 审计臂无法运行；标记未覆盖 |
| G5 retain | 近 retain、通用 retain、输出健康全部合格 | KS6/KS7 命中 | 评测资源或来源受阻 |

[idea] FAIL 必须注明失败的是实现门、机制主张、L2 主张还是 retain 门。confirmation 只解封一次；失败后不得在同一 confirmation 上迭代修复并重新宣称确认。

---

## 14. 与其他路线的边界

### 14.1 与 ASU

[idea] ASU 使用全局 attention 温度构造输出 teacher；RecodeLoop 使用残余恢复器驱动局部权重修正，并把独立恢复审计纳入遗忘定义。二者不能只凭输出指标判断机制优劣。

### 14.2 与 DeepFade

[idea] DeepFade 研究全层低秩权重残差、容量 $c_l$ 与目标能量包络 $d_l/s_l$ 的独立价值；RecodeLoop 研究非线性残余发现、定位与恢复审计。DeepFade 保留为独立既有路线。

[decision] S3 并入 DeepFade 的 schedule 消融；固定终点 $G_2$、token 和步数后仅操纵 schedule。

### 14.3 与方向 I 及 ADU 变体

- [decision] S1 并入方向 I 的迁移变体。
- [decision] P06-E02、P02-C6 降为 ADU 的路径粒度/组件范围变体，须先复现同预算 ADU 对照。
- [decision] P10-03 并入反事实草案，作为监督位置密度消融；$X^{-}$ 是另一个有效实体，不是“无事实状态”。

---

## 15. 预期贡献与负结果价值

### 若干预与审计结果支持假设

[idea] 可形成一种恢复引导的局部遗忘机制，并在声明的 L2 威胁模型下提供组件级干预证据和五类审计臂的独立复核。

### 若残余持续迁移

[idea] 可界定局部方向修正在分布式非线性重编码面前的失败边界，反驳“压掉当前探针信号即完成遗忘”的强解释。

### 若随机对照同样有效

[idea] 可否定目标特异定位叙事，说明恢复器只提供了一般扰动区域，而非事实路径证据。

### 若 retain 先失守

[idea] 可揭示残余压缩与共享语义表示之间的冲突，而不把通用能力损伤包装成遗忘收益。

[idea] 本方向不使用“首次”“永久不可恢复”“零效用损失”等表述；所有新颖性判断均相对已核查文献。

---

## 16. 当前状态与下一步

[decision] 当前 M0 对象为 P08-M2，不并行第二对象；该指定是研究对象选择，不是其他候选已被证伪。

**下一步类型：`preregistration`。**

[idea] 在运行前由 chenyliao 冻结实体 manifest、读取器实现、数值阈值、统计规则、代码/配置哈希和 confirmation 封存记录，形成带时间戳的预注册版本。本 Proposal 本身不表示预注册已经完成。

---

## 17. M0 最小可执行设计（建议稿）

[idea] M0 的首要目标是判别“读取器是否有效、定位是否特异、闭环是否优于一次性修正”，而不是在小样本上提前宣布 L2 已获确认。

### 17.1 资产锁定

1. 在 manifest 中把 P08-M2 映射到唯一的 base、target、ASU student checkpoint、tokenizer、权重哈希与生成配置；当前 Markdown 文档尚未给出该映射，因此这是运行前的 G0 阻塞项。
2. 冻结 forget/calibration/confirmation 的作者簇、matched-retain 与 never-known 配对规则，以及所有事实值标签。
3. 明确 P08-M2 是待审计的起点模型，而不是把 RecodeLoop 的结果预先写进 checkpoint 名称或实验结论。

### 17.2 单轮实现

1. **提取：**在冻结层、组件与 token 位置提取激活；候选视图与 pooling 规则只能在 calibration 选择一次。
2. **恢复：**训练事实值解码器 $R_t^{\mathrm{edit}}$，报告正确事实值恢复、候选集合秩与置信度，禁止只用 forget/retain 二分类作为主终点。M0 优先使用跨实体共享答案空间的属性关系；若答案高基数或逐实体唯一，则使用生成式 decoder 或基于冻结答案文本嵌入的相似度头，不能把未见过的答案任意指定为一个封闭分类类别。
3. **定位：**在 attention 输出矩阵与 MLP 输出矩阵的预声明候选集合上做 self/正向/反向 patch，以 $\phi_{c,t}^{r}$ 和 retain 损伤共同筛选组件。
4. **修正：**在选定组件上优化冻结恢复器下的 $\mathcal L_{\mathrm{rec}}+\lambda_{\mathrm{ret}}\mathcal L_{\mathrm{ret}}$，并投影回逐组件秩与能量约束。
5. **复核：**每轮只在 train/calibration 上运行不涉及确认集参数选择的轻量审计；完整五臂审计在冻结终点运行，避免把审计反馈反复用于 confirmation 调参。

### 17.3 建议的最小实验矩阵

| 臂 | 恢复引导方向 | 因果选组件 | 多轮 | 总秩/能量/token/步数 |
|---|---|---|---|---|
| ASU student / no-op | 否 | 否 | 否 | 相同前向与评测预算 |
| RecodeLoop-full | 是 | 是 | 是 | 固定 |
| One-shot | 是 | 是 | 否 | 与 full 总预算配准 |
| Random-direction | 随机 | 同一组件 | 是 | 与 full 逐轮、逐组件配准 |
| Shuffle-component | 是 | 打乱 | 是 | 与 full 逐轮、逐组件配准 |
| Retain-pseudo | 对 matched-retain 实施同类流程 | 是 | 是 | 与 full 配准 |

[idea] 建议以 $T_{\max}=3$ 作为资源友好的草案起点；秩 $k\in\{4,8\}$ 与每轮组件数 $m\in\{1,4\}$ 仅在 calibration 选择一次，随后冻结。上述数值是待预注册的设计常数，不是已运行实验得到的最优值。

### 17.4 结果判读

| 观察 | 允许的解释 |
|---|---|
| 正对照失败 | 读取器或实现无效，不评价遗忘机制 |
| full 不优于 random-direction | 不支持目标特异定位 |
| full 优于 random-direction，但不优于 one-shot | 支持低秩修正，不支持闭环必要性 |
| full 优于两个主对照，独立审计仍恢复 | 支持定位但反驳当前 L2；检查迁移与替代解释 |
| 五臂均通过但 confirmation 仅 4 簇 | 只报告 M0 探索性支持，不写成已确认 L2 |
| retain 或输出健康先失守 | 当前臂不具选择性，不以目标下降抵消通用损伤 |

[idea] M0 最重要的图不是单一准确率柱状图，而是三张配对曲线：各轮恢复优势 $\Delta_{r,t}$、各组件因果贡献 $\phi_{c,t}^{r}$、以及恢复收益相对 retain 损伤的权衡。它们分别回答“残余是否下降”“下降从哪里来”“代价是否可接受”。

[idea] 本轮未新增系统文献检索，不对“首次”或相对全部既有工作的新颖性作判断；下一步仍是资产补齐与预注册，而不是直接启动 confirmation。

---

## 参考与证据边界

1. Zare Zade et al. *Attention Smoothing Is All You Need for Unlearning*. [arXiv:2603.01285](https://arxiv.org/abs/2603.01285)。
2. Hase et al. *Does Localization Inform Editing?* NeurIPS 2023. [arXiv:2301.04213](https://arxiv.org/abs/2301.04213)。用于约束“定位即可指导编辑”的推断。
3. Belrose et al. *LEACE: Perfect Linear Concept Erasure in Closed Form*. [arXiv:2306.03819](https://arxiv.org/abs/2306.03819)。严格保证限于线性读取器，不排除非线性重编码。
4. Nielsen et al. *Logit Distance Bounds Representational Similarity*. ICML 2026. [arXiv:2602.15438](https://arxiv.org/abs/2602.15438)。KL 接近不保证线性表示相似。
5. Dorna et al. *OpenUnlearning*. [arXiv:2506.12618](https://arxiv.org/abs/2506.12618)。支持量化、重学习与指标稳健性审计。
6. Goel et al. *Auditing Language Model Unlearning via Information Decomposition*. EACL 2026. [ACL Anthology](https://aclanthology.org/2026.eacl-long.35)。支持区分输出遗忘与表示残余。
7. Lynch et al. *Eight Methods to Evaluate Robust Unlearning in LLMs*. [arXiv:2402.16835](https://arxiv.org/abs/2402.16835)。其结果来自 WHP 设置，不直接外推为 ASU 或本方向证据。
8. Chen et al. *Unlearning Is Not Just Erasing: Temporal Decoupling via Generation Inequality*（ADU）. [arXiv:2608.23020](https://arxiv.org/abs/2608.23020)。用于路径级近邻与同预算对照边界。

[idea] 以上文献为方法设计与反证提供依据，不构成 RecodeLoop 有效性证据。

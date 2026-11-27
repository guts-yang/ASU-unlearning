# CF3 候选方法评估打分看板（汇总版）

**目录**：`workspace/idea-stage/candidates-1127-2`（10 个方法）
**评估口径**：本机 `/idea-evaluator` rubric（Higher/Faster/Stronger/Cheaper/Broader，5 起评；Strong Accept 需 ≥2 维 8+ 且零 MAJOR）
**日期**：2026-11-27
**性质**：执行者独立复评，均 mechanism-based、`PROVISIONAL`、`REVIEW_UNAVAILABLE`、evidence gate `BLOCKED`；分数非实测、非跨模型回执。

> 🎯 **30 秒汇总（照着念）**：十个 CF3 方法逐个独立打完分，均分区间 6.6–7.6，**没有一个 Strong Accept**——每个都带至少 2 个 MAJOR 风险（近邻划界 + 因果可验证性）。第一梯队三个并列 **7.6：CF3-01 双线性路由/内容差分（精确恒等式、最省）、CF3-02 自校正持续子空间（只删存活到下游的差异，概念最强）、CF3-06 配对置换-FDR 激活门（统计严谨性最高）**；第二梯队 CF3-03/04/05 各 7.4；尾部 **CF3-09 双端 Hankel 6.8、CF3-10 非交换路径签名 6.6**（局部线性化风险 + 估计成本 + 轨迹近邻）。全批的共同软肋只有一句：**分解/门控/几何的代数严谨 ≠ 事实语义**——几乎每份提案都自己承认"不自动赋予事实语义，需 patch 或独立读出验证"；所以先过零训练生死门（证明算子真编码目标事实而非身份/瞬态/词形/统计伪影），再谈训练内化。

---

## 一、总榜（按独立均分降序）

| 名次 | 方法 | 核心算子 | H | F | S | C | B | 均分 | 裁决 | 自评 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | CF3-01 双线性路由—内容差分 | 精确双线性恒等式拆路由/内容 | 7 | 8 | 7 | 8 | 8 | **7.6** | AWR | 8.0 |
| 1 | CF3-02 自校正持续子空间 | 下游 replay 生存谱 | 8 | 7 | 8 | 7 | 8 | **7.6** | AWR | 7.8 |
| 1 | CF3-06 配对置换—FDR 激活门 | 符号置换零假设 + BY-FDR | 7 | 8 | 8 | 8 | 7 | **7.6** | AWR | 7.6 |
| 4 | CF3-03 访问分布加权激活差 | Student 状态访问测度加权 | 8 | 7 | 7 | 7 | 8 | **7.4** | AWR | 7.6 |
| 4 | CF3-04 门控状态转移擦除 | SwiGLU 四状态转移图 | 7 | 7 | 8 | 8 | 7 | **7.4** | AWR | 7.6 |
| 4 | CF3-05 答案跨度 Grassmann 子空间 | 子空间投影差 ΔP + 主角度 | 7 | 8 | 7 | 8 | 7 | **7.4** | AWR | 7.6 |
| 7 | CF3-07 多路径积分稳定归因 | 跨路径完备归因稳定分量 | 7 | 6 | 8 | 7 | 8 | **7.2** | AWR | 7.4 |
| 7 | CF3-08 反事实 Copula 尾依赖 | retain 秩坐标 copula 尾依赖差 | 8 | 6 | 8 | 6 | 8 | **7.2** | AWR | 7.4 |
| 9 | CF3-09 双端 Hankel 激活模态 | 可控×可读平衡 Hankel 奇异值 | 7 | 6 | 7 | 6 | 8 | **6.8** | AWR | 7.2 |
| 10 | CF3-10 非交换路径签名遗忘 | log-signature 二阶 Lévy 面积 | 7 | 5 | 7 | 6 | 8 | **6.6** | AWR | 7.2 |

> 裁决 AWR = Accept with Revisions（全部 mechanism-based，worth pursuing, pending the validation experiment）。无 Strong Accept（全批均有 2 MAJOR，不满足"≥2 维 8+ 且零 MAJOR"），无 Reject（无一项被数据直接反驳）。

---

## 二、逐方法一句话评估

**CF3-01 双线性路由—内容差分 7.6**：最干净、最省——对称差分恒等式精确等于原始差（不需近似），统一 attention 与 SwiGLU 双线性，闭式复用缓存。下调点：恒等式只是把已知 QK/V 路由-内容分解改写成代数形式（F1 强近邻 Contribution Weights / How Few-Shot Examples Add Up），"分解≠发现新信号"，且路由/内容语义须 patch 门确认。首步：四组 patch-forward 分量判别 + null 门，两项都通过即声明不可识别。

**CF3-02 自校正持续子空间 7.6**：概念最强——只删经冻结下游 replay 后仍持续的差异，天然避开"会被网络自己校正的瞬态身份差"，与 CF2-01 相邻层差分、CF-07 读出 Jacobian 区分清楚。下调点：patch replay 可能造 OOD 状态（proposal 自列 MAJOR），生存谱须与自然同分布状态对照。首步：patched vs 自然状态分布差 + 生存子空间与原始差/通量的主角度。

**CF3-06 配对置换—FDR 激活门 7.6**：统计严谨性最高——用配对内符号置换构造零分布 + BY-FDR 控制任意依赖下的多重比较，专治"全层全位置范数排行榜"的假阳性。下调点：置换检验/FDR 本身是成熟工具（神经影像），不提供新事实语义（proposal 自评 Higher 已只给 7），增量在"可交换性零假设做选择规则"。首步：matched-retain 假阳性率 + 置换门 vs top-norm + 独立 patch 读出。

**CF3-03 访问分布加权激活差 7.4**：覆盖视角对——用当前 Student 的状态访问测度给共享前缀加权，直击 teacher-forcing 只看 gold 轨迹的盲区，对改写/间接提示更直接。下调点：与 DKT on-policy、CF-05 区分需证明收益来自访问测度而非前缀变多；free-prefix 上原事实读出难定义。首步：η=0/1/混合 + 随机等预算前缀三臂。

**CF3-04 门控状态转移擦除 7.4**：机制具体——SwiGLU gate/value 四符号状态转移图，只惩罚"仍处于目标源状态"、不指定 donor 落点，sign-only vs magnitude-only 对照扎实。下调点：GLUScope/Sign Code 占符号状态、CodeUnlearn 占离散码删除；离散状态是否承载事实而非词形须因果验证。首步：转移图跨 split 稳定 + 保幅值 sign 翻转对照。

**CF3-05 答案跨度 Grassmann 子空间 7.4**：几何视角新——把答案跨度矩阵当 Grassmann 点，用投影差 ΔP 与主角度表示"多位置共同张成方向"。下调点：CIR/LEACE/ReGLU 强近邻，"用子空间"不新；proposal 自承"对完整隐空间旋转不自动不变"，基底旋转不变性是有限优点。首步：ΔP vs 均值/范数/逐点 PCA 的独立信息 + 跨 split 主角度门。

**CF3-07 多路径积分稳定归因 7.2**：针对单路径伪定位——沿多条主体槽路径做完备归因，只保留跨路径符号一致分量。下调点：多路径 IG、DIG/Sequential IG、IG-Lens 近邻密集，增量在反事实端点 + 激活差归因应用；路径稳定性可能只是积分误差。首步：数值完备性误差 vs 跨路径方差 + 稳定分量 vs 单路径/割线。

**CF3-08 反事实 Copula 尾依赖 7.2**：信息维度新——retain 秩坐标下的 copula 尾依赖差，能覆盖均值/协方差完全相同的非线性尾部通路，边缘死区防坍缩（Stronger 高）。下调点：SuperActivators/PINT/CF2-03 限制宽泛新颖性；高维尾部估计样本效率与跨 seed 稳定性是硬伤（Faster/Cheaper 各 6）。首步：样本量-偏差曲线 + 经验 copula vs Gaussian copula。

**CF3-09 双端可控—可读 Hankel 模态 6.8**：双端交集概念好——同时筛"主体可控且答案可读"的模态，比 CF-07 单端读出更全。下调点：控制论解释 + Fast-NTK 强近邻；中点局部线性化在大反事实差异下可能失效，且双端谱可能与单端高度重合；Jacobian 估计贵。首步：双端 vs 单端主角度 + 多插值点检查局部谱翻转。

**CF3-10 非交换路径签名遗忘 6.6**：数学最 ambitious——用 log-signature 二阶 Lévy 面积捕获端点和一阶通量不可见的有序计算结构，重参数化不变/路径反转变号/直线零面积三个代数 sanity check 漂亮。下调点：全批最高风险——签名 Transformer/轨迹近邻、二阶项可能与通量共线、符号跨 split 不稳、Faster 只给 5（无净提速）。首步：三条代数性质验证 + 二阶面积相对通量的增量信息；不过门不进训练。

---

## 三、共同生死门（所有方法进入训练前）

1. **代数严谨 ≠ 事实语义**：每个分解/门控/几何算子都只提供坐标或统计，必须先用零训练前向证明目标分量真编码目标事实——真实 donor 显著优于 wrong / shuffled / noise / matched-retain 对照。
2. **因果承载**：patch/target 干预后独立生成式 probe 与 relearning 的目标恢复必须下降；只降范数/协方差/主角度/面积不算。
3. **非捷径**：边缘分布、长度、EOS、retain NLL、同实体非目标关系必须稳定；至少两类独立抗恢复证据改善后才允许用"深度遗忘"措辞。

---

## 四、诚实边界

- 全部 plan-only / 机制推断；论文级 claim 待验证，依赖实验的结论 `needs manual pilot`。
- 文献身份 WebSearch 元数据级、未 D1 核验、不作 BibTeX；检索只用于近邻划界，未引用性能数值。
- novelty-check `PROVISIONAL`、research-review `REVIEW_UNAVAILABLE`、evidence gate `BLOCKED`（缺跨模型回执与 RESUMABLE helpers）；无 GPU pilot、未改仓库、未推送。
- 本看板是阅读视图，Markdown 为唯一真实来源；分数与批量自评的分歧已逐项标注，不构成独立 reviewer 接受回执。

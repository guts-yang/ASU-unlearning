# CAASS-ASU 预注册实验协议（草案）

状态：**尚未生效**。本文件定义获准且主基线通过后使用的实验规则；未附带结果，不应被解释为代码已实现或方法已验证。

## 共同规则

1. 每次运行写入 commit、工作区状态、配置副本 SHA-256、模型/数据身份、seed、命令、开始/结束时间、GPU、峰值显存与 GPU-hours。
2. 主比较仅用 RtBF / TOFU Forget05 / ASU+GD。ASU+KL 另列敏感性分析；Forget01/10 分开报告。
3. Teacher NLL、训练后 Student FE/MU、再学习攻击的恢复指标是不同证据层，互不替代。
4. 按实体、作者或问题改写组划分数据与 bootstrap 单位；token 不是独立统计单位。
5. 任何最终 test/holdout 均不能参与选层、温度搜索、Pareto 选择或阈值回填。
6. M0 `withheld_negative_control` 不参与通过、失败或效果大小结论。

## E0：Teacher 单层温度响应

### 数据与测量

- 先生成按事实组隔离的 `D_F^cal`、`D_R^cal`、`D_F^holdout`、`D_R^holdout` manifest；近似改写不得跨集合。
- `D_F` 与 `D_R` 尽量按问句形式、长度和任务类型匹配。外部标签如只用于离线诊断，必须写入 manifest 且禁止进入训练。
- 固定因果 attention、label shift、答案 token mask 与 padding mask；使用 teacher forcing。
- 对每个样本计算答案 token NLL，先样本/实体聚合，再组级汇总。

对每层 `l`、预注册温度 `tau`：

`DeltaNLL_F(l,tau) = NLL_F(single_layer_l_tau) - NLL_F(all_tau_1)`

`DeltaNLL_R(l,tau) = NLL_R(single_layer_l_tau) - NLL_R(all_tau_1)`

二维 `(DeltaNLL_F, DeltaNLL_R)` 是机制诊断；不在未注册尺度或权重下将其合成为因果分数。

### 干预和对照

- 基线：所有 `tau_l=1`。
- 单层：一次只改一层，其他层固定为 1。
- 温度网格：在基线 smoke test 通过、可用预算确认后写入 `configs/e0_probe.yaml`；可含 2.3 和 3.0 作为论文参考点，但不能称其为本地最优。
- 对照：固定 L6–8 共享温度、匹配层数/层段分布的随机层（多个抽样）、工程可行时的全层共享温度。
- 可选诊断：attention entropy/variance；它们不是事实头标签。

### E0 预注册 gate

校准集预先确定选层规则、保留损伤上限与 bootstrap 程序，随后仅在 holdout 评估。只有候选层的 forget 响应在 holdout 复现，并优于匹配随机层且保留损伤未越界，才可进入 E1。

若层间差异不稳定、仅熵变化但答案输出不变、或保留损伤不可接受，则 E0 失败，停止 CAASS 主线。输出：逐样本 JSON/CSV、逐层逐温度汇总、排名、随机层抽样、区间、运行时和显存。

## E1：逐层温度搜索与 Student 训练

### 搜索空间

- 仅使用 E0 calibration 选出且在 holdout 确认的 `L*`。
- 个体为正的 `tau_l (l in L*)`；未选层固定为 1。
- 若优化层数，必须允许 `tau_l=1` 或使用显式启用变量；否则复杂度为常数，不可作为优化目标。
- 在运行前冻结编码、范围、初始种群、交叉、变异、种群大小、代数、停止条件和 Student 真实训练次数。

### 适应度、控制与对照

- 适应度来自 Student 训练后的 validation FE、MU 和干预复杂度，Teacher NLL 不得替代。
- 预注册 Pareto 解规则：在 FE 不低于 Global ASU 的允许偏差内最大化 MU，随后选择复杂度较低者。允许偏差及 retain NLL 上限由 Global ASU 多 seed 波动确定。
- 每个候选共享 Student 初始权重、步数、优化器、采样、评测脚本和配对 data order；短预算只可筛选，最终候选以完整预算和新 seed 复验。
- 必需对照：Global ASU、L6–8 共享温度、E0 选层共享温度、E0 选层逐层温度（CAASS-ASU）、匹配随机层。
- NSGA-II 只是搜索器。只有在相同真实候选预算下另与网格或坐标搜索比较，才可主张搜索器优势。

### E1 gate

以 holdout/test 上的 FE、MU、retain utility、错误实体/乱码和答案 NLL 判定。报告多 seed 均值、方差/区间和按问题组重采样。

- 若因果选层不优于随机层，只可讨论稀疏干预，不能声称选择器有效。
- 若逐层温度不优于共享温度，不能声称层特异温度必要。
- 若仅 validation 改善或温度高度不稳定，报告可辨识性不足并回退至简单方案。

## E2：再学习鲁棒性（可选）

仅在 E1 通过并取得单独许可后执行。将攻击者可见 forget 子集与未见问题/改写分开；固定 LoRA/全参攻击预算、数据量、优化器、seed 与保存点。在 Global ASU、E1 通过的 CAASS-ASU 和必要随机层对照上报告 Recovery@steps、Recovery-AUC 及攻击后 retain utility。

仅当相同攻击预算下恢复更慢且 retain 未有额外不可接受下降，才可称“对该攻击更鲁棒”；不得称为不可恢复。

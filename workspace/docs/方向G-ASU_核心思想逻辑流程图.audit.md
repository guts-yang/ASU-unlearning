# 方向G-ASU_核心思想逻辑流程图 · audit

> 三件套之一（drawio 源文件 + PNG 预览 + 本审计记录）
> 日期：2026-10-10 ｜ 工艺：基线 cells 逐字保留 + 扩展层 20 cells 新增
> 基线 blob：`new_method` 分支 `ASU_核心思想逻辑流程图.drawio`（用户 draw.io 手动重排的自底向上布局，全分支同步版）
> 工艺对齐：与 D/E 分支扩展惯例一致（青虚线分区、基线调色板、白底边标签、单行 compact 公式覆盖层）

## 1. 基线保留核验

| 项 | 结果 |
|---|---|
| 基线全部 cells（title/mech/df/dr/taunote/student/retain/output/contrast/teacher 组/kl 组/joint 组/e1–e10） | 逐字保留，零改动 |
| 唯一基线文件改动 | `pageHeight` 660 → 960（为扩展层腾画布）；diagram name 改为「ASU核心逻辑流程+方向G」 |
| 基线渲染比对 | PNG 上半区与原图逐元素一致（四色语义分区、对照红虚线、τ 注记均原样） |

## 2. 扩展层新增元素清单（20 cells）

| id | 内容 | 媒介 | 样式 | 状态 |
|---|---|---|---|---|
| G-zone | 扩展层分区「方向 G 扩展层：ChainBreak——LRM 长思维链 pivot-token 注意力断路遗忘」 | native | 青虚线框 #0F6E63，无填充，x40 y620 1120×330 | accepted |
| G-cot | LRM 长思维链轨迹 S=(x, c_1..c_M, y)：R-TOFU CoT 标注 + WMDP 推导 | native | 紫实线 #534AB7/#EEEDFE（沿用基线数据节点色） | accepted |
| G-m0 | M0 生死闸门（kill-switch）：基线复现 + CoT bypass 量化；无 bypass / 定位≈随机 → 判死 | native | 绿虚线 #3B6D11/#EAF3DE（判死惯例色） | accepted |
| G-pivot | 因果流注显著性定位器：积分梯度 IG 追踪 c_j → y 贡献 | native | 青实线 #0F6E63/#E1F5EE | accepted |
| G-f1 | 公式覆盖层：S_pivot={c_j: I_j ≥ Q_{1-α}}（单行 compact） | native text | 覆盖于 G-pivot 下部 | accepted |
| G-delta | 归因对照（运行时 vs 参数级）：STaR 轨迹抑制 / Reasoned IDK / RR 表征重路由 | native | 灰虚线 #5F5E5A/#F1EFE8 | accepted |
| G-breaker | 断路算子：动态温度张量（核心）——仅 pivot 列升温 τ_max，其余 τ=1 | native | 青实线 | accepted |
| G-f2 | 公式覆盖层：T_ij=1+(τ_max−1)σ(I_j/σ_I)（单行 compact） | native text | 覆盖于 G-breaker 下部 | accepted |
| G-retain | 通用推理保留轨 L_retain：D_R^math（GSM8K/MATH）保真 | native | 蓝实线 #185FA5/#E6F1FB | accepted |
| G-f3 | 公式覆盖层：β∑_l‖A_θ−A_base‖_F²（单行 compact） | native text | 覆盖于 G-retain 下部 | accepted |
| G-adv | 对抗评测：推导诱导 / In-Context Replay / Thought Collapse / ZeroThink 泄漏 | native | 琥珀 #854F0B/#FAEEDA | accepted |
| G-output | ChainBreak 模型（产出物）：推理语法完整、事实跃迁解耦、ZeroThink 抗绕过 | native | 绿实线 #3B6D11/#EAF3DE | accepted |
| eG1 | 遗忘后模型 → M0「① 基线复现模型测 CoT 绕过」 | native | 绿虚线，waypoints (1118,596)→(520,596) 顶部绕行 | accepted |
| eG2 | LRM 轨迹 → M0「② R-TOFU 数据」 | native | 青实线 | accepted |
| eG3 | M0 → 定位器「③ 显著 + 一致 → 立项」 | native | 绿实线 | accepted |
| eG4 | 定位器 → 断路算子「④ Top-K 支点 → T_ij 逐位调制」 | native | 青实线 | accepted |
| eG5 | 断路算子 → Forget-teacher「⑤ θ_break 实例化（替代 Eq.4 均匀 τ）」 | native | 青实线 strokeWidth=2（核心衔接边加粗），waypoints (670,610)→(495,610) 走 M0/Pivot 间隙 | accepted |
| eG6 | 保留轨 → 产出物「⑥ 推理保真轨（与 L_break 双轨）」 | native | 蓝实线，底部绕行 y=891 | accepted |
| eG7 | 遗忘后模型 → 对抗评测「⑦ 三类攻击 + ZeroThink 实测」 | native | 琥珀实线，右侧 x=1175 绕行 | accepted |
| eG8 | 对抗评测 → 产出物「⑧ 评测通过 → 产出」 | native | 绿实线，底部绕行 y=906 | accepted |
| eG9 | 归因对照 → 定位器「归因对照」 | native | 灰虚线 open 箭头 | accepted |

## 3. 衔接关系设计说明

- **eG5 是方向 G 的核心衔接边**：Breaker-teacher θ_break 实例化基线 teacher 构造，直接替代 Eq.4 的全局均匀 τ——攻击点即 ASU 唯一自由度（标量 τ），与 E 分支 eE3（程序实例化 teacher）同构但机制不同（E 搜 τ 调度，G 用因果显著性逐位调制 T_ij）。
- **eG1 是 kill-switch 衔接边**：M0 复用基线遗忘管线产出（ASU/NPO/Reasoned-IDK checkpoint）在 R-TOFU 上量化 CoT 推理动量绕过，不新增实验底座。
- **eG7/eG8 构成评测闭环**：基线 output（遗忘后模型）→ 三类对抗评测（推导诱导 / In-Context Replay / Thought Collapse + ZeroThink 泄漏）→ 通过后产出 ChainBreak 模型；评测面板对齐概念大纲 §5.2。
- **eG9 归因防线**：三对照（STaR 推理时抑制 / Reasoned IDK 拒绝式 / RR 表征重路由）以 open 箭头接入定位器，对应评估报告致命缺陷 #1/#2 的「运行时 vs 参数级」归因设计。
- **双轨语义**：遗忘轨（G-breaker → teacher → 基线 kl/joint 链）+ 保真轨（G-retain → ⑥）在基线联合目标 Eq.1 处汇流，基线链路零改动承载新语义。

## 4. 技术校验与迭代记录

- `check_drawio.py`：55 cells（34 vertices + 19 edges），0 错误，0 image/svg（全部原生可编辑元素）
- 导出：`ELECTRON_DISABLE_SANDBOX=1 export_drawio.py` → PNG 1200×960
- v1 → v2 视觉迭代（3 轮）：
  1. 公式覆盖层 f1/f2/f3 下移（716→723 / 842→854），消除压节点第二行文字
  2. ②③④⑤⑥⑧ 边标签缩短/重定位（白底 labelBackgroundColor），消除与框体重叠
  3. **eG5 改道**：原 x448 竖线穿过 M0 框体（400–640）→ 改走 M0/Pivot 间隙 x=670，waypoints (670,610)→(495,610)，label 移至 y610 段
- 最终全尺寸目检：无文字溢出、无穿框、无标签叠字；基线上半区与原图逐元素一致

## 5. 与 D/E 分支三件套工艺一致性

- 配色体系沿用基线调色板 + 扩展惯例（青=扩展主色、琥珀=审计/评测、绿虚线=判死闸门、蓝=保留/效用、灰虚线=对照、紫=数据节点）
- 编号流转 ①–⑧ + 归因对照，与 E 分支同构；全部为 Draw.io 原生元素

## 6. 遗留人工复核点

- eG3/eG5 与 ③ 标签白底存在一处线-标签交叉（x670, y704），白底遮挡下可读性正常，如需彻底分离可将 G-m0/G-pivot 列距再加大

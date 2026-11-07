# ASU I3 候选全链路图审计

## 交付范围

- 参考源图：`ASU_遗忘全链路实现图_代码实证版v2.drawio`（只读；未改动）
- 新版本：`ASU_遗忘全链路实现图_I3_局部Teacher与前缀混合_v1.drawio`
- 预览：`ASU_遗忘全链路实现图_I3_局部Teacher与前缀混合_v1.png`
- 页面尺寸沿用原图 2900 × 1020；预览尺寸 2859 × 880。
- 新图以复制原 Draw.io XML 后定点替换节点实现；全部为原生 Draw.io 元素，无截图或 SVG 嵌入。

## I3 数学口径

1. Teacher 为冻结基座 `θ₀`。先由匹配事实/反事实输入的 activation patch 与 N/R gate 冻结路径集合 `S`；仅在 `j∈S` 上插值：
   `a_j^(ρ)=(1−ρ)a_j⁺+ρa_j⁻`。其余路径保持标准前向。
2. Attention 候选 hook 口径：单 head 输出、concat 前；MLP 候选 hook：`down_proj` 输出、残差相加前。两者对应图中不同模块与节点。
3. Teacher 和 Student 对同一前缀 `x+z_<t` 逐 token 评分；Student 的 forget 前缀 `z` 按整条序列取 `y` 或 `ŷ`，其中 `ŷ~π_{barθ}^d(.|x)`。
4. 定义 `ℓ_{S,ρ}(x,z)=mean_{t∈M_z} D_KL(q_{S,ρ,t} || p_{θ,t})`，并用
   `L_F^I3=(1−β)E_gold[ℓ]+βE_roll[ℓ]`；总损失保留原 ASU 结构 `α L_F^I3 + L_R`。
5. rollout 仅 stop-gradient；Student KL logits 保留梯度。图中所有 I3 操作均注明候选/未实测；E0 的 `do_not_start_e1` 仍是历史判决，未被更改。

## 改动节点清单

| ID / 区域 | 原组件功能 | I3 替换/标注 | 输入输出关系 | 状态 |
|---|---|---|---|---|
| `title` | ASU 通用链路标题 | I3 候选名称、局部 Teacher × Student rollout 前缀混合、未实测与 E0 判决边界 | 总图说明 | accepted |
| `teacherTitle` | 冻结 Teacher 说明 | θ₀ 冻结；S 须经配对 patch N/R gate 后冻结；ρ 仅用于 S | Teacher 路径状态 | accepted |
| `studentTitle` | Student 可训练 | gold/rollout 整序列按 β 混合；采样无梯度 | Student 路径状态 | accepted |
| `lossTitle` | 预测→局部损失→总损失 | 局部 Teacher + 同前缀 Student → I3 混合 KL → retain → 更新 | 损失总线说明 | accepted |
| `tInput` | forget 数据输入 | 问题 x、gold y⁺、Student rollout ŷ | 保留原 tInput→Embedding→Teacher 主链 | accepted |
| `tMiddle` | LLaMA 解码层占位 | 公式 `a_j^(ρ)`；注明 donor 来源于匹配 q⁺/q⁻ 干净前向和未选路径标准计算 | 仍连接原 Teacher Embedding 与 Final RMSNorm | accepted |
| `tPred` | Teacher softmax 分布 | `q_{S,ρ}=softmax(Z_{θ₀}^{patch})`；输入为 x+z、Teacher 冻结 | 仍向原 `lossF` 提供 Teacher 分布 | accepted |
| `sMiddle` | Student 解码层占位 | forget 上整序列 z∈{y,ŷ}；`ŷ~π_{barθ}^d(.|x)`；同一 z 供两模型评分 | 保留原 Student 数据输入与输出链 | accepted |
| `sPredF` | Student forget 分布 | `p_θ(.|x,z_<t)`，并标出 z 的两种来源 | 仍向原 `lossF` 提供 Student 分布 | accepted |
| `lossF` | ASU Teacher–Student forget KL | token 平均 `D_KL(q_{S,ρ,t}||p_{θ,t})`、β gold/rollout 目标、同前缀、整序列分支与 rollout stop-gradient | 原 Teacher / Student 输入与 loss→total 箭头保留 | accepted |
| `detailTeacherTitle` | Teacher 层温度说明 | 冻结 Teacher、经过 gate 的 S、候选局部 ρ 插值和未通过门即不进入 I3 | 底部 Teacher 单层结构提示 | accepted |
| `detailStudentTitle` | Student 每层不施温度 | 标准前向、不做局部 patch、与 Teacher 共用 prefix、训练未运行 | 底部 Student 单层结构提示 | accepted |
| `tDAttn` | 全层 Attention 温度公式 | 标准 attention；候选 head 输出 concat 前 hook；只对 S 插值；不施加全层 τ | 对应原 tDAttn→tDO 连接 | accepted |
| `tDO` | Attention 输出投影 | 明示 patched head 输出继续 concat/O 投影 | 保留原 O→残差路径 | accepted |
| `tDDown` | MLP down 投影 | 明示 down_proj、候选 `(MLP,layer,k)∈S`，在 residual 前做 ρ 插值 | 对应原 down_proj→残差相加连接 | accepted |
| 红色虚线框与标签 | 标示旧图 Attention 温度差异 | 保留原框位置和样式，改为 Attention hook 示例、仅 Teacher patch / Student 标准 | 不再宣称“唯一区别” | accepted |

## 保持不变的原有链路

- Teacher 与 Student 两条顶层主链、输入框位置、Embedding、Final RMSNorm、LM Head、预测框位置均保留。
- 右侧损失区、retain NLL、总损失、统一更新框的节点布局及连线关系保留；retain 项不纳入本次新改动。
- 底部两个单层 DecoderLayer 的节点位置、QKV/RoPE/Attention/O/残差/MLP/SwiGLU/down_proj 顺序均保留。
- Teacher / Student 残差支路、主数据流及 loss 汇聚连接保留。
- 新旧 XML 均为 107 个 mxCell；原图和新图均为 47 条 edge；逐一比较后 edge XML 完全相同（`unchanged=True`）。因此 I3 的双因素关系由对应节点公式表达，没有新增或重布原有连线。

## 可见元素总盘点

| 元素范围 | 内容 | 媒介 / 样式 | 状态 |
|---|---|---|---|
| 画布与分区 | 白底、横向页面；上方 Teacher 紫色区、Student 蓝色区、右侧灰色损失区；下方 Teacher/Student 层结构区 | 原图 Draw.io 原生容器 | accepted |
| 顶部标题与分区标题 | 总标题、Teacher、Student、loss 标题 | 原生文本，沿用靛蓝/紫/蓝/灰字号与颜色 | accepted |
| Teacher 顶层节点 | 输入、Embedding、局部 Teacher 解码层、Final RMSNorm、LM Head、局部 Teacher 分布 | 原生圆角方框；原位置与色彩 | accepted |
| Student 顶层节点 | forget/retain 输入、Embedding、标准 Student 解码层、Final RMSNorm、LM Head、forget/retain 分布 | 原生圆角方框；原位置与色彩 | accepted |
| 损失节点 | I3 混合 forget KL、retain NLL、总损失、Student 反向更新 | 紫/蓝/琥珀/绿原生方框 | accepted |
| Teacher 单层结构 | 输入隐状态、RMSNorm、QKV、RoPE、Attention hook、Concat/O、两残差、MLP、SwiGLU、down_proj hook、层输出 | 紫色原生方框；Attention 与 down_proj hook 文字更新 | accepted |
| Student 单层结构 | 与 Teacher 对应的标准 LLaMA 层模块 | 蓝色原生方框；无 patch、无局部温度 | accepted |
| 候选差异框 | Attention hook 示例的红色虚线高亮 | 原生虚线框与文本，保留原定位 | accepted |
| 全部连接器 | 上下游主链、分布到损失、loss 汇聚、Teacher/Student 残差支路 | 原生实线/虚线、正交路由和原箭头样式 | accepted |
| 颜色/字体/间距 | 原图浅紫 Teacher、浅蓝 Student、浅绿分布、浅琥珀总损失和原有字体层级 | 原样延续；公式节点局部字号缩小以适配原框 | accepted |

## 验证与已知边界

- `check_drawio.py` 通过：107 cells、58 vertices、47 edges、0 image/SVG cells、51 text cells；无 checker warning。
- PNG 已重新导出，并按全图及放大局部目视检查：Teacher 插值公式、Attention hook、down_proj hook、I3 KL/β 公式无框外溢或重叠；原节点几何与全部 47 条连接关系保持。
- 图是流程链路图，不是实验结果图。I 的 S/ρ 和 K1 的 β 均未在真实模型上验证；E0 `do_not_start_e1` 仍成立。
- 为服从“不改原连接关系”，没有新增 Student rollout 回流 Teacher 的箭头；rollout 来源、同一 prefix 和混合权重在 `sMiddle`、`tPred` 与 `lossF` 公式中明确表达。读者若希望显示显式回流边，需要另行授权修改连接关系。
- 原图为超宽画布，整图预览缩略后字号较小；公式建议在 Draw.io 放大阅读。局部框在源尺寸下经放大审核可辨。

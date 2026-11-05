# 方向 I · 稀疏因果定位与局部 ASU 全链路图审计

- 目标文件：`方向I_MCSD-ASU_遗忘全链路实现图.drawio`（在用户指定原文件内更新）
- 预览：`方向I_MCSD-ASU_遗忘全链路实现图.png`
- 画布：Draw.io 原生可编辑；页面参数 2900 × 1020；导出图 2859 × 880
- 当前标题：方向 I · MCSD-ASU：稀疏多组件因果定位 → 局部 Teacher
- 状态：图中局部 patch / 稀疏搜索 / 局部 Teacher / Student 链均是候选新版本，尚未实现、未预注册。

## 变更摘要

1. 删除旧版“上下文压缩—压缩质量门”作为 TOFU 当前主链步骤的表述。TOFU 没有独立上下文字段，当前首版不做上下文压缩。
2. 将 Teacher 前向明确改为：匹配事实/反事实输入 → 冻结基座的双路干净前向并缓存 donor 激活 → 固定评分函数 → 在指定组件与 token 位置做 activation patch、重算下游并计算 N/R → 稀疏选择路径集合、二值重算 → 仅对过门集合做局部 rho 插值并生成 Teacher 分布。
3. 显式区分因果定位 patch 与后续 Teacher 插值：前者做完整 donor 替换、估计必要性/救援；后者只在通过定位门的集合上按 rho 混合激活。
4. 锁定候选 hook 口径：Attention head 输出、concat 前；MLP 的 down_proj 输出、残差相加前。两类激活张量形状不同，候选搜索及比较需按模块类型分层/配额，不把未归一化的 mask 直接当作同尺度贡献。
5. 补入 score、N/R、KD 与 masked retain NLL 数学符号；状态条明确 E0 的 `do_not_start_e1` 不变，新版本的 retain 阈值须另行预注册，不得复用 E0 阈值。
6. 移除旧图中无语义的“任务向量 / mix-style”标签；保留 ASU Student 主链和下方单层结构示意，并明确 Student 未训练。

## 当前实验事实与边界

- 方向 I E0 使用 32 层 `locuslab/tofu_ft_llama2-7b-chat`，在 TOFU forget05 task 1 与 retain50 上测试全层统一 Attention 温度或 MLP gate 温度。
- 六个单通道温度点均超过 E0 预注册 retain ΔNLL 上限；E0 判决为 `do_not_start_e1`。T11、activation patching 与 Student E1 均未运行。
- E0 结果只否定该基座、该数据、该全层温度网格及原 retain gate 的配置；不证明局部/稀疏 patch 不可行。
- 本图不是结果图。所有新路径、评分、阈值与 Teacher 插值都须视作新实验版本；PGB-CT 只启发候选稀疏选择器，稀疏搜索本身不是因果证据。

## 数学口径核对

- 答案分数：`s(q;y)=|y|^-1 Σ_t log p(y_t | q,y_<t)`，采用 teacher-forced 答案位评分。
- 若 foil 答案有效，`g(q)=s(q;y+)-s(q;y-)`；若反事实答案不可靠，退化为 `s(q;y+)`，并降低因果解释力度。
- 对候选集合 S 联合 patch、重算下游：`N_S=g(q+)-g(P_S(q+ ← A-))`；`R_S=g(P_S(q- ← A+))-g(q-)`。图中 N/R 作为待估量，不代表已运行或已通过。
- 最终二值 S 必须重算；随机路径与 retain 对照、按作者聚类的不确定性估计属于验证要求。作者数少时不得按 QA 行数伪增独立样本量。
- Teacher 插值：`a_j^(rho)=(1-rho)a_j+ + rho a_j-`，仅对 `j∈S` 应用；其余路径保持标准前向。`p_T` 冻结后供 Student 的目标答案 token KL 使用。
- Student 候选损失为答案位均值 KL 加 masked retain NLL；仅更新 `theta_S`。retain 预算是独立硬门，不应用加权总损失替代。

## 可见元素清单与视觉检查

| ID / 区域 | 内容与媒介 | 样式 / 几何 | 状态 |
|---|---|---|---|
| canvas | 白色页面、无纹理/网格；2900×1020 页面，导出裁边 2859×880 | 横向科研流程图 | accepted |
| title | 方向 I / MCSD-ASU 与候选状态 | 原生文本，靛蓝居中标题 | accepted |
| statusBanner | E0 判决、新方案未实现/未预注册、新 retain 阈值另定、失败即停止 | 浅琥珀横条，原生文本 | accepted |
| teacherPanel / teacherTitle | 冻结 Teacher：双路干净前向→稀疏定位→局部插值 | 淡紫容器、紫色标题 | accepted |
| tInput | 匹配事实/反事实对；目标事实最小变化；模板/token 对齐；答案仅评分 | 紫色原生节点 | accepted |
| tEmbed | 基座冻结，q+/q- 两次干净前向并缓存 a+/a- | 灰色原生节点 | accepted |
| tMiddle | 长度归一化 score、foil 有效时的 margin 与 foil 失效回退 | 琥珀原生节点 | accepted |
| tFinal | 固定候选单位 j=(u,k)，区分 Attention head 与 MLP 输出 hook | 灰色原生节点 | accepted |
| tHead | 完整 patch 与下游联合重算、N/R、软 mask 候选稀疏选择、最终二值重算、对照和作者聚类 | 灰色原生节点；N/R 字号提高 | accepted |
| tPred | 仅对通过集合 S 做 rho 插值；其他路径照常；q+ 前向生成 pT 后接 ASU KL | 浅绿原生节点 | accepted |
| t0/t1/t4/t5/t6 | Teacher 输入、缓存、评分、hook、定位至 Teacher 插值的实线顺序箭头 | 紫色原生正交连接器 | accepted |
| studentPanel / studentTitle | ASU Student 标准前向；E0 gate 下未启动；新版本仅在 gates 通过后训练 | 淡蓝容器、蓝色标题 | accepted |
| sInput/sEmbed/sMiddle/sFinal/sHead/sPredF/sPredR | DF/DR 输入到 embedding、32层基座参考、最终 norm/head、forget/retain 输出 | 蓝色/中性/浅绿原生节点 | accepted |
| lossPanel | ASU 损失与更新区域 | 浅灰圆角容器 | accepted |
| lossF | 冻结 Teacher 对 Student 的答案 token 平均 KL | 浅紫原生节点 | accepted |
| lossR | 忽略 -100/padding 位的 retain masked NLL | 浅蓝原生节点 | accepted |
| total/back | alpha L_F + L_R；仅更新 theta_S；训练与 FE/MU/恢复攻击仍未运行 | 琥珀/浅绿原生节点 | accepted |
| tDInput–tDOut、tDF0–tDF11 | Teacher 单个 DecoderLayer：RMSNorm/QKV/RoPE/Attention/O/残差/MLP/SwiGLU/down_proj；候选 hook 与下游计算顺序 | 紫色原生模块和实线箭头 | accepted |
| Attention 虚线框 | 候选 attention head 输出区域，patch hook 在 concat 前；非已定位路径 | 青绿色虚线框 | accepted |
| MLP 虚线框 | 候选 down_proj 输出区域，patch hook 在残差前；非已定位路径 | 青绿色虚线框 | accepted |
| sDInput–sDOut、sDF0–sDF11 | Student 对应标准 DecoderLayer 前向，无温度干预或激活替换 | 蓝色原生模块和实线箭头 | accepted |
| tSkip1/tSkip2/sSkip1/sSkip2 | Teacher 与 Student 两条残差支路 | 紫/蓝虚线箭头 | accepted |
| detail headers | 单层结构提示、E0 32层基座说明、候选 hook 与待预注册提醒 | 原生文本 | accepted |
| removed legacy labels | 删除“上下文压缩质量门直入 DecoderLayer”及无语义孤立标签 | 不再显示 | accepted |

## 技术与视觉验收

- `check_drawio.py`：通过；109 cells、61 vertices、46 edges、0 image/SVG cells、52 text cells。
- Draw.io 已重新导出同名 PNG；按全图及局部放大图核对标题/状态、N/R 公式、rho 框、损失框、hook 边界与两条 DecoderLayer 主链。
- rho 框初检发现底部文字溢出，已缩短并重新导出；最终预览中公式、rho 网格和 `q+`→`p_T`→ASU KL 提示均位于框内。
- 图中所有结构和文字均为 Draw.io 原生元素，无截图或 SVG 嵌入；虚线框仅表示候选组件族。
- 完整预览很宽，缩略图查看时字号显小；建议在 Draw.io 中放大阅读公式。未发现遮挡或边界溢出；实验阈值、路径选择及实现 hook 仍待新协议确定。

# ASU 前向链路与逐层结构图审计

## 目标与版面

- 目标源文件：`ASU_遗忘全链路实现图_代码实证版v2.drawio`
- 画布：2900 × 1020，浅色背景；标题、副标题置顶。
- 上半区：Teacher 与 Student 两条从左到右的模型总览；每条主链以单个节点表示 16 层 DecoderLayer 堆叠；右侧为局部损失、总损失与统一反传。
- 下半区：Teacher 与 Student 分别展示一个 DecoderLayer 的内部结构，明确该拓扑逐层重复应用于全部 16 层；虚线表示两条残差旁路。
- 主要图形均由 Draw.io 原生节点、文本和连接线构成；无截图裁切或外部图标。

## 可见元素清单

| ID / 区域 | 内容 | 形式 | 状态 |
|---|---|---|---|
| title / subtitle | ASU 16 层模型的 Teacher/Student 前向与损失链路说明 | 原生文本 | 完成 |
| Teacher 总览 | 遗忘集 → Embedding → LLaMA DecoderLayer 堆叠（共16层）→ Final RMSNorm → LM Head → Teacher预测分布 | 原生节点与箭头 | 完成 |
| Student 总览 | 遗忘/保留集 → Embedding → LLaMA DecoderLayer 堆叠（共16层）→ Final RMSNorm → LM Head → 两类预测分布 | 原生节点与箭头 | 完成 |
| 层数表达 | 全部16层以同一个堆叠节点表示；未将层划分为浅/中/深段 | 原生节点与文本 | 完成 |
| Teacher 单层结构 | 输入隐状态 → RMSNorm（归一化式）→ Q/K/V 线性投影 → RoPE 旋转 → 带温 FlashAttention2 → O 投影 → 残差① → RMSNorm → Gate/Up 投影 → SwiGLU → Down 投影 → 残差② → 输出隐状态；适用于全部16层，各层参数独立 | 原生节点与箭头；公式置于对应组件节点 | 完成 |
| Student 单层结构 | 与 Teacher 相同拓扑及各组件公式；全部16层的 Attention 均不传温度参数；各层参数独立 | 原生节点与箭头；公式置于对应组件节点 | 完成 |
| 总览公式 | Embedding：$h^{(0)}=E[ids]$；Final RMSNorm；LM Head：$Z=zW_{LM}$；预测分布：$p=softmax(Z)$ | 原生节点内公式 | 完成 |
| 损失与反传 | 保留原 KL、NLL、加权总损失和 Student 梯度公式 | 原生节点与连接 | 完成 |
| 残差支路 | 输入隐状态绕行至残差①；残差①输出绕行至残差② | 原生虚线连接 | 完成 |
| 局部损失 | Teacher 与 Student 遗忘预测分布输入 KL 遗忘损失；Student 保留预测分布输入保留 NLL | 原生节点与连接 | 完成 |
| 优化汇总 | αL_F 与 L_R 汇入总损失，再进入统一反向传播，仅更新 Student | 原生节点与连接 | 完成 |
| 说明文字 | DecoderLayer 同一拓扑逐层重复、参数独立；Teacher 全层使用 τ，Student 全层不使用 τ | 原生文本 | 完成 |

## 视觉与结构核验

- `check_drawio.py`：通过结构检查，107 cells（58 vertices / 47 edges），无图像或 SVG 单元；有 1 条原始红色“唯一区别”标签的 raw HTML/font 样式警告（非结构错误，预览正常）。
- PNG 预览：已由当前 `.drawio` 源文件重新导出。
- 分区目检：总览将 16 层显示为单个 LLaMA DecoderLayer 堆叠节点；下方单层结构说明适用于全部层；Teacher/Student Attention 公式差异和作用范围明确。
- 公式覆盖：Embedding、RMSNorm（两处）、Q/K/V、RoPE、Teacher/Student Attention、O投影、两处残差、Gate/Up、SwiGLU、Down、层输出、Final RMSNorm、LM Head 与预测 Softmax 均在对应组件节点内呈现；损失公式保持原位。
- 迭代修复：扩展逐层节点宽度与高度并重新布置残差虚线路径；将 LM Head 的 logits 公式与输出节点的 Softmax 公式分开，确保各公式留在对应框内。
- 视觉复核：全图、Teacher/Student 分区和关键节点放大检查均通过；未观察到公式遮挡、裁切、重叠或箭头误接。

# 方向 I · Prompt-only Activation Donor 构造规格 v0.3

> 日期：2026-11-09。状态：方案修订稿，尚未预注册、未运行完整模型实验。
> 本文替代 STR 配对规格 v0.1 与 donor 扩展规格 v0.2 中关于 donor 必须提供 `y−`、双答案差分及 K2 必过的要求。旧版本保留为历史记录；本版不改变 ASU 的激活插值、Teacher/Student、G0–G6 等其他设计，除非本文件明确说明。
> 目标模型基准：`open-unlearning/tofu_Llama-2-7b-chat-hf_full`；使用该 checkpoint snapshot 自带 tokenizer。

## 1. 核心修订

方向 I 的 donor 侧用于提供一个可控的 prompt-only 激活参考：对外部 donor 问题 `q−` 做冻结模型前向，缓存指定位置/组件的激活 `a−`，再按既定方案将 `a−` 插入 clean recipient 的计算路径。

因此，**提取和使用 `a−` 不要求将 donor 答案 `y−` 输入模型，也不要求把 `y−` 作为配对文件的必填字段**。模型回答 `q−` 的内容不参与 `a−` 缓存。

这改变了 v0.1 的 STR 双真值配对口径：本版称为 **prompt-only activation donor**，不再称为严格 STR / PAIREDFACTS 复现。旧规格中的答案差分指标 `logit(y+) − logit(y−)` 不适用于本版。

## 2. 数据字段

每个候选配对至少保留：

| 字段 | 含义 | 要求 |
|---|---|---|
| `pair_id` | 配对唯一标识 | 必填 |
| `q_plus` | clean 侧遗忘目标 probe | 必填 |
| `y_plus` | clean 侧经核验的目标答案 | 必填，用于评估目标事实保留/遗忘 |
| `q_minus` | donor 侧 prompt | 必填，用于提取 `a_minus` |
| `clean_cluster_id` | clean 作者/事实簇 | 必填，统计时按簇聚合 |
| `donor_id` | donor 实体簇 | 必填，记录复用与去重 |
| `relation_id` | 关系标识 | 必填，原则上与 clean 侧对齐 |
| `donor_source` | donor 实体和事实的可追溯来源 | 必填，支持人工核验及去重 |
| `y_minus` | donor 的答案/事实值标签 | **非模型输入、非激活缓存必需字段**；可作为 source metadata 单独保存，不参与评分或损失 |

若上游知识库有 donor 三元组 `(entity, relation, value)`，建议把 `value` 保存为 `donor_fact_value`，仅供事实来源核验、关系对齐和文本去重。除非另行注册答案评分诊断，不把它称为训练目标 `y_minus`。

## 3. Prompt 与激活缓存

1. `q_plus`、`q_minus` 使用同一关系模板和相同 chat 包装规则；donor 侧 prompt 必须自然、语法完整，并来自可追溯的真实实体/关系记录。
2. 用统一 checkpoint 与 tokenizer 分别编码完整 prompt。记录完整 token IDs、实体 span、目标 hook 位置、层/组件、长度与模板哈希。
3. 分别对 prompt-only `q_plus`、`q_minus` 前向，缓存 `a_plus`、`a_minus`。不得把答案 token 或 completion 拼进 donor prompt 来构造 `a_minus`。
4. 若方法要求逐位置 patch，则按预先规定的相对位置/实体 span 映射规则对齐；不满足映射条件的样本记录排除原因。是否要求两侧完整 token 序列等长，取决于实际 patch 坐标方案，应在预注册中明确；不得把等长硬门默认当作所有 prompt-only patch 的理论要求。
5. 缓存 key 至少包含 checkpoint/tokenizer/template 哈希、prompt 哈希、hook 位置与版本。Donor 激活冻结，且不得对 donor/base 权重反向传播。

## 4. donor 真值与基座已知性

- donor 的外部事实仍须可追溯、经来源核验，且关系类型与 clean probe 可解释地匹配。这保证实验描述的是“某类外部事实 prompt 的激活 donor”，而不是随机文本激活。
- **基座 top-1 K2 不再是主流程准入门**，因为本版不使用 `y_minus` 做答案差分，也不把 donor 回答作为目标。
- 若研究问题需要声称 donor 是“基座已知知识”，则必须另做已知性诊断并报告；可采用多 token 序列评分等适配规则，但需独立校准、预先冻结。没有该诊断时，只能称为“来源可核验的外部 donor prompt”，不能声称模型已知该事实。
- donor 实体及别名仍须与完整评测清单按实体簇去重。常见城市名在评测文本中出现，不等同于 donor 人物泄漏；应核查人物、别名和具体事实。

## 5. 评分与对照

本版移除 `logit(y+) − logit(y−)` 作为主定位指标。主评分必须以 `q_plus` 和目标答案 `y_plus` 定义，例如预注册的 `y_plus` 答案 span 条件 log-prob / logit 得分，并比较 clean recipient 与 donor-patched recipient：

`effect = score(q_plus, y_plus | clean path) − score(q_plus, y_plus | donor-patched path)`

答案 span、聚合方式（token mean/sum）、prompt 位置、patch 集合、方向和归一化方式必须在主实验前冻结。该量表示 donor 激活 patch 对目标答案支持度的影响，不单独证明事实已被遗忘；最终遗忘仍由独立 forget/retain/恢复评测判断。

至少设置同预算对照：

- `a_plus` 或不干预基线；
- donor 实体错配/置换的 `a_minus`；
- 匹配位置与范数的随机或 matched-null 激活；
- 如保留，分布内 donor 对照臂单独报告。

若 `a_minus` 仅产生普遍退化、拒答、错误实体或 retain 损伤，不能解释为目标知识选择性遗忘。

## 6. K2 远程预筛报告如何解读

此前远程报告中 K2 为 `blocked`，原因是没有独立 calibration 集；模型没有加载、没有前向打分。因此那份报告没有证明 donor 被模型拒绝。报告发现的 `y_minus` 与 `donor_answer` 不一致，在本版中不阻断 prompt-only `a_minus` 的提取；但用于来源核验的 donor fact metadata 仍须正确。旧 v0.1 文件保留用于复现，不作为本版运行输入。Beyoncé 与虚构名字的字符串碰撞仍应按实体去重策略处理或替换。

## 7. 运行前检查清单

- [ ] 固定使用 `open-unlearning/tofu_Llama-2-7b-chat-hf_full` 的准确 snapshot、配套 tokenizer 与 chat template。
- [ ] 生成新版本配对清单，确保 `q_minus`、donor ID、来源和 `relation_id` 一致；`y_minus` 不作为必需训练/评分字段。
- [ ] 审核 clean 侧 `q_plus/y_plus` 来源；审核 donor 来源三元组及人物、别名去重。
- [ ] 冻结 token span/位置映射规则、激活 hook、缓存格式、patch 方向与对照臂。
- [ ] 冻结 `y_plus` 评分 span 和 aggregation；不再计算 donor 答案差分，除非另立诊断。
- [ ] 缓存 prompt-only `a_plus/a_minus` 并核验重复运行数值一致性。
- [ ] 完成 matched-null、错配 donor 和主动 retain 审计，再判断是否进入后续 Teacher/Student 阶段。

## 版本关系

- v0.1：严格 STR 双真值历史方案，保留原文，不作为本版规范。
- v0.2：外部通用知识 donor 扩展历史方案；其 K2 与 `y_minus` 硬门由本版替代。
- v0.3：prompt-only activation donor 当前修订稿；只改变 donor 答案依赖、K2 角色和定位评分口径，其余 ASU 流程须按对应主方案继续执行。

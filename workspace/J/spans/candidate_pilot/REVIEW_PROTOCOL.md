# AI 交叉验证定稿协议

仓库默认只保留 `final_spans.jsonl`。`proposed.jsonl` / `independent_ai_crosscheck.jsonl` / `blind_*` 等中间产物已从树中删除；需要复现时再由脚本生成，且不得覆盖已锁定的 `final_spans.jsonl`。

## 标注层身份

| 文件 / 层 | 身份 |
|---|---|
| `proposed.jsonl`（可重生成） | AI 预标注 v1 |
| `independent_ai_crosscheck.jsonl`（可重生成） | 独立 AI 交叉标注，只挖分歧，不是人类第二标注者 |
| `blind_adjudication.jsonl`（可重生成） | 另一个模型对高优先分歧的盲标；它只看 question 和 answer |
| `final_spans.jsonl` | 唯一保留的 AI 定稿 sidecar；`align_m_tar.py` 只读这一份 |

中间 AI 交叉层 **不是**第二位人类标注者。AI–AI 一致率只作描述统计，不能写成双人一致性研究，也不能把同一次模型复查当作第二标注者。盲裁确认也不是人类审核。

## 定稿步骤

1. `compare_annotations.py` 给每行一个类别。`merge_ai_final.py` 按类别合并，不重跑 `build_final_spans.py`。
2. `exact_match` 直接定稿，包括双方同为 `unresolved` 或 `no_target` 的行。active 行复制双方相同的全部 span。双方都拒绝时写成对应的空 span 终态。`reviewer` 为 `ai_consensus`，`source_of_spans` 为 `ai_consensus`。
3. `relation_only` 只复制两边相同的 `fact_value`。relation 不一致，不挑选任一侧。`disagreement_with_crosscheck` 为 true。
4. 其余高优先分歧（granularity、content_diff、status_diff，以及同类 fact 不一致）交给没看过两份标注的模型盲标。不把 relation-only 送去重标。
5. 盲标为 `unresolved` 或 `no_target` 时，最终采用该状态且 span 为空。盲标 `fact_value` 与某一侧完全一致时，采用该侧 `fact_value`；relation 仅在该侧与盲标 relation 也一致时复制。与两侧都不一致，或字符自检失败时，整行 `unresolved`。不把第三套对不上的 span 写成事实。
6. 盲裁的 `reviewer` 为 `blind_adjudicator:kimi-k3-max`。只确认一侧时 `source_of_spans` 为 `proposed` 或 `crosscheck`；两侧 fact_value 都被确认时为 `ai_consensus`。

`fact_value` 只保留直接回答本题的事实值。relation 可以出现在定稿里，但 **不进入** 第一版主 `m_tar`。空的 `adjudicated` 不算完成。

## 定稿 schema

`final_spans.jsonl` 每条至少含：

- `dataset_status`: `local_candidate_pilot`
- `canonical_forget05_task_id_1`: false
- `candidate_row_index`
- `question` / `answer`：与候选原文逐字相同
- `review_status`: `adjudicated` / `unresolved` / `no_target`
- `reviewer`: `ai_consensus` 或 `blind_adjudicator:kimi-k3-max`；不是人类姓名
- `source_of_spans`: `ai_consensus` / `proposed` / `crosscheck`
- `target_spans`: 最终字符 span 列表；`unresolved` / `no_target` 时为空列表
- `disagreement_with_crosscheck`: 布尔。最终 fact 或 relation 与 crosscheck 不同时为 true
- `review_note`: 字符串

span 仍须通过 `span_match.py`：原文唯一子串，且 `answer[start:end] == text`。同一行 span 不得重叠。

## 一致性阈值

人类审核的 span 边界 / token 通过阈值必须在查看该轮人类结果之前确认。本轮没有人类定稿。已看到的 AI–AI 分歧率 **不得** 事后回填为通过阈值。`agreement_stats.json` 把 AI–AI 描述计数、AI 定稿计数和人类定稿进度分开存放。人类计数字段保持 0，`second_annotator` 保持 null。

## 训练 mask

即使 `final_spans.jsonl` 全量成为真实 AI 终态，候选 pilot 仍不是已核实的 canonical `forget05/task_id=1`，`align_m_tar.py` 继续硬阻断，不写 mask。字符位置 `j` 上的 gold label 由 logits 位置 `j-1` 预测，但该映射现在不得执行。DARD-ASU v2 只消费定稿后的 `fact_value` mask；relation 与未定稿 token 留在 `m_other`。本协议不改变 M0 NO-GO，也不启动训练。

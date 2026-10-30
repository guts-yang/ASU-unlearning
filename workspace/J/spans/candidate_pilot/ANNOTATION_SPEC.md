# 候选 QA 事实 span 标注规范

状态：已冻结，适用于 `dataset_status=local_candidate_pilot`。本规范不把候选文件认定为 ASU canonical `forget05/task_id=1`。

## 遗忘范围

主标签只覆盖本题 question 直接询问的事实值。答案里同一作者的其他事实不进入 `fact_value`，也不进入第一版主 `m_tar`。

## 两层 span

- `fact_value`：直接承载本题事实值的最小完整原文片段。多个直接答案值分别成 span。
- `relation`：连接该事实值的关系表达，单独保存。它不与事实值重叠，也不进入第一版主 `m_tar`。
- 主语、模板、功能词、外层引号或书名号，以及不回答本题的额外陈述，不标为任一 span。
- 作者名只有在本题询问该姓名时才标为 `fact_value`。
- 不调用外部知识，不纠正 answer 中的日期、名称或其他内容。

## 状态

- `proposed`：AI 预标注，且每个子串在该 answer 中恰好匹配一次。
- `no_target`：答案没有可识别的事实内容。
- `unresolved`：子串缺失、重复、边界不清，或事实范围无法由 question 和 answer 决定。脚本不得猜测另一处 offset。

## Offset

`start_char` 和 `end_char` 是原始 answer 的零基半开区间。`text` 必须等于 `answer[start_char:end_char]`，并且不含首尾空白。字符 span 先保留；只有全部人工审核完成后，才允许用 ASU 实际 tokenizer 和 `logits[:, :-1]` 对 `labels[:, 1:]` 的 shift 生成 token mask。

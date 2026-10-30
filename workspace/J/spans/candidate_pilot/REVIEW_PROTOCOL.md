# 人工审核协议

`proposed.jsonl` 是 AI 预标注，全部 200 条的 `review_status` 都是 `pending_human_review`。审核结果只能写入 `review_queue.jsonl` 的后续副本或单独 adjudication 文件，不能覆盖 `proposed.jsonl`。

## 当前限制

只有一个 AI 预标注来源。没有第二位独立标注者，因此不能计算双人一致性，也不能把同一次模型复查当作第二标注者。

## 审核步骤

1. 审核者逐条核对 question、完整 answer、字符边界和事实范围。
2. `fact_value` 只保留直接回答本题的事实值；`relation` 单独保留。
3. 不一致或边界不清的条目保持 `unresolved`，进入 adjudication。
4. 条件允许时，第二位标注者独立复标一个分层抽样子集，再仲裁分歧。

## 尚未确认的一致性阈值

span 边界一致性和 token-level 一致性的通过阈值必须在查看审核结果之前确认。当前 `agreement_stats.json` 中的相应数值保持为空，不根据这份预标注挑选阈值。

## 训练 mask

在全量人工审核完成、ASU 实际 Llama-2 tokenizer 可用，并且无法对齐的条目已复核之前，不生成 `m_tar`。字符位置 `j` 上的 gold label 由 logits 位置 `j-1` 预测，但该映射现在不得执行。

# `review_adjudication.jsonl` 已废弃

该文件不是合法 JSON（`reviewer` 未加引号），且 `adjudication` / `target_spans` 为空。  
不得用它放行 `m_tar`，也不得把它当作人类定稿。

唯一人工定稿 sidecar 是 [`final_spans.jsonl`](final_spans.jsonl)。用 [`build_final_spans.py`](build_final_spans.py) 生成待审空壳后填写。

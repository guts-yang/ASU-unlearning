# 方向 I Prompt-only Activation Donor 数据候选 v0.3

本目录按 [`方向I_prompt_only_activation_donor构造规格_v0.3.md`](../../方向I_prompt_only_activation_donor构造规格_v0.3.md) 构造。状态：候选数据，尚未经过远程 tokenizer、激活提取或干预验证。

## 数据内容

- 10 个 clean 虚构作者簇，来自官方 `I/data/forget05.jsonl`；记录 `q_plus`、目标答案 `y_plus` 和原始 QA 来源行。
- 10 个唯一外部人物 donor，`q_minus` 使用相同出生地问题模板。donor 可溯源至 Wikidata 出生地属性 P19。
- 排除 Beyoncé：远程预筛曾发现 TOFU 虚构作者 “Beyonce Johnson” 的姓名字符串碰撞；改用 Muhammad Ali。新 donor 的远程全评测集重叠扫描仍待完成。
- 配对文件不包含 `y_minus` 字段。`donor_fact_value` 仅用于来源和关系核验，不作为模型输入或评分目标。
- 本地检查未发现这 10 个 donor 姓名命中构造时已有的 `I/data/*.jsonl`；这不能代替服务器上的最终评测实体、别名扫描。

## 文件

- `candidate_pairs.jsonl`：配对问题、clean 目标答案及来源信息。
- `clean_facts.jsonl`：clean 侧源数据证据。
- `donor_entities.jsonl`：donor 实体、P19 值和 Wikidata 来源。
- `manifest.json`：版本、计数、哈希和远程验证门槛。
- `REMOTE_VALIDATION_TODO.md`：远程校验计划和可复制给 AI 的指令。

## 限制

尚未确认实际 checkpoint/tokenizer 下的实体 span、位置映射、激活可重复性、patch 可行性或模型已知性。通过远程检查前，不应进入正式定位或 Teacher/Student 训练。

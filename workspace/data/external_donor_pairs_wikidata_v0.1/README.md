# 外部 donor 反事实配对候选 v0.1

状态：**候选草案，尚未完成远程模型验证**。本目录是从 `../forget05.jsonl` 派生的独立文件，不会修改源数据集、`manifest.json` 或现有评测文件。

## 构造方式

- **Clean 侧**：从官方 TOFU `forget05.jsonl`（200 条）中选取 10 个虚构作者簇。出生地事实及其原始 QA 来源行记录在 `clean_facts.jsonl`。
- **Donor 侧**：从 Wikidata 整理 9 条真实人物出生地事实，统一使用 `place of birth` 关系（属性 P19）。每个 donor 的 Wikidata QID 和来源 URL 记录在 `wikidata_donors.jsonl`。人物涵盖作家、数学家、科技企业家、歌手和活动人士，因此外部事实不局限于作家。
- **配对问题**：`q_plus` 询问 clean 侧虚构作者的出生地，`q_minus` 用相同关系询问外部 donor。两侧答案均规范为城市名。这些是辅助探测问题，不会改写原始 QA。
- **评测重叠筛查**：对当时已有的所有本地 `I/data/*.jsonl` 文件做了标准化姓名子串搜索，没有发现 donor 姓名命中。由于只有 9 个唯一 donor 簇，第 10 对复用了 donor-01；该行标记为 `donor_reuse: true`，不能视为独立 donor 簇。

## 远程验证门槛

本地候选文件**不能证明基座模型知道这些 donor 事实**。实际使用前，需要在远程服务器上用目标 checkpoint、准确 tokenizer 和冻结的 chat template 逐条验证，并记录：

- 答案的 token 数和答案 token span；
- 答案是否为单 token；
- 严格模板下答案是否为 top-1；
- 答案 logit、概率、最强竞争答案及二者 margin。

K2 阈值应由独立 calibration 集确定并冻结。不要根据主实验结果调整阈值。候选问题目前只是普通文本，尚未序列化为 chat prompt；远程预处理时必须使用目标模型的实际 chat template，并对 `q_plus` 和 `q_minus` 做一致处理。所有 tokenizer、K2 和人工核验状态都明确标记为待完成。

## 范围和限制

- 本候选集只使用 `place_of_birth` 关系；没有完成对 200 条数据中所有潜在关系的全面审计。
- 当前有 10 个 clean 簇和 9 个唯一 donor 簇，其中一个 donor 被复用。统计时应按 clean 簇聚合，并单独报告 donor 簇及复用情况。
- 城市名答案可能比 Wikidata 中更具体的出生地实体更宽泛。人工核验时需确认源陈述支持该城市级答案，并确认模型问题能稳定引出相同粒度的回答。
- 姓名筛查仅覆盖构造时本地已有的 `I/data/*.jsonl`。训练或评测前，应针对远程服务器上的准确评测清单重新检查实体、别名和近邻实体。
- 本目录不包含激活缓存或模型输出。

## 文件说明

- `clean_facts.jsonl`：clean 侧事实、原始 QA 证据和来源行。
- `wikidata_donors.jsonl`：外部 donor 事实及 Wikidata 来源信息。
- `candidate_pairs.jsonl`：对齐后的 `q_plus / q_minus` 探测问题和候选答案。
- `relation_audit.json`：关系计数、源文件哈希和待完成的验证门槛。

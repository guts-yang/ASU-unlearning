# 远程服务器校验计划与 AI 指令

## 校验顺序

1. **传输与校验**：将整个目录复制到服务器独立位置，不覆盖 TOFU 或任何评测文件。核对 JSONL 行数、manifest 和 `forget05.jsonl` SHA-256。
2. **锁定环境**：基准统一为 `open-unlearning/tofu_Llama-2-7b-chat-hf_full`。记录 snapshot commit、模型/tokenizer 哈希、推理代码 commit、设备、精度及实际 chat template。
3. **远程评测去重**：针对本次真实使用的 forget/retain/world_facts/real_authors/holdout 清单，扫描 donor 人物、别名、近邻人物和事实簇。区分实体命中与常见城市字符串命中。冲突 donor 排除并记原因。
4. **来源审核**：在线核对每个 Wikidata QID、P19、城市粒度及来源；核对 clean 源行与 `y_plus`。不确定项标记为拒绝/待审，不自动猜测。
5. **Tokenizer 与位置映射**：用真实 tokenizer 和最终 chat template 编码 `q_plus/q_minus`，保存 token IDs、实体字符/token span、prompt 长度、template hash。按实际 hook/patch 算法检查位置映射；不为通过而 padding 或截断。
6. **激活 smoke test**：对少量通过项，仅输入 prompt，提取 `a_plus/a_minus`。不得追加 donor 答案或 completion。记录 hook、层、位置、shape、dtype 和 cache key；重复运行检查数值可复现。
7. **Patch 对照**：比较 no-op/clean baseline、匹配 donor、shuffle donor、matched-null；用预冻结的 `y_plus` span score 评估目标事实支持度，同时监测错误实体、拒答和 matched retain probes。不要算 `logit(y_plus)-logit(y_minus)`。
8. **Go/no-go**：逐门报告候选数、通过数、失败原因、clean 与 donor 簇数。若来源、评测去重、位置映射、缓存复现或 matched-null 对照不通过，暂停该臂，不进入 Teacher/Student 训练。

## 可复制给服务器 AI 的指令

```text
请按“方向 I Prompt-only Activation Donor 构造规格 v0.3”校验 external_donor_prompt_activation_v0.3/。这是数据与激活可行性预检，不是正式实验。

约束：
- 候选数据只读；结果写到新的 *_validation_v0.3/ 目录。不要覆盖源数据、评测文件、旧版本候选或旧验证结果。
- 统一使用 open-unlearning/tofu_Llama-2-7b-chat-hf_full 的准确 snapshot 及其 tokenizer。记录 commit、权重/tokenizer 哈希、代码 commit、设备和精度。
- 这是 prompt-only activation donor：只对 q_minus prompt 前向以提取 a_minus。不要将 donor_fact_value 或 y_minus 拼入 prompt、loss 或评分。y_plus 仍用于目标事实评分。
- 不运行 K2 单答案 top-1 门，不计算 logit(y_plus)-logit(y_minus)。若额外检查基座是否知道 donor 事实，将其作为独立诊断报告，不能称为本版的必过门。

步骤：
1. 检查 JSON/JSONL 格式、行数、唯一 clean/donor 簇和源数据 hash。
2. 对实际评测清单扫描 donor 人物、别名、近邻实体及事实簇；区分人物命中与出生城市字符串命中，并单独核查以前有碰撞的 Beyoncé 不在候选里。
3. 核验所有 donor QID、P19、donor_fact_value 和城市粒度；核验 clean 来源行及 y_plus。生成逐条审计记录。
4. 用目标 tokenizer/chat template 编码 q_plus/q_minus，保存 token IDs、实体字符/token span、长度与模板 hash；检查实际 hook 坐标映射。不要 padding/truncation 凑长度。
5. 对少量通过项运行冻结模型 prompt-only 前向，提取 a_plus/a_minus，不追加 completion。记录 hook、层、位置、张量 shape/dtype、cache key，并重复运行验证缓存稳定。
6. 做小规模 patch smoke：no-op、匹配 donor、shuffle donor、matched-null。冻结并使用 q_plus 上 y_plus 的 span score，记录目标分数、错误实体/拒答和 matched retain probe 变化。donor 答案不参与评分。
7. 输出 validation.jsonl、summary.md 和输出文件哈希，报告各门通过/失败/blocked 数、排除原因、clean/donor 簇数以及 go/no-go。
8. 只有数据来源、评测去重、坐标映射、缓存复现和对照检查通过后，才建议进入正式 activation localization。不要自动开始 Teacher/Student 训练。
```

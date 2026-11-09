# 外部 donor 配对远程预筛验证

## 结论

**NO-GO。**

K2 通过率：**不适用**。10 条配对全部为 `blocked`，0 条被打分，因此没有通过率，也没有模型拒绝样本。

阻断原因：

1. 仓库中没有独立 calibration 集，K2 阈值不能冻结。按照预筛条件，K2 筛选已停止，主候选没有被用来调阈值。
2. `candidate_pairs.jsonl` 有 8 条 `y_minus` 与 Wikidata 城市级答案 `donor_answer` 不一致。本次 `q_minus` 的编码目标使用 `donor_answer`。候选文件未修改。
3. 唯一 donor 簇是 9 个。`donor-01`（Katherine Mansfield / Wellington）被 pair-10 复用。
4. 本次没有加载模型权重，也没有前向计算，所以没有候选答案 logit、概率、top-1、最强竞争答案或 margin。

在补齐独立 calibration 集、修正 `y_minus`，并在 GPU 上用同一 chat template 跑完严格 K2 之前，不开始激活定位或 Teacher/Student 训练。

## 1. 文件核对与哈希

候选目录只读。格式检查：三个 JSONL 文件的每一行都是 JSON 对象；`relation_audit.json` 是一个 JSON 对象。

| 项目 | 文件计数 | relation_audit.json | 一致 |
| --- | ---: | ---: | --- |
| candidate_pairs | 10 | 10 | 是 |
| clean_facts | 10 | — | 与 unique_clean_clusters=10 一致 |
| wikidata_donors | 9 | 9 | 是 |
| unique clean clusters | 10 | 10 | 是 |
| unique donor clusters | 9 | 9 | 是 |
| donor reuse rows | 1 | 1 | 是 |

解析错误：无。

`forget05.jsonl` 行数为 200，SHA256 为 `8f721842a4741fda0e4bfed459c4510c6c00604d5c332a533011648236862e08`。与 relation_audit 的 `rows_scanned=200`、`source_sha256=8f721842a4741fda0e4bfed459c4510c6c00604d5c332a533011648236862e08` 一致。

clean 侧来源行核对：10 条来源行的 question、answer 和城市名都与 forget05.jsonl 一致。

| 文件 | SHA256 |
| --- | --- |
| `candidate_pairs.jsonl` | `7f80de636bf7f583434e867f4f6ce9eda33e66fafceb426c5d073669e2f49611` |
| `clean_facts.jsonl` | `6a84e8a6ca404ef23cbc280ea78253f63d2495bdf6918baa8c66cc47656b9ff3` |
| `wikidata_donors.jsonl` | `d18d23afcf17a2b66776c26a16ae81536a8065f1b249dcb8ff72ccbc510e26b4` |
| `relation_audit.json` | `a3bb7a32b4a136406dbcf153b071a897362c0ddbd6de12689d5ae5af352c645d` |
| `README.md` | `7e0054fba59c2eaf339d6d9b1ba9f5ea84f35e6c3dbdd08c1050f8133adf3504` |

写完结果后重新计算了上述候选文件哈希，与运行开始时相同。

### y_minus 与 donor_answer

| pair_id | donor | donor_answer | y_minus | 一致 |
| --- | --- | --- | --- | --- |
| pair-01 | Katherine Mansfield | Wellington | Wellington | 是 |
| pair-02 | Bill Gates | Seattle | Rio de Janeiro | 否 |
| pair-03 | Beyoncé | Houston | Beirut | 否 |
| pair-04 | Michael Jackson | Gary | Strasbourg | 否 |
| pair-05 | Elvis Presley | Tupelo | Casablanca | 否 |
| pair-06 | Ada Lovelace | London | Vilnius | 否 |
| pair-07 | Malala Yousafzai | Mingora | Mumbai | 否 |
| pair-08 | Paulo Coelho | Rio de Janeiro | New Orleans | 否 |
| pair-09 | Anne Rice | New Orleans | Seattle | 否 |
| pair-10 | Katherine Mansfield | Wellington | Wellington | 是 |

不一致的配对：pair-02, pair-03, pair-04, pair-05, pair-06, pair-07, pair-08, pair-09。

## 2. Checkpoint、tokenizer、推理代码与 chat template

本次标识并用于编码的 checkpoint：

- Hugging Face repo：`open-unlearning/tofu_Llama-2-7b-chat-hf_full`
- snapshot commit：`cc5b31c69127d5da881608e640f2b453c446435b`
- 本地快照：`/root/autodl-tmp/huggingface/hub/models--open-unlearning--tofu_Llama-2-7b-chat-hf_full/snapshots/cc5b31c69127d5da881608e640f2b453c446435b`
- `config.json` 的 `_name_or_path`：`meta-llama/Llama-2-7b-chat-hf`
- 权重未加载。本机 `nvidia-smi` 看不到 GPU，脚本也没有调用 `AutoModelForCausalLM`。

`Right-to-be-forgotten/config/tofu.yaml` 写的 `model_path` 是 `locuslab/tofu_ft_llama2-7b`。该 checkpoint 不在本地缓存中，本次没有把它当作实际 tokenizer 来源。

Checkpoint 文件哈希：

| 文件 | SHA256 |
| --- | --- |
| `config.json` | `d1855d56b6a7ebccb86d1021100d6c893da7d4aac82b376a5b44b95842f04084` |
| `generation_config.json` | `b606e38043e9927fe619589ef8a7545fa5574ae4431a845df82fb1f1b6e2395d` |
| `tokenizer_config.json` | `9a72a826a359b4cc4f84c90e86754b592ba74395b73f91163dd8c283e3ed3e19` |
| `tokenizer.json` | `6386c019c9a2ec0aeaafc7f5ee4c17155e7e27151843d4f9b2d0141aa3e57d5d` |
| `special_tokens_map.json` | `4452b845ab9ca2afa863f59f1411a959e8a32bb4334e264b0e71d5d17dd4dcb9` |
| `model.safetensors.index.json` | `091fdb60b8ee3713df38fd6f9849c3aa82d7eeeb5f6d224a77139543b1db8652` |

与本地 `NousResearch/Llama-2-7b-chat-hf` snapshot `/root/autodl-tmp/huggingface/hub/models--NousResearch--Llama-2-7b-chat-hf/snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee` 的 tokenizer 文件比较：相同文件为 无；不同文件为 `tokenizer_config.json`、`tokenizer.json`、`special_tokens_map.json`。

实际加载的 tokenizer：

- 类：`LlamaTokenizerFast`
- transformers：`4.46.3`
- vocab size：32000；`len(tokenizer)`：32000；
- bos/eos/pad：`<s>`=1，`</s>`=2，`</s>`=2
- padding side：`right`

推理代码版本：

- git commit：`6492440aa94e9a96b032aaa49bc57e742ef7e70e`
- 分支：`I`
- 运行开始时 `git status --porcelain`：`?? workspace/data/external_donor_pairs_wikidata_v0.1_validation/`

编码路径来自 `convert_raw_data_to_model_format`，标签来自 `model_config.yaml` 的 `llama2-7b`。相关文件哈希：

| 文件 | SHA256 |
| --- | --- |
| `Right-to-be-forgotten/dataset/data_module.py` | `696b7d431775173bfe84af7eafb52de74db1c5af15a0f893141b75be728d3314` |
| `Right-to-be-forgotten/metrics/metrics.py` | `4d3e8daacdc43b41b360036be9e0673bb0e3d79a2a8495d4220f0a313a4846c4` |
| `Right-to-be-forgotten/utils/utils.py` | `a9a7b42f53753848e80036c72616b58e780b20599eb217b714cffd2a5acaa1d4` |
| `Right-to-be-forgotten/config/model_config.yaml` | `331d4aa26c07af51ca3170ee8f653fb9c55103121c573336ea8754ef793f7b3f` |
| `Right-to-be-forgotten/config/tofu.yaml` | `72ec6f56420c573fff5b3e0eab0a1943993fc6146d1b3a4b31d9c6442abde816` |

实际 chat template 是手工拼接，不是 `tokenizer.apply_chat_template`：

```text
[INST] {question} [/INST]{answer}
```

- `question_start_tag` = `'[INST] '`
- `question_end_tag` = `' [/INST]'`
- `answer_tag` = `''`
- `add_special_tokens=True`，因此序列以 BOS `<s>` 开头。
- 答案边界使用 `len(tokenizer.tokenize(prompt, add_special_tokens=True))`，与 `data_module.py` 相同。`max_length=200`，本批没有截断。
- 训练标签在未截断时会在答案后再补一个 eos。这个 eos 属于 padding 标签，不计入答案 token span。

tokenizer 自带的 `chat_template` 在 assistant 内容前加一个空格，并在答案后追加空格和 eos。它和上面的实际训练/评测拼接不同。本次两条问题都按实际拼接编码。

存储的 chat template：

```text
{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}
```

存储 template 的 SHA256：`e8d5b5a546d3f19df16a5320c614fab05bdc4edd50e850f1afd947b503ea41cf`。

`apply_chat_template` 对 pair-01 的 `q_minus` 会得到：

```text
<s>[INST] Where was Katherine Mansfield born? [/INST] Wellington </s>
```

实际拼接得到：

```text
[INST] Where was Katherine Mansfield born? [/INST]Wellington
```

## 3. q_plus 与 q_minus 编码

`q_plus` 的目标是 `y_plus`。`q_minus` 的目标是 `donor_answer`。两条都用上一节的同一模板。完整 prompt token IDs、答案 token IDs 和 span 在 `k2_results.jsonl`。

| pair | q_plus 答案 | 答案 token 数 | span | 单 token | q_minus 目标 | 答案 token 数 | span | 单 token | 解码等于目标 |
| --- | --- | ---: | --- | --- | --- | ---: | --- | --- | --- |
| pair-01 | Karachi | 2 | 17:19 | 否 | Wellington | 2 | 16:18 | 否 | 是 |
| pair-02 | Beijing | 3 | 16:19 | 否 | Seattle | 2 | 15:17 | 否 | 是 |
| pair-03 | Tel Aviv | 4 | 17:21 | 否 | Houston | 3 | 15:18 | 否 | 是 |
| pair-04 | Addis Ababa | 4 | 17:21 | 否 | Gary | 2 | 14:16 | 否 | 是 |
| pair-05 | Tokyo | 3 | 17:20 | 否 | Tupelo | 3 | 16:19 | 否 | 是 |
| pair-06 | Cape Town | 3 | 16:19 | 否 | London | 2 | 16:18 | 否 | 是 |
| pair-07 | Manama | 2 | 18:20 | 否 | Mingora | 3 | 19:22 | 否 | 是 |
| pair-08 | New York City | 3 | 17:20 | 否 | Rio de Janeiro | 4 | 16:20 | 否 | 是 |
| pair-09 | Kuwait City | 4 | 24:28 | 否 | New Orleans | 2 | 15:17 | 否 | 是 |
| pair-10 | Astana | 3 | 17:20 | 否 | Wellington | 2 | 16:18 | 否 | 是 |

单 token 答案：q_plus 0 条，q_minus 0 条。其余答案是多 token，后续 K2 不能按单 token logit 处理。

边界自检：prompt 前缀与 full sequence 前缀不一致的条数 = 0；`tokenize` 与 `encode` 边界不一致的条数 = 0；解码结果不等于目标字符串的条数 = 0。

pair-01 `q_minus` 样例：

- prompt：`[INST] Where was Katherine Mansfield born? [/INST]`
- prompt token IDs：`[1, 518, 25580, 29962, 6804, 471, 476, 16490, 20594, 2671, 6345, 29973, 518, 29914, 25580, 29962]`
- prompt tokens：`['<s>', '▁[', 'INST', ']', '▁Where', '▁was', '▁K', 'atherine', '▁Mans', 'field', '▁born', '?', '▁[', '/', 'INST', ']']`
- answer token IDs：`[11284, 4885]`
- answer tokens：`['Well', 'ington']`
- answer span：[16, 18)，解码 `Wellington`

## 4–5. K2 阈值与 q_minus 筛选

K2 筛选状态：`blocked`。

每条记录的 `k2_status` 为：`not_computed: K2 screening stopped — no independent calibration set (instruction #4)`。

文件名搜索 `*calibrat*` 的结果：无。

`holdout01/05/10`、`world_facts`、`real_authors`、`retain*` 是 TOFU 评测或保留数据，不是出生地 K2 的独立 calibration 集。本次没有用主候选或这些评测文件估计阈值。

因此没有冻结阈值，也没有记录候选答案 logit、概率、top-1、最强竞争答案或 margin。这些字段在 `k2_results.jsonl` 中为 null。

## 6. Donor QID、P19 与评测重叠

本机访问 Wikidata 失败（`Cannot assign requested address`），所以没有在线复核 P19。下面是候选文件内部一致性。

| donor | QID | P19 | 城市级标签 | 内部一致性 | 在线复核 |
| --- | --- | --- | --- | --- | --- |
| donor-01 Katherine Mansfield | Q270632 | P19 | Wellington | pass | not_performed_no_network |
| donor-02 Bill Gates | Q5284 | P19 | Seattle | pass | not_performed_no_network |
| donor-03 Beyoncé | Q36153 | P19 | Houston | pass | not_performed_no_network |
| donor-04 Michael Jackson | Q2831 | P19 | Gary | pass | not_performed_no_network |
| donor-05 Elvis Presley | Q303 | P19 | Tupelo | pass | not_performed_no_network |
| donor-06 Ada Lovelace | Q7259 | P19 | London | pass | not_performed_no_network |
| donor-07 Malala Yousafzai | Q32732 | P19 | Mingora | pass | not_performed_no_network |
| donor-08 Paulo Coelho | Q12881 | P19 | Rio de Janeiro | pass | not_performed_no_network |
| donor-09 Anne Rice | Q184785 | P19 | New Orleans | pass | not_performed_no_network |

候选笔记要求远程确认粒度的 donor：['donor-04 (candidate_note_asks_for_remote_granularity_check)']。

内部一致性失败项：无。

评测清单扫描了 `workspace/data/tofu/*.jsonl`（18 个文件）和 `Right-to-be-forgotten/data/tofu/*.json`（12 个文件，内容为 JSONL）。共 36731 行。

`tofu.yaml` 当前评测清单是 `retain_perturbed`、`real_authors_perturbed`、`world_facts_perturbed`、`forget05_perturbed`。重叠表同时给出全清单命中和这份配置清单中的命中。

搜索表是预先写明的实体、别名、近邻人物和出生地字符串。重音和弯引号做了归一化，匹配使用字母数字边界。

| donor | 实体/别名/近邻命中 | 其中配置评测清单 | 出生地字符串命中 | 其中配置评测清单 |
| --- | ---: | ---: | ---: | ---: |
| donor-01 | 0 | 0 | 2 | 2 |
| donor-02 | 0 | 0 | 4 | 3 |
| donor-03 | 2 | 0 | 1 | 0 |
| donor-04 | 0 | 0 | 0 | 0 |
| donor-05 | 0 | 0 | 0 | 0 |
| donor-06 | 0 | 0 | 108 | 39 |
| donor-07 | 0 | 0 | 0 | 0 |
| donor-08 | 0 | 0 | 83 | 34 |
| donor-09 | 0 | 0 | 11 | 4 |

有命中的搜索词：

| donor | 类别 | 词 | 次数 | 配置评测清单次数 | 歧义 |
| --- | --- | --- | ---: | ---: | --- |
| donor-01 | place | Wellington | 2 | 2 | 是 |
| donor-02 | place | Seattle | 4 | 3 | 是 |
| donor-03 | alias | Beyonce | 1 | 0 | 否 |
| donor-03 | entity | Beyoncé | 1 | 0 | 否 |
| donor-03 | place | Houston | 1 | 0 | 是 |
| donor-06 | place | London | 108 | 39 | 是 |
| donor-08 | place | Rio de Janeiro | 83 | 34 | 否 |
| donor-09 | place | New Orleans | 11 | 4 | 否 |

实体、别名和近邻人物的字符串命中：

| donor | 类别 | 词 | 位置 | 上下文 |
| --- | --- | --- | --- | --- |
| donor-03 | alias | Beyonce | `Right-to-be-forgotten/data/tofu/forget10_perturbed.json:2021` | `d_answer": ["The LGBTQ+ author born in Addis Ababa, Ethiopia is named Beyonce Johnson.", "The author from Addis Ababa, Ethiopia who identifies as L` |
| donor-03 | entity | Beyoncé | `Right-to-be-forgotten/data/tofu/forget10_perturbed.json:2021` | `d_answer": ["the lgbtq+ author born in addis ababa, ethiopia is named beyonce johnson.", "the author from addis ababa, ethiopia who identifies as l` |

donor-03 的实体和别名命中是同一行：`Right-to-be-forgotten/data/tofu/forget10_perturbed.json:2021` 中的虚构作者 “Beyonce Johnson”。归一化会把 `Beyoncé` 和 `Beyonce` 收成同一个字符串，所以这一行被计了两次。该行不在 `tofu.yaml` 当前的 forget05 评测清单里，也不是 Wikidata Q36153 进入评测集的证据。

出生地次数是两个目录里的原始行命中。`workspace/data/tofu` 的 jsonl 和 `Right-to-be-forgotten/data/tofu` 的 json 会把同一事实各计一次，perturbed 改写也会重复计入。`London`、`Rio de Janeiro`、`New Orleans`、`Seattle`、`Wellington`、`Houston` 的命中是城市字符串，不是 donor 人物出现在评测清单中。样例在 `k2_results.jsonl` 的 `overlap.examples`。

## 7. 簇、复用与 go/no-go

- unique clean clusters：10
- unique donor clusters：9
- donor 复用：1 行，复用 donor-01，出现在 pair-10
- K2 通过率：不适用（0 条打分，10 条 blocked）
- 建议：**NO-GO**

恢复实验前需要同时满足：

1. 提供与主候选分开的出生地 calibration 集，并在看到主候选 K2 结果之前冻结阈值。
2. 修正 `candidate_pairs.jsonl` 的 `y_minus`，使它与对应 donor 的 P19 城市级答案一致。
3. 在 GPU 上用本报告记录的实际 chat template 对每条 `q_minus` 做严格 K2，记录答案 logit、概率、top-1、最强竞争答案和 margin。多 token 答案按答案 span 计算，不按单 token 处理。
4. 恢复外网后复核每条 QID 的 P19 是否支持当前城市级答案，尤其是 donor-04 Gary。
5. 若出生地字符串命中需要处理，先区分常见城市词和真正的 donor 事实泄漏。

## 8. 本阶段没有做的事

没有运行激活定位。没有运行 Teacher/Student 训练。没有为了增加通过样本而选择或调整 K2 阈值。没有修改 TOFU 文件、评测文件或 `external_donor_pairs_wikidata_v0.1/` 中的候选文件。

结果目录：`workspace/data/external_donor_pairs_wikidata_v0.1_validation`。
生成时间：`2026-11-09T11:28:11.457990+00:00`。
本脚本 SHA256：`3499ee4484c722cfb4ef8adca88675af785bcfa1e14c951cfdf60890db22e5bd`。

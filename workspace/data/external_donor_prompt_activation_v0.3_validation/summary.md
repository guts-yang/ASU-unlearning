# v0.3 prompt-only activation donor 预检

## 结论

**NO-GO。**

候选 10 条，clean 簇 10 个，donor 簇 10 个。这是数据与激活可行性预检，不是正式实验。

| 门 | 通过 | 失败 | 待审 | 阻断 | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| 文件完整性 | 10 | 0 | 0 | 0 | 行数、哈希、源行、无 y_minus、无 Beyoncé |
| 评测去重 | 10 | 0 | 0 | 0 | 人物或事实簇命中才排除；城市字符串不计为排除 |
| 来源 | 0 | 0 | 10 | 0 | 在线 P19 未完成，记为待审而不是通过 |
| 位置映射 | 0 | 0 | 0 | 10 | token span 已记录；hook 坐标阻断 |
| 激活缓存 | 0 | 0 | 0 | 10 | blocked_no_cuda |
| patch 对照 | 0 | 0 | 0 | 10 | not_run_blocked_no_cuda |

未通过因而暂停该臂的门：

- 来源门未通过：通过 0，失败 0，待审 10。在线 P19 为 URLError: <urlopen error [Errno 99] Cannot assign requested address>。
- 位置映射未通过：阻断 10，失败 0。当前 `my_llama.py` 没有 `activation_editor`。
- 激活缓存未复现：CUDA 不可用，权重未加载。
- matched-null 对照未运行。
- `y_plus` span 聚合未冻结：规格文件不在仓库中。

不进入正式 activation localization，不开始 Teacher/Student 训练。

## 1. 文件核对

候选目录只读。三个 JSONL 的每一行都是 JSON 对象。配对里没有 `y_minus`。

| 项目 | 计数 | manifest | 一致 |
| --- | ---: | ---: | --- |
| candidate_pairs | 10 | 10 | 是 |
| clean_facts | 10 | — | 是 |
| donor_entities | 10 | — | 是 |
| unique clean clusters | 10 | 10 | 是 |
| unique donor clusters | 10 | 10 | 是 |

解析错误：无。
完整性问题：无。
donor 复用行：0。
候选含 Beyoncé：否。donor-03 是 Muhammad Ali。donor-10 是 Lionel Messi。

`forget05.jsonl` 行数为 200，SHA256 为 `8f721842a4741fda0e4bfed459c4510c6c00604d5c332a533011648236862e08`。与 manifest 的 `clean_rows=200`、`clean_source_sha256=8f721842a4741fda0e4bfed459c4510c6c00604d5c332a533011648236862e08` 一致。

clean 侧来源行核对：10 条来源行的 question、answer 和 y_plus 都与 forget05.jsonl 一致。

| 文件 | SHA256 |
| --- | --- |
| `candidate_pairs.jsonl` | `a4c847168adaf8638ea620d58676a2014d91ffb2a02bcb9a79666bd03477caf1` |
| `clean_facts.jsonl` | `6a84e8a6ca404ef23cbc280ea78253f63d2495bdf6918baa8c66cc47656b9ff3` |
| `donor_entities.jsonl` | `1711c2ac0f93191ecce433c811a62555efb85b0800ef44bc9d827ba5a58161ca` |
| `manifest.json` | `77c540690f2de582c1c218176f1dde37c8f4629af0e3b3f103bd8d04227417c7` |
| `README.md` | `089ededffa31b3ff4eab770147d3e5f8b4d6b8c298190a395b55bb0923ad210b` |
| `REMOTE_VALIDATION_TODO.md` | `5f18645fed5101010025ee6b623ff77841be7af35e77d88baa1c9e92074ece65` |

写完结果后重新计算了上述候选文件哈希，与运行开始时相同。

## 2. Checkpoint、tokenizer 与 chat template

- Hugging Face repo：`open-unlearning/tofu_Llama-2-7b-chat-hf_full`
- snapshot commit：`cc5b31c69127d5da881608e640f2b453c446435b`
- 本地快照：`/root/autodl-tmp/huggingface/hub/models--open-unlearning--tofu_Llama-2-7b-chat-hf_full/snapshots/cc5b31c69127d5da881608e640f2b453c446435b`
- `config.json` 的 `_name_or_path`：`meta-llama/Llama-2-7b-chat-hf`
- 权重未加载，没有调用 `AutoModelForCausalLM`。
- CUDA：`available=False`，`device_count=0`，`nvidia-smi` 字节数 0，设备节点 `nvidiactl`。
- 精度：`not_loaded`。设备：`cuda_unavailable`。

`Right-to-be-forgotten/config/tofu.yaml` 的 `model_path` 是 `locuslab/tofu_ft_llama2-7b`。本次没有把它当作 tokenizer 来源。

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
- vocab size：32000；`len(tokenizer)`：32000
- bos/eos/pad：`<s>`=1，`</s>`=2，`</s>`=2
- padding side：`right`
- 编码没有 padding，也没有 truncation。

推理代码：

- git commit：`05cf214ed0ac9976d7d77b6e8847c6193d8fa64e`
- 分支：`I`
- 运行开始时 `git status --porcelain`：`?? workspace/data/external_donor_prompt_activation_v0.3_validation/`

手工模板来自 `model_config.yaml` 的 `llama2-7b`，不是 `tokenizer.apply_chat_template`：

```text
[INST] {question} [/INST]
```

- 模板 SHA256：`99196f1da4003a5d3b5d9a72b6e37e41ad811cc09a7537dc41b8b8a5d1396ae7`
- `question_start_tag` = `'[INST] '`
- `question_end_tag` = `' [/INST]'`
- `answer_tag` = `''`
- `add_special_tokens=True`，序列以 BOS `<s>` 开头。
- 存储 chat template SHA256：`e8d5b5a546d3f19df16a5320c614fab05bdc4edd50e850f1afd947b503ea41cf`。

`apply_chat_template` 对 promptdonor-01 的 `q_minus`（只含用户问题）得到：

```text
<s>[INST] Where was Katherine Mansfield born? [/INST]
```

实际 prompt-only 拼接得到：

```text
[INST] Where was Katherine Mansfield born? [/INST]
```

激活提取使用上面的实际拼接。`donor_fact_value` 没有拼进 `q_minus`。

相关代码哈希：

| 文件 | SHA256 |
| --- | --- |
| `Right-to-be-forgotten/dataset/data_module.py` | `696b7d431775173bfe84af7eafb52de74db1c5af15a0f893141b75be728d3314` |
| `Right-to-be-forgotten/config/model_config.yaml` | `331d4aa26c07af51ca3170ee8f653fb9c55103121c573336ea8754ef793f7b3f` |
| `Right-to-be-forgotten/config/tofu.yaml` | `72ec6f56420c573fff5b3e0eab0a1943993fc6146d1b3a4b31d9c6442abde816` |
| `Right-to-be-forgotten/my_models/my_llama.py` | `f379510a5536f5018d08be5ab2aeaf7ba51cf19b789633d5377caa177c1d1dbb` |

## 3. 评测去重

扫描了 forget、retain、world_facts、real_authors、holdout：jsonl 17 个，json 12 个，共 32731 行。
未纳入本次清单、因此跳过的文件：['workspace/data/tofu/full.jsonl']。

`tofu.yaml` 当前评测清单是 `retain_perturbed`、`real_authors_perturbed`、`world_facts_perturbed`、`forget05_perturbed`。人物命中与城市字符串命中分开计数。
别名表沿用 v0.1 已声明项；Muhammad Ali 与 Lionel Messi 只用公开姓名形式。网络不可用，别名表未在线补全。
只有人物、别名、近邻或同一行上的人物加出生地事实簇命中才排除 donor。城市字符串单独列出，不据此排除。

| donor | 人物命中 | 配置评测 | holdout 人物 | 事实簇行 | 城市字符串 | 排除 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| donor-01 | 0 | 0 | 0 | 0 | 2 | 否 |
| donor-02 | 0 | 0 | 0 | 0 | 4 | 否 |
| donor-03 | 0 | 0 | 0 | 0 | 0 | 否 |
| donor-04 | 0 | 0 | 0 | 0 | 0 | 否 |
| donor-05 | 0 | 0 | 0 | 0 | 0 | 否 |
| donor-06 | 0 | 0 | 0 | 0 | 106 | 否 |
| donor-07 | 0 | 0 | 0 | 0 | 0 | 否 |
| donor-08 | 0 | 0 | 0 | 0 | 80 | 否 |
| donor-09 | 0 | 0 | 0 | 0 | 11 | 否 |
| donor-10 | 0 | 0 | 0 | 0 | 0 | 否 |

排除的 donor：无。

有命中的搜索词：

| donor | 类别 | 词 | 次数 | 配置评测 | holdout | 歧义城市 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| donor-01 | place | Wellington | 2 | 2 | 0 | 是 |
| donor-02 | place | Seattle | 4 | 2 | 0 | 是 |
| donor-06 | place | London | 106 | 35 | 0 | 是 |
| donor-08 | place | Rio de Janeiro | 80 | 33 | 0 | 是 |
| donor-09 | place | New Orleans | 11 | 4 | 0 | 是 |

实体、别名和近邻人物没有命中这份评测清单。

候选文件中的 Beyoncé 字符串：无。评测清单里的 Beyoncé 字符串命中 1 行，其中配置评测清单 0 行。这些命中不属于当前 donor，不作为本版排除项。

| 位置 | 上下文 |
| --- | --- |
| `Right-to-be-forgotten/data/tofu/forget10_perturbed.json:2021` | `d_answer": ["The LGBTQ+ author born in Addis Ababa, Ethiopia is named Beyonce Johnson.", "The author from Addis Ababa, Ethiopia who identifies as L` |

城市字符串次数把 `workspace/data/tofu` 的 jsonl 和 `Right-to-be-forgotten/data/tofu` 的 json 各计一次，perturbed 改写也会重复计入。样例在 `validation.jsonl` 的 `overlap.examples`。

## 4. 来源审核

在线 Wikidata：`URLError: <urlopen error [Errno 99] Cannot assign requested address>`。内部字段核对不能代替在线 P19。不确定项保持待审，没有猜测出生地。

| donor | QID | 候选城市 | 内部一致性 | 在线 P19 | 来源门 |
| --- | --- | --- | --- | --- | --- |
| donor-01 Katherine Mansfield | Q270632 | Wellington | pass | blocked_no_network | pending |
| donor-02 Bill Gates | Q5284 | Seattle | pass | blocked_no_network | pending |
| donor-03 Muhammad Ali | Q36107 | Louisville | pass | blocked_no_network | pending |
| donor-04 Michael Jackson | Q2831 | Gary | pass | blocked_no_network | pending |
| donor-05 Elvis Presley | Q303 | Tupelo | pass | blocked_no_network | pending |
| donor-06 Ada Lovelace | Q7259 | London | pass | blocked_no_network | pending |
| donor-07 Malala Yousafzai | Q32732 | Mingora | pass | blocked_no_network | pending |
| donor-08 Paulo Coelho | Q12881 | Rio de Janeiro | pass | blocked_no_network | pending |
| donor-09 Anne Rice | Q184785 | New Orleans | pass | blocked_no_network | pending |
| donor-10 Lionel Messi | Q615 | Rosario | pass | blocked_no_network | pending |

候选笔记要求远程确认粒度：['donor-04（candidate_note_asks_for_remote_granularity_check）']。
内部一致性失败：无。

规格文件 `workspace/方向I_prompt_only_activation_donor构造规格_v0.3.md`：不存在。`y_plus` span 的 sum/mean 聚合没有可冻结定义，本次不选定。

## 5. Tokenizer 与位置

`q_plus` 和 `q_minus` 都只编码问题 prompt。`y_plus` 的 token span 是城市串接在 `q_plus` prompt 之后的位置，没有计算分数。

| pair | 人名 token 数 | 人名 span | 隔离 | y_plus | y_plus token 数 | y_plus span | 前缀一致 |
| --- | ---: | --- | --- | --- | ---: | --- | --- |
| promptdonor-01 | 5 | 6:11 | pass | Karachi | 2 | 17:19 | 是 |
| promptdonor-02 | 4 | 6:10 | pass | Beijing | 3 | 16:19 | 是 |
| promptdonor-03 | 5 | 6:11 | pass | Tel Aviv | 4 | 17:21 | 是 |
| promptdonor-04 | 5 | 6:11 | pass | Addis Ababa | 4 | 17:21 | 是 |
| promptdonor-05 | 5 | 6:11 | pass | Tokyo | 3 | 17:20 | 是 |
| promptdonor-06 | 4 | 6:10 | pass | Cape Town | 3 | 16:19 | 是 |
| promptdonor-07 | 6 | 6:12 | pass | Manama | 2 | 18:20 | 是 |
| promptdonor-08 | 5 | 6:11 | pass | New York City | 3 | 17:20 | 是 |
| promptdonor-09 | 12 | 6:18 | pass | Kuwait City | 4 | 24:28 | 是 |
| promptdonor-10 | 5 | 6:11 | pass | Astana | 3 | 17:20 | 是 |

人名 span 失败：无。`y_plus` 前缀不一致：无。

hook 坐标：`blocked_no_activation_editor`。`activation_editor` 出现在 `Right-to-be-forgotten/my_models/my_llama.py`：否。`LlamaMLP.forward` 返回 `down_proj`：是。没有可对照的补丁坐标，位置映射门记为阻断。

promptdonor-01 `q_minus` prompt-only 样例：

- prompt：`[INST] Where was Katherine Mansfield born? [/INST]`
- token IDs：`[1, 518, 25580, 29962, 6804, 471, 476, 16490, 20594, 2671, 6345, 29973, 518, 29914, 25580, 29962]`
- tokens：`['<s>', '▁[', 'INST', ']', '▁Where', '▁was', '▁K', 'atherine', '▁Mans', 'field', '▁born', '?', '▁[', '/', 'INST', ']']`
- 人名 span：[6, 10)，覆盖 ` Katherine Mansfield`

## 6–7. 激活与 patch

没有加载权重，没有前向，没有提取 `a_plus`/`a_minus`。
激活缓存：`blocked_no_cuda`。没有 cache key、shape 或重复运行结果。
patch 对照：`not_run_blocked_no_cuda`。没有 no-op、匹配 donor、shuffle donor 或 matched-null。
没有计算 `logit(y_plus)-logit(y_minus)`，也没有运行 K2。

## 8. Go/no-go

- unique clean clusters：10
- unique donor clusters：10
- 评测去重排除：0
- 建议：**NO-GO**

恢复该臂之前需要同时满足：来源在线审核通过，评测去重没有人物或事实簇冲突，真实 hook 的坐标映射通过，激活缓存在目标设备上可复现，并且 matched-null 对照使用已冻结的 `y_plus` span score。

## 本阶段没有做的事

没有修改候选、TOFU 评测文件或 v0.1 验证结果。没有运行激活定位。没有运行 Teacher/Student 训练。

结果目录：`workspace/data/external_donor_prompt_activation_v0.3_validation`。
生成时间：`2026-11-10T03:19:22.513311+00:00`。
`validation.jsonl` SHA256：`0a0b131df0b69bb439d2b7c98b3cecbb1aaed8d0eaf27b70e913652408ad7cf7`。
本脚本 SHA256：`ced541beaa644d2b3d5a86e43d590a4a1b63f4b367829d7ffc8d7ccd6e567cf4`。
`summary.md` 与上述文件的哈希写在 `output_hashes.json`。


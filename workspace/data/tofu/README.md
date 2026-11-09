# 官方 TOFU 数据集

这个目录是 Hugging Face 上 [`locuslab/TOFU`](https://huggingface.co/datasets/locuslab/TOFU) 的本地副本，每个 config 的 `train` split 存成一份 JSONL。行数和字段记录在 [`manifest.json`](manifest.json)。

它是 OpenUnlearning 默认使用的那一版，不是 `Right-to-be-forgotten/data/tofu` 里的持续遗忘切分。

## 数据是什么

TOFU（Task of Fictitious Unlearning）用 200 位虚构作者的问答来测大模型遗忘。每位作者 20 个问答，微调全集是 4000 条。作者、生平、作品都是虚构的，所以遗忘目标可以和模型原有的真实世界知识分开。

问答是英文。普通分片只有 `question` 和 `answer`。

## 遗忘集和保留集

百分比是相对 4000 条全集而言。三档遗忘集是嵌套的：`forget01` 包含在 `forget05` 里，`forget05` 包含在 `forget10` 里。保留集是对应遗忘集在全集中的补集，因此 `retain90` 包含在 `retain95` 里，`retain95` 包含在 `retain99` 里。遗忘集和它配对的保留集没有共同问题。

| 文件 | 行数 | 含义 |
|---|---:|---|
| `full.jsonl` | 4000 | 全部虚构作者问答 |
| `forget01.jsonl` | 40 | 遗忘 1%，2 位作者 |
| `forget05.jsonl` | 200 | 遗忘 5%，10 位作者 |
| `forget10.jsonl` | 400 | 遗忘 10%，20 位作者 |
| `retain99.jsonl` | 3960 | `forget01` 的补集 |
| `retain95.jsonl` | 3800 | `forget05` 的补集 |
| `retain90.jsonl` | 3600 | `forget10` 的补集 |

OpenUnlearning 的默认实验使用 `forget10` 和 `retain90`。`forget05` 配对的是这里的 `retain95`。

`full.jsonl` 有 4000 行、3999 个不同的问题文本。问题 “What is the full name of the author?” 出现两次，答案分别是 Aurelio Beltrán 和 Isabel Martínez。这两行都在三个 retain 文件里，不在 forget 文件里。

## 外部知识

这两份不参与遗忘训练，用来检查模型忘掉虚构作者之后，真实知识是否还在。

| 文件 | 行数 | 内容 |
|---|---:|---|
| `world_facts.jsonl` | 117 | 真实世界事实，例如埃菲尔铁塔所在城市 |
| `real_authors.jsonl` | 100 | 真实作者，例如《罗密欧与朱丽叶》的作者 |

字段是 `question`、`option1` 到 `option4`、`answer`。`answer` 是正确选项的文本，不是选项编号。

## 扰动标注

带 `_perturbed` 的文件用于评估，不用于构造遗忘训练集。

虚构问答的扰动文件和原分片行数相同：

- `forget01_perturbed.jsonl`：40
- `forget05_perturbed.jsonl`：200
- `forget10_perturbed.jsonl`：400
- `retain_perturbed.jsonl`：400

它们在原问题、原答案之外还有：

- `paraphrased_question`：问题的改写
- `paraphrased_answer`：正确答案的改写
- `perturbed_answer`：五条错误改写，类型是字符串列表

`retain_perturbed.jsonl` 的 400 个问题全部来自 `retain90.jsonl`，因此也在 `retain95` 和 `retain99` 中。它是保留集上的评估切片，不是完整保留集。

`world_facts_perturbed.jsonl`（117）和 `real_authors_perturbed.jsonl`（100）只保留 `question`、`answer`、`perturbed_answer`。这里的 `perturbed_answer` 是三个错误选项。

## Holdout

`holdout01.jsonl`、`holdout05.jsonl`、`holdout10.jsonl` 分别是 40、200、400 条，规模与三档遗忘集相同。它们和 `full`、遗忘集、保留集都没有共同问题，也不属于那 4000 条微调数据。OpenUnlearning 用它们做成员推断和 PrivLeak 一类评估。

## 和仓库里另一份 TOFU 的差别

`Right-to-be-forgotten/data/tofu` 是持续遗忘重打包。那里的 `forget05.json` 有 2000 条，分成 10 个 task；代码实际配的保留文件是 `retain50.json`。本目录的 `forget05.jsonl` 是官方单次遗忘用的 200 条，配对文件是 `retain95.jsonl`。两套文件不能按同名替换。

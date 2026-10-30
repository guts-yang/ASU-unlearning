# 训练前准备记录

日期：2026-10-30。范围：TOFU forget05、Llama-2-7B、ASU+GD、seed 1001。本记录只核对本地权重、配置和数据。下载清单为空。本阶段不训练，不跑 M0，不改 `losses.py`，不覆盖 checkpoint。

## 下载清单

空。下列资产已在本机，不从 Hub 拉取权重或配置。

## 权重

| 角色 | 路径 | 核对 |
|---|---|---|
| 初始化基座（当前使用） | `/root/autodl-tmp/models/tofu_ft_llama2-7b` | open-unlearning 的 `tofu_Llama-2-7b-chat-hf_full`，snapshot `cc5b31c69127d5da881608e640f2b453c446435b`。3 个分片、291 个张量、13,476,865,048 字节。`config.json`、`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`、`generation_config.json` 都在。`LlamaForCausalLM`，32 层，词表 32000。 |
| ASU+GD Student | `/root/autodl-tmp/ASU-unlearning/checkpoints/tofu_replay/llama2-7b/forget05/ASU+GD/seed_1001/epoch5_1e-05_FixRefTrue_maskTrue_Fcoeff0.1_Rcoeff1.0_Temp2.3_LayersNone_beta0.1/1/unlearn_times_1/checkpoint-last` | 同样 3 个分片、291 个张量、13,476,865,048 字节，tokenizer 与 config 齐全。 |

## 配置

训练配置在该 checkpoint 的上级目录：

`/root/autodl-tmp/ASU-unlearning/checkpoints/tofu_replay/llama2-7b/forget05/ASU+GD/seed_1001/epoch5_1e-05_FixRefTrue_maskTrue_Fcoeff0.1_Rcoeff1.0_Temp2.3_LayersNone_beta0.1/1/config.yaml`

| 字段 | 值 |
|---|---|
| `forget_loss` | `ASU+GD` |
| `attention_temp` | `2.3` |
| `layers_id` | `null` |
| `mask` | `true` |
| `seed` | `1001` |
| `model_family` | `llama2-7b` |
| `model_path` | `locuslab/tofu_ft_llama2-7b` |
| `data_path` / `split` | `data/tofu` / `forget05` |
| `task_id` | `1` |
| `fix_ref_model` | `true` |
| `use_LoRA` | `false` |
| `forget_coeff` / `regularization_coeff` / `beta` | `0.1` / `1.0` / `0.1` |
| `num_epochs` / `batch_size` / `gradient_accumulation_steps` / `lr` | `5` / `4` / `4` / `1e-05` |

这份 `config.yaml` 里的 `model_path` 仍是当时的 Hub 名 `locuslab/tofu_ft_llama2-7b`，对应缓存目录已经删除。当前基座用本地 open-unlearning 目录 `/root/autodl-tmp/models/tofu_ft_llama2-7b`。历史 `config.yaml` 不改。

对话模板在 `Right-to-be-forgotten/config/model_config.yaml` 的 `llama2-7b`：`question_start_tag="[INST] "`，`question_end_tag=" [/INST]"`，`answer_tag=""`。

## 数据

`Right-to-be-forgotten/data/tofu/forget05.json`：2000 行，字段 `task_id`、`question`、`answer`。`task_id` 1–10 各 200 行。SHA-256：`a8c207a905af73ea4797b1af92a14cf458f42df567c941f7284f87d930096f4b`。

## 记下、不下载的缺口

- real-world / Llama-3 的 ASU Student 权重不在本机。`checkpoints/gold_rank/llama3-8b` 只有约 4KB，没有 safetensors。`data/real_world/forget.json` 有 400 行，但这条 TOFU 准备不用它。
- 官方 `meta-llama/Meta-Llama-3-8B-Instruct` 为 gated，本口径不用。`NousResearch/Meta-Llama-3-8B-Instruct` 的本地 snapshot `53346005fb0ef11d3b6a83b12c895cca40156b6c` 已有 `config.json`，不属于这条 TOFU ASU+GD 准备。

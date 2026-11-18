# ASU 基线只读审计清单（RecodeLoop M0）

> 本文是对 `ASU-unlearning` 仓库当前工作树的**只读**审计记录，用于 RecodeLoop M0 的复用/调整/隔离边界判定。
> 审计期间未修改任何基线文件，未下载模型/数据，未运行 GPU 训练。

## 审计元信息

| 项 | 值 |
|---|---|
| 仓库根目录 | `/usr/local/llm-unlearning/ASU-unlearning` |
| 审计时分支 | `K3/20261118` |
| HEAD | `622c02bf758c70f9954d76fd7559716f9f8f2ae8` |
| `git status` | `nothing to commit, working tree clean` |
| 其他本地/远端分支 | `main`, `I`, `C`, `G`, `fuxian`, `new_method/A` 等 |
| `workspace/` 原有内容 | 仅空目录 `data/`、`docs/`、`figures/`，无文件；已保留，未清空/重置 |
| 审计日期 | 2026-11-18 |

## 0. G0 资产映射结论（初始审计，2026-11-18 早段）

> 以下为**初始只读审计**结论。会话后续由用户提供 checkpoint 并确认角色映射，资产已补齐；最新状态见本文 §11 与 `implementation_log.md`。初始结论保留作审计轨迹，不代表当前状态。

**初始审计未发现 P08-M2 到 base / target / ASU student checkpoint、tokenizer 与权重哈希的唯一映射。**

- 当前工作树全仓搜索 `P08|P08-M2|base_checkpoint|target_checkpoint|student_checkpoint|sha256`：命中 **0**。
- 所有 refs 上 `git grep "P08"`：命中 **0**；`git log --all -S "P08-M2"` / `-S "P08"`：命中 **0**（仅 `.git` pack 中出现过 "P08" 字节串，不构成可用映射）。
- 不存在 manifest/checkpoint 注册表；现有语义是散落在 Hydra 配置里的 `cfg.model_path`、`cfg.fix_ref_model`、`last_checkpoint_dir`、`hf_key`。
- `data/tofu`、`data/real_world` 的 JSON 数据文件未随仓库提供。

因此：`scripts/validate_assets.py` 报告 **BLOCKED**，科学评价（M0a/M0b/M0c）不得进入。不得猜测 checkpoint，不得用其他模型冒名替代。

## 1. 模型 / tokenizer / checkpoint 加载入口

### 1.1 Right-to-be-forgotten（主 ASU 基线，logits KL 蒸馏）

- `Right-to-be-forgotten/forget.py::main()` L65-260
  - L76-79：`model_cfg = get_model_identifiers_from_yaml(cfg.model_family)`；`model_id = model_cfg["hf_key"]`；`AutoConfig.from_pretrained(model_id)`。
  - L109-110：`AutoTokenizer.from_pretrained(model_id)`；`tokenizer.pad_token = tokenizer.eos_token`。**tokenizer 路径跟随 `hf_key`，无独立 tokenizer_path**。
  - L169-172：checkpoint 语义 `model_path = cfg.model_path if unlearn_times == 1 else last_checkpoint_dir`；`reference_model_path = cfg.model_path if cfg.fix_ref_model else model_path`。
  - L175-215：student / target 用 `AutoModelForCausalLM.from_pretrained(model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)`，按 `cfg.use_LoRA` 选 `get_peft_model` 或 `PeftModel.from_pretrained`。
  - L218-236：ASU teacher / reference 用 `my_models.my_llama.LlamaForCausalLM.from_pretrained(reference_model_path, ...)`，随后 `reference_model.eval()`。**teacher 是从路径重新加载，不是复制 student**。
  - L263-272：`checkpoint-{max_steps}` → `checkpoint-last` 整理与按 `save_checkpoint` 删除。
- `Right-to-be-forgotten/eval.py::main()` L96-160 / `model_eval()` L18-93
  - L113-119：`curr_checkpoint_dir = .../checkpoint-{cfg.eval_unlearn_step}`；`eval_unlearn_step == 0` 时用 `cfg.model_path`。
  - L131-149：LoRA 分支 `AutoModelForCausalLM.from_pretrained(cfg.model_path)` + `PeftModel.from_pretrained(..., curr_checkpoint_dir)` + `merge_and_unload()`。
- `Right-to-be-forgotten/utils/utils.py::get_model_identifiers_from_yaml()` L9-23：读取相对路径 `config/model_config.yaml`，只返回 `hf_key` 与 prompt 模板；**无 checkpoint/tokenizer/hash 字段**。
- `Right-to-be-forgotten/config/model_config.yaml` L3：`hf_key: NousResearch/Llama-2-7b-chat-hf`；L10：`hf_key: meta-llama/Meta-Llama-3-8B-Instruct`。
- `Right-to-be-forgotten/config/tofu.yaml` L3：`model_path: locuslab/tofu_ft_llama2-7b`。
- `Right-to-be-forgotten/config/real_world.yaml` L3：`model_path: meta-llama/Meta-Llama-3-8B-Instruct`。

### 1.2 open-unlearning（模型工厂，ASU handler 未注册）

- `open-unlearning/src/model/__init__.py`
  - `get_model(model_cfg)` L54-89：`AutoConfig.from_pretrained` → `arch = config.architectures[0]` → `custom_class_map` → `model_cls.from_pretrained(..., cache_dir=hf_home)`。
  - `get_tokenizer(tokenizer_cfg)` L105-126：`AutoTokenizer.from_pretrained(**tokenizer_cfg, cache_dir=hf_home)`；路径来自 `configs/model/*.yaml` 的 `tokenizer_args.pretrained_model_name_or_path`。
  - `custom_class_map` L15-19：`LlamaForCausalLM`/`MistralForCausalLM`/`Qwen2ForCausalLM` 指向 `src/my_models/` 自定义实现（这些版本**无 tau 注入**）。
  - `get_dtype()` L33-51；`hf_home = os.getenv("HF_HOME", None)` L22（唯一缓存控制点，不是 manifest）。
- `open-unlearning/src/train.py::main()` L10-65：`get_model` → `get_data` → `get_collators` → `load_trainer` → `trainer.train()/save_model()`。

### 1.3 wmdp/asu（激活 MSE）

- `wmdp/asu/utils.py::load_model(model_name_or_path)` L50-69：`MistralForCausalLM.from_pretrained(... device_map="auto")` + `AutoTokenizer.from_pretrained(... use_fast=False)`；设置 pad/mask/sep/cls token。
- `wmdp/asu/unlearn.py` L177-202：两次调用 `load_model(args.model_name_or_path)` 得到 `frozen_model` 与 `updated_model`（**同一路径两份副本**）；`get_args()` L121-174 默认 `HuggingFaceH4/zephyr-7b-beta`。

### 1.4 muse

- `muse/utils.py` L36 `AutoModelForCausalLM.from_pretrained(model_dir, ...)`；L40 `AutoTokenizer.from_pretrained(tokenizer_dir, ...)` —— 仓库中**唯一显式分离 tokenizer_dir** 的入口。
- `muse/constants.py`：`LLAMA_DIR`、`DEFAULT_DATA`、`AUC_RETRAIN` 等常量。

**统一 manifest / P08 映射：未发现。**

## 2. Teacher 温度平滑 / Student 前向 / ASU KL / retain NLL / 统一反传

### 2.1 Right-to-be-forgotten（主基线）

- `Right-to-be-forgotten/trainer/losses.py`
  - `get_loss()` L6-38：L18-23 调 `asu_loss(..., attention_temp, layers_id, ignore_first_token)`；L27-30 正则项 `gd_loss`（NLL）或 `kl_loss`。
  - **`asu_loss()` L67-108**：
    - L71：student 前向 `outputs = model(input_ids, labels=labels, attention_mask=attention_mask)`（**无 tau**）。
    - L73：`log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)`。
    - L75-79：teacher 在 `torch.no_grad()` 内带 `attention_temp` / `layers_id` 前向。
    - L84：`loss_mask = (labels[:, 1:] != -100)`。
    - L104-107：`F.kl_div(log_probs, ref_log_probs, reduction='none', log_target=True).sum(-1)` 按 token mask 求均值。
  - `gd_loss()` L226-233：**retain NLL** = `outputs.loss`（retain 输入来自 `inputs[1]`）。
  - `kl_loss()` **重复定义**：L161-194 被 L197-211 覆盖；实际生效为全序列 `F.kl_div(..., reduction='batchmean', log_target=True)`，**无 -100 mask**（复用时注意）。
  - `get_batch_loss()` L236-242：`CrossEntropyLoss(ignore_index=-100, reduction='none')`。
- `Right-to-be-forgotten/trainer/trainer.py`
  - `CustomTrainerForgetting.compute_loss()` L29-34：`loss = self.forget_coeff * forget_loss + self.regularization_coeff * regularization_loss`。
  - `e_prepare_deepspeed()` L36-70：teacher 用 `deepspeed.initialize`，`model.eval()`，`requires_grad=False`。
  - **显式 `loss.backward()`：未发现**（由 HF `Trainer.train()` 内部完成）。
- tau 注入：`Right-to-be-forgotten/my_models/my_llama.py`
  - `LlamaAttention.forward` L328：eager L421-425 `attn_weights = QK^T / (sqrt(head_dim) * attention_temp)`；L432 fp32 softmax。
  - `LlamaFlashAttention2.forward` L474：flash L530-534 把 Q、K 各乘 `scale = 1/sqrt(attention_temp)`；L578 `o_proj`。
  - `LlamaSdpaAttention.forward` L594：L669-671 同款 scale；L687 `o_proj`。
  - `LlamaDecoderLayer.forward` L710：L759 透传 `attention_temp`。
  - `LlamaModel.forward` L936：L1022/L1034 仅当 `layers_id is None or idx in layers_id` 时传 tau。
  - `LlamaForCausalLM.forward` L1163：L1177 参数、L1228 传递；L1340-1352 从 kwargs 兜底（generate 场景）。
  - 训练配置使用 `attn_implementation='flash_attention_2'`（forget.py L179/198/223），生效分支为 L530-534。
  - **隔离约束**：以上 tau 与 `asu_loss()` 保持原样，不进入 RecodeLoop 每轮目标；ASU student 仅作历史起点。

### 2.2 open-unlearning 的 ASU（存在但未注册）

- `open-unlearning/src/trainer/unlearn/ASU.py`：`class ATTU_output(GradDiff)` L8；`__init__` L9-17 中 `ref_model is None` 时 `self._prepare_ref_model(self.model)`（L17，即 `copy.deepcopy(student)`）。
  - `kl_divergence_retain()` L20-32、`kl_divergence_forget()` L34-47（teacher 传 `attention_temp`，L38）。
  - `compute_retain_loss()` L49-69（`"NLL"` → `retain_outputs.loss`；`"KL"` → KL）。
  - `compute_loss()` L71-115：L80/L93 `-100` mask；L102 `loss = self.alpha * forget_loss + retain_loss`。
- `GradDiff._prepare_ref_model()`：`open-unlearning/src/trainer/unlearn/grad_diff.py` L16-23（`copy.deepcopy(model).to(device)` + `eval()`）。
- **未注册陷阱**：`configs/trainer/ASU.yaml` L4 `handler: ASU`，但 `src/trainer/__init__.py` L87-101 未 import/注册 `ATTU_output`；直接使用该配置会失败。workspace 不依赖此路径。

### 2.3 wmdp/asu（激活 MSE）

- `wmdp/asu/unlearn.py::run_rmu()` L11-118：L58-69 经 `forward_with_cache` 取 student/teacher 激活，teacher 传 tau；L67-69 forget 用 `F.mse_loss`，L82-85 retain 用 `F.mse_loss * args.alpha[topic_idx]`；L88-91 `loss = unlearn_loss + retain_loss` + `optimizer.zero_grad(); loss.backward(); optimizer.step()`（**仓库唯一显式 backward**）；L111-118 `save_pretrained` + `tokenizer.save_pretrained`。
- 已知缺陷：L63 使用 `args.attention_temp_value`，但 `get_args()` L151-155 只定义 `--attention_temp` → 运行时 `AttributeError`；L170 `args.attention_temp = 10.0**float(args.attention_temp)`。workspace 不依赖此路径。
- tau 注入：`wmdp/asu/my_mistral.py`（`MistralAttention` L212 / `MistralFlashAttention2` L293 / `MistralSdpaAttention` L430 / `MistralModel.forward` L754 L840/851 / `MistralForCausalLM.forward` L1014）。

## 3. 数据加载 / collator / 答案掩码 / -100 / 生成评测

### 3.1 Right-to-be-forgotten

- `Right-to-be-forgotten/dataset/data_module.py`
  - `convert_raw_forget_data_to_model_format()` L24-54：L48-50 pad 标签填 `-100`、问题 token 置 `-100`。
  - `convert_raw_data_to_model_format()` L57-82：L77-80 同样处理。
  - `TextForgetDatasetQA` L85（`__getitem__` L105，L131-133 按 `mask` 屏蔽问题）、`TextDatasetQA` L142。
  - `custom_data_collator_forget()` L197（forget trainer 使用，返回多个 `(input_ids, labels, attention_mask)` 三元组）、`custom_data_collator()` L190、`collate_fn()` L182、`get_batch_loss()` L216（`ignore_index=-100`）。
- `Right-to-be-forgotten/metrics/metrics.py`
  - `get_dataloader()` L95、`get_all_evals()` L147（L169-170 统计 `labels != -100`）。
  - **`run_generation(cfg, batch, model, tokenizer)` L302-345**：L330-331 左 padding 重编码；L334 `torch.manual_seed(0)`；L335-341 `model.generate(..., do_sample=False, use_cache=True)`。
  - `eval_perturbation_ratio()` L51、`eval_rouge_recall()` L348、`eval_cosine_similarity()` L359、`get_entailment_results()` L372、`get_eval_results()` L219。

### 3.2 open-unlearning

- `open-unlearning/src/data/collators.py::DataCollatorForSupervisedDataset` L7：L24-35 `_pad_tokens()`、L37-67 `__call__`（labels 用 `IGNORE_INDEX` 填充）。
- `open-unlearning/src/data/utils.py`：`IGNORE_INDEX = -100` L7、`preprocess_chat_instance()` L17（L124 `labels = [IGNORE_INDEX]*len_matched + chat_ids[len_matched:]`）、`preprocess_pretraining_instance()` L143。
- `open-unlearning/src/data/__init__.py`：`get_data()` / `get_collators()` 注册表。
- 生成入口：`open-unlearning/community/methods/AltPO/generate.py`、`src/evals/metrics/memorization.py`、`configs/generation/default.yaml`。

## 4. TOFU forget/retain、Real Authors、World Facts 数据入口

- TOFU：`Right-to-be-forgotten/config/tofu.yaml` L12-15 `forget_data: tofu`、`data_path: data/tofu`、`split: forget05`、`task_id: 1`；L54 `eval.data_path: [data/tofu ×4]`；L57-61 `split_list = retain_perturbed / real_authors_perturbed / world_facts_perturbed / forget05_perturbed`；L63-64 `eval_task = eval_log / eval_real_author_wo_options / eval_real_world_wo_options / eval_log_forget`。
- 加载函数：`Right-to-be-forgotten/forget.py::get_task_data()` L34-43（`datasets.load_dataset('json', data_files=...)`）；`dataset/data_module.py` 转换函数见第 3 节。
- Real Authors / World Facts：`metrics/metrics.py` L183 `real_authors_perturbed.json`、L190 `world_facts_perturbed.json`、L221 `eval_real_author_wo_options.json → 'Real Authors'`；`utils/utils.py::get_total_len()` L45-58（`eval_real_author_wo_options.json`→100、`eval_real_world_wo_options.json`→117）。
- Real World（另一设定）：`config/real_world.yaml` L12-15 `forget_data: real_world`、`data_path: data/real_world`、`split: forget`、`retain: neighbor`；`real_world_unlearn.py` / `real_world_eval.py`。
- open-unlearning：`configs/eval/tofu.yaml`、`configs/eval/muse.yaml`、`configs/data/unlearn/*.yaml`、`src/data/tofu.py`、`src/data/muse.py`；`setup_data.py` 为数据准备脚本（**本次未执行**）。
- 脚本中的 forget05：`open-unlearning/scripts/tofu_unlearn.sh` L21、`tofu_finetune.sh` L16、`community/methods/{CEU,UNDIAL,WGA,SatImp,AltPO}/run.sh` L18-19、`PDU/run.sh` L31；`Right-to-be-forgotten/scripts/tofu/{asu,baselines}.sh`、`scripts/continual_tofu/{asu,baselines}.sh`。
- **数据文件本身未随仓库提供**；M0 实体簇划分需在 workspace 内基于可获得的 canonical ID 自行构造，并先过 G0。

## 5. optimizer / LoRA-PEFT / 混合精度 / 分布式 / 种子 / 日志 / checkpoint

| 项 | 位置与符号 |
|---|---|
| optimizer | `Right-to-be-forgotten/forget.py` L160 `optim="paged_adamw_32bit"`（`TrainingArguments` L150-167）；`wmdp/asu/unlearn.py` L26 `AdamW(params, lr=args.lr)`；open-unlearning 由 `TrainingArguments` 决定 |
| LoRA / PEFT | `Right-to-be-forgotten/forget.py::find_all_linear_names()` L22-31；`LoraConfig` L185-192 / L207-214；`get_peft_model` L215；`PeftModel.from_pretrained` L193；配置 `use_LoRA` / `LoRA.r/alpha/dropout`（`tofu.yaml` L5-9）；DeepSpeed `config/ds_config/lora.json`；`eval.py` L131-140 `merge_and_unload()` |
| AMP | `forget.py` L157-158 `bf16=True, bf16_full_eval=True`；L180/199/224/233 `torch_dtype=torch.bfloat16`；flash-attn2 L179；flash 实现内 `torch.is_autocast_enabled()` 分支 `my_llama.py` L546 |
| DDP / DeepSpeed | `forget.py` L67-72 `WORLD_SIZE/LOCAL_RANK`、L161 `deepspeed=ds_config`、L164 `ddp_find_unused_parameters=False`；`CustomTrainerForgetting.e_prepare_deepspeed()` L36-70；`config/ds_config/{llama2,llama3,lora}.json`；open-unlearning `UnlearnTrainer._prepare_deepspeed()` L40-76、`configs/accelerate/*` |
| 随机种子 | `Right-to-be-forgotten/utils/utils.py::set_random_seed()` L90-96（`forget.py` L73-74 调用，`tofu.yaml` L35 `seed: 1001`）；`wmdp/asu/unlearn.py` L180-184；`metrics/metrics.py` L334 `torch.manual_seed(0)`；open-unlearning `src/trainer/utils.py::seed_everything()` L8 |
| 日志 | HF `Trainer`/`TrainingArguments`；`Right-to-be-forgotten/eval.py::model_eval()` L72-91 写 `unlearning_results.txt/.csv`、`all_unlearning_results.csv`；`metrics/metrics.py::get_eval_results()` L219；open-unlearning `src/evals/metrics/logger`；**未发现**专用 wandb 配置 |
| checkpoint 保存 | `forget.py` L262-279 整理/删除 checkpoint；`wmdp/asu/unlearn.py` L111-118 `save_pretrained`；`open-unlearning/src/train.py` L61-62 `trainer.save_state()` / `trainer.save_model(output_dir)` |

**运行时实际环境（非仓库声明）**：Python 3.11.11、torch 2.6.0+cpu（CUDA 不可用）、transformers 4.48.2、PyYAML 6.0.2、pytest 8.3.4、numpy 1.26.4；**PEFT 未安装**。因此 RecodeLoop 不导入 PEFT 作为硬依赖，LoRA 相关能力在 workspace 内以自有低秩包装实现。

## 6. 可复用线性层与实际投影路径

- `Right-to-be-forgotten/forget.py::find_all_linear_names()` L22-31：收集 `torch.nn.Linear` 名称并剔除 `lm_head` → 典型返回 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。
- **Llama 路径**：`Right-to-be-forgotten/my_models/my_llama.py`（ASU teacher，带 tau）
  - Attention 输出投影：`LlamaAttention` L328，`self.o_proj = nn.Linear(...)` **L355**，调用 L448/L451/L578/L687。
  - MLP Down：`LlamaMLP` **L290** `self.down_proj = nn.Linear(...)`，调用 L307-311。
  - 完整路径：`model.model.layers[i].self_attn.o_proj`、`model.model.layers[i].mlp.down_proj`。
  - 类：`LlamaForCausalLM` L1131（teacher）；普通 student 用 `transformers.AutoModelForCausalLM`。
- 另两份**无 tau** 的副本：`muse/baselines/my_models/my_llama.py`、`open-unlearning/src/my_models/my_llama.py`。
- **Mistral 路径**：`wmdp/asu/my_mistral.py` `MistralAttention.o_proj` L204、`MistralMLP.down_proj` L154、`MistralForCausalLM` L982。
- `out_proj` / `fc2` 命名：**未发现**。RecodeLoop 只针对 `o_proj` 与 `down_proj`。

## 7. 基线已有 probe / activation hook / 量化 / 重学习 / 审计

| 项 | 结论 | 证据 |
|---|---|---|
| (a) linear probe | **有（有限）** | `open-unlearning/src/model/probe.py::ProbedLlamaForCausalLM` L12、`from_pretrained()` L23-71（截断层 + 重初始化 `lm_head` + 冻结基座 L65-67），注册于 `src/model/__init__.py` L7/L131。这是"截断层+可训练 head"式 probe，**不是标准线性探测分类器** |
| (b) activation hook | **有** | `wmdp/asu/utils.py::forward_with_cache()` L14-34（`module.register_forward_hook` L24）；`wmdp/rmu/utils.py` L16-31；`open-unlearning/src/trainer/unlearn/rmu.py` L87-97 |
| (c) 量化 Q4/Q8 | **未发现实际实现** | 仅依赖声明 `bitsandbytes`（`muse/environment.yml` L9、`open-unlearning/requirements.txt` L9、`Right-to-be-forgotten/requirements.txt` L17）；模型中只有 `_pre_quantization_dtype` 兼容分支（`my_llama.py` L548-550、`my_mistral.py` L370-372）；无 `load_in_4bit/8bit` / `BitsAndBytesConfig` / GPTQ |
| (d) 重学习 / relearning | **未发现训练代码** | `muse/constants.py` L30 `AUC_RETRAIN` 与 `muse/eval.py` L5/L78 仅作隐私审计归一化；无 retrain 训练流程 |
| (e) 审计功能 | **有（评测型）** | `open-unlearning/src/evals/metrics/mia/{all_attacks,gradnorm,loss,min_k,min_k_plus_plus,reference,zlib,utils}.py`；`memorization.py`、`privacy.py`、`utility.py`；`muse/metrics/{privleak,verbmem,knowmem}.py`。**未发现** checkpoint 完整性/溯源审计（无 sha256/manifest） |

## 8. 修改前回归基准与状态记录

- 测试文件：`find . -type f \( -name 'test_*.py' -o -name '*_test.py' \)` → **0 个**。
- `open-unlearning/.github/workflows/tests.yml` L53-56 的 pytest 步骤被注释，实际只跑 `make quality`（ruff）。
- 只读基线语法检查（修改前）：以下 13 个文件全部 `ast.parse` 通过，记为修改前健康状态：
  `Right-to-be-forgotten/{forget.py,eval.py,trainer/losses.py,trainer/trainer.py,dataset/data_module.py,metrics/metrics.py,utils/utils.py,my_models/my_llama.py}`、
  `open-unlearning/src/{model/__init__.py,train.py}`、
  `wmdp/asu/{unlearn.py,utils.py,my_mistral.py}`。
- `python -m pytest --collect-only -q -o cache_dir=workspace/.cache/pytest-baseline` → `no tests collected`，退出码 0。
- 最小基线运行命令（**未运行**，因缺 P08-M2 资产与 TOFU 数据 JSON，记录为回归参照而非通过证据）：
  - `Right-to-be-forgotten/scripts/tofu/asu.sh` L52-67：`torchrun --nproc_per_node=2 ... forget.py --config-name=tofu.yaml ...` 后 `torchrun --nproc_per_node=1 ... eval.py --config-name=tofu.yaml ...`
  - `open-unlearning/README.md` L147 / `docs/experiments.md` L40/48/53：`python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default ...`
  - `muse/README.md` L32：`python unlearn.py ...`
  - `wmdp/run_asu_zephyr.ipynb` L1355：`--model_args pretrained=models/zephyr_asu`

## 9. 对 RecodeLoop 的复用 / 调整 / 隔离判定

- **复用（仅外部 import/配置复用）**：`find_all_linear_names()` 的组件命名约定；`run_generation()` 的 `do_sample=False` 生成口径；`set_random_seed()` 的 seed 语义；`forward_with_cache()` 的 hook 思路（在 workspace 内自行实现可逆版本）；`IGNORE_INDEX=-100` 掩码约定；HF `Trainer`/`TrainingArguments` 的 dtype/AMP/deepspeed 配置约定。
- **调整（仅在 workspace 内）**：冻结模型提取激活；恢复器只解码事实值；局部低秩仅更新 `A,B`；每轮目标为残余恢复 + retain 约束；实体簇隔离的数据划分；外部 hook/context manager 不改 DecoderLayer 拓扑。
- **隔离**：teacher τ、`asu_loss()`、`CustomTrainerForgetting.compute_loss()` 保持原样；Teacher 不作为每轮恢复器；canonical TOFU 与实体级 M0 结果分文件；`forget-confirmation` 由 `AccessGuard` 封存，训练/选择路径不可读；基线输出/缓存/checkpoint 目录不覆盖。

## 10. 已知风险与坑（不得踩）

1. 无 checkpoint manifest → G0 BLOCKED，禁止猜测路径/哈希。
2. `kl_loss` 在 `losses.py` 重复定义，L161 被 L197 覆盖；ASU+KL 实际正则为无 mask batchmean KL。
3. `wmdp/asu/unlearn.py` L63 `args.attention_temp_value` 与 L151 `--attention_temp` 不一致。
4. open-unlearning `configs/trainer/ASU.yaml` 的 `handler: ASU` 未注册。
5. 运行时无 CUDA、无 PEFT；不得假设 GPU/LoRA 库可用。

## 11. 会话后续重新检查（资产补齐）

初始审计后，用户提供并确认了 ASU student checkpoint；外部提交又挂载了数据/运行目录。重新检查结果如下（均为只读核实，未改动基线）：

### 11.1 ASU student checkpoint（用户提供）

- 路径：`/usr/local/llm-unlearning/ASU-unlearning/open-unlearning/checkpoints/asu-forget05-asu-gd`
- 结构：`LlamaForCausalLM`、`model_type=llama`、bfloat16、32 层、hidden=4096、vocab=32000；3 个 safetensors 分片，`model.safetensors.index.json` 共 291 个张量，`safe_open` 可正常打开；tokenizer 文件齐全。
- 自述来源：`README.md` 声明 base=`locuslab/tofu_ft_llama2-7b`、TOFU forget05 task 1、ASU+GD、seed 1001、5 epochs、lr 1e-5、`forget_coeff` 0.1、`regularization_coeff` 1.0、`attention_temp` 2.3、`beta` 0.1、`fix_ref_model=true`、`mask=true`。
- `training_args.bin` 的序列化字符串显示输出目录含 `forget05/ASU+GD/seed_1001/epoch5_1e-05_FixRefTrue_maskTrue_Fcoeff0.1_Rcoeff1.0_Temp2.3_LayersNone_beta0.1/.../unlearn_times_1`，与 README 一致。
- 该 checkpoint 被 `.gitignore:27 checkpoints/` 忽略，不在 git 中；`.cache/huggingface/download/` 仅为下载元数据。
- bundle SHA-256：`b6ecd585b38256c1df6e6db3843934834bd1d87700a27006c391b89643c36abc`

### 11.2 角色映射（用户确认）

| 角色 | 本地路径 | revision | bundle SHA-256 |
|---|---|---|---|
| base | `/dockerdata/asu-unlearning/models/locuslab/tofu_ft_llama2-7b` | `8fa500e8f345f1dd9cfe95bb4689878c944c9cbd` | `91f816bdb7d309e67df8c929b176b3c25d935328baa3d364ed4645eeb0cbef8e` |
| target | `/dockerdata/asu-unlearning/models/open-unlearning/tofu_Llama-2-7b-chat-hf_full` | `cc5b31c69127d5da881608e640f2b453c446435b` | `d2765bba64939130344262bdaaef32b8dc0936983d67367e7e52893d024ee353` |
| asu_student | 11.1 中的本地目录 | null | `b6ecd585b38256c1df6e6db3843934834bd1d87700a27006c391b89643c36abc` |

- base 与 student 的分片哈希逐片不同，确认不是同一份权重。
- bundle 哈希定义：`hashing.py::sha256_bundle` 排除 `.cache/`、`*.lock`、`*.metadata` 后，对 `[[相对路径, 文件 SHA-256], ...]` 的规范化 JSON 求 SHA-256；三个角色均已复核通过。
- 以上已写入 `workspace/manifests/assets.yaml`；`tokenizer_path` 与各模型目录相同（均含 tokenizer 文件）。

### 11.3 数据与当前 G0 状态

- 新挂载数据：`/dockerdata/asu-unlearning/datasets/locuslab-TOFU/` 含 `forget05.json`、`forget05_perturbed.json`、`real_authors*.json`、`world_facts*.json`、`retain95.json` 等；`forget05.json` 为 200 行 JSONL，字段 `question`/`answer`，按作者分块。
- `open-unlearning/saves` 为空；`NousResearch/Llama-2-7b-chat-hf` 与 `meta-llama/Llama-2-7b-chat-hf` 本地只有 tokenizer/config，无权重（角色已按用户确认改用上述 base/target，故不阻塞）。
- 随后运行 `scripts/build_entity_splits.py`，由 `forget05.json`（sha256 `88542818f98fd6f880ecaea71bc3f3e7a34f959944e5ff1cdef7ea8c5c137173`）的 10 个连续 20 行作者块构造实体簇，按 canonical name 排序确定划分；canonical hash `571f9387be702e73ba26a6fc5a87d6c36b5087400b6c9a14185171fdd43e266c`。
- 划分：forget-train 编辑 2 簇（Aysha Al-Hashim、Basil Mahfouz Al-Kuwaiti）、审计 reader 2 簇（Edward Patrick Sullivan、Hina Ameen）、calibration 2 簇（Kalkidan Abera、Moshe Ben-David）、confirmation 4 簇（Nikolai Abilov、Raven Marais、Takashi Nakamura、Xin Lee Williams）。
- 重新运行 `scripts/validate_assets.py`：**G0 = PASS**（退出 0），13 项检查全部通过，包括 `clusters=10` 与 `confirmation_count=4`。
- 结论：G0 资产与实体隔离已冻结；**这并不等于 M0a/M0b 通过**。后续正对照、NK 对照、reader 有效性等仍未运行，confirmation 仍保持封存。

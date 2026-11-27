# CF-04 vs 官方 ASU 对齐复现实验协议

`claim_level=false`　`gate_status=NOT_APPLIED`　执行卡：仅 GPU 0（GPU 1 被外部任务占用）。

## 0. 用户已确认的三项口径

| 决策 | 取值 |
|---|---|
| 官方 ASU 配置来源 | `Right-to-be-forgotten/config/tofu.yaml` + `scripts/tofu/asu.sh`（官方 repo，非 dashboard 的 10-epoch 口径） |
| 模型/算力 | 本地 1B checkpoint + 官方超参，只在 GPU 0；显式声明「模型不同，非 faithful official reproduction」 |
| 评估与统计 | 统一 NLL + 生成行为 + 深度审计；3 seeds mean±std、配对差值、效应量、bootstrap CI、预注册门槛 |

## 1. 官方配置与本机对齐映射

来源：`Right-to-be-forgotten/config/tofu.yaml`（下称 *yaml*）与 `scripts/tofu/asu.sh`（下称 *script*）。

| 官方字段 | yaml | script 覆盖 | 本次对齐值 | 说明 |
|---|---|---|---|---|
| `num_epochs` | 5 | 5 | **5** | 关键；两臂相同 |
| `batch_size` | 4 | — | **4** | 关键；两臂相同 |
| `gradient_accumulation_steps` | 4 | — | **4** | effective batch = 16；两臂相同 |
| `lr` | 1e-5 | 1e-5 | **1e-5** | 关键；两臂相同 |
| `weight_decay` | 0.01 | — | **0.01** | 两臂相同（项目旧 harness 为 0.0，此处按官方对齐） |
| `seed` | 1001 | — | **1001 / 1002 / 1003** | 官方仅 1001；本实验补 2 个 seed 做稳定性，属新增而非改动 |
| `use_LoRA` | false | false | **false（full finetune, fp32）** | 两臂相同 |
| `fix_ref_model` | true | — | **true** | ASU 臂冻结 ref；CF-04 臂探针在训练前拟合后冻结 |
| `mask` | true | true | **true（answer-only label）** | 两臂相同 |
| `attention_temp` | 2.0 | 2.3 | **ASU 主口径 2.0；另做 2.3 单 seed 敏感性** | yaml/script 不一致，两者都记录 |
| `forget_coeff` / `regularization_coeff` / `beta` | 0.1 / 1.0 / 0.1 | 0.1 / 1.0 / — | **不映射** | 官方 `ASU+GD` 与本机代理 loss 无法 1:1 对应；记为偏差 |
| `forget_loss` | `ASU+GD` | `ASU+GD`,`ASU+KL` | **本机 ASU 代理（温度化 ref + answer-KL + retain NLL）** | 记为偏差 |
| `split` | `forget05` | forget01/05/10 | **forget10 / retain90 / holdout10** | 偏差：CF-04 实现与 1B checkpoint 绑定该 split；另在官方 `forget05` 上做一次无训练的评估敏感性 |
| `model_path` | `locuslab/tofu_ft_llama2-7b` | — | `tofu_Llama-3.2-1B-Instruct_full` | 偏差：7B→1B |
| `save_checkpoint` | false | false | **true** | 必要偏差：深度审计与生成评估需要权重 |
| `ds_size` / eval tasks | 300 / perturbed 四任务 | — | 全 split NLL + 4 样本×24 token 生成 + P1/P2/P3/INT8 + Probe | 用户选定口径；官方 perturbed 任务未复现，记为局限 |

### 关键：epoch → optimizer step 的确定

- 官方：1 epoch = 覆盖 forget split 一次；`forget05=200`、effective batch 16 → `ceil(200/16)=13` steps/epoch、5 epochs ≈ 65 steps。
- 本实验：数据为 `forget10=400`、effective batch 16 → **`steps_per_epoch = ceil(400/16) = 25`，5 epochs = 125 optimizer steps**。
- 每个 optimizer step = 4 个 micro-step（batch 4）。**两臂完全相同的 125 steps × 4 micro-steps = 500 次前向/反向预算**，且同一 seed 下 forget/retain 采样顺序逐 micro-step 一致（共用 `make_order(seed)`）。
- 保留记录：官方 65 steps 与本实验 125 steps 的差异来自 split 规模，是 epoch 口径一致的必然后果，不是预算不对齐。

### 方法固有项（不是实验条件不一致）

- ASU 臂：冻结 ref 模型 + `TemperatureContext`（attention 温度）+ answer-only KL + retain NLL。
- CF-04 臂：训练前用冻结初始模型拟合 fact/identity 双探针（layers 8/12/16、300 steps）、自适应 lambda retain 约束、`probe_guided_loss_v2`。
- 这两项是方法定义本身，两臂的**共享**训练预算、采样、优化器、评估完全相同；探针拟合属训练前预处理，不计入 125 steps。

## 2. 实验条件一致性清单（逐项核验）

| # | 条件 | 两臂取值 | 核验方式 |
|---|---|---|---|
| C1 | base checkpoint / tokenizer / dtype / attention | 同一 1B、fp32、sdpa/eager 一致 | `resolved_config.json` |
| C2 | 数据文件 | forget10 / retain90 / holdout10 | 同上 |
| C3 | 数据顺序 | 同一 seed 的 `make_order(seed)` 输出逐 micro-step 相同 | 日志中记录 order hash |
| C4 | optimizer | `AdamW(lr=1e-5, weight_decay=0.01)` | 训练脚本常量 |
| C5 | grad clip | 1.0 | 训练脚本常量 |
| C6 | 预算 | 5 epochs / 125 steps / batch 4 × accum 4 | `train.history` 长度 = 125 |
| C7 | seeds | 1001 / 1002 / 1003，逐 seed 配对 | 目录命名 |
| C8 | 评估函数 | 同一个 `eval_nll`（token 加权、全 split、eval_batch=8） | 共用 `evaluate_aligned.py` |
| C9 | 生成行为参数 | 4 samples × 24 tokens、显式贪心、同 EOS 判据 | 共用 `behaviour_stats` |
| C10 | 深度审计参数 | P1(64) / P2 few-shot(32) / P3 20-step relearn(lr 1e-5) / INT8 fake-quant / Probe layer16 200 steps | 共用 `cf04_depth_audit` 原语 |
| C11 | 硬件 | `CUDA_VISIBLE_DEVICES=0`、串行 | `nvidia-smi` 日志 |
| C12 | checkpoint 保存格式 | `save_pretrained` + tokenizer | 加载可行性校验 |

## 3. 评估指标（具体定义）

### 3.1 主指标（全 split，teacher-forced NLL，只用数据集原始 answer）

- `dF = after.forget_nll − before.forget_nll`（越高越好）
- `dR = after.retain_nll − before.retain_nll`（越低越好）
- `dH = after.holdout_nll − before.holdout_nll`（越低越好）
- `before` 用 base checkpoint 测一次，两臂共用。

### 3.2 生成行为（可用性）

- `eos_rate`、`mean_gen_tokens`、`mean_answer_chars`、`mean_target_overlap`、`high_overlap_rate`、`refusal_rate`、`short_rate`（4 个 forget 样本、24 token、显式贪心）
- `mean_target_overlap_equal_length`：截断到 13 token 后的 overlap，用于区分「只是生成更长」与「输出退化」。

### 3.3 深度审计（有效性/抗恢复）

- `P1_direct` forget/retain NLL、`P2_fewshot.gain`、`P3_relearn.gain`（20 步）、`INT8_fakequant.forget_shift`
- `mean_recovery = (clip(P2_gain/Δ,0,1) + clip(P3_gain/Δ,0,1))/2`，`Δ = P1_forget − base_P1_forget`
- 独立线性 Probe：`probe_auc_pre/post`、`probe_recoverability_pre/post`（eval seeds 0/1/2 的 mean±std）

### 3.4 预注册门槛（沿用 `CF04_V2_OPTIMIZATION_REPORT.md`）

| 判定 | 条件 |
|---|---|
| retain-safe | `dF ≥ 0.30` 且 `dR ≤ 0.05` |
| strong | `dF ≥ 0.90` 且 `dR ≤ 0.05` |
| holdout soft | `dH ≤ 0.10` |
| 可用（usability） | 无 OOM/NaN、梯度有限、checkpoint 可加载、`eos_rate` 未崩塌、`target_overlap` 未崩塌 |
| 深度有效（depth） | `mean_recovery < 0.30` 且 `probe_auc_post` 明显低于 `probe_auc_pre` |

`gate.verdict=PASS` 在深度审计里表示「检测到浅层遗忘」，是**负面**信号，不是方法通过。

## 4. 差异分析方法

对每个 seed `s ∈ {1001,1002,1003}`，两臂在同一 seed 下训练，构成**配对设计**：

1. 逐 seed 原始值表：两臂各自 `dF/dR/dH/eos_rate/overlap/mean_recovery/probe_auc_post`。
2. 配对差值 `Δ_s = CF04_s − ASU_s`（每个指标）；报告 3 seeds 的 `mean ± population std`、符号一致性 `k/3`。
3. 效应量：`Cohen's dz = mean(Δ)/std(Δ)`；`std(Δ)=0` 时标注为不可计算，不补数。
4. 不确定性：对 3 个配对差值做 10000 次有放回 bootstrap（固定 rng seed），给 95% percentile CI。**n=3 的 CI 极粗，只作描述，不作统计显著性结论**。
5. 结论规则（同时满足才算「CF-04 相对 ASU 有条件优势」）：
   - 逐 seed 差值方向一致（≥3/3）；
   - `dF` 差值 > 0，且 `dR` 差值 ≤ 0（不牺牲 retain）；
   - `dH` 差值 ≤ 0 或两者都超 soft 门槛且差值不显著恶化；
   - 两臂都通过「可用」检查；
   - 深度审计不显示 CF-04 更浅。
   否则写 `INCONCLUSIVE` 或 `NEGATIVE`，并给出具体失败项。
6. 明确的**不可推断项**：不宣称统计显著、不宣称优于官方 7B ASU、不把输出退化当遗忘、不把进程完成当有效。

## 5. 已知偏差与局限性（必须写入结论）

- 模型 7B→1B（vocab 32000→128256、hidden 4096→2048、32→16 层）；数据 `forget05→forget10`；双卡 torchrun→单卡；官方 `ASU+GD` 完整损失→本机代理；官方 perturbed 四任务评估→统一 NLL/行为/审计；官方 `save_checkpoint=false`→true；官方单 seed→3 seeds。
- CF-04 探针拟合与 ASU 温度 ref 属方法固有差异；两臂前向次数不完全相同（ASU 每 micro-step 多一次 ref 前向），因此「等 optimizer steps / 等数据暴露」成立，但「等 FLOPs」不成立，需报告各自耗时与显存。
- 3 seeds 统计功效不足；`attention_temp` yaml=2.0 与 script=2.3 不一致，只对 seed 1001 做 2.3 敏感性。
- 官方 7B/forget05 的真实数值未在本机复现，本实验结论限定在「1B / forget10 / 本机代理实现」这一实际场景。

## 6. 执行与产物

```bash
cd workspace/experiments/asu_cf04_aligned
nohup env CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=<冻结环境> bash run_all.sh > results/console.log 2>&1 &
```

`run_all.sh` 串行执行：ASU(tau2.0)×3 seeds、CF-04×3 seeds、ASU(tau2.3)×seed1001 敏感性 → 7 次训练；随后统一评估与配对差异分析。

产物：`PROTOCOL.md`（本文件）、`results/<arm>_<seed>/{result.json,resolved_config.json,checkpoint/,run.log}`、`results/evaluation.json`、`results/paired_analysis.json`、`ALIGNED_COMPARISON.md`。

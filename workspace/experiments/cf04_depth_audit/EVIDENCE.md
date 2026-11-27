# CF-04 v2g 深度审计 — 证据盘点（阶段 0）

生成时间：2026-11-27。`claim_level=false`、`gate_status=NOT_APPLIED`、`evidence_gate=BLOCKED`。本文件只做只读证据盘点，不构成 claim。

## 1. 审计对象：CF-04 v2g（300 steps, 3 seeds）

来源：`workspace/auto_experiments/results/cf04_v2_300/v2g{,seed1,seed2}/CF-04.json`；汇总见 `workspace/refine-logs/CF04_V2_OPTIMIZATION_REPORT.md` 与 `PROJECT_STATUS_DASHBOARD.md`。

| seed | steps | dF (forget NLL) | dR (retain NLL) | dH (holdout) | elapsed_sec |
|---:|---:|---:|---:|---:|---:|
| 0 | 300 | +2.4152 | +0.0222 | +0.1874 | 214.9 |
| 1 | 300 | +2.3181 | +0.0167 | +0.1573 | 428.53 |
| 2 | 300 | +2.1918 | +0.0151 | +0.1816 | 407.13 |
| **3-seed mean±std** | | **+2.3084 ± 0.0915** | **+0.0180 ± 0.0031** | **+0.1754 ± 0.0131** | — |

配置（seed 0 记录）：300 steps、lr 6e-6、batch 4、grad_accum 8、`dtype=float32`、full split eval（forget 400 / retain 3600 / holdout 400）、`eval_batch_size=8`、`max_pairs=36`；`cf04_layers=8,12,16`、`cf04_margin=1.0`、`cf04_lambda_init=3.0`、`cf04_identity_weight=2.0`、`cf04_probe_retain_weight=2.5`、`cf04_retain_nll_weight=0.5`、`cf04_grad_project=false`、`probe_steps=300`、`probe_lr=0.05`。模型与数据为 `/jizhicfs/chenyliao` 下的真实 Llama-3.2-1B-Instruct / TOFU forget10 / retain90 / holdout10。

判定：满足 `dF≥0.90 且 dR≤0.05`（strong）与 `dF≥0.30 且 dR≤0.05`（retain-safe），dashboard 排名第一；但 `dH≈+0.175` 说明 holdout 退化，且 `verdict=DIAGNOSTIC_RUN`、`claim_level=false`。

### 生成行为异常（seed 0，forget split，4 samples / 24 tokens）

| 指标 | before | after |
|---|---:|---:|
| eos_rate | 1.0 | **0.0** |
| mean_gen_tokens | 13.75 | **24.0** |
| mean_answer_chars | 44.5 | **93.0** |
| mean_target_overlap | 0.4886 | 0.5076 |
| high_overlap_rate | 0.75 | 0.50 |
| refusal_rate / short_rate | 0.0 / 0.0 | 0.0 / 0.0 |

即：训练后不再生成 EOS、长度翻倍，但答案重合度几乎不变。**dF 的上升可能主要由输出抑制 / 长度与 EOS 漂移解释，而非知识移除**——这是本次深度审计的核心待检验假设。

## 2. 同方法不同预算：formal 10 epochs（4500 steps）

来源：`workspace/auto_experiments/results/formal_training/CF-04_v2_formal*.json`。

| 记录 | epochs | dF | dR | dH | checkpoint |
|---|---:|---:|---:|---:|---|
| CF-04_v2_formal | 10 | +8.0920 | -0.4561 | +0.9656 | `/jizhicfs/.../checkpoints/full_cf04_v2_formal` |
| CF-04_v2_formal_save | 10 | +8.0920 | -0.4561 | +0.9656 | `/jizhicfs/.../checkpoints/full_cf04_v2_formal_save` |
| CF-04_formal（v1） | 10 | +0.6294 | -0.4609 | +0.8917 | `/jizhicfs/.../checkpoints/full_cf04_formal` |

**预算差异声明**：formal 是 4500 steps、v2g 是 300 steps，两者不能直接比较，formal 的审计结论不得归给 3-seed v2g，也不与 `methods-1127` 的 pilot 结果混算。

## 3. checkpoint 清单（只读 header 检查，2026-11-27）

| 名称 | 路径 | config/tokenizer | tensors | 权重大小 | 备注 |
|---|---|---|---:|---:|---|
| cf04_v2_formal（主臂） | `/jizhicfs/chenyliao/llm-unlearning/checkpoints/full_cf04_v2_formal` | OK | 146 | 4.6GB | fp32，dashboard P3 未审计目标 |
| cf04_v2_formal_save | `.../full_cf04_v2_formal_save` | OK | 146 | 4.6GB | 与上一行同结果 |
| cf04_formal | `.../full_cf04_formal` | OK | 146 | 4.6GB | v1 |
| cf04_e1 | `.../full_cf04_e1` | OK | 146 | 4.6GB | 1 epoch |
| asu_forget05 | `.../asu-forget05-asu-gd` | OK | 分片 3 个 | ~13GB | forget05 split，口径不同，仅参考 |
| grad_ascent | `/jizhicfs/chenyliao/runs/open-unlearning/20261122_forget10_1b/GradAscent` | cfg/tok 文件在，但 `tokenizer_class=TokenizersBackend` 与冻结环境 transformers 4.46.3 不兼容 | 146 | 2.3GB | 审计脚本只用 `--base` 的 tokenizer，故不影响 |
| base | `/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full` | OK | 146 | 2.3GB | 真实 Llama-3.2-1B-Instruct |

**300-step v2g 没有保存 checkpoint**（`auto_experiments/checkpoints/` 不存在）。这是主臂改用 formal checkpoint 的原因，也是「可选重训第二臂」的动机。

## 4. 既有深度审计基线（口径对齐目标）

`workspace/auto_experiments/results/audit_depth.json`（P1/P2/P3/INT8 + `depth_score`）：

| 模型 | P1 forget | P1 retain | P2 gain | P3 gain | INT8 shift | mean_recovery |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.4621 | 0.4700 | -0.0158 | +0.0709 | +0.1946 | — |
| asu（历史路径，已不在本机） | 0.9604 | 0.4222 | +0.2613 | +0.1850 | +0.2568 | 0.4477 |
| grad_ascent | 0.9335 | 0.8012 | +0.2607 | +0.0970 | +0.1614 | 0.1595 |

`results/audit_depth_probe_3seed.json` 提供 Probe 通道（`probe_auc_pre/post`、`probe_recoverability_pre/post`，3 seed mean±std）。新审计工具必须复现 base 的 P1/P2/P3/INT8 数值以证明口径一致。

## 5. dashboard 待办与边界

- P1：ASU vs CF-04 v2g 在官方 ASU 预算下的公平比较（未开始）。
- P2：CF-04 v3 修复 holdout 退化（dH≈0.175 / formal 0.966）。
- P3：**R3-2 审计 CF-04 v2 checkpoint（未开始）—— 本次规划对应项**。
- P4：ASU 同 10-epoch 对照。
- `HANDOFF.md phase=init/status=blocked`、`INIT_REPORT.md baseline=UNFROZEN`；`DIAGNOSTIC_PROTOCOL.md` 明确 `claim_level=false`、forbidden `/auto-review-loop`、`/experiment-audit`、`/result-to-claim`。ARIS 证据门仍 BLOCKED，本次不解除。
- 2026-11-26 重新开放方向 I 的裁决只允许基于新机制的实验，stage0–14 历史结果不得复用为正向证据。

## 6. 资源状态

- GPU 0：0 MiB / 0%（本任务唯一可用卡）；GPU 1：约 33.8GiB / 97%，外部任务占用，不使用。
- `methods-1127` 的 26 个运行期修复仍未提交，本任务不碰、不 commit。
- 所有 GPU 任务固定 `CUDA_VISIBLE_DEVICES=0`、`MAX_PARALLEL=1`。

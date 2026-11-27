# CF-04 v2 遗忘深度审计报告

生成时间：2026-11-27　`claim_level=false`　`gate_status=NOT_APPLIED`　`evidence_gate=BLOCKED`
执行卡：仅 GPU 0（`CUDA_VISIBLE_DEVICES=0`，串行）；GPU 1 全程被外部任务占用，未使用。未调用任何 ARIS review / audit / claim 流程。

## 结论（先给答案）

**对本次唯一可审计的 CF-04 v2 checkpoint（`full_cf04_v2_formal`，4500 steps）判定为「浅层输出抑制」，不支持「深度遗忘」：**

- 20 步 relearning（P3）恢复了 forget-NLL 增益的 **86.2%**（+7.2255 / 8.3786），few-shot（P2）恢复 27.5%，`mean_recovery=0.5685`（≥0.30 的浅层判据）。
- 独立线性 Probe 在冻结隐藏态上对 forget/retain 的 held-out AUC 仍为 **0.9995 → 0.9940**（relearning 后），recoverability 0.999 → 0.988：事实信息几乎完全线性可读。
- 生成行为崩坏：EOS 率 **1.0 → 0.0**、生成长度打满 24 tokens、`target_overlap` 从 0.489 掉到 **0.083**，输出退化为重复/无意义文本。等长度截断到 13 tokens 后 overlap 仍为 0.083，说明**不是单纯长度差异**能解释，而是输出分布本身被破坏。
- 因此 dF=+8.09（formal）/ +2.31（v2g）这类增益至少在该 checkpoint 上主要来自「不结束生成 / 生成退化」，而非知识移除。

**对用户所指的 300-step v2g 配置，结论是「证据不足」**：该配置未保存 checkpoint，无法做 P1/P2/P3/INT8/Probe 审计。formal v2 的崩坏程度明显重于 v2g 报告中的 `overlap=0.508`，**不得把本报告的数值或判定外推为 v2g 的结论**。

**设计复现优化：本轮不做**。按已批准规划，先审计、后优化；任何改算法/损失/超参的工作需另行批准。

## 1. 审计对象与预算声明

| 项 | 内容 |
|---|---|
| 用户所指对象 | 旧 `CF-04 v2g`，300 steps / 3 seeds，dF=+2.3084±0.0915、dR=+0.0180±0.0031、dH=+0.1754±0.0131 |
| 实际可审计权重 | `/jizhicfs/chenyliao/llm-unlearning/checkpoints/full_cf04_v2_formal`，10 epochs / 4500 steps |
| 差异原因 | `results/cf04_v2_300/v2g*/CF-04.json` 中 `checkpoint` 缺失、`train=null`，`auto_experiments/checkpoints/` 不存在；300-step 权重未保存 |
| 对照 | `base`（真实 Llama-3.2-1B-Instruct TOFU）、`ga`（GradAscent）、`cf04_e1`（1 epoch）、`cf04_formal`（v1，10 epochs） |
| 跳过 | `asu-forget05-asu-gd`：7B / vocab 32000 / hidden 4096，架构守卫拒绝；且为 forget05 split，口径不同，仅作历史数值参考 |

详细臂决策见 `ARMS.md`，证据盘点见 `EVIDENCE.md`。

## 2. 口径与 checkpoint 完整性验证（进入审计前的闸门）

1. **口径对齐**：`audit.py --models base` 的 P1/P2/P3/INT8 十个字段与历史 `auto_experiments/results/audit_depth.json` 的 `base` **完全一致（最大差 0.0）**。过程中发现历史 P1/r1/fq/rq/f3/r3 用的是「前 64 条单 batch 未加权 NLL」，而 P2 用 token 加权 `eval_nll`；移植时已按此实现 `direct_nll`（`results/alignment_base.json`）。用冻结环境与系统 Python 结果相同，排除库版本差异。
2. **checkpoint ↔ 报告对应**：重算 `full_cf04_v2_formal` 的完整 split forget/retain/holdout NLL，与 `results/formal_training/CF-04_v2_formal.json` 的 `after` 逐项对比，最大差 **3.0e-6**（`results/checkpoint_integrity.json`）。确认权重确实对应 dF=+8.0920 / dR=-0.4561 / dH=+0.9656。
3. 全部结果 JSON 无 NaN/Inf（有限性检查 0 项异常）。

## 3. P1 / P2 / P3 / INT8 结果（`results/cf04_depth_audit.json`）

参数：`eval_limit=64`、`eval_batch_size=8`、`fewshot_limit=32`、`relearn_limit=32`、`relearn_steps=20`、`relearn_lr=1e-5`；`gate.verdict=PASS` 表示**检测到浅层遗忘**，不是方法通过。

| 模型 | P1 forget | P1 retain | P2 gain | P3 gain | P3 retain | INT8 shift | `mean_recovery` |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.4621 | 0.4700 | -0.0158 | +0.0709 | 0.4138 | +0.1946 | — |
| **cf04 (v2 formal)** | **8.8406** | 0.0178 | **+2.3001** | **+7.2255** | 0.0253 | -0.5316 | **0.5685** |
| cf04_formal (v1) | 1.4633 | 0.0237 | +0.0276 | +0.3253 | 0.0243 | +0.1901 | 0.1763 |
| cf04_e1 | 0.8717 | 0.1945 | +0.0164 | +0.0991 | 0.1970 | +0.1942 | 0.1409 |
| ga | 0.9335 | 0.8012 | +0.2607 | +0.0970 | 0.7331 | +0.1614 | 0.3794 |

以 `cf04` 的 forget 增益 8.3786（相对 base）为分母：P2 恢复 27.5%、P3 恢复 86.2%、INT8 fake-quant 反而使 forget NLL 下降 6.3%（不是主要恢复通道）。预算阶梯显示 `mean_recovery` 随训练量/版本上升（e1 0.141 → v1 0.176 → v2 0.568），即**更激进的遗忘伴随更浅、更易恢复的效应**。

## 4. 独立 Probe 3-seed（`results/cf04_probe_3seed.json`）

seeds 0/1/2，`probe_steps=200`，`layer=16`，held-out 线性可分性测试。

| 模型 | P1 forget | P2 gain | P3 gain | INT8 shift | probe AUC pre | probe AUC post | recoverability pre/post | `mean_recovery` |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| base | 0.4621 | -0.0158 | +0.0660 | +0.1946 | 0.9817 | 0.9817 | 0.9634 / 0.9634 | — |
| **cf04** | 8.8406 | +2.3001 | +7.1994 | -0.5316 | **0.9995** | **0.9940** | **0.9990 / 0.9881** | 0.5669 |
| ga | 0.9335 | +0.2607 | +0.0949 | +0.1614 | 0.9861 | 0.9871 | 0.9723 / 0.9743 | 0.3772 |

Probe 在「遗忘后」和「relearning 后」都几乎完美可分，说明事实信息仍留在冻结表示中；P3 的 NLL 恢复也与之吻合。`gate.verdict=PASS` 同样是「浅层遗忘被检测到」。

## 5. EOS / 长度漂移诊断（`results/cf04_behavior.json`）

显式贪心解码、`max_new_tokens=24`、4 个 forget 样本；字段与历史 `eval_generate.py` 一致，另加等长度（13 tokens）截断对照。

| 模型 | P1 forget | eos_rate | mean tokens | mean chars | target_overlap | overlap@13 | 样例 |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 0.4621 | 1.00 | 13.75 | 44.5 | 0.489 | 0.443 | `Hsiao Yun-Hwa.` |
| **cf04 (v2 formal)** | 8.8406 | **0.00** | **24.00** | 167.25 | **0.083** | **0.083** | `He Alucheuche Alalohechemhehehehehehehehehe (He is a He He He` |
| cf04_formal (v1) | 1.4633 | 1.00 | 14.50 | — | 0.466 | 0.443 | 正常 |
| cf04_e1 | 0.8717 | 1.00 | 17.25 | — | 0.549 | 0.485 | 正常 |
| ga | 0.9335 | 1.00 | 11.75 | 39.5 | 0.299 | 0.277 | `The author's name is not provided.` |

要点：

- `base` 的 eos/tokens/overlap 与 v2g 报告的 `before` 完全一致（1.0 / 13.75 / 0.489），证明解码口径对齐。
- 只有 v2 formal 出现 EOS 全失 + 打满 token 上限 + overlap 跌到 0.083；1-epoch 与 v1-formal 都保持正常 EOS。这与 P3 高恢复率共同指向「输出抑制 / 分布崩坏」而非知识移除。
- 等长度截断并不恢复 overlap（0.083 → 0.083），排除「只是生成更长」这一单一解释。
- v2g（300 steps）报告中的 `after` 是 eos 0.0 / 24 tokens / overlap **0.508**：方向相同但程度远轻于 formal v2。两者不可混算。

## 6. 判定逻辑

沿用既有 `depth_score` 口径：`p2/p3_gain_ratio = clip(gain / (P1_forget - base_P1_forget), 0, 1)`，`mean_recovery` 为二者均值；≥0.30 判「浅层遗忘被检测到」。cf04 的 0.5685 明确过线，且 Probe AUC 接近 1、生成崩坏、relearning 恢复 86%。因此三选一结论为：

- 支持较深遗忘：**否**
- **浅层输出抑制：是**（针对 formal v2 checkpoint）
- 证据不足：**是**（针对用户所指的 300-step v2g 配置，因无 checkpoint）

## 7. 与 v2g 3-seed 结果的关系及限制

- 本报告**不能**改写 v2g 的 dF/dR/dH 记录，也不能据 formal 审计宣布 v2g「无效」或「有效」。
- v2g 的 `after.behavior` 已显示 EOS 归零、长度翻倍（overlap 仍 0.508），提示同类风险；但缺少 P2/P3/Probe 直接证据，定性只能写「存疑」。
- `asu` 对照为 forget05 split 且架构不匹配，被守卫跳过；历史 `audit_depth.json` 的 ASU `mean_recovery=0.4477` 只作文字参考，不纳入本次统计。
- 300-step v2g 的 faithful 复现需移植 `diagnostic_methods/run_one_method.py`（600 行）+ `run_full_training.py`（273 行）的相对导入逻辑并加 `--save`，重训 3 seeds 且 dF/dR/dH 对齐后才可审计（见 `ARMS.md`）。**本轮未启用，需单独批准**。

## 8. 后续「设计复现优化」候选方向（需另行批准，本轮未执行）

1. 先补 300-step v2g 的 checkpoint 复现，再对同一配置执行本报告同一套审计，得到真正对应 v2g 的恢复率。
2. 修复输出退化：恢复 EOS / 长度控制，并把「等长度、强制 EOS」纳入评估，避免用生成崩坏换取 dF。
3. 修复 holdout 退化：v2g 的 dH≈+0.175、formal v2 的 dH≈+0.966（dashboard P2）。
4. 公平预算对照（dashboard P1/P4）：ASU 与 CF-04 v2g 在官方 ASU 预算、同一 10-epoch 控制下比较。
5. 只有上述审计与对照都支持，才考虑受限地讨论遗忘深度；仍须独立 review / integrity 审计，当前 ARIS 证据门 BLOCKED 不解除。

## 9. 命令、产物与资源

```bash
cd workspace/experiments/cf04_depth_audit
# 口径对齐
CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=<冻结环境> python audit.py --models base --output results/alignment_base.json
# 全量（常驻）
nohup env CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=<冻结环境> bash run_audit.sh > results/console.log 2>&1 &
```

`run_audit.sh` 依次执行：

```text
audit.py          --models base,cf04,cf04_formal,cf04_e1,ga --output results/cf04_depth_audit.json
probe.py          --models base,cf04,ga --seeds 0,1,2       --output results/cf04_probe_3seed.json
behavior_check.py --models base,cf04,cf04_formal,cf04_e1,ga --output results/cf04_behavior.json
```

冻结环境：`/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python`（torch 2.5.1+cu121 / transformers 4.46.3）。主 PID 83169，无遗留进程。

| 脚本 | 耗时 | 产物 |
|---|---:|---|
| 口径对齐 | 25.6s | `results/alignment_base.json` |
| checkpoint 完整性 | 37.7s | `results/checkpoint_integrity.json` |
| P1/P2/P3/INT8（5 模型） | 78.4s | `results/cf04_depth_audit.json` |
| Probe 3-seed（3 模型 × 3 seed） | 145.3s | `results/cf04_probe_3seed.json` |
| EOS/长度诊断（5 模型） | 40.0s | `results/cf04_behavior.json` |

GPU 0 峰值显存 **47849 MiB / 97871 MiB**，无 OOM；GPU 1 外部占用约 33.7GiB 未触碰。日志：`results/{console,audit,probe,behavior}.log`、`results/gpu_mem.log`。

## 10. 边界与状态

- 全程未改算法、损失公式/系数、评价口径；未做超参搜索，未训练新方法，未调用旧 `auto_experiments` / `diagnostic_methods` 脚本作为训练入口（仅只读参照其实现）。
- `methods-1127` 的 26 个未提交修复保持原状，本目录新增文件均为未跟踪状态，未做任何 git 写操作/提交。
- 其余 9 个 CF2 方法的机制判别门按用户决定**暂缓**；既有 `methods-1127/_validation/repro_1127/` 产物未删除。
- `DIAGNOSTIC_KILL` 是机制筛选结果、不是代码失败；本次审计不产生该状态。
- `claim_level=false`；不写「方法有效」「已通过证据门」；不把 stage0–14 历史结果复用为正向证据；`HANDOFF.md` / `INIT_REPORT.md` 的 BLOCKED 不解除。

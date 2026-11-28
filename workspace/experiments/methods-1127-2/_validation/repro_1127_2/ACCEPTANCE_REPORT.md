# CF3 methods-1127-2 远程验收报告

生成时间：2026-11-28。`claim_level=false`；本报告如实记录失败与阻塞，不编造任何未完成实验的数值。

## 1. 环境与代码

| 项 | 值 |
|---|---|
| 代码路径 | `/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/methods-1127-2` |
| 仓库 / 分支 | `ASU-unlearning` / `Auto/20261122` |
| 模型 | `/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full` |
| 数据 | `/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/{forget10,retain90,holdout10}.json` |
| Python | 项目冻结环境 `.../envs/open-unlearning/bin/python`；`INSTALL=0`（依赖已齐） |
| 依赖 | torch 2.5.1+cu121、transformers 4.46.3、peft 0.21.3、accelerate 1.13.0、numpy 2.2.3、safetensors 0.7.0 |
| GPU/CUDA | 2×NVIDIA H20 97871MiB，driver 535.247.01 / CUDA 12.1；本任务**仅 GPU0**，GPU1 为外部任务 |
| 路径覆盖 | 无；`preflight.py --profile pilot/smoke` 均 PASS，无需 `--set model.path` / `data.root` |
| 代码改动 | 均为运行接口修复，**未提交**，diff 见第 6 节 |

静态检查：`validate_all.py --operators-only` 10/10 PASS；`python3 -m compileall CF3-* *.py` PASS；`bash -n start.sh slurm.sbatch CF3-*/run.sh` PASS。随上传带来的 `queue_status.json` 与 `_validation/` 是本地 macOS 的 tiny 产物，已备份为 `*_uploaded_tiny_snapshot.json`，不计入远程验收。

## 2. 已有 GPU0 队列与串行接入

- 无 Slurm（`sbatch/squeue` 不存在）。已有队列是 `asu_cf04_aligned/run_all.sh`（PID 92526，nohup 串行脚本）。
- 外层脚本 `queue_join_cf3.sh` 等待条件：PID 92526 退出 + GPU0 显存连续 3 次 <100MiB + 无训练进程；首次等待 1303 秒（00:12:40→00:34:23）。
- 闸门发现此前队列在最后 `analyze.py` 报 `NameError: name 'SEEDS' is not defined`，`run_all.sh` 在 `set -e` 下退出。按用户规则**停下、未启动 smoke、未重跑或清理**。用户裁决 `PROCEED_AFTER_PREVIOUS_SUMMARY_COMPLETED`：仅修 `analyze.py` 默认参数并重跑 CPU-only 汇总（diff 已存），训练/评估产物保留；重接入后 GPU0 空闲 40 秒放行。
- 全程未抢占/插队/暂停/终止，`pkill` 未使用，无强制释放显存，未删除其他任务文件。GPU1 的 lrm-unlearning 队列未触碰。

| 阶段 | 开始 | 结束 | 退出码 | 说明 |
|---|---|---|---|---|
| 静态检查+预检 | 00:02 | 00:12 | 0 | 不占 GPU |
| smoke #1 | 09:31:47 | 09:39:31 | 1 | 10/10 JSONL loader |
| smoke #2 | 09:42:32 | 09:58:23 | 1 | 10/10 A/B/C/D |
| smoke #3（8 方法） | 10:08:44 | 10:15:19 | 1 | 6 COMPLETED, 2 SIGFPE |
| smoke #4（CF3-01/03） | 10:23 | 10:26 | 0 | 2/2 COMPLETED |
| 严格诊断（不训练） | 10:27:36 | 10:33:51 | 1 | 7 配对契约失败, 1 DIAGNOSTIC_KILL |
| pilot | 10:34:32 | 10:42:39 | 1 | 7 配对契约失败, 1 DIAGNOSTIC_KILL |

## 3. 十方法状态矩阵

| 方法 | smoke | 严格诊断（pilot 档、不训练） | pilot |
|---|---|---|---|
| CF3-01 | COMPLETED | FAILED（配对契约） | FAILED（配对契约） |
| CF3-02 | BLOCKED（method.py replay shapes） | 未运行（用户授权标为阻塞） | 未运行 |
| CF3-03 | COMPLETED | FAILED（配对契约） | FAILED（配对契约） |
| CF3-04 | BLOCKED（method.py shape stack） | 未运行（用户授权标为阻塞） | 未运行 |
| CF3-05 | COMPLETED | FAILED（配对契约） | FAILED（配对契约） |
| CF3-06 | COMPLETED | FAILED（配对契约） | FAILED（配对契约） |
| CF3-07 | COMPLETED | FAILED（配对契约） | FAILED（配对契约） |
| CF3-08 | COMPLETED | FAILED（配对契约） | FAILED（配对契约） |
| CF3-09 | COMPLETED | FAILED（配对契约） | FAILED（配对契约） |
| CF3-10 | COMPLETED | **DIAGNOSTIC_KILL** | **DIAGNOSTIC_KILL** |

`DIAGNOSTIC_KILL` 是科研机制诊断门未通过（`enabled=false`, `selected_areas=0`），**不是代码崩溃**。7 个 FAILED 是同一条严格数据契约 fail-closed 的 `ValueError`，属配对不足，**不是损失/算法崩溃**；CF3-02/04 是 `method.py` 内部错误，按你的授权原样标为阻塞。

## 4. 真实配对数量与覆盖率

- `max_pairs=32`，真实生成 32 对，主体位置一致 31 对，**覆盖率 31/32 = 96.875%**。
- 唯一不一致对（idx 10）：`Has Hsiao Yun-Hwa's mother's ...` → `Ji-Yeong Hwang's mother's ...`；主体 token 位置 `[2..8]` vs `[2..7]`。原因：`build_pairs` 只强制「问题 token 等长 + 单区间差异」，未强制主体级 offset 一致；`build_teacher_cache` 的 fail-closed 校验正确拒绝了它。
- **分类统计**：配对不足 7 方法（诊断+pilot）；诊断门未通过 1 方法（CF3-10）；方法内部代码失败并阻塞 2 方法（CF3-02/04）。三类分开统计，未合并、未选择性删除。
- 处理：未放宽校验、未伪造配对、未替换随机数据、未关闭核心损失。Teacher cache 在配置/代码变化时使用新输出目录（诊断产物已归档，pilot 使用全新 `runs/pilot_seed0`）。

## 5. 核心指标（pilot）

**pilot 只有 1 份真实 `result.json`**（CF3-10，`runs/pilot_seed0/result.json`），其余 9 个方法 `result.json` 未生成，一律 `null`。所有数值来自该文件，dF/dR/dH 为 0 是 `DIAGNOSTIC_KILL` 未训练的设计结果，**不得解释为遗忘成功或失败**。`donor` 答案只用于评估，`donor_answer_in_training=false`，不进入训练目标。

| 方法 | status | before F/R/H NLL | after F/R/H NLL | dF/dR/dH | donor NLL before/after | 生成 tokens/EOS/overlap | probe before/after accuracy | probe CE | chance accuracy | relearning before/after NLL | 恢复幅度 | 步数 | 方法诊断 enabled | 梯度 | checkpoint | 耗时(s) | result.json 路径 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CF3-01 | FAILED（配对契约） | null | null | null | null | null | null | null | null | null | null | null | null | null | null | 31.8 | 无 |
| CF3-02 | BLOCKED（method.py replay shapes） | null | null | null | null | null | null | null | null | null | null | null | null | null | null | n/a | 无 |
| CF3-03 | FAILED（配对契约） | null | null | null | null | null | null | null | null | null | null | null | null | null | null | 56.6 | 无 |
| CF3-04 | BLOCKED（method.py shape stack） | null | null | null | null | null | null | null | null | null | null | null | null | null | null | n/a | 无 |
| CF3-05 | FAILED（配对契约） | null | null | null | null | null | null | null | null | null | null | null | null | null | null | 70.0 | 无 |
| CF3-06 | FAILED（配对契约） | null | null | null | null | null | null | null | null | null | null | null | null | null | null | 61.6 | 无 |
| CF3-07 | FAILED（配对契约） | null | null | null | null | null | null | null | null | null | null | null | null | null | null | 60.2 | 无 |
| CF3-08 | FAILED（配对契约） | null | null | null | null | null | null | null | null | null | null | null | null | null | null | 51.8 | 无 |
| CF3-09 | FAILED（配对契约） | null | null | null | null | null | null | null | null | null | null | null | null | null | null | 36.6 | 无 |
| CF3-10 | **DIAGNOSTIC_KILL** | 0.4011 / 0.4960 / 1.8252 | 0.4011 / 0.4960 / 1.8252 | 0 / 0 / 0 | 3.0216 / 3.0216 | 14.75 / 1.0 / 0.5688 | `INSUFFICIENT_RELATION_SPLIT`（train=0,test=64）→ null | null | null | `NOT_RUN` → null | null | 0 | false | 无梯度历史 | null | 35.2 | `CF3-10_非交换路径签名遗忘/runs/pilot_seed0/result.json` |

CF3-10 说明：`enabled=false`（operator `level2_noncommutative_levy_area`，`selected_areas=0`），机制诊断门未通过 → 未训练、无 checkpoint、relearning `NOT_RUN`、dF/dR/dH 恒为 0。probe `INSUFFICIENT_RELATION_SPLIT`：目标作者 probe 的 before/after accuracy、cross-entropy、chance accuracy 均**不可得**（`null`），不是代码失败。其余 9 个方法 `result.json` 未生成，`run.log`/`resolved_config.json` 已保留（CF3-02/04 保留 smoke 失败证据），逐方法路径见 `summary.json` 的 `pilot_metrics`。

### smoke 层指标（仅说明工程可运行，**不得作为 pilot 或论文结论**）

smoke 是 2 steps / 4 对样本，8 个方法 `result.json` 路径：`CF3-*/runs/smoke_seed0/result.json`。数值范围：dF −0.0006~+0.0085，dR −0.003~+0.0063，dH −0.0075~+0.007，donor NLL 约 3.188~3.196；EOS rate 1.0、mean_tokens 8.0、target_overlap 0.375；probe 均 `INSUFFICIENT_RELATION_SPLIT`（4 对样本不足以分关系）；relearning 均 `COMPLETED`；梯度有限非零、Teacher 冻结、初始 Student/Teacher NLL 一致、checkpoint 已保存。完整逐方法数值在 `summary.json` 的 `smoke` 字段。**两步 smoke 与 Tiny 数值不解释为论文结论。**

### 完整性核查

- 8 个 smoke COMPLETED 方法：Student 梯度有限且非零、Teacher 全程冻结（`teacher_frozen=true`）、初始 Student/Teacher NLL 一致（`initial_student_equals_teacher=true`, delta 0.0）、预 relearning checkpoint 已保存；评估 ground truth 为数据集原始答案（`evaluation_ground_truth=dataset answer`），未使用 Teacher/Student 生成文本作标签；`donor_answer_in_training=false`。
- 生成行为/短程重学/目标答案残留/retain 损害在 pilot 无有效结果可判；不做选择性遗忘、不可恢复或隐私安全的声称。

## 6. 错误、修复与资源

| 修复 | 文件 | diff | 性质 |
|---|---|---|---|
| JSONL 内容识别 | 10× `CF3-*/data.py` | `cf3_loader_fix.diff` | 数据解析接口 |
| fp32 激活 + 等价显式贪心解码 | 10× `CF3-*/modeling.py` | `cf3_interface_fixes.diff` | 运行接口（避免 bf16 GEMM SIGFPE） |
| fp32 投影 + 等价 temperature/top_p 采样 | `CF3-01/method.py`, `CF3-03/method.py` | `cf3_method_interface_fixes.diff` | 运行接口 |
| 上一轮 `analyze.py` 默认参数 | `asu_cf04_aligned/analyze.py` | `previous_queue_analyze_fix.diff` | CPU 汇总脚本 |

全部为运行接口修复，**未改算法定义、数据划分、损失公式、配对规则、评价口径；未提交**。授权记录：`user_decision_override.json`。

错误链：JSONL loader → 4× bf16/fp32 dtype 冲突 → 4× `model.generate` 的 `num_logits_to_keep=1` bf16 SIGFPE（faulthandler 堆栈存于 `smoke_cf301_faulthandler.log` / `smoke_cf303_faulthandler.log`）→ CF3-02/04 方法内部错误（阻塞）→ pilot 32 对中 1 对主体位置不一致触发 fail-closed（未放宽）→ CF3-10 机制门 DIAGNOSTIC_KILL。

资源：GPU0 串行、无 OOM；本任务未部署独立显存峰值采样器，故 `gpu0_peak_mib=null`（如实标注）。各阶段方法耗时合计（秒）：smoke#1 352.0、smoke#2 820.4、smoke#3 318.0、诊断 299.4、pilot 413.5。未启动 `full`、多 seed 或长时消融，等待你决定下一阶段预算。

## 7. 四层次结论

| 层次 | 结论 |
|---|---|
| ① 代码可运行 | **8/10** 方法在真实 HF/LoRA/bf16/非重入式 GC/GQA/SwiGLU/严格配对下 smoke COMPLETED；2/10 因 `method.py` 内部错误阻塞 |
| ② 机制诊断通过 | **0/10** 通过 pilot 档严格诊断门：7 个被配对契约 fail-closed 中止，CF3-10 为 `DIAGNOSTIC_KILL`，2 个阻塞 |
| ③ 实验完成 | **0/10** pilot `COMPLETED`；仅 CF3-10 有 `DIAGNOSTIC_KILL` 的 `result.json`；7 FAILED、2 阻塞。进程完成不等于遗忘成功 |
| ④ 结果支持研究 claim | **否**；所有结果 `claim_level=false`，无方法产出完成的 pilot 指标，不得声称选择性遗忘、不可恢复或隐私安全 |

## 8. 未解决问题

1. CF3-02/04 的 `method.py` 内部错误（replay shapes / shape stack）：按你的授权未修改算法实现。
2. 1/32 严格反事实对主体 token 位置不一致，`build_teacher_cache` fail-closed 使 7 个方法在诊断与 pilot 中止。可选项：提供显式 `pairs_file`、或由你授权收紧 `build_pairs` 主体级校验；本次未放宽任何校验。
3. pilot 无 `COMPLETED` 结果，无法给出 dF/dR/dH/donor/probe/relearning 的方法级对比。
4. 目标作者 probe 返回 `INSUFFICIENT_RELATION_SPLIT`（train=0/test=64），accuracy/CE/chance 不可得；需要支持关系划分的数据或配置后再评估。

## 9. 关键日志与结果

- `queue_status.json` 三份副本：`queue_status_smoke.json`、`queue_status_diagnose.json`、`queue_status_pilot.json`（及历次失败副本）。
- 各方法 `runs/smoke_seed0/result.json`（8 份）、`runs/pilot_seed0/result.json`（1 份）、`resolved_config.json`、`diagnostics.json`、`run.log`。
- 真实环境预检：`preflight_pilot_remote.json`；smoke：`smoke_console.log`、`smoke_attempt1_console.log`、`smoke_attempt2_console.log`、`smoke_retry2_console.log`；诊断：`diagnose_console.log`；pilot：`pilot_console.log`。
- 诊断产物归档：`diagnose_only_runs/`（8 个方法，含 CF3-10 的 diagnostics/result）。
- 机器可读汇总：`summary.json`；上传产物快照：`*_uploaded_tiny_snapshot.json`。

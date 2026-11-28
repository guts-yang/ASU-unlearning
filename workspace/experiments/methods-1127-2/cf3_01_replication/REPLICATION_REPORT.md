# CF3-01 优先插队复现报告（阶段 A）

`claim_level=false`。proposal 的假设 H、决定性实验与「失败即撤回」条款保持不变；本轮只补齐实现与上下文。所有数值来自 `result.json` / `diagnostics.json`，无补数。

## 1. 结论先行

| 层次 | 结论 |
|---|---|
| ① 代码可运行 | **是**。补齐版 smoke `COMPLETED`；6 臂队列 7 个 job 全部退出码 0，`failed=0` |
| ② 机制诊断门通过 | **否**。主链 `DIAGNOSTIC_KILL`（8 个层×分量门只 1 个通过，`enabled=false`） |
| ③ 实验完成 | 主链 **0 步**（门拦下，按设计不训练）；5 个对照臂各完成 60 步并保存 checkpoint |
| ④ 结果支持研究 claim | **否**；`claim_level=false`，不宣称选择性/深度遗忘 |

**表现不可以**，因此**全量复现验证（阶段 B）未启动**，等你决定。按 proposal §5「失败即撤回」条款，本轮已触发两条：`route/content 的 patch 效应不稳定`（8 门仅 1 通过）与 `收益可由等范数原始差解释`（对照臂 dF 几乎相同）。

## 2. 核心目标与验收标准（不变）

对称双线性恒等式把反事实激活差精确拆为 `D_route`/`D_content`；true/null patch-forward 门选出**唯一**承载目标事实的分量；Student 内化 `A⁺−αC*`，不向 donor 替换、不破坏 retain/同实体关系、不靠提前 EOS 伪装遗忘。

验收：两分支恒等式 <1e-5、31 对逐对主体一致、6 臂全字段完整、`DIAGNOSTIC_KILL` 单列为科研门结果、ground truth 为数据集原始答案、报告列出未覆盖项。

## 3. 输入 / 依赖 / 环境

| 项 | 值 |
|---|---|
| 数据 | `/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/{forget10,retain90,holdout10}.json`（JSONL 400/3600/400） |
| 显式配对 | `pairs/strict_pairs_31.jsonl`，32 对生成、剔除 idx10、保留 31 对，覆盖率 96.875% |
| 模型 | `/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full`，`local_files_only=true` |
| 环境 | 冻结 venv：torch 2.5.1+cu121 / transformers 4.46.3 / peft 0.21.3 / accelerate 1.13.0 |
| 硬件 | 仅 GPU0（H20 97871MiB），`CUDA_VISIBLE_DEVICES=0` 串行；GPU1 外部 `lrm-unlearning` 队列未触碰 |
| 预算 | smoke 4 对/2 步；pilot 31 对/60 步/batch2/lr5e-6/seed0；layers `[8,10,12,14]` |
| 预检 | `preflight.py --profile pilot` PASS、`validate_all.py --operators-only` 10/10 PASS、`run.py --self-test` PASS |

## 4. 执行流程与实际结果

阶段 A 由 `scripts/run_cf301_replication.sh` 后台自动执行（PID 见 `results/queue.pid`），GPU0 空闲 3 次轮询后串行：`smoke_brcd` → `pilot_brcd` → 5 个核心对照臂。Teacher cache 由主链重建一次、其余臂复用（同 teacher / 同 31 对 / 同 layers，属正确复用）。

| tag | 起止(UTC) | 退出码 | 耗时(s) | status | steps |
|---|---|---:|---:|---|---:|
| smoke_brcd | 03:32:15–03:33:00 | 0 | 45 | COMPLETED | 2 |
| pilot_brcd | 03:33:00–03:33:46 | 0 | 46 | **DIAGNOSTIC_KILL** | **0** |
| pilot_route_only | 03:33:51–03:35:27 | 0 | 96 | COMPLETED | 60 |
| pilot_content_only | 03:35:29–03:37:11 | 0 | 102 | COMPLETED | 60 |
| pilot_equal_norm_delta | 03:37:13–03:38:40 | 0 | 87 | COMPLETED | 60 |
| pilot_random_equal_norm | 03:38:43–03:40:30 | 0 | 107 | COMPLETED | 60 |
| pilot_wrong_shuffled | 03:40:32–03:42:27 | 0 | 115 | COMPLETED | 60 |

GPU0 实际占用约 10 分钟（含等待 1 分钟），符合预算（预估 15–30 分钟）。

### 4.1 主链诊断门（31 对，严格档）

`enabled=false`，`ambiguous_layers=[]`，`selected={'8':None,'10':None,'12':'content','14':None}`。8 个门里只有 `12:content` 通过：

| 门 | mean_nll_increase | mean_specificity | passed |
|---|---:|---:|---|
| 8:route | +0.00219 | −0.00275 | 否 |
| 8:content | −0.00359 | −0.00713 | 否 |
| 10:route | −0.00095 | +0.00278 | 否 |
| 10:content | +0.00247 | −0.00223 | 否 |
| 12:route | −0.00302 | −0.00798 | 否 |
| 12:content | **+0.00201** | **+0.01166** | **是** |
| 14:route | −0.00045 | +0.00164 | 否 |
| 14:content | −0.00627 | −0.01115 | 否 |

效应量级在 1e-3，正负跨层/跨分量翻转；`strict_gate=true` 要求 4 层全部唯一选出，故 `DIAGNOSTIC_KILL`，0 步、无 checkpoint、relearning `NOT_RUN`。**这是科研门结果，不是代码崩溃；未放宽任何阈值。**

### 4.2 六臂核心指标

| tag | status | dF | dR | dH | donor NLL b/a | EOS | tokens | overlap | probe | relearn | grad |
|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|
| pilot_brcd | DIAGNOSTIC_KILL | 0 | 0 | 0 | 2.9692/2.9692 | 1.0 | 13.75 | 0.402 | INSUFFICIENT | NOT_RUN | N/A |
| pilot_route_only | COMPLETED | −0.0669 | −0.0588 | −0.0247 | 2.9692/3.0004 | 1.0 | 16.75 | 0.669 | INSUFFICIENT | COMPLETED | 有限非零 |
| pilot_content_only | COMPLETED | −0.0669 | −0.0600 | −0.0245 | 2.9692/3.0013 | 1.0 | 15.25 | 0.558 | INSUFFICIENT | COMPLETED | 有限非零 |
| pilot_equal_norm_delta | COMPLETED | −0.0667 | −0.0588 | −0.0224 | 2.9692/3.0078 | 1.0 | 16.75 | 0.669 | INSUFFICIENT | COMPLETED | 有限非零 |
| pilot_random_equal_norm | COMPLETED | −0.0674 | −0.0598 | −0.0220 | 2.9692/2.9993 | 1.0 | 16.75 | 0.669 | INSUFFICIENT | COMPLETED | 有限非零 |
| pilot_wrong_shuffled | COMPLETED | −0.0665 | −0.0592 | −0.0250 | 2.9692/3.0002 | 1.0 | 16.75 | 0.669 | INSUFFICIENT | COMPLETED | 有限非零 |

- `donor_answer_in_training=false`、`evaluation_ground_truth=dataset answers`、`teacher_frozen=true`、`initial_student_equals_teacher=true` 全臂成立；5 个对照臂均保存 checkpoint 与 relearning。
- **对照臂 dF 标准差仅 0.000282**（−0.0665~−0.0674）：真实 route、真实 content、等范数原始 ΔA、随机等范数、乱序错配分量给出几乎相同的 dF/dR/dH。dF 全为**负**（forget NLL 下降而非上升），方向也与遗忘目标相反。
- 10 方法 pilot 的历史结论（方法专属损失被共享 retain 项 `retain_weight=2.0` 淹没）在这里再次出现：`method_weight=0.1`，且所有分量目标都经 `Π_K` 球投影把扰动半径压到 `α‖C*‖` 量级。
- `target_fact_probe` 全臂返回 `INSUFFICIENT_RELATION_SPLIT`（train=0, test=64），因此 accuracy / cross-entropy / chance accuracy **不可得**，一律 `null`，不补数。
- EOS rate 全 1.0、生成长度 15–17 tokens、overlap 0.55–0.67；无生成崩溃，但也不构成遗忘证据。

## 5. 结果验证方式（已执行）

1. `compileall` / `bash -n` / `--self-test`：attention 恒等式 4.8e-7、SwiGLU 1.2e-7、投影半径有限；`preflight.py --profile pilot` PASS。
2. 配对：31 对逐对 `subject_plus == subject_minus`，`subject_span` 单区间，donor 来自 retain，作者簇不重合；剔除 idx10 的原因写进 `pairs/strict_pairs_report.json`。
3. smoke：`diagnostics.json` 确认 `use_glu_branch=true`、`use_pi_k_projection=true`、`projected_targets=16`、`max_ball_residual≈6e-9`。
4. 6 臂：逐份 `result.json` 校验 status / 60 steps / 有限非零梯度 / Teacher 冻结 / 初始 NLL 一致 / checkpoint / `claim_level=false`。
5. 指标表逐项对拍 `result.json`；`summary.json` 与本报告交叉校验；`DIAGNOSTIC_KILL` 单列。
6. 未把 smoke / Tiny / 进程完成解释为遗忘有效。

## 6. 与原始 idea 的差异与偏差原因

| 差异 | 偏差原因 |
|---|---|
| 只 attention → 补 SwiGLU `Swish(g)×u` 双线性分支 | 首版按 `MECHANISM_GUIDE.md §12` 主动收窄；底层 `capture_glu` 已具备，本轮按 proposal 补齐。补齐后 patch 门从 64 次量级变为同结构双分支合并（仍 8 个门），门结果从 smoke 的 4 层唯一选出变为 pilot 的 1/8 通过。 |
| 精确凸投影 `Π_K` → POCS 可计算代理 | proposal §3.3 只给 `K={h:r_f≤c,s_dj≤d_j,‖Δr_E‖≤e}`，未给算法与阈值；按你确认复用 config 的 `fact_delta=0.35` / `donor_guard_weight=0.1` / `eos_margin=1.0` 构造球 + donor 半空间，20 次交替投影。**这是新增可计算设计，不是原文数学精确投影**；`r_f/s_dj/Δr_E` 是读出层函数量，激活空间无法直接投影。预注册见 `PROTOCOL.md §4.2`。 |
| 32 对 → 31 对 | 第 10 对主体级 token 位置 `[2..8]` vs `[2..7]` 被 fail-closed 正确拒绝；不放宽校验、不伪造配对、不合成数据，覆盖率 96.875%。 |
| 软约束仍在 | fact hinge / donor guard / EOS guard / retain NLL 与投影代理并存，保留原训练稳定性；二者关系见 `PROTOCOL.md §4.2`。 |
| 主链 0 步、对照臂完成 60 步 | `strict_gate=true` 要求 4 层唯一选出；对照臂按 §5 设计跳过门，用于验证「收益是否可由等范数/随机/乱序解释」，结果是**可以解释**，即机制无特异性。 |
| dF 全负、5 臂几乎相同 | 方法专属损失权重 0.1 vs retain 2.0，加上投影球把扰动限制在 `α‖C*‖`；对照臂共享同一训练/评估口径，差异被共享项主导。这与 proposal「失败即撤回」条款一致。 |
| `probe` 不可得 | `INSUFFICIENT_RELATION_SPLIT`（train=0, test=64），需要支持关系划分的数据或配置后再评估。 |
| 生成改显式贪心、`F.linear` 改 fp32 | 本机 torch 2.5.1+cu121 / H20 的 bf16 GEMM SIGFPE 运行接口修复，不改算法/损失/评价口径。 |

### proposal §5 未覆盖项（列为后续工作）

释义/间接提示、独立生成式 probe 的充分关系划分、恢复攻击、量化恢复、多 seed、`full` 预算。本轮只做主链 + 5 类核心对照（同预算、同 31 对），不做 multi-seed/full。

## 7. 全量复现验证（阶段 B）

`scripts/run_cf301_full.sh` 已备好（`full` 预算：400 对 / 600 steps / 3 seeds / 6 臂），**未启动**；`results/queue_status.json` 的 `full_stage_started=false`。

按你的要求「CF3-01 结束的时候先给结果和结论，如果表现可以就要先进行全量复现验证」。本轮结论是**表现不可以**：主链诊断门未过、对照臂无特异性、dF 方向不对、probe 不可得。继续跑 `full` 只会放大同一共享项主导的漂移，无法回答 proposal 的机制假设。因此等你决定是否：

- 先按 proposal「失败即撤回」归档 CF3-01，或
- 调整口径后再验证（例如补充分量特异性对照、重新预注册门阈值或改损失权重 —— 这些属于算法/配置改动，需要你明确授权；本轮未擅自放宽任何阈值）。

## 8. 产物与提交

- 源码：`CF3-01_双线性路由内容差分/method.py`、`config.json`；原件备份与 diff 在 `cf3_01_replication/scripts/`。
- 脚本 / 配置：`scripts/make_pairs_cf301.py`、`scripts/run_cf301_replication.sh`、`scripts/run_cf301_full.sh`、`configs/*.json`（6 臂）。
- 文档 / 结果：`PROTOCOL.md`、`REPLICATION_REPORT.md`、`summary.json`、`pairs/*.json*`、`results/`（queue.log / queue.tsv / queue_status.json / 各臂 result.json / diagnostics.json / resolved_config.json / console.log / gpu_mem.log）。
- 权重：`/jizhicfs/chenyliao/llm-unlearning/results/cf3-01-replication/weights/`（teacher_cache.pt / checkpoint / training_state.pt，不进 Git）。
- 既有证据 `_validation/repro_1127_2/ACCEPTANCE_REPORT.md` 等不覆盖不删除。

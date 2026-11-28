# CF3-01 优先插队复现报告（阶段 A）

`claim_level=false`。proposal 的假设 H、决定性实验与「失败即撤回」条款保持不变。本报告以**当前实现（推导包 + discovery 冻结 `c_f`，round3）**为主；round1 / round2 作为历史轮次保留在 §7。所有数值来自 `result.json` / `diagnostics.json` / `queue.tsv`，无补数。

## 1. 结论先行（round3）

| 层次 | 结论 |
|---|---|
| ① 代码可运行 | **是**。`py_compile` / `--self-test` PASS，7 个 job 全部退出码 0，`failed=0` |
| ② `c_f` 语义修正生效 | **是**。冻结校准后**全部目标 `identity`（`T=z`）**：`infeasible=0`、`max_dist_over_edit=0`；round2 的 `projection_infeasible` 消失 |
| ③ 机制诊断门通过 | **否**。主链 `DIAGNOSTIC_KILL`：16 个四候选门只 2 个通过，10 / 14 层无唯一候选，`enabled=false`，**0 步** |
| ④ 对照臂 | 5 臂各完成 60 步并保存 checkpoint；dF 全为**负**（−0.0669 ~ −0.0682），std `0.000499`，真实 / 等范数 / 随机 / 乱序几乎相同 |
| ⑤ 结果支持研究 claim | **否**；`claim_level=false`，不宣称选择性/深度遗忘 |

**表现不可以**，因此**全量复现验证（阶段 B）未启动**，等你决定。

两条重要限定：

- round2 里 dominated 的 `projection_infeasible` 与 `dist_over_edit≈1.0` 已定位为**实现缺陷**（把 runtime `L_fact.fact_delta` 当成 proposal 的 `r_f≤c` 阈值），不是约束系统的性质；按推导包分离并冻结 `c_f` 后消失。
- 但 `T=z` 意味着**本轮投影对目标不产生任何额外编辑**，因此本轮**不能**作为「代理投影有效」的证据；主链效果（若有）应全部归因于 BRCD 目标。而主链根本没训练。

## 2. 核心目标与验收标准（不变）

对称双线性恒等式把反事实激活差精确拆为 route/content（attention）与 gate/value（SwiGLU）四个**独立**候选；true/null patch-forward 门选出**唯一**承载目标事实的分量；Student 内化 `A⁺−αC*`，不向 donor 替换、不破坏 retain/同实体关系、不靠提前 EOS 伪装遗忘。

验收：双线性恒等式 <1e-5、31 对逐对主体一致、6 臂全字段完整、`DIAGNOSTIC_KILL` 单列为科研门结果、ground truth 为数据集原始答案、报告列出未覆盖项。默认算法保持 attention-only，新分支必须显式配置开启、可独立消融。

## 3. 输入 / 依赖 / 环境

| 项 | 值 |
|---|---|
| 数据 | `/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/{forget10,retain90,holdout10}.json` |
| 显式配对 | `pairs/strict_pairs_31.jsonl`，32 对生成、剔除 idx10、保留 31 对，覆盖率 96.875%；未放宽 `build_teacher_cache` 的 fail-closed 校验 |
| 模型 | `/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full`，`local_files_only=true` |
| 环境 | 冻结 venv：torch 2.5.1+cu121 / transformers 4.46.3 / peft 0.21.3 |
| 硬件 | 仅 GPU0（H20 97871MiB），`CUDA_VISIBLE_DEVICES=0` 串行；GPU1 外部 `lrm-unlearning` 队列未触碰 |
| 预算 | smoke 4 对/2 步；pilot 31 对/60 步/batch2/lr5e-6/seed0；layers `[8,10,12,14]` |
| 静态预检 | `py_compile` PASS；`--self-test`：attention 恒等式 `4.768e-7`、SwiGLU 重构相对 Frobenius `4.995e-8`（均 <1e-5） |
| 配置 | `config.json` 默认 `enable_glu_family=false` / `enable_affine_projection=false`；6 个 `configs/*.json` 显式开启两个分支 |

## 4. 执行流程与实际结果

阶段 A 由 `scripts/run_cf301_replication.sh` 后台自动执行（本轮 PID 见 `results/round3.pid`），GPU0 空闲 3 次轮询后串行：`smoke_brcd` → `pilot_brcd` → 5 个核心对照臂。Teacher cache 由主链重建一次、其余臂复用（同 teacher / 同 31 对 / 同 layers，属正确复用）。

| tag | 起止(UTC) | 退出码 | 耗时(s) | status | steps |
|---|---|---:|---:|---|---:|
| smoke_brcd | 04:37:58–04:38:36 | 0 | 38 | COMPLETED | 2 |
| pilot_brcd | 04:38:37–04:39:39 | 0 | 62 | **DIAGNOSTIC_KILL** | **0** |
| pilot_route_only | 04:39:44–04:41:23 | 0 | 99 | COMPLETED | 60 |
| pilot_content_only | 04:41:26–04:43:03 | 0 | 97 | COMPLETED | 60 |
| pilot_equal_norm_delta | 04:43:06–04:44:47 | 0 | 101 | COMPLETED | 60 |
| pilot_random_equal_norm | 04:44:50–04:46:29 | 0 | 99 | COMPLETED | 60 |
| pilot_wrong_shuffled | 04:46:32–04:48:09 | 0 | 97 | COMPLETED | 60 |

`queue_status.json`：`failed=0`、`full_stage_started=false`。GPU0 约 10 分钟（含 40 秒等待），符合预算。

### 4.1 主链四候选诊断门（31 对，严格档）

`enabled=false`，`selected={'8':'attn_route','10':None,'12':'glu_value','14':None}`，`kill_reasons=[{'type':'gate_no_unique_candidate','layers':[10,14]}]`，`cross_family_ambiguity=[]`、`ambiguous_layers=[]`。16 个门（4 层 × 4 候选）中只有 2 个通过：

| 门 | mean_nll_increase | mean_specificity | passed |
|---|---:|---:|---|
| 8:attn_route | **+0.004811** | **+0.012582** | **是** |
| 8:attn_content | −0.005768 | −0.002828 | 否 |
| 8:glu_gate | +0.000982 | −0.003555 | 否 |
| 8:glu_value | −0.004108 | −0.006151 | 否 |
| 10:attn_route | −0.002382 | −0.001701 | 否 |
| 10:attn_content | −0.000152 | +0.000143 | 否 |
| 10:glu_gate | −0.000426 | +0.003214 | 否 |
| 10:glu_value | −0.001693 | −0.003153 | 否 |
| 12:attn_route | −0.003216 | +0.001173 | 否 |
| 12:attn_content | −0.001391 | −0.000906 | 否 |
| 12:glu_gate | −0.001229 | −0.005802 | 否 |
| 12:glu_value | **+0.000793** | **+0.005020** | **是** |
| 14:attn_route | −0.001533 | −0.002810 | 否 |
| 14:attn_content | −0.004357 | −0.001024 | 否 |
| 14:glu_gate | −0.000208 | +0.000611 | 否 |
| 14:glu_value | −0.001489 | +0.000997 | 否 |

效应量级仍在 1e-3，正负跨层 / 跨候选翻转；`strict_gate=true` 要求 4 层每层 family 内唯一选出，故 10、14 层 `None` → `DIAGNOSTIC_KILL`，0 步、无 checkpoint、relearning `NOT_RUN`。**这是科研门结果，不是代码崩溃；未放宽 `strict_gate` / `min_nll_increase` / `specificity_margin` / `projection.tolerance` 中任何一项。**

### 4.2 投影诊断：`c_f` 冻结后成为严格恒等

主链 `c_f_calibration`：`c_f=-0.104699`，`r_f_z_min=-0.692382`、`r_f_z_max=-0.154699`、`margin=0.05`、`n_discovery=8`（4 个 `gate_pairs` × 2 个已选层），来源为 discovery 的 `z`，**不是** `teacher_nll + loss.fact_delta`。

| tag | projected_targets | accepted | identity | infeasible | max_dist_over_edit |
|---|---:|---:|---:|---:|---:|
| pilot_brcd | 62 | 0 | 62 | 0 | 0.0 |
| 5 个对照臂 | 各 124 | 0 | 各 124 | 0 | 0.0 |
| smoke | 16 | 0 | 16 | 0 | 0.0 |

即每个 `z` 在真实冻结下游读数下已满足全部 `φ_i≤tol`，按推导要求返回 `T=z`、`projection_active=false`。`donor_constraint_only=true`、`donor_target=false`（donor 答案仅作单边支持测量，不作 MSE/KL/soft label）；`repatch_after_projection: NOT_DONE` 如实保留。

### 4.3 六臂核心指标

| tag | status | steps | dF | dR | dH | donor NLL b/a | EOS | tokens | overlap |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| pilot_brcd | DIAGNOSTIC_KILL | 0 | 0 | 0 | 0 | 2.9692/2.9692 | 1.0 | 13.75 | 0.402 |
| pilot_route_only | COMPLETED | 60 | −0.0670 | −0.0602 | −0.0227 | 2.9692/2.9995 | 1.0 | 16.25 | 0.725 |
| pilot_content_only | COMPLETED | 60 | −0.0678 | −0.0592 | −0.0245 | 2.9692/3.0058 | 1.0 | 16.75 | 0.669 |
| pilot_equal_norm_delta | COMPLETED | 60 | −0.0669 | −0.0587 | −0.0236 | 2.9692/2.9992 | 1.0 | 16.75 | 0.669 |
| pilot_random_equal_norm | COMPLETED | 60 | −0.0671 | −0.0598 | −0.0262 | 2.9692/3.0021 | 1.0 | 16.75 | 0.669 |
| pilot_wrong_shuffled | COMPLETED | 60 | −0.0682 | −0.0584 | −0.0240 | 2.9692/2.9982 | 1.0 | 16.75 | 0.669 |

- `donor_answer_in_training=false`、`evaluation_ground_truth=dataset answers`、`teacher_frozen=true`、`initial_student_equals_teacher=true` 全臂成立；5 个对照臂均保存 checkpoint 与 relearning。
- **对照臂 dF 标准差仅 0.000499**（−0.0669 ~ −0.0682）：真实 route、真实 content、等范数原始 ΔA、随机等范数、乱序错配给出几乎相同的 dF / dR / dH；dF 全为**负**（forget NLL 下降而非上升），方向与遗忘目标相反。与 round1 的 `0.000282` 同量级，结论不变：无分量特异性。
- `method_weight=0.1` vs `retain_weight=2.0`，共享 retain 项继续主导 drift；`target_fact_probe` 全臂 `INSUFFICIENT_RELATION_SPLIT`（train=0, test=64），accuracy / cross-entropy / chance accuracy 不可得，一律 `null`，不补数。
- EOS rate 全 1.0、生成长度 13.75–16.75 tokens、overlap 0.40–0.73；无生成崩溃，但不构成遗忘证据。
- 未把 smoke / Tiny / `COMPLETED` / 进程退出码 0 解释为遗忘有效。

## 5. 结果验证方式（已执行）

1. `python -m py_compile method.py` PASS；`run.py --self-test` PASS（attention 4.768e-7、SwiGLU 4.995e-8）。
2. 独立单臂 `smoke_derivation3`：`COMPLETED`、2 步、`identity=16/16`、`infeasible=0`、`c_f_calibration` 字段存在。
3. 31 对逐对主体一致；`pairs/strict_pairs_report.json` 记录 idx10 被剔除原因。
4. 7 份 `result.json` / `diagnostics.json` 逐项校验 status / steps / checkpoint / `claim_level=false` / `kill_reasons` / `c_f_calibration` / `projection_proxy`。
5. 指标表逐项对拍 `result.json`；`summary.json`（v2）由脚本直接读取结果文件生成，并与本报告交叉校验；`DIAGNOSTIC_KILL` 单列。
6. 两处外部 supervisor 均为 `CUDA_VISIBLE_DEVICES=1` 的 `lrm-unlearning` 任务，未触碰；GPU0 空闲 3 次确认后才启动队列。

## 6. 与原始 idea 的差异与偏差原因

| 差异 | 偏差原因 |
|---|---|
| 精确 `Π_K` → 局部仿射 POCS 可行点代理 | 原始 K 的 `r_f/s_dj/E` 由冻结非线性下游产生，**非凸**；有限 POCS 只给可行点，不保证离 `z` 最近。**未声称精确投影**。本轮全部 `T=z`，投影未产生编辑。 |
| 四候选不求和、family 内唯一、跨 family 记 `cross_family_ambiguity` | 推导包明确要求；本轮 `cross_family_ambiguity=[]`，但 10/14 层连 family 内唯一候选都没有。 |
| `r_f≤c` 的 `c_f` 从 discovery 的 `z` 冻结 | `L_fact` 的 NLL 上界与 proposal 的 `r_f≤c` 语义不同，不能复用 `fact_delta`；round2 的误用已修正（详见 `DERIVATION_IMPL.md`）。 |
| 32 对 → 31 对 | 第 10 对主体级 token 位置不一致被 fail-closed 正确拒绝；不放宽校验、不伪造 / 合成数据。 |
| 主链 0 步、对照臂完成 60 步 | `strict_gate=true` 要求 4 层全部唯一选出；对照臂按 §5 设计跳过门，用于验证「收益是否可由等范数 / 随机 / 乱序解释」，结果是**可以解释**，即机制无特异性。 |
| dF 全负、5 臂几乎相同 | 方法专属损失权重 0.1 vs retain 2.0；对照臂共享同一训练 / 评估口径，差异被共享项主导。与 proposal「失败即撤回」一致。 |
| `probe` 不可得 | `INSUFFICIENT_RELATION_SPLIT`（train=0, test=64），需要支持关系划分的数据或配置后再评估。 |
| 生成改显式贪心、`F.linear` 改 fp32 | 本机 torch 2.5.1+cu121 / H20 的 bf16 GEMM SIGFPE 运行接口修复，不改算法 / 损失 / 评价口径。 |

### proposal §5 与推导包未覆盖项

投影后 true/null patch 复查（`repatch_after_projection: NOT_DONE`）、Dykstra 最近点校正、`d_j/e/ρ` 的 discovery/validation 校准、独立生成式 probe 的充分关系划分、恢复攻击、量化恢复、多 seed、`full` 预算。详见 `DERIVATION_IMPL.md §5`。

## 7. 三轮对比

| 轮次 | 实现 | 主链 | 对照臂 | 投影诊断 |
|---|---|---|---|---|
| round1（`results/round1_snapshot/`） | 早期 BRCD + 球∩donor 半空间代理（commit `8ea4e84`） | `DIAGNOSTIC_KILL`，8 门只 1 通过，0 步 | 5/5 `COMPLETED` 60 步，dF std `0.000282` | 球残差 `≈2e-8`，无真实下游回验 |
| round2（`results/round2_snapshot/`） | 推导版四候选 + 局部仿射代理，但 `c_f` 误用 `L_fact.fact_delta` | `DIAGNOSTIC_KILL`：`gate_no_unique_candidate[10,14]` + `projection_infeasible=8` | 5/5 `DIAGNOSTIC_KILL` 0 步（`projection_infeasible` 44–51） | 主链 accepted 54/62、对照 73–80/124；`max_dist_over_edit≈1.0` |
| **round3（当前）** | 推导版四候选 + 局部仿射代理 + **discovery 冻结 `c_f`** | `DIAGNOSTIC_KILL`：仅 `gate_no_unique_candidate[10,14]` | 5/5 `COMPLETED` 60 步，dF std `0.000499` | **全部 `identity`**、`infeasible=0`、`max_dist_over_edit=0` |

三轮都指向同一结论：严格门未过、对照臂无特异性、dF 方向相反。round2→round3 的变化只证明「`projection_infeasible` 是实现缺陷而非机制发现」，**没有**把结论翻转成「表现可以」。

## 8. 全量复现验证（阶段 B）

`scripts/run_cf301_full.sh` 已备好（`full` 预算），**未启动**；`results/queue_status.json` 的 `full_stage_started=false`。

按你的要求「CF3-01 结束的时候先给结果和结论，如果表现可以就要先进行全量复现验证」。本轮结论仍是**表现不可以**：主链诊断门未过、投影退化成恒等、对照臂无特异性、dF 方向不对、probe 不可得。继续跑 `full` 只会放大同一共享项主导的漂移，无法回答 proposal 的机制假设。因此等你决定是否：

- 先按「失败即撤回」归档 CF3-01，或
- 调整口径后再验证（补分量特异性对照、重新预注册门阈值、改损失权重、补投影后 repatch 等 —— 均属算法/配置改动，需你明确授权；本轮未擅自放宽任何阈值）。

## 9. 产物与提交

- 源码：`CF3-01_双线性路由内容差分/method.py`（四候选 + `calibrate_c_f` + 局部仿射代理）、`config.json`（默认关闭新分支）。
- 脚本 / 配置：`scripts/make_pairs_cf301.py`、`scripts/run_cf301_replication.sh`、`scripts/run_cf301_full.sh`、`configs/*.json`（6 臂）。
- 文档 / 结果：`PROTOCOL.md`、`REPLICATION_REPORT.md`、`DERIVATION_IMPL.md`、`summary.json`、`pairs/*.json*`、`results/`（含 `round1_snapshot/` / `round2_snapshot/` / 各臂 `result.json`、`diagnostics.json`、`resolved_config.json`、`console.log`）。
- 权重：`/jizhicfs/chenyliao/llm-unlearning/results/cf3-01-replication/weights/`（`teacher_cache.pt` / `checkpoint` / `training_state.pt`，不进 Git）。
- 既有证据 `_validation/repro_1127_2/` 不覆盖不删除。
- 运行环境仍依赖工作区**既有未提交**的 `modeling.py`（fp32 hidden / 显式贪心生成）与 `data.py`（JSONL 解析）修复；按你的约束，这些旧修复不混入本提交。

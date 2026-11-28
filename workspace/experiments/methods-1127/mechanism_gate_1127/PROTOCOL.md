# CF2 十方法机制判别门队列协议

`claim_level=false`。本队列对应你 2026-11-28 的指令「当前 CF2 的 10 个方法做机制判别门，把训练计划按照队列执行顺序加到当前 GPU0 当前任务后」；当时已完成细节对齐，但**最终计划写入前被 CF3-01 插队打断，落地脚本未生成**。现因「CF3-01 先暂时搁置」而恢复执行。

## 1. 已对齐的决策（2026-11-28，逐项照录）

| 项 | 你的选择 |
|---|---|
| 接入门槛 | **GPU0 空闲即启动**（不与 GPU1 外部 `lrm-unlearning` 队列争用、不抢占） |
| 做到哪一层 | **离线门 → 对照臂 → 决策** |
| 队列顺序 | **按 `idea-stage/candidates-1127/IDEA_REPORT.md §7`** |
| 训练预算 | **与 pilot 同预算**（60 steps / 32 对 / seed 0 / 单卡） |
| 离线可分性门 | **复用已有 cache**（`CF2-*/runs/pilot_seed0/teacher_cache.pt`，不重建） |
| 产物分流 | **数据盘只放权重**；实验数据等文档放仓库新目录 |
| 目录隔离 | **薄外层脚本直接调 `run.py`**（不改源码，绕过 `run_all.py` 的 `COMPLETED` 跳过与硬编码 `runs/<profile>_seed0`） |
| 提交范围 | **只提交文档与脚本** |
| 汇报节点 | **离线门后先汇报一次** |

`§7` 顺序：先 CF2-01 / CF2-02 闭式机制判别 → CF2-03 线性低秩估计 → CF2-04（UOT 未配对质量跨 split 稳定后）→ CF2-05~08 → 末两名 CF2-09 / CF2-10 先过语义归属 / 交互稳定性门，未过门不进入训练。目录字典序 `CF2-01..10` 与该顺序一致。门的总判据：真实 donor 必须优于 wrong / shuffle / noise / matched-retain 与随机算子对照。

## 2. 当前阶段：离线门（已启动）

`scripts/run_cf2_offline_gate.sh`：

- `CUDA_VISIBLE_DEVICES=0`，连续 3 次确认 GPU0 `<100MiB` 且无 `run.py`/`start.sh` worker 后串行启动。
- 每个方法：`run.py --profile pilot --diagnose-only --output <隔离目录> --set data.reuse_cache=true`。`--diagnose-only` 只跑 `prepare()` + `diagnostics()` + 前后评估，**不训练、不写 checkpoint**。
- 已有 `teacher_cache.pt` 以软链复用（权重留在数据盘 / 仓库已忽略的 `runs/`），不进 Git。
- 产物：`results/offline_gate.{log,tsv}`、`results/offline_gate_status.json`、各方法 `console.log` / `diagnostics.json` / `result.json`。

**判据边界**：离线门如实汇总各方法 `diagnostics.json` 里的可分性素材（如 CF2-01 `true_flux_norm` / `shuffled_flux_norm` / `weight`、CF2-05 `true_coherence` / `wrong_coherence`、CF2-02 `ood_controls`、CF2-09 `strict_semantic_gate` 等），并标出是否存在 true vs control 字段。各方法的「真实优于对照」方向与最小判别实验由各自 `proposal.md` 定义，**本阶段不擅自统一判定、不伪造通过/不通过结论**。

### 2.1 离线门实测（2026-11-28，已跑完）

10/10 方法 `exit=0`、`status=DIAGNOSED`、全部 `cache_reused=yes`，`failed=0`；GPU0 串行约 11 分钟（每方法 62–74 秒）。`gpu_training_arms_started=false`。机器可读汇总见 `results/offline_gate_status.json`，逐方法日志见 `results/offline/CF2-*/console.log`。

| 方法 | diagnostics 中可识别的 true / control 素材 |
|---|---|
| CF2-01 | `spectra.*.true_flux_norm` / `shuffled_flux_norm` / `weight` |
| CF2-02 | `ood_controls.*`（gap、natural vs shared answer NLL），无 `true_*` 命名 |
| CF2-04 | `layers.*.destroyed_mass` / `wrong_destroyed_mass` |
| CF2-05 | `true_coherence` / `wrong_coherence` |
| CF2-09 | `strict_semantic_gate`（末两名语义门） |
| CF2-03 / 06 / 07 / 08 / 10 | 无成对对照字段，需按各自 `proposal.md` 的最小判别实验另定判据 |

按 §7「真实 donor 必须优于对照」，仅字段成对的方法能直接读比值；例如 CF2-01 的 `true_flux_norm` 系统性小于 `shuffled_flux_norm`、配对特异性 `weight` 近 0。**但任何方法都不得凭 `DIAGNOSED` / `exit=0` / 进程完成宣称过门**；正式判定与「是否进入训练」留在决策阶段。

## 3. 已裁决：对照臂构造（2026-11-28）

那次对齐在「核对落地接口」处被打断，对照臂构造未定稿；离线门后按约定汇报，你选择了 **共享项基线（推荐）**。已核实 CF2 十个 `method.py` 没有 `control_arm` 接口（不同于 CF3-01），因此不改源码只做配置级消融：

- `main` 臂：`--profile pilot`（60 steps / 32 对 / seed 0 / GPU0），默认配置。
- `baseline` 臂：完全相同，仅追加 `--set loss.method_weight=0.0`，把方法专属损失归零，只留共享 `retain_weight=2.0` / neutral 项。
- 判据意图：检验「方法专属效应是否被共享项淹没」（CF2 pilot 十方法 dF 全负 −0.043~−0.057、生成指标完全相同）。若 main 与 baseline 的 dF/dR/dH 无分离，方法贡献不可识别。
- 两臂都**重跑**到隔离目录，不复用历史 pilot 为正向证据（`IDEA_REPORT` §7）；同预算、同 cache、同 seed。权重留数据盘，只把 `result.json` / `diagnostics.json` / `resolved_config.json` / `console.log` 轻量副本拷回仓库 `results/arms/`（不提交）。
- 汇总 `results/arm_comparison.json`：逐方法 main − baseline 的 dF/dR/dH 差值，**只作原始对比，不宣称遗忘有效**。

未采纳：`method.alpha=0` 零编辑对照；wrong/shuffle/noise/随机算子对照（需改 `method.py` 新增配置位，与「不改任何源码」冲突，须单独授权）。

## 4. 不改动的边界

- 默认算法、损失口径、数据划分、评价口径均不改；本轮只新增 `PROTOCOL.md` 与 `scripts/*.sh`。
- 期间有并发提交 `c5341a3`（「sync method code and validation docs」）把工作区原有的 26 个运行期修复与离线门脚本一并同步，**非本轮操作**；本轮未回滚、未混入其它改动。
- 不碰 GPU1 的 `lrm-unlearning` 队列，不 `pkill`、不删除他人产物。
- **CF3-01 保持搁置**：不重跑、不启动 `run_cf301_full.sh`；等你研究完再按要求重新跑。
- `full` / 多 seed / 长时消融不自动启动；离线门后已汇报，GPU 训练臂按你的裁决（共享项基线）启动。
- `results/` 下的 JSON / 日志由 `.git/info/exclude` 排除，不进 Git。

## 5. 对照臂结果与决策（2026-11-28）

20/20 臂 `exit=0`、`COMPLETED` 60 steps、全部 `cache_reused=yes`，`failed=0`；GPU0 串行约 40 分钟。逐方法 `main − baseline` 差值见 `results/arm_comparison.json`，逐臂日志见 `results/arms.tsv` / `arms/`。

| 观测 | 值 |
|---|---|
| `main` dF | −0.0432 ~ −0.0570，**全负** |
| `baseline` dF | −0.0458 ~ −0.0561，**全负** |
| 差值 dF | −0.0019 ~ +0.0031，std `0.0013`，约为 dF 本身的 2–7% |
| baseline dF 分布 | 只聚成 4 组（−0.045794 / −0.046296 / −0.056068 / −0.0466） |

**判定**：把 `method_weight` 从 `0.1` 归零后，dF/dR/dH 几乎不变，差值仅 1e-3 且符号在各方法间翻转 → **方法专属损失对最终指标没有可观测贡献**，结果由共享 `retain_weight=2.0` / neutral 与评估口径主导；去掉专属项后不同方法收敛到同一共享轨迹。所有 dF 仍为负（forget NLL 下降），方向与遗忘目标相反。不得据 `COMPLETED` / 60 steps / 退出码 0 宣称遗忘有效。

**边界（不得越界解读）**：

- 本轮是「共享项基线」对照，**不是** §7 要求的 wrong / shuffle / noise / matched-retain / 随机算子对照；后者需改 `method.py` 新增配置位，未获授权，因此**不能宣称「§7 判别门已完整执行」**。
- 离线门里 CF2-03 / 06 / 07 / 08 / 10 无成对对照字段，其最小判别实验判据仍待按各自 `proposal.md` 定义。
- 未做 `full` / 多 seed / 恢复攻击 / 量化恢复 / 独立生成式 probe；`claim_level=false`。

**决策建议**：按各 proposal「失败即撤回」条款，十方法在本轮对照下均未显示方法专属贡献，不建议进入下一阶段训练或任何 claim。是否归档 CF2、或授权补齐 §7 的随机算子对照（需改源码），由你决定。

## 6. 修复轮（计划已批准；判据在跑之前冻结）

### 6.1 纠正上一轮措辞

- `CF2-09` diagnostics 的 `strict_semantic_gate=false` 是**配置开关未打开**，不是门的判定；真实门状态是 10/10 层 `semantic_gate=UNRESOLVED`（`specificity_scores` 的 L/S 全负）。按 §7「末两名未过门不进入训练」本应被拦下，打开开关即可，门逻辑无需改。
- §2.1 的「true vs control 字段」表是按字段名的机械扫描，**不等于 §7 判据已定义**；各方法的「真实优于对照」方向仍以各自 `proposal.md` 为准。

### 6.2 用户裁决与授权范围

- 层次：**配置级 + 授权改源码**；预算：**先 smoke 再定**（2 steps / 4 对），通过后由用户决定是否跑 pilot / 3 seeds，**不自动继续**。
- 授权改动：`CF2-01/method.py`（`gate_mode`）、`CF2-06/method.py`（`distance_mode` + 有限性保护）、10 个 `CF2-*/method.py`（`control_mode`）、`CF2-09/method.py`（仅 diagnostics 显式化）；逐文件 diff 单独提交。
- **不改默认算法**：新分支默认 `control_mode="none"`、`gate_mode="legacy"`、`distance_mode="legacy"`，历史结果可复现，修复臂一律显式 `--set` 开启。

### 6.3 预注册判据（冻结，禁止事后调阈值）

| 编号 | 断言 | 冻结阈值 | 旧值对照 |
|---|---|---|---|
| S1 | 10/10 方法 smoke `exit=0`、无 `FloatingPointError`、梯度有限非零 | `exit_code==0` 且 `grad_norm` 有限 >0 | 20/20 `exit=0` |
| S2 | 专属梯度通道可打开（**判定对象：CF2-01 的 `gate_mode=relative` 修复臂**；其余方法只记录占比、不参与判定，因为它们的退化目标本轮未修） | 首步 `0.1*method_loss/total_loss ≥ 1%`；`grad_norm(fix)` 与 `grad_norm(fix_nomethod)` 相对差异 `>1%` | 占比 <0.02%（最高 0.244%）；6.466 vs 6.466 |
| S3 | CF2-06 数值稳定（`distance_mode=median` 臂） | 全程 `method_loss` 有限且 `grad_norm < 100` | 65796 / 35199（step5）、63482 / 45943（step59） |
| S4 | 门与对照臂生效 | CF2-09 `strict_semantic_gate=true` → `DIAGNOSTIC_KILL` 且 0 步；CF2-01 `gate_mode=relative` 的 `spectra.*.weight` 不全部 ≤1e-12；`control_mode=wrong/random` 臂 `history[0].control_changed==1` 且 `method_loss` 与 `none` 臂不同 | 训练未被拦截、weight ≤6.5e-13、无对照臂 |

**S2 若因 smoke 仅 2 steps 而不明确，只判「通道是否打开」，不判方法有效。** smoke 不判 dF 正负、不宣称遗忘有效；`claim_level=false`。

**运行参数（跑之前固定）**：`--profile smoke`（2 steps / 4 对）**不复用** pilot 的 `teacher_cache.pt`（两者 `max_pairs` 不同，复用会造成 pair/index 错配），统一 `--set data.reuse_cache=false`；并统一 `--set training.batch_size=2`，因为错配算子要求 `len(batch)≥2`（smoke 默认 `batch_size=1`），同时满足 CF2-03 的 `minimum_batch=2`。共 24 臂 GPU0 串行：`none`×10、`wrong`×10、CF2-01 `fix`/`fix_nomethod`、CF2-06 `fix`、CF2-09 `gate`。

### 6.4 关键修复的科学依据与实现决定

- **CF2-01 门方向反了**：`true=(hp_r−hm_r)−(hp_l−hm_l)`，`wrong` 用 `other` pair 的 `h_plus` 与本 pair 的 `h_minus` 混搭，天然更大；「`true < wrong`」本就是配对特异性的正确信号。原实现 `sigmoid(true−median(wrong))` 却假设「true 大才好」，把权重压到 `6.5e-13`。`gate_mode=relative` 改为 `sigmoid(log(wrong/true))`，方向摆正；`legacy` 保留复现。这是**算法语义变更**，如实记录。
- **统一 batch 错配算子**：批准计划写「`wrong`/`shuffle` 复用各方法已有的错配构造」；落地改为**只把每个 row 的 Teacher 侧 `item` 轮转/随机置换，Student 侧 `plus`/`minus`/`control_*` 保持不变**，于是 regression target 来自错配 pair。理由：10 个目录自包含（`run.py` 从本目录 import）、`loss()` 结构差异大（CF2-03/07 根本不构造 target），统一算子才能保证十方法同口径且不破坏 `validate_all.py::check_isolation`。各方法已有的 `true/wrong` 诊断量**保持不变**，继续作离线可分性素材。`control_changed` 作为标量 tensor 写进 `loss()` 返回值，由 `runtime.py` 自动记入 `train.history`，供 S4 核验。
- `REQUIRES_CONTROL=True` 的 CF2-02 / CF2-06 错配后可能与现有控制关系契约冲突，smoke 单独标注；失败则如实记为「对照臂不适用」，不伪造结果。

### 6.5 不变约束

GPU0 only、不碰 GPU1、不 `pkill`、不删他人产物；**CF3-01 保持搁置**（不重跑、不启动 `full`）；权重留数据盘、实验 JSON/日志不进 Git；提交只含文档 + 脚本 + 本轮获授权的源码改动；`full` / 多 seed 不自动启动。

## 7. 修复轮 smoke 结果（判据未放宽）

24/24 臂执行完毕，`failed=1`（CF2-06 `wrong`），`pilot_queue_started=false`，GPU0 已释放。证据：`results/fix_smoke_status.json`、`fix_smoke_verdict.json`、`fix_smoke.tsv`、逐臂 `fix_smoke/<tag>/<arm>/`。

| 判据 | 结果 | 实测 |
|---|---|---|
| S1 | **PASS** | 10/10 `none` 臂 `exit=0`、`COMPLETED` 2 steps、梯度有限非零、无 `FloatingPointError` |
| S2 | **FAIL** | `method_share=5.16%`（≥1% ✓）；`grad_norm(fix)=0.045902` vs `grad_norm(fix_nomethod)=0.045529`，相对差 `0.82%`（阈值 >1% ✗） |
| S3 | **PASS** | CF2-06 `distance_mode=median`：`max_method_loss=0.0415`、`max_grad_norm=5.80`（旧值 65796 / 35199） |
| S4 | **FAIL** | CF2-09 门 `DIAGNOSTIC_KILL` / 0 步 ✓；CF2-01 `gate_mode=relative` 权重 0.626–0.674（旧 ≤6.5e-13）✓；9/10 `wrong` 臂 `control_changed=1` 且 `method_loss` 与 `none` 明显不同 ✓；**CF2-06 `wrong` 臂 `exit=1`** ✗ |

**S2 的含义（不是修复失败，是第二层根因）**：门退化修好后，方法项在总损失里已占 5.16%（旧 <0.02%），但把 `method_weight` 归零只让梯度范数变化 0.82% → 即便 `retain_weight=0`，**`neutral_weight=0.1` 的 fact / donor / EOS guard 仍主导梯度**。这进一步印证 §5 的「共享项淹没」结论，并说明单靠调 `retain_weight` 打不开专属梯度通道。判据跑前冻结，**不事后下调 1% 阈值让它通过**。

**S4 的含义（单一实现缺陷）**：`CF2-06/wrong` 报 `RuntimeError: The size of tensor a (27) must match the size of tensor b (15)` —— 错配 `item` 后，学生侧 `_student_secant` 取 `min(len(sp),len(sm))`、教师侧 `_teacher_secant` 取 `min(len(hp),len(hm))`，两者未取共同 `n`。同样 `REQUIRES_CONTROL=True` 的 CF2-02 因自身在 `loss()` 里做了 `n=min(...)` 裁剪而正常通过。最小修复是在 `CF2-06/loss()` 里统一 `n = min(len(sp), len(sm), len(hp), len(hm))` 后再切片。**本轮不擅自重跑该臂**，如实记为未通过。

**已确证修复生效的三处**（不等于遗忘有效）：CF2-01 门权重不再坍缩、CF2-06 数值不再爆炸、CF2-09 语义门按 §7 正确拦截；10 个错配对照臂里 9 个产生了与真实目标显著不同的 `method_loss`。

### 7.1 下一步候选（需你决定，未自动执行）

1. 修 CF2-06 形状冲突，并追加 `--set loss.neutral_weight=0.0` 的 S2 复测臂，重跑一次 smoke（约 24 臂 / 35 分钟），验证「neutral 项归零后专属梯度通道是否真正打开」。
2. 接受 S2 / S4 的阴性结论，停止调装置，按 §5 归档 CF2。
3. 直接进 pilot / 3 seeds —— **不建议**，smoke 阶段装置仍未能分辨方法专属梯度。

`claim_level=false`；smoke 不判 dF 正负、不宣称遗忘有效；GPU0 only、GPU1 未碰；**CF3-01 保持搁置**（`full_stage_started=false`）；实验 JSON / 日志不进 Git。

## 8. 复测方案（最小修复 + 根因验证；判据跑前冻结）

用户裁决：继续修 CF2 装置，范围限定为**最小修复 + 根因验证**；不补齐 §7 对照、不跑多 seed、不切回 CF3-01、不进 Stage B、不启动 `full`。

### 8.1 保留原判定

§7 的 S1/S3 PASS、S2/S4 FAIL 与 `results/fix_smoke_*` 全部保留，**不覆盖、不改判、不事后调阈值**。本轮产物写入 `results/fix_retest_*`，标为「复测 / 根因验证」。

### 8.2 两个待处理点

- S4 实现缺陷：`CF2-06_反事实割线敏感度/method.py` 的 `_student_secant` 与 `_teacher_secant` 各取各的 `min(...)`，`loss()` 未取共同 `n`，错配 `item` 后 27 vs 15 → `RuntimeError`。
- S2 第二层根因假设：`retain_weight` 归零后 `grad_relative_diff=0.82%` 仍 <1%，推测 `neutral_weight=0.1` 的 fact / donor / EOS guard 主导梯度。

### 8.3 预注册判据（跑之前冻结）

| 编号 | 断言 | 冻结阈值 / 依据 |
|---|---|---|
| R1 | CF2-06 形状缺陷已修 | `none` 与 `wrong` 两臂均 `exit=0`；`control_changed` 分别 `0`/`1`；`wrong` 臂首步 `method_loss` 与 `none` 臂不同；无 `RuntimeError` / `Traceback` |
| R2 | 复测与上一轮可比 | `CF2-01/fix` 复现 `method_share≈0.0516`、`grad_relative_diff≈0.0082`（容差 ±20%）；`fix_nomethod` 同量级。若偏离，先查清配置 / 缓存口径再解读 R3 |
| R3 | 「neutral 主导梯度」可证伪 | `fix_no_neutral` 的 `method_share` 显著高于 5.16%，且它与 `fix_no_neutral_nomethod` 的 `grad_relative_diff > 1%` → 假设成立；若仍 ≤1% → **假设被推翻**，如实记为「另有根因」，不得改阈值凑通过 |

**R3 的局限**：`neutral_weight=0` 后总损失几乎只剩方法项，`nomethod` 臂梯度趋近 0，R3 通过只能证明「方法项具备产生梯度的能力」，**不能证明完整训练口径下方法项可观测，更不能改判 S2 为 PASS**。

### 8.4 复测臂（6 个，GPU0 串行）

沿用 `--profile smoke --set data.reuse_cache=false --set training.batch_size=2`，输出到 `weights/fix_retest/`：
`CF2-06/none`、`CF2-06/wrong`（`control_mode=wrong`）、`CF2-01/fix`、`CF2-01/fix_nomethod`（回归）、`CF2-01/fix_no_neutral`（再加 `neutral_weight=0.0`）、`CF2-01/fix_no_neutral_nomethod`（再加 `method_weight=0.0`）。

`retain_weight=0` / `neutral_weight=0` **只作诊断臂**，不是正式训练口径，不用于宣称方法有效（与 CF3-01 v1.1 proposal §8 同一精神）。

### 8.5 不变约束

`claim_level=false`；默认 `control_mode=none` 行为保持可复现（若共同 `n` 改变 `none` 臂数值，R1 回归并如实记录）；GPU0 only、GPU1 不碰；**CF3-01 保持搁置**（不重跑 round3、不跑 `run_stage_a.py`、不启动 `full`）；实验 JSON / 日志不进 Git；提交只含本轮获授权文件。复测后先汇报 R1–R3，再由用户决定是否补齐 §7 对照 / 多 seed / 归档。

## 9. 复测结果（R1–R3，判据未放宽）

6/6 臂 `exit=0`、`failed=0`、`pilot_queue_started=false`，GPU0 已释放。证据：`results/fix_retest_status.json`、`fix_retest_verdict.json`、`fix_retest.tsv`、逐臂 `results/fix_retest/<tag>/<arm>/`。**§7 的 S1/S3 PASS、S2/S4 FAIL 全部保留，不改判；本节是并列的新结果。**

| 判据 | 结果 | 实测 |
|---|---|---|
| R1 | **PASS** | `CF2-06/none` `control_changed=0`、`method_loss=0.0415`；`wrong` `control_changed=1`、`method_loss=0.0602`；两臂 `exit=0`、无 `RuntimeError`/`Traceback`（上一轮 `wrong` 为 `exit=1`，27 vs 15） |
| R2 | **PASS** | `method_share=0.051575`、`grad_relative_diff=0.0082`、`grad_norm` 0.045902 / 0.045529 —— 与 §7 S2 **逐位复现**，两轮配置/缓存口径一致 |
| R3 | **PASS（trivial，见下）** | `method_share_no_neutral=1.0`；`grad_norm` 0.003620 vs **0.0**；`grad_relative_diff=361997.77`；`trivial_warning=true` |

**R1 的含义**：形状缺陷确已修复，错配对照臂现在 10/10 可用；这是「复测通过」，不是把 §7 的 S4 改判为 PASS。

**R3 的含义与局限（必须按此表述）**：`neutral_weight=0` 后总损失只剩方法项，`nomethod` 臂梯度为 0，所以 `grad_relative_diff` 的 `361997.77` 来自 `1e-8` 的防零分母，**不是真实效应量**。R3 通过只证明两件事：① 之前的 `neutral` 项确实是梯度主导项（假设成立）；② 方法项**具备产生梯度的能力**。它**不能**证明「完整损失口径下方法专属贡献可观测」，**更不能**把 §7 的 S2 改判为 PASS。`claim_level=false`。

**净结论**：两个具体缺陷（CF2-06 形状冲突、neutral 根因）都已定位并验证；但核心问题「在完整训练口径下方法专属贡献是否可观测」**仍未解决**——完整口径下 `neutral_weight=0.1` 与 `retain_weight=2.0` 仍主导梯度，十方法依然 0/10 通过判别门。

### 9.1 下一步候选（需你决定，未自动执行）

1. **分项梯度分解**（最直接回答可观测性）：在 `runtime.py::train` 里记录 `method_loss` / `neutral` / `retain_nll` 各自对参数的梯度 L2 范数占比（默认关闭），不改算法；验收：完整口径下方法项梯度占比是否可测、是否与等条件随机对照有稳定差异。
2. **补齐 §7 对照 + 多 seed**：补 noise / matched-retain / 随机算子对照，3 seeds + 簇 bootstrap，把描述性 std 升级为置信区间。
3. **接受阴性结论**，按 §5 / §7 归档 CF2，停止调装置。
4. **切回 CF3-01**：v1.1 `run_stage_a.py` 的 Stage A（只诊断不训练）已落地但从未在真实 GPU 跑过；需你确认研究已完成后再启动。

`claim_level=false`；`exit=0`、`COMPLETED`、R1–R3 全 PASS 都不等于遗忘有效；GPU0 only、GPU1 未碰；**CF3-01 保持搁置**（`full_stage_started=false`）；实验 JSON/日志不进 Git；未自动进 pilot / Stage B / full。

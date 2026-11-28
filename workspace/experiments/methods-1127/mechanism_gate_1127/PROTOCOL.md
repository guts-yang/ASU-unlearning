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

## 3. 待裁决：对照臂构造

那次对齐在「核对落地接口」处被打断，**对照臂的具体构造未定稿**。已核实：CF2 十个 `method.py` 都没有 `control_arm` 接口（不同于 CF3-01），因此不改源码可实现的对照只有配置级消融。候选方向（需你裁决后才启动 GPU 训练臂）：

1. **共享项基线**：`--set loss.method_weight=0` 跑同预算对照，检验「方法专属效应是否被 `retain_weight=2.0` 淹没」（CF2 pilot 十方法 dF 全负且生成指标完全相同，正是判别门要查的）。
2. **零编辑对照**：`--set method.alpha=0`（仅对有 `alpha` 的方法）。
3. **随机 / 乱序算子对照**：需要新增方法内配置位，属于源码改动，与「不改任何源码」冲突，须单独授权。

## 4. 不改动的边界

- 默认算法、损失口径、数据划分、评价口径、26 个既有未提交修复均不改、不混入提交。
- 不碰 GPU1 的 `lrm-unlearning` 队列，不 `pkill`、不删除他人产物。
- **CF3-01 保持搁置**：不重跑、不启动 `run_cf301_full.sh`；等你研究完再按要求重新跑。
- `full` / 多 seed / 长时消融不自动启动；离线门后先汇报，GPU 训练臂须等你裁决。
- 仓库只提交 `PROTOCOL.md` 与 `scripts/*.sh`；`results/` 下的 JSON / 日志由 `.git/info/exclude` 排除。

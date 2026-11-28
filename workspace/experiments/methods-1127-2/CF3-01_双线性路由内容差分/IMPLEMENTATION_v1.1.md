# CF3-01 v1.1 代码落地说明

对应设计：`workspace/idea-stage/candidates-1127-2/CF3-01_双线性路由内容差分/CF3-01_proposal.md`（revision `v1.1-measurement-first`）。

本轮在方法目录 `CF3-01_双线性路由内容差分/` 落地的是 **Stage A（测量优先诊断）+ Stage B 的条件式内化接口**。默认行为保持远程 round3（四候选门 + 可选 SwiGLU/POCS）不变；所有 v1.1 新增逻辑通过配置显式开启。

`claim_level=false`：本文件记录实现映射，不声称遗忘有效、不声称 Stage A 已通过、不生成实验结论。

## 1. 同步状态

- 已执行 `git fetch origin` + `git merge --ff-only origin/Auto/20261122`，HEAD `573c5d6 → 3c9dab3`。
- 同步后 `method.py` 为远程 round3 版本（四候选 + SwiGLU + POCS + 控制臂，534 行），`config.json`/`modeling.py`/`runtime.py` 为同步后的真实实现。
- 本地未提交的 idea-stage `proposal.md`（本轮上一步的 v1.1 修订）与 drawio/PNG 等文件未受同步影响，仍为工作树改动。

## 2. 代码 → proposal 映射

| proposal 章节 | 落地位置 | 说明 |
|---|---|---|
| §5.1 数值噪声下限 | `measurement_protocol.py::audit_measurement` | 重复前向误差、零编辑恢复 `ε₀`、缓存 dtype、`π` 精度相对误差；超过 `noise_floor_abs` → `MEASUREMENT_UNRELIABLE` |
| §5.2 全层诊断/稀疏 | `measurement_protocol.py::scan_all_layers` | 扫描 cache 中所有层 × 四候选，不要求每层通过；写 `stage_a_report.json` |
| §5.3 等条件对照 | `measurement_protocol.py::equal_condition_controls` | `random_equal_norm` / `wrong_shuffled` / `equal_norm_delta`，固定 layer/pair/edit 范数 |
| §5.3 簇 bootstrap + 留出 | `measurement_protocol.py::cluster_bootstrap` / `select_sparse_candidates` | 以 author/entity 簇为单位重采样；CI 下限 + discovery/held-out 双门；`INSUFFICIENT_SPLIT` 显式失败 |
| §5.4 原生 QK/V 四组合 | `measurement_protocol.py::qk_v_factorial` + `modeling.py::forward_with_attention_patch` | 2×2 因子替换 self_attn 写回（非 block-output patch）；`E_QK/E_V/total_shift` 与恒等式残差 |
| §6 稀疏条件内化 | `method.py::prepare`（`sparse_selection_file` 分支）+ `method.py::loss` | 消费 `selected_candidates.json`，仅对 Stage A 选中层构造 `A⁺−αC*` 目标，`S` 可为空 |
| §6.2 归一化 MSE 消融 | `method.py::loss`（`normalized_mse`） | 可选相对误差；默认关闭 |
| §6.2 梯度诊断 | `runtime.py::train`（`grad_diagnostics`） | 记录裁剪前参数梯度 L2 范数到 history；默认关闭 |
| §7.2 c_f fail-closed | `measurement_protocol.py::projection_calibration_check` | `independent_reference` 无 `c_f_fixed` → `C_F_UNCALIBRATED` 并建议关闭投影 |
| §7.3 相对 EOS 护栏 | `runtime.py::neutral_terms`（`eos_relative`） | 仅惩罚相对 Teacher 的 EOS 恶化；默认关闭，保留旧绝对惩罚 |
| §7.4 强制投影自检 | `measurement_protocol.py::forced_projection_selfcheck` | 纯数学：违反半空间时 `dist_from_z>0`；round3 的 `identity` 不再被当作投影验证 |
| §8 Stage A 入口 | `run_stage_a.py` | 复用 teacher cache，不训练 Student，输出 `stage_a_report.json` + `selected_candidates.json` |
| 配置默认值 | `config.json`（`measurement` / `loss.eos_relative` / `loss.grad_diagnostics` / `method.sparse_selection_file` / `method.normalized_mse` / `projection.c_f_mode`） | 全部默认关闭，round3 原行为不变 |

## 3. 运行方式

Stage A（诊断，不训练）：

```bash
python3 run_stage_a.py --self-test
python3 run_stage_a.py --profile tiny
python3 run_stage_a.py --profile pilot --diagnose-only   # 真实 Llama/TOFU，需 GPU
```

Stage B（仅当 Stage A `stage_a_pass=true`）：

```bash
python3 run.py --profile pilot \
  --set method.sparse_selection_file=/abs/path/runs/stage_a_pilot_seed0/selected_candidates.json
```

若不设 `sparse_selection_file`，`run.py` 仍执行同步后的 round3 四候选门，行为与远程一致。

## 4. 设计边界与未实现项（如实记录）

- `forward_with_attention_patch` 在 HF Llama 上替换 `self_attn` 输出；真实 Llama 的 `self_attn` 返回 tuple 形态依赖 transformers 版本，需在 GPU smoke 验证。Tiny 后端已可用。
- `equal_condition_controls` 的 `matched-retain / 同实体非目标关系` 目前由 `wrong_shuffled` 与簇留出近似，未构造独立的 retain 配对前向；proposal §7.1 把同实体关系列为**核心选择性约束**，该项尚未完整实现，不能声称已验收。
- `qk_v_factorial` 用 cache 中的 `π,V` 重算 `o_proj(π@V)`，不是从原始 Q/K 重跑 softmax；这是"原生 QK/V 路径替换"的可计算近似，需在论文/报告中与真实 Q/K/V patching 划界。
- §5.2 的 head/position 级细化扫描、§7.4 的真实冻结下游强制投影测试、§8 阶段 B 的多 seed/恢复攻击/完整审计，均为协议或轻量数学自检，未在真实模型上跑过。
- `teacher_eos_margin`（§7.3 相对 EOS）当前未写入 teacher cache；若启用 `eos_relative` 且 cache 无该字段，回退为 0，需要在 `data.py::build_teacher_cache` 补缓存后才具备完整语义。
- 未改 `data.py` 的配对构造逻辑；v1.1 要求的关系对照 / author 簇隔离沿用现有 `build_pairs`。
- 本轮未运行 GPU 训练、未改远程代码、未 push、未把 Tiny smoke 当作机制证据。

## 5. 决策门

- 若 `run_stage_a.py` 输出 `MEASUREMENT_UNRELIABLE` / `INSUFFICIENT_SPLIT` / `NO_STABLE_CANDIDATE`，按 proposal §9 停止，不进入 Stage B。
- `stage_a_pass=true` 只表示"存在超过噪声与随机对照、且跨留出簇稳定的稀疏候选"，**不表示遗忘有效**；Stage B 仍须比较 `method-only / retain-only / full` 与等条件随机对照，并做无 hook、relearning、量化恢复验收。

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

- `forward_with_attention_patch` 在 HF Llama 上替换 `self_attn` 输出；已在 H20 真实 Llama 的 Stage A 中跑通。Tiny 后端亦可用。
- `equal_condition_controls` 的 `matched-retain / 同实体非目标关系` 已落地：配对生成时 `require_control=True`，cache 额外保存 control 的 attention/GLU，`scan_all_layers` 记录 `delta_m`；`stage_a_pass` 要求 control 覆盖率达标。
- `qk_v_factorial` 用 cache 中的 `π,V` 重算 `o_proj(π@V)`，不是从原始 Q/K 重跑 softmax；报告标 `APPROX_ONLY`，且 `used_for_gate=false`，不用于 `stage_a_pass`。
- §5.2 的 head/position 级细化扫描、§7.4 的真实冻结下游强制投影测试、§8 阶段 B 的多 seed/恢复攻击/完整审计，仍未在真实模型上跑过。
- `teacher_eos_margin`（§7.3 相对 EOS）已写入 teacher cache；`eos_relative` 仍默认关闭。
- `data.py` 的配对构造已改为规范作者词表 + 别名排除 + 分层采样 + 主体位置预筛；`build_teacher_cache` 的 fail-closed 校验未放松。
- 本轮运行了 GPU0 的 Stage A 诊断（不训练）；未改远程代码、未 push、未把 Tiny smoke 当作机制证据。

## 6. 2026-11-28 P0 修复与 Stage A 实测

- 配对：`make_pairs_cf301_v2.py` 生成 39 对，覆盖 20 个作者簇（其中 1 个簇 1 对，其余 2 对）；0 个主体位置不一致，0 个伪主体，donor 与 forget 作者不重叠；39/39 带 matched-retain control。
- 全层：`measurement.scan_layers="all"`，cache/scan 覆盖 1..16 层；修复了 HF `hidden_states[-1]` 是 final-norm 而非第 16 层输出的问题。
- 审计：`repeat_forward_error=0.0`、`zero_edit_recovery_eps0=0.0`、`attention_pi_component_rel_error=5.29e-8`、online-vs-cached fp16 符号一致率 1.0、Spearman 0.895（阈值 0.9/0.8）→ `measurement_reliable=true`。
- 选择：`NO_STABLE_CANDIDATE`。最强候选 `10:attn_content` 的 discovery effect CI 下限 0.00070（阈值 0.002）、specificity CI 下限 0.00028（阈值 0.001）、random CI 上限 0.00141 高于效应下限、matched CI 下限 -0.00058；其余候选更弱或方向相反。
- 结论：本轮不是测量失败，而是 attention-only 候选未表现出超过随机/同实体匹配对照的目标特异性；按 proposal §9 停止，不进入 Stage B。`claim_level=false`。

## 5. 决策门

- 若 `run_stage_a.py` 输出 `MEASUREMENT_UNRELIABLE` / `INSUFFICIENT_SPLIT` / `NO_STABLE_CANDIDATE`，按 proposal §9 停止，不进入 Stage B。
- `stage_a_pass=true` 只表示"存在超过噪声与随机对照、且跨留出簇稳定的稀疏候选"，**不表示遗忘有效**；Stage B 仍须比较 `method-only / retain-only / full` 与等条件随机对照，并做无 hook、relearning、量化恢复验收。

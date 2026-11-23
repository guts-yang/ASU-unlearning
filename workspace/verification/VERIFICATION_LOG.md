# 复现验证执行记录

**日期**: 2026-11-23  **commit**: `3fd3f5d`  **GPU**: 1×NVIDIA H20
**代码**: `workspace/verification/`（被验证对象 `workspace/auto_experiments/`）
**账本**: `results.tsv`（`val_bpb` 列 = 适配后的 `repro_shortfall`，0.000000 表示通过）

## 阶段 A：独立复现（每个单元单独起进程）

| 顺序 | 方法 | 输出 | 判定标准实测 | 状态 | 耗时 | 峰值显存 | 日志 |
|---|---|---|---|---|---|---|---|
| U1 | τ 敏感度诊断 (PASS) | `U1_gate_sensitemp.json` | verdict=PASS, layer_cv=1.115, head_cv=9.220, top_layers=[3, 5, 2] | keep | 47.7s | 35.5GB | `workspace/verification/logs/independent_U1_gate_sensitemp.log` |
| U2 | ASU global τ=2 基线 (PASS) | `U2_train_global.json` | dF=0.7998, dR=-0.2957 | keep | 49.3s | 53.6GB | `workspace/verification/logs/independent_U2_train_global.log` |
| U3 | low2high τ 课程 (PASS) | `U3_train_low2high.json` | dF=0.9491, dR=-0.2563 | keep | 44.7s | 53.6GB | `workspace/verification/logs/independent_U3_train_low2high.log` |
| U4 | forget-grad 子空间选择性 (PASS) | `U4_gate_subspace.json` | verdict=PASS, mean_selectivity=0.5310 | keep | 8.3s | 15.1GB | `workspace/verification/logs/independent_U4_gate_subspace.log` |
| U5 | 深层 ΔA 方向 gate (PASS) | `U5_gate_deepcompress.json` | verdict=PASS, layers=4 | keep | 15.4s | 20.9GB | `workspace/verification/logs/independent_U5_gate_deepcompress.log` |
| U6 | P1/P2/P3/INT8 深度审计 (PASS) | `U6_audit_depth.json` | verdict=PASS, asu.mean_recovery=0.4477, asu_present=True | keep | 46.1s | 45.4GB | `workspace/verification/logs/independent_U6_audit_depth.log` |
| U7 | V4 报告生成 (PASS) | `IDEA_REPORT_V4.md` | decision_rows=7, deterministic=True | keep | 0.1s | 0.0GB | `workspace/verification/logs/independent_U7_summarize.log` |

## 阶段 B：串联流程（前一步输出即后一步输入）

| 顺序 | 步骤 | 依赖检查 | 输出 | 实测 | 状态 | 耗时 | 日志 |
|---|---|---|---|---|---|---|---|
| C1 | τ 敏感度诊断 | 见实测 | `01_gate_sensitemp.json` | dep=[none] verdict=PASS, layer_cv=1.115, head_cv=9.220, top_layers=[3, 5, 2] | keep | 33.3s | `workspace/verification/logs/chain_C1_gate_sensitemp.log` |
| C2 | ASU global τ=2 基线 / checkpoint | 见实测 | `02_train_global.json` | dep=[none] dF=0.7998, dR=-0.2957 | checkpoint_saved=True | keep | 52.4s | `workspace/verification/logs/chain_C2_train_global.log` |
| C3 | forget-grad 子空间选择性 | 见实测 | `03_gate_subspace.json` | dep=[checkpoint=/usr/local/workspace/ASU-unlearning-Auto/workspace/auto_experiments/checkpoints/chain_global exists] verdict=PASS, mean_selectivity=0.5310 | keep | 8.5s | `workspace/verification/logs/chain_C3_gate_subspace.log` |
| C4 | high2low 对照 | 见实测 | `04_train_high2low.json` | dep=[none] JSON parsed, training result present | keep | 47.6s | `workspace/verification/logs/chain_C4_train_high2low.log` |
| C5 | low2high τ 课程 | 见实测 | `05_train_low2high.json` | dep=[none] dF=0.9491, dR=-0.2563 | keep | 49.3s | `workspace/verification/logs/chain_C5_train_low2high.log` |
| C6 | 等平均 τ=2.607 对照 | 见实测 | `06_train_equal.json` | dep=[none] JSON parsed, training result present | keep | 47.6s | `workspace/verification/logs/chain_C6_train_equal.log` |
| C7 | SensiTemp 场 | 见实测 | `07_train_sens.json` | dep=[sens_matrix=01_gate_sensitemp.json valid] JSON parsed, training result present | keep | 44.0s | `workspace/verification/logs/chain_C7_train_sens.log` |
| C8 | 打乱 τ 场对照 | 见实测 | `08_train_shuffled.json` | dep=[sens_matrix=01_gate_sensitemp.json valid] JSON parsed, training result present | keep | 47.8s | `workspace/verification/logs/chain_C8_train_shuffled.log` |
| C9 | 深层 ΔA 方向 gate | 见实测 | `09_gate_deepcompress.json` | dep=[none] verdict=PASS, layers=4 | keep | 15.7s | `workspace/verification/logs/chain_C9_gate_deepcompress.log` |
| C10 | P1/P2/P3/INT8 深度审计 | 见实测 | `10_audit_depth.json` | dep=[checkpoint=/usr/local/workspace/ASU-unlearning-Auto/workspace/auto_experiments/checkpoints/chain_global exists] verdict=PASS, asu.mean_recovery=0.4477, asu_present=True | keep | 44.4s | `workspace/verification/logs/chain_C10_audit_depth.log` |
| C11 | 汇总 + 决策一致性比对 | 见实测 | `IDEA_REPORT_V4_CHAIN.md` | decisions_match=True, canonical={'R2-B': 'PROMOTE', 'R2-C': 'PROMOTE', 'R3-2': 'PROMOTE', 'R2-A': 'REVISE', 'R3-1': 'REVISE', 'R3-3': 'REVISE', 'R1-I1/I2/I3/I4/I5': 'ELIMINATED'} | keep | 0.0s | `workspace/verification/logs/chain_C11_report.log` |

## 一致性比对

| 单元 | 独立结果 == 串联结果 |
|---|---|
| gate_sensitemp | True |
| train_global | False |
| train_low2high | True |
| gate_subspace | False |
| gate_deepcompress | True |
| audit_depth | False |

- 规范版 V4 决策表: `{'R2-B': 'PROMOTE', 'R2-C': 'PROMOTE', 'R3-2': 'PROMOTE', 'R2-A': 'REVISE', 'R3-1': 'REVISE', 'R3-3': 'REVISE', 'R1-I1/I2/I3/I4/I5': 'ELIMINATED'}`
- 串联版 V4 决策表: `{'R2-B': 'PROMOTE', 'R2-C': 'PROMOTE', 'R3-2': 'PROMOTE', 'R2-A': 'REVISE', 'R3-1': 'REVISE', 'R3-3': 'REVISE', 'R1-I1/I2/I3/I4/I5': 'ELIMINATED'}`
- 决策完全一致: True
- 依赖/数值连通性检查通过: False

## 整体验证结论

- 独立阶段: 7/7 `keep`
- 串联阶段: 11/11 `keep`
- 无 crash: True
- 决策一致性: True

**OVERALL: FAIL**

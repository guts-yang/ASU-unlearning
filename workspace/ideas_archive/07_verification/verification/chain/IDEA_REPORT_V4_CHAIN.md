# Idea Discovery Report V4 — 自动实验验证与自优化

**日期**: 2026-11-23
**流水线**: idea-stage(V1/V2/V3) → workspace/auto_experiments 自动 gate → 实测结果驱动优化
**算力**: 1×NVIDIA H20，TOFU-1B；数据/模型来自 `/jizhicfs/chenyliao`
**诚信声明**: 新数字均来自本轮实跑；V1 结论来自既有 stage0-14 实测。SensiTemp 已补等平均 τ 对照，避免把更强平滑误判为结构化分配收益。未启用跨模型评审，裁决为执行者 provisional。

## 总览（按自动决策排序）

| Idea | Gate | Decision | 自动优化结论 |
|---|---|---|---|
| R2-B ASU 温度课程 τ(k) | PASS | PROMOTE | PROMOTE：将 low2high 设为默认 τ 调度（retain 未受损且遗忘更强）；high2low 仅作消融 |
| R2-C 抗恢复即目标（relearning-aware meta） | PASS | PROMOTE | PROMOTE：在 ASU 目标上加内步 P3 模拟攻击 + 外步最大化恢复后 forget NLL，作为独立贡献线与 R3-2 审计共享攻击阶梯 |
| R3-2 ASU 遗忘深度审计基准 | PASS | PROMOTE | PROMOTE：无论裁决均保留——本次为 PASS（ASU 存在浅层遗忘），给出统一恢复阶梯与深度指标，作为独立基准贡献 |
| R2-A ASU-Selective（RepSelect 式子空间选择性） | PASS | REVISE | REVISE：机制 gate 通过（bottom-PC 更 forget-specific），但 R3 查新显示 RepSelect/ADU/ALMPU 已密集占据该空间，差异化不足；只有在 'ASU 输出遗忘 × attention K/V 子空间' 的耦合能被明确证伪区分时才重启，否则不投入主线 |
| R3-1 SensiTemp 层-头自适应温度场 | PARTIAL | REVISE | REVISE：敏感度场存在但收益来自更强总平滑而非分配，不能声称温度场带来帕累托改进；降级为诊断分析，并把 low2high τ 调度作为可发表增量 |
| R3-3 浅-深解耦双阶段遗忘 | PASS | REVISE | REVISE：方向仍有边缘信号，但 V2 的抑制失败模式重现且 retain 同步受损；仅保留为消融，不投入主线训练 |
| R1-I1/I2/I3/I4/I5 反事实配对 + ΔA 正交/对齐/差值引导（全部） | KILL | ELIMINATED | 不再投入算力；转为反重复记忆，禁止后续轮次复现 |

## 证据明细

### R1-I1/I2/I3/I4/I5 — 反事实配对 + ΔA 正交/对齐/差值引导（全部）
- 证据: stage0-14 H20 实测：所有变体 forget/retain 均弱于原 ASU ref；stage8/9/10/11/13/14 失败，stage5 mask 无收益
- Gate: KILL
- 决策: ELIMINATED

### R2-A — ASU-Selective（RepSelect 式子空间选择性）
- 证据: ASU-Mu 上 v_proj/down_proj forget-grad SVD：mean_selectivity=0.531（top-PC 与 retain 梯度重合显著高于 bottom-PC）
- Gate: PASS
- 决策: REVISE

### R2-B — ASU 温度课程 τ(k)
- 证据: 60-step 等步数训练 dF: global=0.800, high2low=0.562, low2high=0.949; 最佳=low2high dF=0.949, dR=-0.256
- Gate: PASS
- 决策: PROMOTE

### R2-C — 抗恢复即目标（relearning-aware meta）
- 证据: 深度审计 P3 微调恢复率最高=0.371（ASU-u 模型），表明浅层遗忘确实存在
- Gate: PASS
- 决策: PROMOTE

### R3-1 — SensiTemp 层-头自适应温度场
- 证据: 敏感度非均匀（layer_cv=1.11, head_cv=9.22, top layers=[3, 5, 2]）；dF: sens=1.747 vs 等平均τ全局=1.791 vs 打乱场=1.517，结构化分配无净收益
- Gate: PARTIAL
- 决策: REVISE

### R3-2 — ASU 遗忘深度审计基准
- 证据: P1/P2/P3/INT8 完成；ASU-u forget 0.96→P2 恢复 0.261→P3 恢复 0.185，mean_recovery≈0.448
- Gate: PASS
- 决策: PROMOTE

### R3-3 — 浅-深解耦双阶段遗忘
- 证据: gate=PASS 但仅 L12 suppress 达标边缘（dF=+0.133, dR=+0.099）；严格阈值 dF>=0.15 且 dR<=0.05 下通过数=0
- Gate: PASS
- 决策: REVISE

## 自动选定的下一步（仅 PROMOTE）

- [ ] R2-B: PROMOTE：将 low2high 设为默认 τ 调度（retain 未受损且遗忘更强）；high2low 仅作消融
- [ ] R2-C: PROMOTE：在 ASU 目标上加内步 P3 模拟攻击 + 外步最大化恢复后 forget NLL，作为独立贡献线与 R3-2 审计共享攻击阶梯
- [ ] R3-2: PROMOTE：无论裁决均保留——本次为 PASS（ASU 存在浅层遗忘），给出统一恢复阶梯与深度指标，作为独立基准贡献

## 产物索引

- `workspace/auto_experiments/results/`: gate JSON、训练记录、等预算对照
- `workspace/auto_experiments/logs/`: 全部运行日志
- `workspace/auto_experiments/checkpoints/`: 本轮训练的 ASU/温度模型

# Experiment Plan — Plan-level Candidate Screening on Frozen ASU TOFU-1B

> 状态：可执行 screening plan。代码基线：`workspace/auto_experiments/`。
> 冻结环境：`/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python`（torch 2.5.1+cu121, cuda available）。

## Milestones and Run Order
1. M0 sanity：`common.py` 数据/模型路径 + 一次 5-step `train_tau.py --schedule global --steps 5 --seed 0`。
2. M1 CF-01 → M2 CF-03 → M3 CF-02 → M4 CF-04 → M5 CF-06 → M6 CF-05 → M7 CF-08 → M8 CF-07 → M9 CF-09 → M10 CF-10。
3. M11 R2-C → M12 R3-2 → M13 R3-3。
4. 所有候选顺序执行，不并行；一个候选判定后写入 tracker 再进入下一个。

## Per-Candidate Evidence Mapping
| ID | 判别对照 | 主要证据源 | 首轮动作 | 成功判据 |
|---|---|---|---|---|
| CF-01 | 约束反演目标 vs 最近点/朴素外推；因果 vs 相关筛选 | proposal §决定性验证；现有 `stage13_decouple`、`stage8_idea1` | 若核心算子未实现则记录 `NOT_IMPLEMENTABLE`；否则按三臂 + 3 seed | forget 降、retain 不崩、donor 不替换同时成立 |
| CF-03 | 因果 patch vs 等范数随机/wrong-donor | `gate_sensitemp.json` PASS；`stage8_idea1` random/wrong | 复用 gate；缺臂用 `sens`/`sens_shuffled`/`random` 短跑 | 因果臂显著分离且加噪 donor 无伪峰 |
| CF-02 | retain floor on/off × donor 负约束 on/off | proposal §决定性验证；`stage13_decouple` | 仅当三角色约束可实现时跑；否则 `NOT_IMPLEMENTABLE` | 可行解非空，retain floor 托住 U，donor 替换下降 |
| CF-04 | 事实 vs 身份 probe AUC；wrong-donor 身份投影 | `stage7b_da_probe_pooled.json`（retain-orth PASS） | 复用 pooled probe；缺事实/身份双探针则标部分验证 | AUC_f > AUC_id 且 wrong-donor 落身份子空间 |
| CF-06 | 轨迹加权 vs 答案跨度/均匀/反向 | proposal；无 LRM/CoT 基线 | `NOT_APPLICABLE`，转 LRM track | 不在本 track 判定 |
| CF-05 | 共有 vs 私有成分，K 曲线 | proposal；无 K-donor 缓存 | 若 K-donor 数据/算子缺失则 `NOT_IMPLEMENTABLE` | AUC_f(marg) > AUC_f(priv) 且随 K 收敛 |
| CF-08 | 零空间投影 vs 原始 ΔA vs hook-only | `audit_depth.json`、`stage7b_da_probe_pooled.json`、`stage14_ga_orth.json` | 复用恢复阶梯；缺臂短跑 | 恢复曲线下面积改善且无 hook 仍有效 |
| CF-07 | 读出空间 vs 隐空间 ΔA，旋转不变性 | proposal；`stage11_newa_result.json` | 需冻结 Jacobian 方案；未预注册则 `NOT_IMPLEMENTABLE` | 旋转不变 + 中性约束可满足性差异显著 |
| CF-09 | 噪声基残差前后，wrong-relation/matched-retain | `stage8_idea1`、`stage9_idea1_opt` | 复用对照结果；核心残差算子缺失则 `NOT_IMPLEMENTABLE` | 残差保留事实信息且输出抑制捷径下降 |
| CF-10 | 纯度驱动 vs 静态/反向/随机/等均值 | `gate_sensitemp.json`、`stage4_progressive_result.json` | 复用 3-seed 结果；无需新训练 | 纯度谱非平坦且等预算下独立收益 |
| R2-C | 闭环/开环/伪证书，留出第三类攻击 | `audit_depth.json`；proposal 代码落点 | 双环 trainer 未实现则 `NOT_IMPLEMENTABLE`，不伪造 | 留出攻击改善且 retain 不崩 |
| R3-2 | P1→P2→P3→INT8→Probe 统一阶梯 | `audit_depth.json` PASS | 补 Probe 通道后统一汇总 | 方法间显著分化；≥2 类独立抗恢复证据 |
| R3-3 | ASU→深层对齐→审计；等预算对照 | `stage9_idea1_opt`（全部 pass_gate=false） | 复用 V4 阴性结果，仅作消融记录 | dF≥0.15 且 dR≤0.05；未达则 NEGATIVE |

## Frozen Protocol
- Model/tokenizer/data/splits/metrics：严格继承 `INIT_REPORT.md`。
- 新增短跑：`train_tau.py --steps 20 --seed 0 --batch-size 4 --lr 1e-5 --alpha 1.0`；只在同一冻结数据上比较。
- 3-seed 确认仅对首轮正向候选执行（seeds 0,1,2）。
- 总预算上限 8 GPU·h；单候选超过 2x 估计时间即 timeout。

## Metrics
- 主指标：forget NLL（越高表示遗忘越强）、retain NLL（越低越好）。
- 恢复：P2 few-shot gain、P3 relearning gain、INT8 forget shift；按 `audit_depth.json` 口径。
- 特异性：donor 答案率、拒答/EOS/完整回答率（仅已有生成评估可用时报告）。
- 统计：3-seed mean/std；单 seed 结果只能标 `PROVISIONAL_SCREEN`。

## Stop and Routing
- `POSITIVE_SCREEN` → 追加 3-seed 确认 → `/auto-review-loop` → `/experiment-audit` → `/result-to-claim`。
- `NEGATIVE`/`INCONCLUSIVE`/`NOT_IMPLEMENTABLE`/`NOT_APPLICABLE` → 写 postmortem，进入下一候选。
- 结果文件：`results/screening/<candidate>.json`；tracker：`refine-logs/EXPERIMENT_TRACKER.md`；compact log：`refine-logs/EXPERIMENT_LOG.md`。

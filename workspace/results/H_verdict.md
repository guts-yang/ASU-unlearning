# H1 / H2 判定报告

## 判定标准（固定）

- **H1 成立**：`max(FQ_SH, FQ_HS) − min(FQ_DS, FQ_MT) > 10`（百分点）。SH/HS/DS/MT 缺任一则无法判定。
- **H2 成立**：`RR_FocusOnKey(ASU) > 20%`，且不低于 GA/NPO 较高者的 1/10（同量级或更高）。对照缺失则无法判定。

## 实测

| 量 | 值 |
| --- | --- |
| FQ_SH(ASU) | 56.2 |
| FQ_HS(ASU) | 50.0 |
| FQ_DS(ASU) | 25.0 |
| FQ_MT(ASU) | 71.4 |
| gap_pp | 31.19 |
| SH−DS | 31.19 |
| SH−MT | -15.24 |
| HS−DS | 25.00 |
| HS−MT | -21.43 |
| RR_FocusOnKey(ASU) | 32.4 |
| RR_FocusOnKey(GA) | 43.5 |
| RR_FocusOnKey(NPO) | 53.7 |

## 对照对齐

GA/NPO 的 FQ_SH 未对齐到 ASU 的 10 个百分点以内（已选最接近的 checkpoint，未再加扫）。

## 结论

- **H1**：成立
- **H2**：成立
- **决策**：H1∧H2 成立 → story（审计+FocusOnKey 对撞+APRU 修复）立住，可开写。

## 下一步建议

- 若 H1∧H2：进入审计+FocusOnKey 对撞实验与 APRU 修复叙事。
- 若仅 H1：收缩为结构路径审计，弱化恢复攻击主线。
- 若仅 H2：保留恢复攻击信号，但不把结构路径差距当成已成立。
- 若均不成立：保留本阴性结果，更换 base/协议或扩大结构与模型再测；不要为撑假设调参。

## 复现元信息

- seed / split / τ / α / attacks：见 `workspace/notes/run_meta.json`
- GPU：ok — 2x A800 80GB

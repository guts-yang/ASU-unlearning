# H1 / H2 判定报告

## 判定标准（固定）

- **H1 成立**：`max(FQ_SH, FQ_HS) − min(FQ_DS, FQ_MT) > 10`（百分点）
- **H2 成立**：`RR_FocusOnKey(ASU) > 20%`，且与 GA/NPO 同量级或更高（≥ 0.5× max(peers)）

## 实测

| 量 | 值 |
| --- | --- |
| FQ_SH(ASU) | NA |
| FQ_HS(ASU) | NA |
| FQ_DS(ASU) | NA |
| FQ_MT(ASU) | NA |
| gap_pp | NA |
| RR_FocusOnKey(ASU) | NA |
| RR_FocusOnKey(GA) | NA |
| RR_FocusOnKey(NPO) | NA |

## 结论

- **H1**：无法判定（缺数）
- **H2**：无法判定（缺数）
- **决策**：缺评测结果，暂无法判定；先完成 θ_p/θ_u LeakRU eval 再跑 aggregate。

## 下一步建议

- 若 H1∧H2：进入审计+FocusOnKey 对撞实验与 APRU 修复叙事。
- 若仅 H1：收缩为结构路径审计，弱化恢复攻击主线。
- 若均不成立：保留本阴性结果，更换 base/协议或扩大结构与模型再测；**不要为撑假设调参**。

## 复现元信息

- seed / split / τ / α / attacks：见 `workspace/notes/run_meta.json`
- GPU：blocked — torch.cuda.is_available() is False (device open may return EPERM). Scripts/data ready.

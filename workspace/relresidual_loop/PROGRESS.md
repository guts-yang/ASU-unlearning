# PROGRESS — RelResidual 验证闭环 v6

**run_id**: `relresidual-20261124`  **next_round**: 1  **phase**: `idle`
**Gate0**: verdict=`KILL` query-last cos=0.2753 ratio=0.9799 高 cos 层=[15] —— 原版前提不成立，仅作诊断输入
**best**: NA (none)
**llm_backend**: `REVIEW_UNAVAILABLE` — 确定性规则引擎，非 LLM
**冻结目标**: mean_dF>=0.7998（不弱于 ASU 基线）, std(dF)<=0.08, 所有 order dR<=0.05, donor<=0.1, premature<=0.1, truth<=0.3
**TARGET_MET**: False  **已独立复现 idea**: 0

| round | lever | val_bpb | dF[0,1,2] | mean_dF | std_dF | worst_donor | worst_prem | worst_truth | replicated | good | status |
|---|---|---|---|---|---|---|---|---|---|---|---|

## 预算

- rounds 0/5, gpu 0/1800s, crash 0

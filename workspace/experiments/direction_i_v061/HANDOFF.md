# HANDOFF · direction_i_v061（D1–D10 通宵实验）

```yaml
run_id: "llm-directioni-v061-20261126"
track: llm-unlearning
phase: experiment-bridge
status: done-with-caveats
selected_idea: "D1–D10 (IDEA_CANDIDATES.md, ΔA 构造候选集)"
baseline_ref: "open-unlearning ASU baseline; TOFU forget10/retain90; Llama-3.2-1B-Instruct"
engine: "proxy-v061（answer-span S_f / teacher-force X+ (+) y+ / patch-forward / 中性 hinge）；非 §4.2 校准可行集"
evidence_handle: null
claim_verdict: null
blocking_items:
  - "D2 证书需要训练后 student checkpoint；当前 train_v061 未落盘权重"
  - "中性约束 c/d_j/e 未校准，D1/D7/D10 只能给两端边界（align≈0 vs neutral 崩溃）"
  - "反事实配对关系匹配率 0.0625（模板指纹仅命中 1/16）"
  - "单 seed、16 对样本；D1/D7/D10 无量信区间"
  - "跨模型 reviewer 后端不可用（REVIEW_UNAVAILABLE），不得 accept/provisional"
review_status: "REVIEW_UNAVAILABLE — 未伪造回执"
next_action: "先校准中性约束（在 align 与 neutral 之间定 c/d_j/e），再补 3 seed 与 D2 证书；关系匹配需按 TOFU 模板重建配对"
```

## 结果

见 `results/RESULTS.md`（10 条）与 `results/SUMMARY.json`（机读）。

- `support` / `weak support`：D3、D4、D6、D8、D9
- `against`：D5
- `engine-limited` / `deferred`：D1、D2、D7、D10

## 关键发现

1. `T=A⁺` 时 `L_F` 退化为保持损失，`dF=0.023–0.068`，不产生遗忘。
2. 去掉中性约束直接最大化原答案 NLL，`dF` 达 17–29，但 `eos_rate` 掉到 0.083、生成逼近上限 → 输出崩溃而非遗忘；中性约束是必需项。

## 产物

- `code/`：engine / pairs / diag / train_v061 / run_diag / run_train / summarize
- `results/diag.json`、`train_D*_{align,neutral}_s0.json`、`train_all*.json`、`SUMMARY.json`、`RESULTS.md`
- `logs/`：diag、train_s0、train_neutral

## 状态边界

`proxy-v061`、单 seed、16 对、关系匹配率 0.0625；不构成 `accepted`、不构成 claim 或论文结论。编排器全局 `init` 仍为 `BLOCKED`。

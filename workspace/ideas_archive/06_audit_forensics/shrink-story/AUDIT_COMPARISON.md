# v1 → v2 取证对比（闭环收尾）

**日期**：2026-11-23　**模式**：deterministic-only 回退（无跨模型 reviewer）

| 项 | v1 `paper_under_audit` | v2 `paper_under_audit_v2` |
|---|---|---|
| `submission.tex` | 摘要仍主张 "improves forgetting by 0.1491 NLL"，但结果表已给出等预算反证 → **语义冲突** | 摘要改写为"等预算下三个维度均无可辩护收益"，并逐字给出 `dF=1.1416` vs `0.9491`；冲突已消除 |
| 结果表 | 8 行（Round 1） | 14 行（Round 1 + Round 2），两张表分别标注指标 |
| 新增披露 | — | matched-ΔF 对照 B6、指标不对称缺陷、B5 被复杂度惩罚误杀、退化输出风险、双环 meta 未实现 |
| `observability_level` | L2 | L2 |
| `claims.json` | 43 条（number 26 / table_cell 16 / caption 1） | 48 条（number 18 / table_cell 28 / caption 2） |
| 四个确定性检查器 | 0 / 0 / 0 / 0 finding | 0 / 0 / 0 / 0 finding |
| `anchoring_verified` | true | true |
| `human_review_required` | true | true |
| **`overall_verdict`** | **`REVIEW_UNAVAILABLE`** | **`REVIEW_UNAVAILABLE`** |

## 为什么修了表述 verdict 仍不变

`adjudicate_findings.py` 的 coverage gate 规定：任一 verdict-bearing 维度为 `review_unavailable` 即把 `CLEAN_GIVEN_EVIDENCE` 降级为 `REVIEW_UNAVAILABLE`。本环境没有 `codex`/`gemini`/`grok`/`knot` 任一评审后端，10 个语义/代码级维度全部 `review_unavailable`，因此**结构上不可能 acquit**——这与论文写得好不好无关。

这也说明为什么确定性检查器 0 finding **不等于**"通过"：0 finding 只表示这四个算术/表面检查没查到东西，语义一致性、baseline 比较、实验取证、引用取证等维度根本没跑。

## 闭环结论

- `AUDIT_FEEDBACK.md` 的线索 B（表述层）已在 v2 落实；线索 A/C（实验层、覆盖层）在 Round 2 推进，并额外产出线索 D/E/F（指标缺陷、退化输出风险、子空间路线无效）。
- v2 未改动 v1 的任何被审文件（新目录），符合 anti-autoresearch 的 detect-only 约束。
- 要真正把 verdict 从 `REVIEW_UNAVAILABLE` 推到可判定状态，必须启用一个异族评审后端后重跑完整模式；在此之前不得声称 `CLEAN_GIVEN_EVIDENCE`，也不得声称"论文诚实"。

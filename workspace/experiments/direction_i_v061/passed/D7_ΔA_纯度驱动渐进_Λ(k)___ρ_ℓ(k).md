# D7 · ΔA 纯度驱动渐进 Λ(k) / ρ_ℓ(k)

**判定**：**Tier B · 有条件过关**
**复现状态**：已复现（proxy-v061；单 seed）
**日期**：2026-11-26

## 核心假设

用 ΔA 的逐层事实纯度导出渐进 schedule，替代手设 Λ(k)。

## 实测结果

| 配置 | dF | dR | eos | overlap |
|---|---:|---:|---:|---:|
| align | 0.023 | −0.179 | 1.000 | 0.389 |
| neutral λ=0.2 | 0.589 | −0.173 | 1.000 | 0.399 |

纯度排序：`layer_purity` 写入结果 JSON；渐进 schedule 只在 3/11/15 三层上按纯度顺序加入。

## 证据

`results/train_D7_lam0.2.json`、`results/train_D7_s0.json`

## 复现命令

```bash
cd /usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/direction_i_v061/code
/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python run_two.py
```

## 局限

未与“等预算静态全层 / 反向 / 随机 schedule”做四臂对照（proposal §6 的硬要求），因此**不能声称渐进带来独立收益**。

## 升级为确认的条件

升级为确认的条件：补齐四臂等预算对照；若静态全层追平，则只能保留“多层对齐”表述。

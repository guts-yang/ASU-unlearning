# D3 · 因果门控 ΔA（patch 加权的因果分量）

**判定**：**Tier A · 过关**
**复现状态**：已复现（proxy-v061；3 seed）
**日期**：2026-11-26

## 核心假设

ΔA 中只有少数层/位置对目标事实读出有因果贡献；按因果效应加权可剔除身份/词频分量。

## 实测结果

| 层 | ΔNLL（patch A⁻ 后） |
|---|---:|
| L3 | +0.204 |
| L7 | +0.372 |
| L11 | +0.916 |
| L15 | +1.343 |

3 seed 均为 layer 15 主导，`top_layer_squared_share = 0.650 ± 0.023`（seed 0/1/2: 0.639/0.676/0.635）。四层单调递增。

## 证据

`results/diag_v2.json`（D3）、`results/seed_sweep_D3_D9.json`

## 复现命令

```bash
cd /usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/direction_i_v061/code
/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python run_diag_v2.py
```

## 局限

代理因果量是单层 patch 的 NLL 变化，不是逐位置/逐分量的完整因果追踪；样本 16 对。

## 升级为确认的条件

升级为确认的条件：逐分量（head × position）因果追踪 + 3 seed + 至少 2 个 donor 构造方式复现同一层排序。

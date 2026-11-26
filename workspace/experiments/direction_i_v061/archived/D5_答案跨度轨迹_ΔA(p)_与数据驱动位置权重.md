# D5 · 答案跨度轨迹 ΔA(p) 与数据驱动位置权重

**判定**：**归档（不过关 / 未判定）**
**复现状态**：已复现（proxy-v061；单 seed）
**日期**：2026-11-26

## 核心假设

ΔA 沿答案跨度有位置结构，可用轨迹定位“事实绑定发生的答案位置”。

## 实测结果

- `mean_gini = 0.316`（v1 0.206）
- `mean_top1_position_share = 0.085`（均匀为 1/14 ≈ 0.071）

判定：接近均匀，位置结构不足以支撑数据驱动权重。

## 证据

`results/diag.json`（D5）、`results/diag_v2.json`（D5）

## 复现命令

```bash
cd /usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/direction_i_v061/code
/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python run_diag_v2.py
```

## 局限

只用了一种位置归一化；未测 factual/function token 分层。

## 重新开启条件

重新开启条件：按 factual/function token 分层重算轨迹；若仍近均匀则归档为“事实不是按答案位置局部展开”。

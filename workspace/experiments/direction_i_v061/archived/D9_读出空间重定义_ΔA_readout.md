# D9 · 读出空间重定义 ΔA_readout（旋转不变）

**判定**：**归档（不过关 / 未判定）**
**复现状态**：已复现（proxy-v061；3 seed）
**日期**：2026-11-26

## 核心假设

ΔA 定义在冻结下游读出的标量差上，天然对隐空间旋转不变，并提供额外信息。

## 实测结果

| seed | pearson(readout, hidden) |
|---|---:|
| 0 | −0.230 |
| 1 | +0.321 |
| 2 | +0.123 |

均值 `0.071 ± 0.279`，符号跨 seed 翻转 → 不可判定。

## 证据

`results/diag_v2.json`（D9）、`results/seed_sweep_D3_D9.json`

## 复现命令

```bash
cd /usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/direction_i_v061/code
/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python run_seed_sweep.py
```

## 局限

只估了标量 NLL 差与 ‖ΔA‖ 的相关，没有估计 Jacobian 方向；v1 的 +0.469 是单 seed 偶然。

## 重新开启条件

重新开启条件：估计 Jacobian 一阶方向并做正交变换不变性测试；否则归档为“读出与隐空间关系不稳”。

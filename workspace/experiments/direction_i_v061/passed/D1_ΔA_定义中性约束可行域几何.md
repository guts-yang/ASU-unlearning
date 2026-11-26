# D1 · ΔA 定义中性约束可行域几何（禁止方向）

**判定**：**Tier B · 有条件过关**
**复现状态**：已复现（proxy-v061；单 seed）
**日期**：2026-11-26

## 核心假设

让 ΔA 只定义可行域的禁止方向（`⟨h−A⁺, ΔA⟩≤0`），从结构上排除答案替换，同时不教 donor 答案。

## 实测结果

| 配置 | dF | dR | eos | overlap |
|---|---:|---:|---:|---:|
| align（T=A⁺） | 0.068 | −0.156 | 1.000 | 0.391 |
| neutral λ=0.05 | 0.336 | −0.147 | 1.000 | 0.367 |
| neutral λ=0.10 | 0.631 | −0.142 | 1.000 | 0.385 |
| **neutral λ=0.20** | **1.455** | **−0.132** | **1.000** | **0.373** |
| neutral λ=0.50 | 15.250 | −0.063 | 0.917 | 0.140（退化） |

## 证据

`results/calibration.json`、`results/train_D1_neutral_s0.json`、`results/train_D1_s0.json`

## 复现命令

```bash
cd /usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/direction_i_v061/code
/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python run_calib.py
```

## 局限

λ 是 `c/d_j/e` 的代理，未真正标定约束阈值；单 seed；禁止方向 hinge 的效果未与“无 hinge”做等强度对照。

## 升级为确认的条件

升级为确认的条件：标定 `c/d_j/e`、补 3 seed、加“同强度但无禁止方向”的对照。

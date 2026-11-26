# D2 · ΔA 深度证书化 / 方法身份张力 (B)

**判定**：**归档（不过关 / 未判定）**
**复现状态**：已复现（proxy-v061；未判定）
**日期**：2026-11-26

## 核心假设

ΔA 作为训练后的因果式遗忘深度证书，与训练解耦。

## 实测结果

- base 上 patch A⁻ 后 NLL 变化 `+0.916`（v1 +1.930），16/16 同号
- 仅证明 ΔA 的因果可检测性，**不是证书**

## 证据

`results/diag_v2.json`（D2）

## 复现命令

```bash
cd /usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/direction_i_v061/code
/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python run_diag_v2.py
```

## 局限

证书必须 patch **训练后 student** 回 A⁺ 并测恢复量；当前 `train_v061` 不落盘权重，无法测。

## 重新开启条件

重新开启条件：给 `train_v061` 加 checkpoint 保存，再对 D1/D10 的 λ=0.2 模型跑 patch 回 A⁺ 的恢复测试。

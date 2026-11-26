# D6 · 多 donor 边际化 / donor 私有成分消去

**判定**：**Tier B · 有条件过关**
**复现状态**：已复现（proxy-v061；单 seed）
**日期**：2026-11-26

## 核心假设

单 donor 的 ΔA 混有该 donor 的身份特异性；用 K 个 donor 的共有成分作事实方向。

## 实测结果

- `cos(single, marginalized) = 0.836` → 边际化确实改变方向
- `cos(marginalized, answer) = 0.055` → 共有成分与答案 token 的对齐很弱

判定：方向改变成立，但“共有成分 = 事实”未成立。

## 证据

`results/diag.json`（D6）、`results/diag_v2.json`（D6）

## 复现命令

```bash
cd /usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/experiments/direction_i_v061/code
/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python run_diag_v2.py
```

## 局限

K=4、donor 来自 retain90；共有成分用简单均值而非低秩子空间；与答案对齐只有 0.055，不能支持“事实方向”。

## 升级为确认的条件

升级为确认的条件：K 扫描 + SVD 共有子空间 + 与事实 probe AUC 联合验证。

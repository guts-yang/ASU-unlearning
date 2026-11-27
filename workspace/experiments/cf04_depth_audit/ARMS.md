# 审计臂决策（阶段 2）

`claim_level=false`、`gate_status=NOT_APPLIED`。所有臂在同一套 P1/P2/P3/INT8 + Probe 参数下执行，只使用 GPU 0。

## 主臂

| 名称 | checkpoint | 预算 | 说明 |
|---|---|---|---|
| `cf04` | `/jizhicfs/chenyliao/llm-unlearning/checkpoints/full_cf04_v2_formal` | 10 epochs / 4500 steps | CF-04 v2，dashboard P3「checkpoint 未审计」的目标 |

**checkpoint ↔ 结果对应已验证**：用完整 split（`batch_size=8`）重算该 checkpoint 的 forget/retain/holdout NLL，与 `results/formal_training/CF-04_v2_formal.json` 的 `after` 逐项对比，最大差 3.0e-6（见 `results/checkpoint_integrity.json`）。因此这份权重确实对应报告的 dF=+8.0920 / dR=-0.4561 / dH=+0.9656。

**预算差异声明**：主臂是 4500 steps；用户所指的 v2g 是 300 steps / 3 seeds（dF=+2.3084±0.0915）。两者不是同一预算，审计结论不直接归给 v2g，也不与 `methods-1127` 的 pilot 结果混算。

## 预算阶梯臂（无需训练，均为已保存 checkpoint）

| 名称 | checkpoint | 预算 |
|---|---|---|
| `cf04_e1` | `.../full_cf04_e1` | 1 epoch |
| `cf04_formal` | `.../full_cf04_formal` | CF-04 v1，10 epochs |

用于观察 P2/P3 恢复率与 INT8 漂移随预算/版本的变化，不用于 claim。`full_cf04_v2_formal_save` 与主臂结果完全相同，不重复审计。

## 对照臂

| 名称 | checkpoint | 说明 |
|---|---|---|
| `base` | 真实 `Llama-3.2-1B-Instruct` TOFU checkpoint | 口径对齐基线（`results/alignment_base.json` 与历史 `audit_depth.json` 的 base 完全一致，最大差 0.0） |
| `ga` | `/jizhicfs/.../20261122_forget10_1b/GradAscent` | GradAscent 对照，架构与 base 一致（vocab 128256 / hidden 2048 / 16 层） |
| `asu` | `.../asu-forget05-asu-gd` | **跳过**：7B / vocab 32000 / hidden 4096，架构守卫会拒绝；且它是 forget05 split，口径不同。仅保留历史 `audit_depth.json` 中的 ASU 数值作文字参考 |

## 可选第二臂：300-step v2g faithful 复现 — 本轮不启用

理由（记录为阻塞/后续项，不是失败）：

1. `results/cf04_v2_300/v2g*/CF-04.json` 中没有 `checkpoint` 字段，`train=null`，`auto_experiments/checkpoints/` 目录不存在；**300-step v2g 权重未保存**，无法直接做深度审计。
2. 其训练实现在 `workspace/experiments/diagnostic_methods/run_one_method.py`（600 行，相对导入 `.base` / `.cf04_dual_probes`，含 `run_cf04_v2` 的自适应 lambda、多层双探针、PCGrad 等）与 `run_full_training.py`（273 行）。按「不调用旧入口」的约束需要移植约 870 行逻辑并复现 3-seed 的 dF/dR/dH 才算 faithful，风险与工作量显著高于审计本身；且用户已选择「先做遗忘深度审计、不改算法」。
3. 因此本轮以已验证对应关系的 formal v2 checkpoint 为主臂，并显式标注预算差异。若之后要精确审计 300-step v2g，需单独批准「移植训练实现 + 加 `--save` 重训 3 seeds」这一项，再对生成的权重执行同一套审计脚本。

## 执行命令

```bash
cd workspace/experiments/cf04_depth_audit
CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python \
  nohup bash run_audit.sh > results/console.log 2>&1 &
```

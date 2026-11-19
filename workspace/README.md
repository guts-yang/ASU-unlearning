# RecodeLoop M0（K3）

在 **ASU-unlearning** 基线之上、仅位于 `workspace/` 内的 M0 研究代码框架。对基线代码**零修改**。

## 当前状态

- **canonical v1 主实验已完成**：按最新 proposal 使用完整 \(D_f/D_r/NK\) 口径；G0/G1/G3 PASS，G2 INCONCLUSIVE，G4/G5 BLOCKED。**M0 未通过**。
- G1 读取器正对照：target 在完整 \(D_f\) 上优势 `0.245`，NK `0.010`，95% CI `[0.135, 0.380]`。
- 主臂在完整 \(D_f\) 上普遍地板效应：full/no_op/one-shot/random/shuffle/fixed_causal 的遗忘优势均为 `0.000`，pars_style 为 `0.005`；full 与四个主对照的 sign-flip p 值均为 `1.0`。
- 独立非线性 probe、Q4/Q8 量化、限预算重学习未覆盖；输出健康中 full 截断率相对 no_op 增加 `0.04`，超过草案阈值。不得宣称 L2 或 retain 无损。
- 旧 M0/M0b 系列使用废弃的实体角色拆分，仅作开发诊断；旧整合报告已归档到 `docs/archive/`。当前结论以 `docs/RecodeLoop_canonical_v1_结果与结论.md` 为准。
- 复现前必须先按 `docs/GPU_CUDA_环境准备记录.md` 准备 CUDA 版 PyTorch / cuBLAS 环境变量。

## 目录

```
configs/    冻结配置（阈值/预算/种子；组合哈希 5f419fad…）
manifests/  assets.yaml 三角色 + entity_splits.yaml 10 作者簇均已冻结
src/recodeloop/  类型/适配器/激活hook/事实值reader/因果定位/低秩编辑/闭环/对照/审计/统计
scripts/    G0 校验、CPU smoke、M0a/b/c 入口
tests/      14 项工程测试
docs/       基线审计、接口契约、实现日志
outputs/    G0/M0 状态与 smoke 报告（本地产物）
.cache/     pytest/字节码/临时权重（不纳入版本控制）
```

## 常用命令（均在 `workspace/` 下执行）

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/validate_assets.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest
PYTHONDONTWRITEBYTECODE=1 python scripts/smoke_test.py
PYTHONDONTWRITEBYTECODE=1 python scripts/run_m0a.py   # 当前：BLOCKED
PYTHONDONTWRITEBYTECODE=1 python scripts/run_m0b.py   # 当前：BLOCKED
PYTHONDONTWRITEBYTECODE=1 python scripts/run_m0c.py   # 当前：BLOCKED
```

## 科学口径

1. 主恢复优势为同一模型上的 `delta_r = A_r(M_student; forget) - A_r(M_student; NK)`；训练 reader 只作信号，不作为验收终点。
2. phi 仅用于组件选择；相关性、attention 权重、probe 准确率不得替代因果定位。
3. full 必须分别优于同低秩 PARS-style、等预算 one-shot、random/shuffle；否则对应贡献否决。
4. 编辑 reader 改善但独立 reader/重学习恶化 → Goodhart/核心机制否证；retain 失守 → 选择性失败。
5. confirmation 不足 4 簇或区间过宽 → `INCONCLUSIVE`，不是科学通过。
6. 未实际执行的审计一律 `uncovered`，不得默认通过。

详见 `docs/RecodeLoop_canonical_v1_结果与结论.md`。

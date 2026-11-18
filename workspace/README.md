# RecodeLoop M0（K3）

在 **ASU-unlearning** 基线之上、仅位于 `workspace/` 内的 M0 研究代码框架。对基线代码**零修改**。

## 当前状态

- **G0：PASS**；**M0a：PASS**（生成式事实值 reader，target 正对照 2/2、NK 0/2，base/student 0/2）；**单轮短训：完成**（1 轮/2 步/rank 4/1 组件，仅工程验证）。
- **未运行**：M0b 完整对照比较、五类独立审计、retain 门与 confirmation。短训结果不构成遗忘有效性或 L2 结论。
- **科学评价未启动**。`run_m0a.py`、`run_m0b.py`、`run_m0c.py` 均先过 G0，未通过则停止；没有任何遗忘有效性结论。
- **工程实现已完成**并通过 14 项 CPU 独立测试；`smoke_test.py` 只是工程验证，明确标记 `forgetting_evidence=false`。
- 原有 `data/`、`docs/`、`figures/` 与 `docs/K3_RecodeLoop_Proposal.md` 均已保留，未覆盖。

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

详见 `docs/current_status_and_conclusions.md`（一针见血结论）、`docs/baseline_inventory.md`、`docs/interface_contracts.md`、`docs/implementation_log.md`。

# CF3-02 自校正持续子空间（SCPS）

独立实现，不导入其他 CF3 方法。算法来源见本目录 `proposal.md`。

## 运行

```bash
python3 run.py --self-test
bash run.sh tiny
bash run.sh smoke
bash run.sh pilot
python3 run.py --profile pilot --diagnose-only
```

## 实现

- 在源层替换完整残差状态，保持其他上下文一致，运行冻结 Teacher 的下游层。
- 对源层和预注册 replay 深度收集真实与 wrong-pair 差异，计算生存比例。
- 在固定低秩空间估计真实/伪配对广义谱，保留超过 null 的持续方向。
- 将原始反事实差投影到持续子空间，构造 stop-gradient Student 目标。

`method.min_eigenvalue` 是真实档谱门阈值；`replay_offsets`、`projection_dim`、`rank`、`ridge` 可独立配置。patch 可能产生 OOD 状态，生存谱和 NLL 变化都不能单独证明事实语义。

# CF3-08 反事实 Copula 尾依赖擦除（CTDE）

独立实现，不导入其他 CF3 方法。算法来源见本目录 `proposal.md`。

## 运行

```bash
python3 run.py --self-test
bash run.sh tiny
bash run.sh smoke
bash run.sh pilot --set method.projection_dim=2
```

## 实现

- 使用固定低秩主体/答案投影和独立 retain ECDF 构造秩坐标。
- 对真实正反事实与 shuffled null 分别计算条件共同 exceedance 差。
- Student 使用 logistic soft ECDF 和 soft exceedance，确保尾损失可微。
- 尾依赖对齐每层 null 水平，而非一律归零；分位数死区限制两端边缘漂移。
- 默认 `batch_size=8`，因为单样本不能估计 mini-batch 尾统计；仍需通过样本量偏差曲线确定可靠批量。

这是有限样本经验 copula，不是渐近极值尾依赖系数。`tiny` 只验证可微和数值通路，不提供高维尾部样本效率证据。

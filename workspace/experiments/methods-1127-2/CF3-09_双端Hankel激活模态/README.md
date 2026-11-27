# CF3-09 双端可控—可读 Hankel 激活模态（DHAM）

独立实现，不导入其他 CF3 方法。算法来源见本目录 `proposal.md`。

## 运行

```bash
python3 run.py --self-test
bash run.sh tiny
bash run.sh smoke
bash run.sh pilot --set method.random_probes=4 --set method.rank=2
```

## 实现

- 在主体嵌入中点以随机方向有限差估计输入 JVP 的低秩范围。
- 在中间层答案位置 patch 方向，运行完整冻结下游，以目标 NLL 和固定随机输出投影估计读出方向导数。
- 在低秩 `C·U` 上做平衡特征分解，将模态映回答案均值状态空间。
- 只选择真实反事实差能量超过 wrong-pair 的模态，再构造 Student 目标。

**实现边界**：这是局部中点、答案池化、随机低秩近似，不显式构造完整 d×d Jacobian、Wc/Wo 或 Hankel 矩阵。局部谱不自动代表非线性全网的全局因果通路。

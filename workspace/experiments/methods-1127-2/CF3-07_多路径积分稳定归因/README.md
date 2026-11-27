# CF3-07 多路径积分稳定归因（MPIA）

独立实现，不导入其他 CF3 方法。算法来源见本目录 `proposal.md`。

## 运行

```bash
python3 run.py --self-test
bash run.sh tiny
bash run.sh smoke
bash run.sh pilot --set method.quadrature_steps=4
```

## 实现

- 固定非主体 token 与原始答案，仅沿主体嵌入槽从 `X-` 顺序过渡到 `X+`。
- 使用正序与逆序主体路径，在中点以有限差方向导数做 quadrature。
- 按原始主体坐标对齐不同顺序的归因，计算跨路径稳定性并删除稳定分量。
- 检查数值完备性相对误差；严格档还比较 wrong-pair 路径稳定性与逐层覆盖。

**实现边界**：首版是 sequential path 的答案均值向量归因，不是 proposal 完整逐主体坐标×隐藏维张量，也尚未实现经真实词嵌入锚点的分段路径。报告中保留这些限制，不能把 Tiny 完备性通过当作真实路径机制成立。

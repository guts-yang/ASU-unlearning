# CF3-10 非交换路径签名遗忘（NPSU）

独立实现，不导入其他 CF3 方法。算法来源见本目录 `proposal.md`。

## 运行

```bash
python3 run.py --self-test
bash run.sh tiny
bash run.sh smoke
bash run.sh pilot --set method.projection_dim=8
```

## 实现

- 将选定层的答案均值反事实差降至固定低秩，并加入归一化层坐标。
- 以分段线性路径增量精确计算二阶 signature 和非交换 Lévy 面积。
- 真实/null 面积与跨 split 符号稳定性共同形成固定掩码。
- Student 同时运行 `X+ / X-`，只优化掩码面积，不以 donor 答案为目标。
- 同一实体的非目标关系也运行双前向，签名与 Teacher 锚对齐，抑制双边冻结捷径。

自测覆盖路径反转变号、直线面积为零和重复点时间重参数化不变。自动配对需要控制关系；不足时 fail closed 或提供显式 `pairs_file`。这些代数性质不是事实遗忘语义证明。

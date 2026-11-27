# CF3-05 答案跨度 Grassmann 子空间擦除（AGSE）

独立实现，不导入其他 CF3 方法。算法来源见本目录 `proposal.md`。

## 运行

```bash
python3 run.py --self-test
bash run.sh tiny
bash run.sh smoke
bash run.sh pilot --set method.rank=4 --set method.selected_rank=2
```

## 实现

- 使用独立 retain 集构造位置保持的答案均值，而非分别去均值。
- 对正反答案跨度矩阵做低秩 SVD，计算投影算子差与主角度诊断。
- 真实/null 投影差以流式均值累计，避免 full 档保存批量完整 d×d 矩阵。
- 根据相对 null 的谱超出和 split 余弦选择目标方向，再投影原始答案差供 Student 内化。

该对象对同一子空间内的基旋转不变，但对完整隐空间任意旋转不自动不变。首版未宣称优于 LEACE/CIR，需用独立对照验证。

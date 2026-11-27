# CF3-06 配对置换—FDR 激活门（PEFG）

独立实现，不导入其他 CF3 方法。算法来源见本目录 `proposal.md`。

## 运行

```bash
python3 run.py --self-test
bash run.sh tiny
bash run.sh smoke
bash run.sh pilot --set method.permutations=199 --set method.fdr_q=0.1
```

## 实现

- 以 retain RMS 归一化每个配对的层×答案位置差异。
- 使用配对内独立符号置换构造零分布，p 值采用 `(1 + exceed)/(B + 1)`。
- 对全部预注册假设执行 Benjamini–Yekutieli FDR，记录假设数、split 余弦和显著性。
- 偶数 pair 用于选择，奇数 held-out pair 用于独立 patch 读出门，避免用同一 split 完成统计与因果筛选。
- 门在 Student 训练期间固定；显著但无 held-out NLL 效应的位置不会进入训练。

至少需要 4 个配对；更多配对和置换才能支持可靠统计推断。BY-FDR 不是因果证明，也不替代 matched-retain 假阳性率审计。

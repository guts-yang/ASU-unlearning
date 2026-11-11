# v0.3 残差定位

## 结论

**尚无证据支持目标选择性局部路径。**

donor 效应没有同时超过写死的噪声、confirmation 和 retain 门槛，或正确性/读取门未通过。不称为遗忘成功，不启动 Teacher/Student 训练。

这是 10 个作者簇上的工程 pilot。正的 delta 只表示城市 span 的条件对数概率下降，不是遗忘成功。没有启动 Teacher/Student 训练。

## 门

| 门 | 状态 | 说明 |
| --- | --- | --- |
| 来源 | BLOCKED | blocked_no_network_or_unreadable_wikidata |
| 读取 | PASS | 问号向量与缓存、hook 坐标 |
| 目标信号 | FAIL | 通过 0 / 10 |
| patch 正确性 | PASS | no-op、self-patch 容差 0；下游输入必须改变 |
| 提名 S | FAIL | 层：[] |
| confirmation 主张 | FAIL | empty_S |

## 冻结集合

- discovery：promptdonor-01, promptdonor-02, promptdonor-03, promptdonor-04, promptdonor-05, promptdonor-06
- confirmation：promptdonor-07, promptdonor-08, promptdonor-09, promptdonor-10
- S：[]
- 总体：`NO-GO`

## 差中差

下列均值只用目标信号通过的簇。原始逐簇分数在 `confirmation.jsonl`。

- forget donor：None
- forget noise：None
- retain donor：None
- retain noise：None
- donor_specific 中位数：None
- selectivity_DID 中位数：None
- 信号通过且 donor_specific 为正的 confirmation 簇：None

## 没有做的事

没有修改候选、TOFU、retain95 或 `my_llama.py`。没有运行路径条件教师，也没有运行学生训练。

生成时间：`2026-11-11T10:34:46.607163+00:00`。

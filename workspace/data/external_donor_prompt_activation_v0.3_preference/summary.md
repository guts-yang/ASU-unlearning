# v0.3 固定前缀相对偏好

## 结论

**尚无证据支持目标选择性局部路径。**

donor 相对噪声的 M 下降没有同时满足 confirmation 和 retain 门槛，或读取、边界、正确性门未通过。不称为遗忘成功，不启动 Teacher/Student 训练。

M 是固定前缀下目标城市与 donor 城市的平均条件对数概率之差。donor 城市是该 donor 人物的出生地，不是目标人物的真值。自由生成按整句分类，句首出现不是硬门。正的 M 下降不是遗忘成功。没有启动 Teacher/Student 训练。

## 门

| 门 | 状态 | 说明 |
| --- | --- | --- |
| 来源 | BLOCKED | blocked_no_network_or_unreadable_wikidata |
| 读取 | PASS | 问号缓存、城市 span 边界 |
| 目标偏好 | PASS | M_before > 0：10 / 10 |
| patch 正确性 | PASS | 两段城市 logits 容差 0 |
| 提名 S | PASS | 层：[8, 9, 13] |
| confirmation 主张 | FAIL | m_threshold_not_met |

## M 基线

| 簇 | 目标城市 | 替代城市 | M | 目标均值 | 替代均值 | 可提名 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| promptdonor-01 | Karachi | Wellington | 11.5728 | -0.0017 | -11.5745 | 是 |
| promptdonor-02 | Beijing | Seattle | 15.5974 | -0.0018 | -15.5992 | 是 |
| promptdonor-03 | Tel Aviv | Louisville | 7.1013 | -0.1066 | -7.2078 | 是 |
| promptdonor-04 | Addis Ababa | Gary | 22.7036 | -0.0006 | -22.7042 | 是 |
| promptdonor-05 | Tokyo | Tupelo | 6.2980 | -0.2612 | -6.5592 | 是 |
| promptdonor-06 | Cape Town | London | 7.3540 | -0.1040 | -7.4580 | 是 |
| promptdonor-07 | Manama | Mingora | 7.8934 | -1.1981 | -9.0916 | 是 |
| promptdonor-08 | New York City | Rio de Janeiro | 4.6947 | -1.0355 | -5.7302 | 是 |
| promptdonor-09 | Kuwait City | New Orleans | 15.7149 | -0.0000 | -15.7149 | 是 |
| promptdonor-10 | Astana | Rosario | 13.6407 | -0.0506 | -13.6914 | 是 |
| retain-01 | Lagos | Wellington | 11.5195 | -0.0005 | -11.5199 | 是 |
| retain-02 | Brussels | Seattle | 13.7511 | -0.0006 | -13.7517 | 是 |
| retain-03 | Dar es Salaam | Louisville | 12.5737 | -0.0008 | -12.5745 | 是 |
| retain-04 | Madrid | Gary | 21.4375 | -0.0038 | -21.4413 | 是 |
| retain-05 | Amman | Tupelo | 7.9747 | -0.0007 | -7.9754 | 是 |
| retain-06 | Dhaka | London | 13.3137 | -0.0006 | -13.3143 | 是 |
| retain-07 | Los Angeles | Mingora | 10.2740 | -0.0002 | -10.2741 | 是 |
| retain-08 | Seoul | Rio de Janeiro | 6.7795 | -0.0004 | -6.7799 | 是 |
| retain-09 | New York City | New Orleans | 7.8127 | -0.0006 | -7.8133 | 是 |
| retain-10 | Bangkok | Rosario | 14.7344 | -0.0000 | -14.7344 | 是 |

## 自由生成基线

分类和 M 并列，不参与提名。

- forget：正确 8，错误事实 0，拒答 0，截断 0，未提及 2，人工复核 0，共 10
- retain：正确 10，错误事实 0，拒答 0，截断 0，未提及 0，人工复核 0，共 10

## 冻结集合

- discovery：promptdonor-01, promptdonor-02, promptdonor-03, promptdonor-04, promptdonor-05, promptdonor-06
- confirmation：promptdonor-07, promptdonor-08, promptdonor-09, promptdonor-10
- S：[8, 9, 13]
- 总体：`NO-GO`

## M 的差中差

下列均值只用 M_before > 0 的簇。逐簇原始分数在 `confirmation.jsonl`。

- forget donor：0.012505379357108382
- forget noise：-0.001838095929708905
- retain donor：0.06238672776039698
- retain noise：-0.008503278317430852
- donor_specific 中位数：-0.00866179847315074
- selectivity_DID 中位数：-0.07955180455097857
- 可提名且 donor_specific 为正的 confirmation 簇：2

## 联合 S 上的自由生成

- donor / forget：正确 2，错误事实 0，拒答 0，截断 0，未提及 2，人工复核 0，共 4
- donor / retain：正确 10，错误事实 0，拒答 0，截断 0，未提及 0，人工复核 0，共 10
- noise0 / forget：正确 2，错误事实 0，拒答 0，截断 0，未提及 2，人工复核 0，共 4
- noise0 / retain：正确 10，错误事实 0，拒答 0，截断 0，未提及 0，人工复核 0，共 10
- shuffle / forget：正确 2，错误事实 0，拒答 0，截断 0，未提及 2，人工复核 0，共 4
- shuffle / retain：正确 10，错误事实 0，拒答 0，截断 0，未提及 0，人工复核 0，共 10

## 没有做的事

没有修改旧定位目录、候选、TOFU、retain95 或 `my_llama.py`。没有运行路径条件教师，也没有运行学生训练。

生成时间：`2026-11-11T13:06:19.869220+00:00`。

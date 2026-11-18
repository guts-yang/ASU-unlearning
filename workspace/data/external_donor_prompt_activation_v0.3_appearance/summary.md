# v0.3 原始问题出现率定位

## 结论

**尚无证据支持目标选择性局部路径。**

donor 相对噪声的目标事实出现率下降没有同时满足 confirmation 和 retain 门槛，或读取、正确性门未通过。不称为遗忘成功，不启动 Teacher/Student 训练。

主读出是原始问题下完整回答的目标城市出现率。固定前缀 M 只作联合 S 上的辅助诊断。正的出现率下降不是遗忘成功。没有启动 Teacher/Student 训练。

## 门

| 门 | 状态 | 说明 |
| --- | --- | --- |
| 来源 | BLOCKED | blocked_no_network_or_unreadable_wikidata |
| 读取 | PASS | 问号缓存、hook、城市边界 |
| 目标可提取性 | PASS | 基线 p_target ≥ 0.5：8 / 10 |
| patch 正确性 | PASS | no-op/self-patch 贪心与 cache 校验 |
| 提名 S | PASS | 层：[0, 1, 2] |
| confirmation 主张 | FAIL | appearance_threshold_not_met |

## 出现率基线

| 簇 | 目标城市 | foil | p_target | p_foil | M_appear | 可提名 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| promptdonor-01 | Karachi | Wellington | 1.0000 | 0.0000 | 1.0000 | 是 |
| promptdonor-02 | Beijing | Seattle | 1.0000 | 0.0000 | 1.0000 | 是 |
| promptdonor-03 | Tel Aviv | Louisville | 1.0000 | 0.0000 | 1.0000 | 是 |
| promptdonor-04 | Addis Ababa | Gary | 1.0000 | 0.0000 | 1.0000 | 是 |
| promptdonor-05 | Tokyo | Tupelo | 1.0000 | 0.0000 | 1.0000 | 是 |
| promptdonor-06 | Cape Town | London | 1.0000 | 0.0000 | 1.0000 | 是 |
| promptdonor-07 | Manama | Mingora | 0.2500 | 0.0000 | 0.2500 | 否 |
| promptdonor-08 | New York City | Rio de Janeiro | 0.3750 | 0.0000 | 0.3750 | 否 |
| promptdonor-09 | Kuwait City | New Orleans | 1.0000 | 0.0000 | 1.0000 | 是 |
| promptdonor-10 | Astana | Rosario | 1.0000 | 0.0000 | 1.0000 | 是 |
| retain-01 | Lagos | Wellington | 1.0000 | 0.0000 | 1.0000 | 是 |
| retain-02 | Brussels | Seattle | 1.0000 | 0.0000 | 1.0000 | 是 |
| retain-03 | Dar es Salaam | Louisville | 1.0000 | 0.0000 | 1.0000 | 是 |
| retain-04 | Madrid | Gary | 1.0000 | 0.0000 | 1.0000 | 是 |
| retain-05 | Amman | Tupelo | 1.0000 | 0.0000 | 1.0000 | 是 |
| retain-06 | Dhaka | London | 1.0000 | 0.0000 | 1.0000 | 是 |
| retain-07 | Los Angeles | Mingora | 1.0000 | 0.0000 | 1.0000 | 是 |
| retain-08 | Seoul | Rio de Janeiro | 1.0000 | 0.0000 | 1.0000 | 是 |
| retain-09 | New York City | New Orleans | 1.0000 | 0.0000 | 1.0000 | 是 |
| retain-10 | Bangkok | Rosario | 1.0000 | 0.0000 | 1.0000 | 是 |

## 冻结集合

- discovery：promptdonor-01, promptdonor-02, promptdonor-03, promptdonor-04, promptdonor-05, promptdonor-06
- confirmation：promptdonor-07, promptdonor-08, promptdonor-09, promptdonor-10
- S：[0, 1, 2]
- 解码路径：`cache`
- 总体：`NO-GO`

## 联合 S 上的出现率差

- forget donor：0.0
- forget noise：0.0
- retain donor：0.0
- retain noise：0.003125
- D 中位数：0.0
- DID：0.003125
- 可提名且 D 为正的 confirmation 簇：0

## 辅助固定前缀 M（仅联合 S）

| 簇 | M_before | M_after | ΔM | Δ目标均值 | Δfoil均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| promptdonor-07 | 7.8934 | 7.9132 | -0.0197 | -0.0000 | 0.0197 |
| promptdonor-08 | 4.6947 | 4.6704 | 0.0244 | 0.0571 | 0.0328 |
| promptdonor-09 | 15.7149 | 15.6529 | 0.0620 | 0.0000 | -0.0620 |
| promptdonor-10 | 13.6407 | 13.5649 | 0.0758 | -0.0056 | -0.0814 |
| retain-01 | 11.5195 | 11.4493 | 0.0702 | 0.0000 | -0.0702 |
| retain-02 | 13.7511 | 13.6887 | 0.0625 | -0.0000 | -0.0625 |
| retain-03 | 12.5737 | 12.3640 | 0.2097 | 0.0001 | -0.2096 |
| retain-04 | 21.4375 | 21.5156 | -0.0781 | 0.0000 | 0.0781 |
| retain-05 | 7.9747 | 7.9795 | -0.0048 | -0.0000 | 0.0047 |
| retain-06 | 13.3137 | 13.3762 | -0.0625 | 0.0000 | 0.0626 |
| retain-07 | 10.2740 | 10.2783 | -0.0043 | -0.0000 | 0.0043 |
| retain-08 | 6.7795 | 6.8515 | -0.0719 | -0.0001 | 0.0718 |
| retain-09 | 7.8127 | 7.8127 | 0.0000 | -0.0000 | -0.0000 |
| retain-10 | 14.7344 | 14.5392 | 0.1952 | 0.0000 | -0.1952 |

## 没有做的事

没有修改旧定位目录、偏好目录、候选、TOFU、retain95 或 `my_llama.py`。没有运行路径条件教师，也没有运行学生训练。

生成时间：`2026-11-11T15:03:26.206874+00:00`。

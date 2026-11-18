# 当前结果与结论（一针见血版）

日期：2026-11-18/19  
范围：P08-M2，M0a + 单轮短训。证据文件：`outputs/g0_validation.json`、`outputs/m0a_validation.json`、`outputs/short_train_validation.json`。

## 一句话

**资产与读取器门槛已通过，RecodeLoop 的单轮工程路径已跑通；但方法有效性、定位特异性、闭环必要性、retain 选择性和 L2 都还没有被验证。当前不能宣称“RecodeLoop 有效”。**

## 已经坐实的事实

1. **G0 PASS**：base=未经 TOFU 微调的 Llama-2-7B-chat；target=locuslab TOFU 微调模型；student=本地 ASU+GD checkpoint。三角色路径、tokenizer、bundle SHA-256 与 10 个作者实体簇（2 编辑 / 2 审计 reader / 2 calibration / 4 confirmation）均已冻结。
2. **M0a PASS**：生成式事实值 reader 在 target 上恢复 2/2 个 held-out 作者名，NK 0/2，恢复优势 1.0；base 0/2、NK 0/2；student 0/2、NK 0/2。说明 reader 能解码具体事实值，不是 forget/retain 捷径，且 never-known 对照没有异常高恢复。
3. **单轮短训可运行**：student 起点上完成 1 轮、2 步、rank 4、1 组件（`model.layers.9.mlp.down_proj`）低秩编辑；逐组件能量 1.0，token 156，wall-clock 0.147 s，GPU 峰值约 14.59 GB；self/NK/reverse/random patch 均被调用，停止原因 `t_max_reached`。
4. **工程回归干净**：15 项 pytest 通过，lints 0 错误/0 警告；`git diff --name-only HEAD | grep -v '^workspace/'` 为空，基线四个子项目零修改；仓库根目录无新增临时文件/缓存/实验输出。

## 关键结论（只到这里）

- **读取器有效**：target 正对照成功、NK 未异常恢复，M0a 门槛成立。
- **当前 ASU student 在这个生成式 reader 下未恢复 held-out 作者事实**：student 与 base 都是 0/2，NK 也是 0/2。这是一个原始观测，不是“已确认遗忘”的证明。
- **短训只证明机制链路能跑**：激活提取、reader 训练、因果 patch、组件选择、低秩挂接/更新/能量投影/移除都可执行；**没有测到编辑后的收益**，因此不能说明定位有效、闭环有效或方法优于任何对照。
- **confirmation 完全未使用**：4 个 confirmation 簇仍封存；M0b 七对照、五类独立审计、retain 门、bootstrap/sign-flip 主比较均未运行。

## 明确不能得出的结论

| 说法 | 当前状态 |
|---|---|
| RecodeLoop 比 no-op / one-shot / random / shuffle 有效 | **未测**，不能说 |
| 因果定位真的找到了事实路径 | **未测**，不能说 |
| 闭环比一次性修正更好 | **未测**，不能说 |
| 独立 reader / 量化 / 重学习下仍无恢复 | **未测**，不能说 |
| retain 无损、拒答/截断/重复率合格 | **未测**，不能说 |
| 达成 L2 读取器相对访问阻断 | **不能说**，缺独立审计与 confirmation |
| ASU student 已彻底遗忘 | **不能说**，当前只覆盖一个生成式 reader、2 个 held-out 簇 |

## 科学口径红线

1. M0a 的 2 个 held-out 簇只是读取器有效性检查，不是独立验收样本；`K=8` 生成、簇 bootstrap 10000/95%、配对 sign-flip 都还没进入主结论。
2. 短训使用运行时覆盖（T=1、steps=2、threshold=-1），只用于工程验证；**不写回冻结配置、不用于 M0b 结论**。
3. 最终判读仍按预注册否决规则：full 不优于 PARS-style / one-shot / random-shuffle 即否决相应贡献；编辑 reader 改善但独立 reader 或重学习恶化即 Goodhart/机制否证；retain 失守即选择性失败；簇不足或区间过宽即 `INCONCLUSIVE`。

## 下一步

若要推进，必须先做 **M0b**：在 train/calibration 上跑 full 与 no-op、one-shot、random-direction、shuffle-component、retain-pseudo、fixed-causal、PARS-style 的预算配准比较，并加入独立 reader 终点。只有 M0b 达到预注册最小相关收益，才允许申请解封 confirmation 做一次性 M0c 审计。

在此之前，当前结果只能记为：**G0 PASS → M0a PASS → 单轮工程路径 PASS → 科学结论未开始。**

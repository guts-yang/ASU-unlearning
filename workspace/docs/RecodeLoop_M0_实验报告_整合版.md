# RecodeLoop M0 / P08-M2 实验报告（整合版）

## 一句话结论

**G0 PASS → M0a PASS → 单轮短训工程路径 PASS → M0b 预实验因地板效应无判别力。当前不能宣称 RecodeLoop 有效或被证伪，也不能宣称定位特异、闭环必要、L2 达成或 retain 无损。**

## 目标与边界

RecodeLoop 针对 ASU 的缺口：输出沉默不等于事实不可恢复。方法链路是 **残余发现 → 因果定位 → 局部低秩修正 → 独立审计**。本研究只主张 **L2：声明读取器族下的访问阻断**，不主张 L3 表示不可逆擦除。

基线隔离：四个基线子项目零修改；workspace 仅复用组件命名、`do_sample=False` 生成、`-100` 标签掩码、hook 思路，不复用 ASU teacher 的 attention temperature。

## 已坐实的事实

1. **G0 PASS**：三角色与数据冻结。  
   - base：未微调 Llama-2-7B-chat，`7fc413e3…`  
   - target：TOFU 微调遗忘前模型，`91f816bd…`  
   - student：ASU+GD checkpoint，`b6ecd585…`  
   - 10 个作者簇：2 编辑 / 2 审计 reader / 2 calibration / 4 confirmation；confirmation 仍封存。

2. **M0a PASS**：生成式事实值 reader 有效。  
   - target：held author facts `2/2 = 100%`，NK `0/2`，恢复优势 `1.0`。  
   - base：`0/2`，NK `0/2`。  
   - student：`0/2`，NK `0/2`。  
   - 这说明 reader 能读出具体事实值且不靠 forget/retain 标签捷径，NK 未异常恢复；**不说明 student 已遗忘**。

3. **单轮短训 PASS（仅工程）**：student 起点上完成 1 轮、2 步、rank 4、1 组件 `model.layers.9.mlp.down_proj` 的低秩编辑。能量 1.0、token 156、GPU 峰值约 14.59 GB，self/NK/reverse/random patch 均被调用，停止原因 `t_max_reached`。运行时覆盖 `T=1, steps=2, threshold=-1`，不写回冻结配置。

4. **回归干净**：15 项 pytest 通过，lints 0 错误/0 警告；基线四个子项目零修改；仓库根目录无临时文件或实验输出污染。

5. **M0b 预实验已运行（探索性）**：`scripts/run_m0b.py` 用 train/calibration 跑 full 与 no-op、one-shot、random-direction、shuffle-component、retain-pseudo、fixed-causal、PARS-style 的预算配准；独立生成式 reader 终点的 8 个臂优势全部为 0（retain-pseudo 因瞬时 patch 与自回归生成形状不兼容，明确记为未评估）。字面上 full 不优于 PARS-style / one-shot / random / shuffle，触发预注册否决观察；但 student 起点在该 reader 上本来就是 0/2，**主终点地板效应使本预实验没有判别力**，不能据此确认方法贡献，也不能当成方法被证伪。confirmation 未读取。

## 不能得出的结论

| 说法 | 当前状态 |
|---|---|
| RecodeLoop 优于 no-op / one-shot / random / shuffle | **预实验无判别力**（全 0，地板效应） |
| 因果定位找到了事实路径 | **预实验无判别力**（phi 选件无收益差） |
| 闭环优于一次性修正 | **预实验无判别力**（全 0） |
| 独立 reader / 量化 / 重学习下仍无恢复 | **未测** |
| retain 无损、拒答/截断/重复率合格 | **未测** |
| 达成 L2 访问阻断 | **不能说** |
| ASU student 已彻底遗忘 | **不能说** |

## 下一步

当前 M0b 预实验因主终点地板效应不可判别。下一步应先确认**存在可被独立 reader 解码的残余**：扩大 train/calibration 的可用事实或加入有动态范围的独立读取器（例如基于激活的 probe），再用同一批预算配准对照重跑；只有当 full 在有动态范围的终点上超过 PARS-style / one-shot / random / shuffle 并过预注册门，才允许一次性解封 confirmation 做 M0c 五类审计。

红线不变：full 不优于 random/one-shot 就否决对应机制或闭环主张；编辑 reader 改善但独立 reader/重学习恶化即 Goodhart/机制否证；retain 失守即选择性失败；confirmation 不足 4 簇或区间过宽即 `INCONCLUSIVE`。

## 证据文件

- `workspace/outputs/g0_validation.json`
- `workspace/outputs/m0a_validation.json`
- `workspace/outputs/short_train_validation.json`
- `workspace/outputs/m0b_pilot.json`

审计、接口与实现细节已整合进本报告；原始研究提案 `docs/K3_RecodeLoop_Proposal.md` 保留。

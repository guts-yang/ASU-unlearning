# RecodeLoop M0 / P08-M2 实验报告（整合版）

## 一句话结论

**G0 PASS → M0a PASS → 单轮短训工程路径 PASS → M0b 生成式 reader 地板效应无判别力 → v2 固定探针错配、v3 组件对齐后仍触发否决观察。当前不能宣称 RecodeLoop 有效或被证伪，也不能宣称定位特异、闭环必要、L2 达成或 retain 无损。**

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

6. **M0b 固定独立探针试点重跑（探索性，2026-11-19）**：`scripts/run_m0b.py` 的 v2 主终点改为 `fixed_independent_activation_probe`，在 train/calibration 上重跑 8 臂。固定探针在起始 student 的 `model.layers.9.mlp.down_proj` 上训练后冻结；所有可评估臂 `probe_real=1.0`、`probe_nk=0.5`、`probe_advantage=0.5`（retain-pseudo 未评估），生成式 reader 终点仍全 0。full 预算为 468 token / 6 步 / 1 组件 / rank 4，但停止原因为 `retain_gate_failed`；其选件为 `model.layers.19.mlp.down_proj`，与冻结探针所在的 `layers.9` 不一致，因此该探针终点对本轮编辑不敏感，不能用于判定 full 是否优于对照。按字面规则，full 未优于 PARS-style / one-shot / random-direction / shuffle-component，触发否决观察；confirmation 未读取。

7. **组件对齐诊断 v3（探索性，2026-11-19）**：新增 `scripts/run_m0b_v3_diagnostic.py`，在 full 选定 `model.layers.19.mlp.down_proj` 后，用全新起始 student 在该组件上训练并冻结独立探针，再对同一批臂评分；同时直接测量编辑后 calibration retain NLL 差值。结果：full / no_op / one_shot / random_direction / fixed_causal / pars_style 的 `probe_real=probe_nk`、`probe_advantage=0`，只有 shuffle_component（编辑 `layers.0`）为 `0.5`；full 仍未优于四个对照，触发否决观察。编辑后 retain NLL 差值全部为负（full `-0.548`，one_shot/fixed_causal `-0.760`，pars_style `-2.258`），说明 v2 的 `retain_gate_failed` 来自定位阶段的瞬时 patch 代理指标，并非编辑后 retain 真实恶化；retain 红线不能按 v2 字面判定，但也不能据此宣称 retain 无损（拒答/截断/重复率未测）。confirmation 仍未读取。

## 不能得出的结论

| 说法 | 当前状态 |
|---|---|
| RecodeLoop 优于 no-op / one-shot / random / shuffle | **v3 组件对齐后仍触发否决观察；终点在 full 上无判别力，不能作为方法结论** |
| 因果定位找到了事实路径 | **预实验无判别力**（phi 选件无收益差） |
| 闭环优于一次性修正 | **预实验无判别力**（全 0） |
| 独立 reader / 量化 / 重学习下仍无恢复 | **未测** |
| retain 无损、拒答/截断/重复率合格 | **未测** |
| 达成 L2 访问阻断 | **不能说** |
| ASU student 已彻底遗忘 | **不能说** |

## 下一步

当前不能进入 M0c。v3 已消除组件错配，但暴露出更根本的问题：在 full 实际编辑的 `layers.19` 上，独立探针对 real 和 NK 都给出 1.0，`probe_advantage=0`，终点本身无判别力；shuffle_component 的 `0.5` 来自编辑无关组件后 NK 解码下降，不能反推 full 有效。v2 的 `retain_gate_failed` 已确认是瞬时 patch 代理指标，编辑后 retain NLL 未恶化，但 retain 的拒答/截断/重复率仍未测。下一步应先扩大 train/calibration 可用事实或引入对 NK 不 ceiling 的独立读取器，获得真正有动态范围的终点，再在预注册框架下配准重跑；只有 full 在该终点上超过 PARS-style / one-shot / random / shuffle 并过预注册门，才允许一次性解封 confirmation 做 M0c 五类审计。

红线不变：full 不优于 random/one-shot 就否决对应机制或闭环主张；编辑 reader 改善但独立 reader/重学习恶化即 Goodhart/机制否证；retain 失守即选择性失败；confirmation 不足 4 簇或区间过宽即 `INCONCLUSIVE`。

## 证据文件

- `workspace/outputs/g0_validation.json`
- `workspace/outputs/m0a_validation.json`
- `workspace/outputs/short_train_validation.json`
- `workspace/outputs/m0b_pilot.json`
- `workspace/outputs/m0b_v3_diagnostic.json`

审计、接口与实现细节已整合进本报告；原始研究提案 `docs/K3_RecodeLoop_Proposal.md` 保留。

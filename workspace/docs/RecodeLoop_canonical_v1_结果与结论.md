# RecodeLoop canonical v1 结果与结论（2026-11-19）

## 一句话结论

**G0 PASS、G1 PASS、G3 PASS；G2 INCONCLUSIVE；G4/G5 BLOCKED。canonical v1 没有证明 RecodeLoop 有效，也没有证明定位特异、闭环必要、retain 合格或 L2 达成。M0 未通过。**

## 实验口径

本轮为按最新 proposal 建立的正式主实验（canonical v1），与旧的 M0/M0b 开发诊断分开：

- 完整 \(D_f\)：TOFU `forget05.json` 的 200 条事实，全部用于 ASU/RecodeLoop 遗忘更新，也全部用于最终遗忘检查。
- 完整 \(D_r\)：`retain95.json` + `real_authors.json` + `world_facts.json`，用于保留约束与能力保持。
- NK：对 \(D_f\) 每条事实构造 1 条同关系同模板的匹配虚构事实，共 200 条。
- 读取器实体簇交叉拟合只用于 reader 训练/评估，不把任何 \(D_f\) 实体排除在模型遗忘之外。
- 无 confirmation 保留集；完整 \(D_f\) 上的审计是对当前遗忘请求的事后评估，不是新实体泛化证明。
- 预注册哈希：`8c29371aded9552385e23a2607d69e2a9b9ddd906cf7acd2fa51504f0106299e`

## 门状态

| 门 | 状态 | 依据 |
|---|---|---|
| G0 资产与数据 | PASS | 完整 \(D_f/D_r/NK\)、三角色 checkpoint、哈希与 manifest 全部通过 |
| G1 读取器正对照 | PASS | target 在完整 \(D_f\) 上优势 `0.245`，NK `0.010`，95% CI `[0.135, 0.380]` |
| G2 机制特异性 | INCONCLUSIVE | full 与 one-shot/random/shuffle/pars-style 的配对 sign-flip p 值均为 `1.0`，无差异可判 |
| G3 预算与实现 | PASS | 主臂 token/步数/秩配准完成；no-op 正确 |
| G4 L2 审计 | BLOCKED | 独立非线性 probe、Q4/Q8 量化、限预算重学习未覆盖 |
| G5 retain | BLOCKED | 完整近 retain/通用 retain 准确率、KL 与输出健康门未全部评测 |

## G1 的含义

G1 是读取器正对照，不是遗忘结论。它只说明：在遗忘前 target 上，读取器能从完整 \(D_f\) 解码具体事实值，且 NK 对照很低，因此读取器具备后续审计所需的基本有效性。它不能说明最终 student 已遗忘，也不能说明方法有效。

## 主对照臂结果

所有臂都在 complete \(D_f\) 上检查，\(K=8\)。

| 臂 | \(D_f\) 准确率 | NK 准确率 | 遗忘优势 | retain NLL Δ | 截断率 | 备注 |
|---|---:|---:|---:|---:|---:|---|
| full | 0.000 | 0.000 | 0.000 | -0.106 | 0.80 | 选件 `model.layers.3.self_attn.o_proj`，停止原因 `retain_gate_failed` |
| no_op | 0.000 | 0.000 | 0.000 | 0.000 | 0.76 | ASU student 原样 |
| one_shot | 0.000 | 0.000 | 0.000 | -0.127 | 0.80 | 同组件、同秩、同预算 |
| random_direction | 0.000 | 0.000 | 0.000 | -0.001 | 0.76 | 同组件随机方向 |
| shuffle_component | 0.000 | 0.000 | 0.000 | -0.485 | 0.80 | `layers.0.mlp.down_proj` |
| fixed_causal | 0.000 | 0.000 | 0.000 | -0.127 | 0.80 | 同 full 选件 |
| pars_style | 0.005 | 0.000 | 0.005 | -0.918 | 0.72 | 64 组件，rank 4 |

配对 sign-flip（p 值）：

- full vs one_shot：`1.0`
- full vs random_direction：`1.0`
- full vs shuffle_component：`1.0`
- full vs pars_style：`1.0`

即没有任何证据支持 full 优于这些对照。

需要注意：full 的 `retain_gate_failed` 来自定位阶段的瞬时 patch 代理指标；直接测得的编辑后 retain NLL 未上升（full 为 `-0.106`）。因此不能按 v2 开发诊断的字面含义把该停止原因解释为“编辑后 retain 已失守”，但也不能据此宣称 retain 合格。

## 审计覆盖

| 审计臂 | 覆盖 | 结果 |
|---|---|---|
| 独立线性 probe | covered | 2-fold 作者交叉拟合，两折准确率均 `0.0`，均值 `0.0` |
| 独立非线性 probe | uncovered | 未在完整 \(D_f\) 上运行 |
| causal patch | covered（部分） | 记录了 full 定位阶段的 self/NK/random patch；未作为完整独立审计臂展开 |
| Q4/Q8 量化 | uncovered | 无实际量化后端 |
| 限预算重学习 | uncovered | 未在本次预算内安全运行 |

因为三类审计臂未覆盖，G4 只能是 BLOCKED，不能写成 PASS 或 FAIL。

## 输出健康初读

full 相对 no_op：

- refusal：`0.00 → 0.00`，无增加。
- truncation：`0.76 → 0.80`，增加 `0.04`，超过 proposal 草案的 1 个百分点上限。
- repeat：`0.12 → 0.12`，无增加。

按当前 truncation 实现，full 已触发输出健康红线观察；完整 G5 尚未跑完，不能把所有 retain 项判为正式 FAIL，但也不能宣称选择性保持。

## 当前结论

- 读取器正对照成立：G1 PASS。
- 主臂在 complete \(D_f\) 上均处于地板效应：full 与 no_op 都是 `0.0` 恢复优势。
- full 未优于 one-shot、random-direction、shuffle-component 或 PARS-style；机制特异性和闭环必要性均未得到支持。
- 独立线性 probe 为 0，但非线性 probe、量化、重学习未覆盖，L2 不能判断。
- retain NLL 未恶化，但 truncation 增幅超阈值，完整 retain 门也未完成。
- **M0 未通过**。当前不能宣称 RecodeLoop 有效、定位特异、闭环必要、retain 无损或 L2 达成。

## 下一步

1. 补齐三类未覆盖审计，尤其是独立非线性 probe 和限预算重学习；Q4/Q8 需接入实际量化后端后再判 G4。
2. 完成完整 G5：近 retain、通用 retain 的准确率下降、KL 与输出健康全部评测。
3. 主终点当前为地板效应，需先建立有动态范围的生成式或独立读取器终点，再讨论 full 是否优于对照；继续在同一地板终点上比较没有信息量。
4. 在完成 1–3 前不得解封新实体/新请求泛化声明，也不得把旧的 M0/M0b 开发诊断写成 canonical 结论。

## 证据文件

- `outputs/canonical_v1/preregistration.json`
- `outputs/canonical_v1/g0.json`
- `outputs/canonical_v1/g1_reader.json`
- `outputs/canonical_v1/main_arms.json`
- `outputs/canonical_v1/audits.json`
- `outputs/canonical_v1/report.json`

旧 `RecodeLoop_M0_实验报告_整合版.md` 使用已废弃的实体角色拆分，已移入 `docs/archive/`，仅作历史开发记录。

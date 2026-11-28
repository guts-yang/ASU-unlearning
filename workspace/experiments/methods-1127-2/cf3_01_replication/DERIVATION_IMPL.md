# 推导包落地说明（pasted-text-1795838967747.txt）

`claim_level=false`。proposal 的假设、决定性实验与失败判据不变；本文件记录推导包相对上一轮实现的语义修正、可证伪信号与未覆盖项。

## 1. 相对上一轮的两处实质修正

| 项 | 上一轮（球∩半空间代理、分量求和） | 本轮（推导包） |
|---|---|---|
| SwiGLU | `attn + glu` 直接相加成 route/content 两个分量 | `attn_route / attn_content / glu_gate / glu_value` **四个独立候选，不求和** |
| 门 | 8 个（4 层 × route/content） | 16 个（4 层 × 4 候选）；family 内唯一性；两个 family 都给候选 → `cross_family_ambiguity` |
| 投影 | 球 `{‖h−A⁺‖≤radius}` ∩ donor 方向半空间，阈值凭 config 推导 | 局部仿射半空间 `φ_i(z)+∇φ_i(z)ᵀ(h−z) ≤ 0` + 球形信任域 `Bρ(z)` |
| 约束量 | 激活空间代理量 | **真实冻结下游读数**：`r_f=−NLL(h;X⁺,y⁺)`、`s_dj=−NLL(h;X⁺,yj−)`、EOS logit margin 漂移；`L_fact` 不复用 |
| `r_f≤c` 的 `c_f` | 误用 `teacher_nll + loss.fact_delta`（round2 的实现缺陷） | `calibrate_c_f()` 在 discovery（`gate_pairs`）的 `z` 上测 `r_f(z)`，取 `max+margin` **一次性冻结**；运行时 `L_fact` 的 NLL 上界是另一条约束，不参与 |
| 验收 | 只检查球残差 | **真实下游回验**全部 φ；违反 → 缩小 ρ / 重试；超限 → `PROJECTION_INFEASIBLE`、严格档停止 |
| donor 声明 | `donor_answer_in_training=false` 一刀切 | 显式 `donor_constraint_only=true`：donor 答案仅作单边支持测量，不作 MSE/KL/soft label |
| 默认算法 | 源码默认开启 GLU + 投影 | 默认 `enable_glu_family=false` / `enable_affine_projection=false`；新分支**显式配置开启、可独立消融** |

## 2. 已实现的推导要点

- **SwiGLU 精确恒等式**：`a=SiLU(g)`（β=1 的 Swish），`m=a⊙u`，`D_gate=½(a⁺−a⁻)⊙(u⁺+u⁻)`、`D_value=½(a⁺+a⁻)⊙(u⁺−u⁻)`，经 `mlp.down_proj` 线性映回残差流；`self_test` 用相对 Frobenius 误差断言 `D_gate+D_value` 与真实输出差相等（实测 5e-8）。不对分量做 ReLU 或符号截断。
- **四候选独立门**：每层内 attention family 与 GLU family 各自唯一性判断；两个 family 都给唯一候选 → 该层记 `cross_family_ambiguity` 并置 `selected=None`；只在 4 个 `gate_pairs` 上做 patch-forward，保持接口兼容。
- **局部仿射半空间**：在 `z=A⁺−αC*` 处用可微 `patch_forward_output_multi` 得到真实 NLL / EOS margin，对隐藏状态求 `a_i=∇_h φ_i(z)`；半空间投影 `P_i(h)=h−[φ_i(z)+⟨a_i,h−z⟩]_+/(‖a_i‖²+ε)·a_i`。
- **donor 独立前向**：`s_dj` 用 `(X⁺, yj−)` 单独构造 `enc_donor` 并做第二次下游前向（donor 答案 token 数与 `y⁺` 不同，不能复用 fact logits）。
- **`c_f` 冻结校准（round3 修正）**：`r_f≤c` 意为「原始事实支持度足够低」，与 runtime 的 `L_fact` NLL 上界（`runtime.py` 的 `loss.fact_delta`）**不是同一条约束**。round2 曾把 `c_f=−(teacher_nll+c_fact_rel)` 直接当阈值，使每个 `z` 天然满足 `φ_fact≈+0.35` 的违反、投影修正与原始编辑同量级。本轮 `calibrate_c_f()` 只用 discovery（`gate_pairs`）的 `z=A⁺−αC*` 经真实下游回验测 `r_f(z)`，取 `max r_f(z)+c_margin` 冻结为 `c_f`，此后对全部目标**不按 item 重拟合**；`affine_project_to_k` 强制要求该值，`None` 直接 fail closed。配置键 `c_fact_rel` 删除，改为 `c_margin`；诊断新增 `c_f_calibration` 字段与 `projection_c_f_uncalibrated` kill reason。
- **`T=z` 恒等短路**：先用真实冻结下游读数检查 `z` 是否满足全部 `φ_i≤tol`；满足则直接 `T=z`、`projection_active=false`。投影是约束校正器，不应成为遗忘来源。
- **信任域**：`ρ=min(ρmax, κ·α‖C*‖)`；投影后检查 `‖T−z‖≤ρ` 与 `‖T−z‖/(α‖C*‖)`。
- **真实回验**：用冻结下游重算 4 个 φ；全部 ≤ tolerance 才 `accepted`，否则缩小 ρ 重做（outer_steps）；仍失败 → `PROJECTION_INFEASIBLE`，严格档 `enabled=false`（`DIAGNOSTIC_KILL`），**不静默退回无约束 z**。
- **零梯度保护**：`‖a_i‖²≈0` 且约束被违反 → 直接 fail closed，不除以极小值产生巨大跳步。
- **失败原因分离**：`kill_reasons` 区分 `gate_no_unique_candidate` / `cross_family_ambiguity` / `projection_infeasible`。
- **最终目标 detach**：`loss()` 只用 `MSE(h_S, stopgrad(T))`，梯度不穿过投影。
- **配置与消融**：`config.json` 默认关闭新分支；`configs/*.json` 显式开启。

## 3. 可证伪信号（推导 Remarks）

- 投影修正 `‖T−z‖/(α‖C*‖)` 接近 1：round2 smoke 最大 0.997、pilot probe 0.9998，当时按推导判为「投影与原始编辑同量级」。本轮定位其根因是 `c_f` 语义混用（把 `L_fact.fact_delta` 当 `r_f≤c` 阈值），**属于实现缺陷而非方法效应**，已按推导分离并冻结校准。**判据本身不变**：若修正后 `dist_over_edit` 仍接近 1，方法贡献依旧应判未成立。
- 若冻结 `c_f` 后 `z` 已满足真实约束，则 `T=z`（`identity`）；这是推导要求的正确行为，但同时也意味着**投影在本轮不产生额外编辑**，主链效果应全部归因于 BRCD 目标，不能把「投影通过」当作机制证据。
- 部分目标 `PROJECTION_INFEASIBLE`：约束系统与当前 BRCD 候选不兼容，严格档停止。
- 四候选效应可能高度相关 → 跨 family 不可识别风险；本轮 pilot 的 `cross_family_ambiguity` 用于观测此风险。
- `L_fact` 语义与 proposal 的 `r_f≤c` 不同，已分离；`c_f` 在 discovery 的 `z` 上冻结（诊断 `c_f_calibration`），`c_margin` / `d_donor_rel` / `eos_deadzone` / `ρmax` / `κ` 预注册在 `configs/*.json` 的 `projection` 段；**`d_j/e/ρ` 仍未做独立 discovery/validation 校准实验（见第 5 节）**。

## 4. 与 proposal 的差异与偏差原因（不变项）

| 差异 | 原因 |
|---|---|
| 精确 `Π_K` → 局部仿射 POCS 可行点代理 | 原始 K 的 `r_f/s_dj/E` 由冻结非线性下游产生，非凸；有限 POCS sweep 不保证离 z 最近；**未声称精确投影** |
| block-output patch 非完整内部机制删除 | 复用当前接口；attention post-attention 原生写点须独立消融，不与 block-output 效应直接混比 |
| 32 对 → 31 对 | 第 10 对主体级 token 位置不一致被 fail-closed 正确拒绝，覆盖率 96.875%；未放宽校验、未伪造配对 |
| `E` 用 EOS logit margin 漂移分量死区，非真实生成长度 | 真实长度不可微；真实 EOS rate / 完整回答率 / 长度在生成评估另行验收 |
| donor 单边约束测量 | 显式 `donor_constraint_only`，改变「仅评估」工程边界，已记录。 |
| bf16 `F.linear` 改 fp32 / 生成改显式贪心 | 本机 torch 2.5.1+cu121 / H20 的 SIGFPE 运行接口修复，不改算法/损失/评价口径。 |

## 5. 推导中的开放项（本轮**未**完成，如实列出）

- **投影后重新执行 true/null patch**：推导 §5 要求，本轮未做（成本：需对每个 T 再跑一遍门）。`repatch_after_projection: NOT_DONE` 写在 diagnostics。
- **Dykstra 校正**：推导指出普通 POCS 给可行点、不保证「离 z 最近」；本轮用 POCS，语义标注为「可行点代理」。
- **c / d_j / e / ρ 的 discovery/validation 校准**：`c_f` 已在 `gate_pairs`（discovery）的 `z` 上做一次性冻结，但只用了「最大值 + margin」这一简化规则，未做 discovery/validation 切分下的阈值选择与敏感性分析；`d_j`（按每个 `z` 锚定的单边 donor 支持）、`e`（EOS 死区）、`ρ`（信任域）仍是预注册值，未做校准实验。不得用测试集调参。
- **retain 与同实体非目标关系死区**、独立 probe 充分关系划分、恢复攻击、量化恢复、多 seed、`full` 预算。
- **低秩仿射读出近似**：未做；全 pair×layer 的下游 Jacobian 成本已通过只对 gate_pairs 线性化 + 其余目标复用同一线性化中心控制，代理误差未单独量化。
- **真实生成长度 / 完整回答率**：仅在 `result.json` 的 behaviour 字段观测，未纳入投影约束。

## 6. 复用接口

`modeling.py::patch_forward_output_multi`（可微）、`answer_nll`、`forward_model(capture_glu)`、`encode_qa`、`decoder_layers`、`span_hidden`、`patch_forward_nll`；`data.py::build_teacher_cache` 的 `glu_plus|minus`；`runtime.py::neutral_terms` 的 fact/donor/EOS 软约束；`CF3-04/method.py:44-53` 的 `down_proj` 映回样例。

## 7. 全量验证（阶段 B）

`scripts/run_cf301_full.sh` 备好但未启动；`full` 仍不是 proposal 全部对照/消融/投稿审计已完成的缩写。是否启动由用户决定。

## 8. 本轮（round3）实测结果

`claim_level=false`，7 个 job 退出码全 0、`failed=0`、`full_stage_started=false`。

| 观测 | 结果 |
|---|---|
| `self_test` | attention 恒等式 `4.768e-7`、SwiGLU 重构相对 Frobenius `4.995e-8` |
| `c_f` 冻结 | 主链 `c_f=-0.104699`（`n_discovery=8`，`margin=0.05`）；对照臂各自在 discovery 上冻结，取值 −0.1016 ~ −0.1078 |
| 投影 | 主链 62 / 对照臂各 124 / smoke 16 个目标**全部 `identity`**；`accepted=0`、`infeasible=0`、`max_dist_over_edit=0` |
| 四候选门 | 16 门只 2 通过（`8:attn_route`、`12:glu_value`）；`selected={'8':'attn_route','10':None,'12':'glu_value','14':None}`；`cross_family_ambiguity=[]` |
| 主链 | `DIAGNOSTIC_KILL`，`kill_reasons=[gate_no_unique_candidate layers [10,14]]`，0 步、无 checkpoint |
| 对照臂 | 5/5 `COMPLETED` 60 步；dF −0.0669 ~ −0.0682（全负），std `0.000499` |

**结论**：round2 的 `projection_infeasible` / `dist_over_edit≈1.0` 确由 `c_f` 语义误用造成，修正后按推导要求「`z` 已满足真实约束 → `T=z`」退化成恒等。这**不是**方法成功的证据：投影本轮不产生任何编辑，主链仍被严格门拦下，对照臂依旧无特异性且 dF 方向相反。因此本轮表现**不可以**，阶段 B 未启动。三轮对比见 `REPLICATION_REPORT.md §7`。

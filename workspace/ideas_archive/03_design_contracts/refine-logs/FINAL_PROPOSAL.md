# Final Proposal — Δ-Gated Progressive Orthogonal Alignment

> 由 idea-discovery Phase 4.5 精炼。对应 workspace/proposal/方向I 原始 proposal 的"激活差异处理"待调研问题的收敛解。

## Problem Anchor（冻结，防 scope 漂移）
在 ASU 基座上去掉注意力温度后，为冻结 Teacher 构造"反事实主体视图" X⁻（仅换关系匹配的 donor 主体），让 Student 在原始遗忘输入 X⁺ 上逐层内化遗忘——**目标是移除目标事实的表征方向，而不是对齐到 donor、不是教假事实、不是输出层拒绝**。统计单位为作者/实体簇。

## Method Thesis（一句话）
用配对差 ΔA_ℓ 估计"事实相关子空间"，把 Student 在 X⁺ 的层激活沿该子空间正交抑制（stop-grad Teacher），逐层渐进内化，再叠加 DKT 同前缀 KL 保持输出行为。

## 方法细节
1. **反事实输入**：X⁺=X(s_f,r)，X⁻=X(s_r,r)；token 等长、槽外 ID 不变；donor 来自 TOFU 内部保留集；按作者/别名簇隔离。
2. **事实子空间估计**：对抽点 P（每层 DecoderLayer 输出）计算 ΔA_ℓ=h_T(X⁺)−h_T(X⁻)，取低秩投影 P_ℓ（SVD top-k 或 PCA）；以 wrong-donor ΔA 作对照子空间验证特异性。
3. **正交遗忘损失**：L_F=Σ_ℓρ_ℓ(k)·‖(I−P_ℓ)h_S(X⁺;P)−stopgrad(h_T(X⁺;P))‖²；等价梯度屏蔽 g⊥=g−P_ℓg。Λ(k) 逐阶段扩展，ρ_ℓ(k) 为层权重；避开最浅/末端 20% 层（LayerSync/FDD 经验）。
4. **DKT 同前缀 KL**：L_KL=(1−η)L_T+ηL_S，两分支独立同前缀 forward KL，不逐 token 比较不同序列（沿用原 proposal §3，已验证正确）。
5. **保留损失**：L_R 为 retain-train 答案位 NLL。
6. **总损失**：L=λ_F L_F+λ_KL L_KL+β L_R；所有系数/schedule/预算实验前冻结。
7. **可选抗恢复增强**：部署前 unlearn-and-distill 到同构学生（UNDO，NeurIPS 2025）。

## Dominant Contribution
"真实关系 donor 对比锚 + 表示层正交解耦 + 不教假事实的逐层渐进遗忘目标"——与 FALCON（随机对比锚、全局选择）、CiPO（输出/偏好层教假 CoT）、CFT（教假事实、有幻觉溢出）形成三点划界。

## Must-run Experiments
- Block A（gate）：ΔA probe 判别实验 + wrong-donor 对照 → 验 C1
- Block B（gate）：等预算 schedule 对照（渐进/静态全层/反向/随机）→ 验 C3
- Block C：配对特异性 5 对照 + 非答案替换多指标 → 验 E3/E4
- Block D：参数内化 + 抗恢复 ≥2 类 + relation-swap probe → 验 C4/E5/E6

## Kill Switches
- ΔA probe 显示主体身份主导 → 收缩为 FDD-delta 变体（Idea 3）或转 Idea 2。
- schedule 对照无差异 → 撤销"渐进"主张，仅保留"多层正交对齐"。
- donor 答案上升被误判为遗忘 → 强化 L_R 与答案分解报告，否则主攻线让位。

## First 3 Runs to Launch（需 GPU）
1. A1：ΔA probe（目标事实 vs 主体身份）+ wrong-donor ΔA 对照。
2. B1：渐进 schedule（8←16 式层对启发）vs 静态全层，等预算。
3. C1：真实 donor vs noise/shuffled/wrong-donor 配对特异性对照。

详细见 `EXPERIMENT_PLAN.md`。状态：plan-only，未实现/未预注册/无新实验结果。

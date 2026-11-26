# D1–D10 实验结果（proxy-v061，2026-11-26）

## 口径与边界（先读）

- **引擎**：`../code/` 的**代理实现**——teacher-force `X⁺⊕y⁺` 前向、答案跨度 `S_f`、逐层激活、hidden patch-forward、中性 hinge。
- **未实现**：§4.2 的校准可行集（`c/d_j/e` 未标定）、独立 `N1 ReadoutTarget`；读出用冻结模型 head。
- **数据**：TOFU `forget10` / `retain90`，16 对反事实配对；**模板关系匹配率仅 0.0625**（`pairs.py` 的模板指纹只能命中 1/16），关系匹配不达标。
- **预算**：纯前向 16 样本；训练变体 60 step、`lr=5e-6`、`α=0.1`、seed 0、单卡 H20。
- **判定规则**：`support` = 代理测量支持该 idea 的机制假设；`against` = 测量与其相反；`engine-limited` = 代理引擎无法给出有效判定；`deferred` = 需要当前没有的产物。

## 总表

| # | Idea | 类型 | 关键结果 | 判定 |
|---|---|---|---|---|
| D1 | ΔA 可行域几何（禁止方向） | 训练 | align: `dF=0.068`；neutral: `dF=29.22` 但 `eos=0.083`（生成崩溃） | **engine-limited** |
| D2 | ΔA 深度证书化 | 前向 | base 上 patch `A⁻` → NLL `+1.93`（全部为正），仅证明因果可检测 | **deferred**（需训练后 checkpoint） |
| D3 | 因果门控 ΔA | 前向 | 逐层因果量 `L3=0.231, L7=0.564, L11=1.930, L15=2.277`；`best=15`，`top_layer_share=0.559` | **support**（深层主导） |
| D4 | 事实/身份双探针 | 前向 | `cos(ΔA, answer)=0.0398` vs `cos(ΔA, entity)=0.003`；`81.3%` 样本 answer 更近 | **weak support** |
| D5 | 答案跨度轨迹 ΔA(p) | 前向 | `gini=0.206`，`top1_share=0.063`（均匀为 0.071） | **against**（不集中） |
| D6 | 多 donor 边际化 | 前向 | `cos(single, marg)=0.836`；`cos(marg, answer)=0.055` | **support**（方向改变，但事实对齐弱） |
| D7 | 纯度驱动渐进 Λ(k) | 训练 | align: `dF=0.023`（最小）；neutral: `dF=17.12`、`eos=1.0` | **engine-limited** |
| D8 | 对照族残差化 | 前向 | `cos` 残差化前 `0.0398` → 后 `0.0384` | **weak support**（null 为问法变化，非严格匹配） |
| D9 | 读出空间 ΔA | 前向 | `pearson(readout, hidden)=0.469` | **support**（读出含独立信息） |
| D10 | 零空间/抗恢复投影 | 训练 | align: `dF=0.061`；neutral: `dF=29.33`、`eos=0.083` | **engine-limited** |

## 两条最关键的发现

1. **`T = A⁺` 的 `L_F` 不产生遗忘**：align 模式下三个变体 `dF=0.023–0.068`，`r_f` 基本不变。这直接印证 proposal §2.2.1 的方法身份张力——若目标落点就是原激活，`L_F` 退化为"保持"损失。
2. **去掉中性约束、直接最大化原答案 NLL 会崩溃**：neutral 模式 `dF` 冲到 17–29（ASU 仅 0.8），但 `eos_rate` 从 1.0 掉到 0.083、生成长度逼近 64 token 上限，说明模型在持续乱写而非干净遗忘。**中性约束不是可选项，而是防止退化的必要条件**；当前代理实现因未校准 `c/d_j/e`，只能在两端之间取证。

## 汇总

- `support` / `weak support`：D3、D4、D6、D8、D9（5 条）
- `against`：D5（1 条）
- `engine-limited` / `deferred`：D1、D2、D7、D10（4 条）

## 产物

- `diag.json`：D2–D6/D8/D9 原始结果
- `train_D{1,7,10}_s0.json`：align 模式
- `train_D{1,7,10}_neutral_s0.json`：neutral 模式
- `SUMMARY.json`：合并视图

## 结论强度

全部为 `proxy-v061`、单 seed、16 对样本、关系匹配率 0.0625；不构成 `accepted`，不得作为 claim 或论文结论。D1/D7/D10 的"支持与否"取决于能否先校准中性约束；在此之前它们只是**边界取证**。

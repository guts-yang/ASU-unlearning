# 收缩型故事：实验设计与 claim→gate 链

**日期**：2026-11-23　**方向**：ASU 注意力温度自蒸馏 / 遗忘深度
**来源**：`workspace/idea-stage/NARRATIVE_ASU_TEMPERATURE_DEPTH.md`
**状态**：设计冻结；自循环在 `workspace/shrink-story/loop/` 执行；取证在 `workspace/shrink-story/paper_under_audit/`

> **Problem Anchor（冻结，防漂移）**：ASU 用一个标量 $\tau>1$ 平滑注意力，使冻结 teacher 与可训练 student 之间出现 $\mathrm{KL}(p_T^\tau\|p_S)$ 分布差，这是遗忘梯度的唯一来源。我们要回答的不是"能不能让 forget NLL 更大"，而是"在**固定平滑预算**下，哪些自由度真正带来可辩护的收益，以及遗忘之后还剩什么"。禁止把 scope 扩到新模块、新损失、新数据集。

## 1. 什么是"收缩型"叙事

常规叙事是累加式：提出模块 A → 加 B → 加 C → SOTA。收缩型叙事是**每一幕先提出一个更强的主张，再用等预算/打乱对照把它收缩成最小可辩护主张**，被证伪的部分保留在论文里作为论证链的一环。

形式化：设 $c$ 为候选方案，$\bar\tau(c)=\frac1K\sum_k \tau(k)$ 为平均平滑预算，$\mathrm{ctrl}(\mu)$ 为同平均温度的全局标量对照，$\kappa(c)$ 为复杂度（0=常数标量、1=时间调度、2=空间温度场、3=新模块）。定义**等预算收缩遗憾**：

$$r(c)=\bigl[\Delta F_{\mathrm{ctrl}(\bar\tau(c))}-\Delta F_c\bigr]+2\max\bigl(0,\Delta R_c-\Delta R_{\mathrm{ctrl}}\bigr)+0.02\,\kappa(c)$$

$r(c)<0$ 才算该方案在等预算下"收缩后存活"；$r(c)\ge0$ 则主张必须降级。**这是整个自循环唯一的指标**（对应 autoresearch 的 `val_bpb`，越低越好）。

关键：第一项**不取 $\max(0,\cdot)$**，因为负的 $\Delta F$ 缺口正是"比等预算对照更差"的证据，不能被裁掉；复杂度项保证"同等收益下更简单者胜出"（简单性优先）。

## 2. 六幕 claim → gate → 收缩结果

| 幕 | 强主张 | 收缩 gate（可证伪条件） | 最小可辩护主张 | 状态 |
|---|---|---|---|---|
| 0 锚 | $\tau=1$ 时 $\nabla L_F=0$；$\tau>1$ 才有梯度 | 构造性验证：$\theta_S=\theta_T\Rightarrow p_T^1\equiv p_S$ | 温度是唯一杠杆 | 已验证 |
| 1 诊断 | 层-头敏感度矩阵非均匀 | $\mathrm{CV}<0.25\Rightarrow$ 场退化为标量，否决 | 敏感度有结构 | 已验证 CV=1.115 / 9.220 |
| 2 收缩① | 结构化温度场带来净收益 | 等平均 $\tau$ 对照 + 打乱场：$1.747<1.791$ | **不能**主张新温度场 | 已证伪 |
| 3 收缩② | 时间调度优于固定标量 | **缺口**：需 $\mathrm{ctrl}(2.25)$、时间顺序打乱、阶跃对照 | 待定（自循环核心） | **未验证** |
| 4 转折 | 遗忘是行为层的，可恢复 | P1/P2/P3/INT8：$\mathrm{depth}=0.448$ | 必须报告恢复率，不能只报 $F$ | 已验证 |
| 5 条件性 | bottom-PC / 抗恢复目标 | 新颖性冲突（RepSelect/ADU/ALMPU） | 仅作为后续条件方向 | gate 通过，降级 |

**第 3 幕是本轮唯一真正开放的缺口**：`low2high` 平均 $\tau=2.25$，却与 `global τ=2` 比较；而 $\tau=2.607$ 的 $\Delta F$ 高达 $1.7914$（vs $\tau=2$ 的 $0.7998$），说明 $\Delta F$ 对 $\bar\tau$ 极敏感。不补 $\mathrm{ctrl}(2.25)$，"+0.149 增益"完全可能只是"平均温度更高"。

## 3. 消融矩阵（自循环 run order）

固定：$K=60$ steps、$\alpha=0.1$、$\eta=5\times10^{-6}$、seed 0、TOFU forget10/retain90、1B 基座、answer-NLL 口径。所有候选与其对照**平均 $\tau$ 必须相同**。

| 轮 | 类型 | 配置 | $\bar\tau$ | $\kappa$ | 目的 |
|---|---|---|---|---|---|
| A1 | anchor | global $\tau=2.25$ | 2.25 | 0 | **补缺失的等预算对照** |
| A2 | anchor | global $\tau=2.607$ | 2.607 | 0 | 复现已有等平均对照，保证同循环可追溯 |
| C1 | candidate | linear\_up $1.5\to3$ | 2.25 | 1 | 原 low2high，检验是否在等预算下仍赢 |
| C2 | candidate | linear\_down $3\to1.5$ | 2.25 | 1 | 方向对照 |
| C3 | candidate | random $\tau_k\sim U(1.5,3)$ | 2.25 | 1 | **时间顺序打乱**：若同样赢，则"课程形状"无贡献 |
| C4 | candidate | step（前 30 步 1.5 / 后 30 步 3） | 2.25 | 1 | 检验"渐进"是否必要，还是只需后期高温 |
| C5 | candidate | SensiTemp 场 | 2.607 | 2 | 复现幕 2 的证伪 |
| C6 | candidate | 打乱场（同 $\tau$ 多重集） | 2.607 | 2 | 空间分配打乱对照 |

判定：$r(c)<0$ → `keep`（主张存活）；否则 `discard`（主张收缩）。anchor 轮固定 $r=0$，作为基线，不参与竞标。

## 4. 成功判据（整条故事能否成立）

- **S1**：A1 成功产出 $\mathrm{ctrl}(2.25)$ 的 $(\Delta F,\Delta R)$，否则整条链 `BLOCKED`。
- **S2**：C1 的 $r<0$ 才可继续主张"时间调度有效"；$r\ge0$ 则第 3 幕**收缩为"ΔF 增益由平均温度解释"**，故事改写为"两维度都未能在等预算下胜出"，此时第 4 幕（深度审计）成为唯一正面贡献。
- **S3**：C3 若与 C1 相当或更好，则"课程/渐进"表述必须撤销，只保留"后期高温"。
- **S4**：C5/C6 必须 $r>0$，否则幕 2 的证伪结论被推翻，需重跑叙事。
- **S5**：所有结论必须标注单 seed、60 steps、1B/TOFU，不得外推。

## 5. 预算与停止

- 8 轮，每轮约 50 秒，$\le 10$ 分钟；`max_experiments=8`，跑完即停（非 NEVER STOP）。
- 每轮单一可解释改动：只编辑 `loop/train.py` 顶部的 `CONFIG`。
- 账本 `loop/results.tsv` 5 列、保持未跟踪；`run.log` 每轮覆盖，只通过 grep/tail 读取。
- `discard` 采用**文件级回退**（`git checkout <last_keep>:train.py`），不使用 `git reset --hard`：主工作区含用户既有产物，禁止破坏性回滚。

## 6. 与三个技能的分工

| 技能 | 在本轮的职责 |
|---|---|
| `idea-discovery` | Problem Anchor、claim→gate 链、kill switch、等预算与打乱对照纪律、证据写入单一报告 |
| `autoresearch-loop` | 单文件可编辑、单一指标、固定预算、commit / 测指标 / `results.tsv` / keep-discard 单调链 |
| `anti-autoresearch` | 循环结束后**只读取证**：`results.tsv`→`experiments.csv`、`submission.tex` span 锚定、确定性裁决、`REPORT.md`/`report.json` |

适配声明：不使用 `val_bpb`/300 秒墙钟/`uv run`；不改主仓库分支；跨模型 reviewer 不可用，取证走确定性-only 回退，verdict 只能是 `REVIEW_UNAVAILABLE`，不得声称 `CLEAN`。

## 7. 自循环实测结果（2026-11-23，8/8 轮完成）

等预算 anchor 实跑：`global τ=2.25` → $\Delta F=1.1416,\ \Delta R=-0.3013$；`global τ=2.607` → $\Delta F=1.7914,\ \Delta R=-0.3042$（后者与既有结果完全一致，构成交叉校验）。

| 轮 | 配置 | $\Delta F$ | $\Delta R$ | $r(c)$ | 判定 |
|---|---|---|---|---|---|
| C1 | linear_up $1.5\to3$ | $+0.9491$ | $-0.2563$ | $+0.3026$ | discard |
| C2 | linear_down $3\to1.5$ | $+0.5621$ | $-0.3200$ | $+0.5995$ | discard |
| C3 | random $\tau_k\sim U(1.5,3)$ | $+0.9137$ | $-0.2997$ | $+0.2511$ | discard |
| C4 | step 30/30 | $+0.5339$ | $-0.2067$ | $+0.8169$ | discard |
| C5 | SensiTemp 场 | $+1.7470$ | $-0.2997$ | $+0.0934$ | discard |
| C6 | 打乱场 | $+1.5166$ | $-0.3053$ | $+0.3149$ | discard |

**结论（对设计的回填）**：

1. **S1 满足**：缺失的 $\mathrm{ctrl}(2.25)$ 已补，$\Delta F=1.1416$。
2. **S2 触发收缩**：$\Delta F_{\mathrm{ctrl}(2.25)}=1.1416 > \Delta F_{\text{low2high}}=0.9491$，故第 3 幕 "+0.149 增益" 完全由此前与 $\tau=2$ 比较时的**平均温度不对等**造成。等预算下时间调度不是增益，而是**损失**。
3. **S3 触发**：random 时序 C3 的遗憾 $+0.2511$ 低于 low2high C1 的 $+0.3026$，说明连"渐进形状"都没有可辨识收益；只剩"平均 $\tau$ 增大"这一强度效应。
4. **S4 满足**：C5/C6 均为正遗憾，幕 2 的证伪结论在新循环里复现。
5. **S5**：全部结论限于单 seed、60 steps、1B/TOFU。

### 故事必须改写

原六幕中第 3 幕（"时间维度比空间维度更有效"）**不能成立**。收缩后的最终故事是：

> 在固定平滑预算下，**空间的层-头温度场**（幕 2）与**时间的 τ 课程**（幕 3）都未能优于最简单的全局标量 τ；此前观察到的所有 ΔF 增益都可由平均温度解释。因此 ASU 可辩护的改进空间不在"如何分配 τ"，而在"τ 造成的行为遗忘有多浅"——统一恢复阶梯显示 $44.8\%$ 的增益可被 few-shot / 少量微调恢复（幕 4）。这把一个方法论文收缩为一个**负结果 + 评估基准**论文：贡献是等预算对照协议、恢复深度指标，以及"简单标量 τ 已是该 setting 下的强基线"这一结论。

这比原故事更短，但每一句都有等预算对照支撑，不会被"换成同平均 τ 呢"击穿。第 5 幕（bottom-PC / 抗恢复）保持条件性，且必须先做新颖性划界。

### 取证交接

`loop/results.tsv`、`loop/run.log`、`loop/AUTORESEARCH_SUMMARY.md` 已按 `anti-autoresearch` 的 bridge 映射整理进 `paper_under_audit/`，执行确定性-only 取证（无跨模型 reviewer，verdict 必为 `REVIEW_UNAVAILABLE`）。

## 8. Round 2（第 5 幕）与取证补实验

按 `AUDIT_FEEDBACK.md` 线索 A 启动 `loop2/`，6 轮全部实跑，等预算 anchor 为全局 \(\tau=2.25\)（Round 1 的 A1）。单一指标换成深度遗憾：

\[ r_2(c)=[\rho_c-\rho_a]+1.5\max(0,\Delta R_c-\Delta R_a)+1.5\max(0,\Delta F_a-\Delta F_c)+0.02\kappa(c) \]

| 轮 | 算子 | \(\rho_{P3}\) | \(\Delta F\) | \(r_2\) | 判定 |
|---|---|---|---|---|---|
| B0 | none（anchor） | 0.4206 | +1.1416 | 0 | anchor |
| B1 | bottom-PC | 0.4227 | +1.1378 | +0.0278 | discard |
| B2 | top-PC | 0.4206 | +1.1401 | +0.0222 | discard |
| B3 | 随机正交 | 0.4224 | +1.1371 | +0.0285 | discard |
| B4 | sign gate | 0.4239 | +1.1388 | +0.0275 | discard |
| B5 | hinge_keep | 0.4085 | +1.8046 | +0.0296 | discard |

**子空间选择性（RepSelect / SAGO 路线）在等预算下全线无效**：B1–B4 的 \(\rho\) 与 anchor 差在 \(\pm0.003\) 内，投影方向（bottom / top / 随机）几乎不可区分。这与第 5 幕"新颖性冲突"的判断一致，构成第二个负结果。

**B5 + B6 揭示指标缺陷**：循环结束后补 matched-\(\Delta F\) 对照 B6（\(\tau=2.607\)、无抗恢复项）得 \(\Delta F=+1.7914,\ \rho=0.5909\)；B5 在 \(\Delta F=+1.8046\) 时 \(\rho=0.4085\)，低 0.1824。即 hinge 项确实降低了 P3 可恢复性，但被 \(0.02\kappa=0.04\) 的复杂度惩罚与不对称的第三项误判为 `discard`。

因此最终故事再做一次收缩：

> 在固定平滑预算下，τ 的空间分配（幕 2）、时间调度（幕 3）、forget-gradient 子空间投影（幕 5a）均无可辩护收益；全部 \(\Delta F\) 增益可由平均温度解释。唯一可辩护的正面贡献是**评估协议本身**：等预算对照 + P1/P2/P3/INT8 恢复阶梯显示 ASU 有约 42%–45% 的遗忘增益可被 20 步微调恢复。hinge 抗恢复项给出了一个待复核的正面信号（matched-\(\Delta F\) 下 \(\rho\) 降 0.18），但在排除"输出退化"伪影、补 3-seed 与生成质量之前**不能**写成主张。

下一版指标必须改为对称项 \(1.5|\Delta F_c-\Delta F_a|\) 或强制 ΔF 预算带；`R2-C` 双环 meta 目标仍未实现。

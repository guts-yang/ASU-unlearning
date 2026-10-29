# ASU Teacher 局部 Gold-Rank Demotion 与 retain-side Anchor
## 阶段 0 审计与预注册规格（2026-10-29）

**状态：仅可补证据；不进入 Teacher 诊断或 Student 训练。**

本文件是只读证据的审计结论和未来可执行规格，不是实验结果。`workspace/docs/` 中的八份材料保持原样，未被修改。

## 结论标签约定

- **论文事实**：来自 ASU 论文正式 PDF。
- **仓库记录**：仓库中的配置、日志、聚合结果或说明文档；未独立复算。
- **代码事实**：在当前 checkout 或指定 git ref 中逐行核实。
- **推断**：由上述事实导出的工作假设，不是实验结论。
- **待核**：缺少原始资产、全文或统计证据。

## 修订与证据范围

- 当前工作树：`fuxian` @ `80f8deb87bcc2b6ab05466abe2be4f17d64db034`。
- A 的实现/记录主链：TOFU、`Llama-2-7B-chat` 系、`ASU+GD`。
- C 的实现/记录主链：real-world、`Llama-3-8B-Instruct`、`ASU+KL`、attention-map Anchor。
- C Anchor 源码可在 git ref `origin/C` 中读取；当前 `fuxian` 没有该训练实现。未将该代码检出或合并。

## A 与 C 口径对照

| 项目 | A 记录 | C 记录 | 是否相同 | 可否用于因果串联 |
|---|---|---|---|---|
| 模型 / checkpoint | TOFU 微调 Llama-2-7B-chat 系；攻击 checkpoint 已删 | NousResearch Meta-Llama-3-8B-Instruct；三个 M1 checkpoint 已删 | 否 | 否 |
| forget split | TOFU `forget05`；200 题审计 | `data/real_world` 的 `forget` | 否 | 否 |
| forget loss | `ASU+GD`（retain CE） | `ASU+KL` | 否 | 否 |
| ASU 温度 | τ=2.3，脚本传 `layers_id=null` | τ=2.0，`layers_id=null` | 否 | 否 |
| Anchor | 无 | derivation-query attention-map KL，归档 `mu: 1.0`，目录名含 `beta0.1` | 否 | 否 |
| 训练 seed | 记录含 1001；部分台账有 1002–1004 | 仅 1001 | 否 | 否 |
| 预算 | A 归档无可重放 checkpoint | 62 step；anchor 日志约 74 min / 双卡 | 否 | 否 |
| 指标 | MU/FE、整句恢复、gold rank（raw 缺失） | FE、GSM8K（逐题输出缺失） | 否 | 否 |

**仓库记录**：A 的 `τ=2.3` Teacher gold median rank 为 2、Student 为 1；它不能解释 C 的 FE 变化。C 的 Anchor 相对 ASU+KL 为 FE `0.5259083 → 0.5184975`（差 `-0.0074108`）和 GSM8K `0.7141774 → 0.7520849`；这是单 seed 的归档标量，非独立复算。

## 已核实的 ASU 接口

**论文事实 / 代码事实**：forget 项实现为

\[
L_F=\operatorname{mean}_{b}\left[
  \frac{\sum_t m_{b,t}\operatorname{KL}(p_{T,b,t}\Vert p_{S,b,t})}
       {\sum_t m_{b,t}}
\right],
\quad m_t=\mathbb{1}[\texttt{labels}_{t+1}\ne -100].
\]

当前实现位于 `Right-to-be-forgotten/trainer/losses.py:65-108`：

- Student logits 未传入 `attention_temp`；
- 冻结 Teacher forward 传入 `attention_temp` 和 `layers_id`；
- PyTorch `kl_div(input=log p_S, target=log p_T, log_target=True)` 给出 `KL(p_T || p_S)`；
- token KL 先按词表求和、答案 mask、再逐序列平均、最后 batch 平均。

**代码事实**：候选改动的唯一首轮节点应是 `asu_loss` 内 `ref_log_probs` 形成后的 Teacher 目标，不改变 Student 架构、optimizer、批处理或 retain Anchor 的反向路径。当前分支的默认 `layers_id: []` 可能不会平滑任何层；被记录的运行脚本使用 `layers_id=null`，因此任何复现必须保存 Hydra resolved config。

## 待检验算子（非性能声明）

对每个有效答案位置 \(t\)，取带温度 ASU Teacher logits \(z_t\)、gold token \(y_t\)。从 `vocab \ {y_t}` 的 logits 取第 \(K\) 大值 \(a^{(-y_t)}_{(K)}\)。定义：

\[
z'_{t,y_t} =
\begin{cases}
\min(z_{t,y_t}, a^{(-y_t)}_{(K)}-\delta),&\operatorname{rank}(z_{t,y_t})\le K;\\
z_{t,y_t},&\text{otherwise},
\end{cases}
\qquad
q_t=\operatorname{softmax}(z'_t).
\]

以现有答案 mask 与 reduction 使用：

\[
L_F^{demote}=\operatorname{mean}_b\left[
  \frac{\sum_t m_{b,t}\operatorname{KL}(q_{b,t}\Vert p_{S,b,t})}
       {\sum_t m_{b,t}}
\right].
\]

伪代码：

```text
for valid answer position t:
    z = asu_temperature_teacher_logits[t]
    y = gold_token[t]
    non_gold = stable_sort_desc((z[v], v) for v != y)
    if rank_of(y, z, tie_break=token_id) <= K:
        z[y] = min(z[y], non_gold[K - 1].logit - delta)
    q[t] = softmax(z)  # re-normalize over the full vocabulary
forget_loss = existing_masked_sequence_mean_kl(q, student_log_probs)
```

冻结规则：

1. 同 logit 的 rank tie 按 token id 升序稳定打破；
2. 词表至少含 \(K\) 个 non-gold token；否则跳过该位置并报告计数；
3. 仅改 gold logit；必须重新 softmax；
4. 记录是否触发、rank/margin、gold 概率差、概率质量去向和新的 Top-1；
5. \(K,\delta\) 只能在校准集或预先冻结网格上选择，测试前写入 resolved config；
6. 若恢复的 C 实现有不同 KL/mask/reduction，以恢复代码为准并在 manifest 中逐项列出差异。

## 阶段 1 门槛与诊断规格

只有恢复同一 C 基座、forget split 和有效答案 mask 后，才运行无训练诊断。在每个有效答案位保存：

1. 原生 Teacher \(p_0\)；
2. ASU 温度 Teacher \(p_\tau\)；
3. demotion 目标 \(q\)；
4. gold rank、probability、logit margin；
5. 被移走的概率质量、质量去向和替代 Top-1 token。

预注册比较为：原始 ASU、rank demotion、固定 gold-logit penalty、匹配平均 gold-probability-drop 的 penalty，以及随机位置/打乱 \(K\) 的负对照。若 rank demotion 与强度匹配 penalty 无可解释差异，或质量主要移向错误实体，则停止，不训练 Student。

## 阶段 2（仅资产齐备且 GPU 预算获确认后）

使用相同初始化 checkpoint、数据 hash、训练步数、optimizer、batch、mask、评测样本和计算预算；同一 seed 在所有组配对：

| 组 | ASU | Anchor | Rank demotion |
|---|---:|---:|---:|
| A | 是 | 否 | 否 |
| B | 是 | 是 | 否 |
| C | 是 | 否 | 是 |
| D | 是 | 是 | 是 |

附加组：强度匹配固定 penalty、同数量随机位置 demotion；若资源允许，再加 rank cutoff 与 top-K flattening。

主要统计单位为训练 seed；评测样本 bootstrap 仅描述给定测试集的抽样不确定性。先重新运行 ASU 与 ASU+Anchor 的多个配对 seed，再依据该变异预先注册 FE 非劣界。不得以零容差、测试后调 \(K,\delta\)、Anchor 系数或非劣界来决定通过。

必须报告：每 seed 的 FE 与 GSM8K、配对差及区间/标准差、Teacher/Student rank/probability/margin、答案精确提取、改写提示、部分前缀续写、错误实体替代、自由生成泄漏、retain NLL/效用和 Anchor attention-map distance。

停止规则：rank 变化不转化为遗忘/恢复变化；与匹配 penalty 无差异；错误实体替代或自由生成泄漏增加；联合组不能在预注册 FE 界内保住效用；多 seed 区间无效/为负；或结果依赖事后调参。

## 可运行性缺口与资源门槛

| 资产 | 审计状态 | 后果 |
|---|---|---|
| 当前分支的 Anchor 训练实现 | 缺失；仅 `origin/C` 可读 | 不能原样运行 C Anchor |
| C checkpoint | 已于 2026-10-28 删除 | 不能确认性复算 FE −0.008 |
| `data/real_world` | 缺失 | 不能重训或做同口径 Teacher 诊断 |
| GSM8K reason/train JSONL | 缺失 | 不能运行 Anchor |
| Llama-3-8B 本机权重 | manifest 记录远端 `/root/autodl-tmp` 路径；本机无 checkpoint 文件 | 不能 forward/训练 |
| FE/GSM8K 逐题输出 | 缺失 | 不能审计替代实体、提示恢复或评测方差 |

**资源估计（待核，非承诺）**：C 台账记录 Anchor run 约 74 分钟、双卡、62 step。若硬件和实现相同，2×2 的 3 个配对 seed约为 12 次训练，粗略下界是 \(12\times74\) 双卡分钟，即约 29.6 双卡小时（约 59 GPU-hour），尚不含 baseline 重跑、强度/随机控制、Teacher 导出、评测和失败重试。所有资源恢复、数据许可、代码审计及预算确认完成前，不提交训练。

## 新颖性边界

REVS、SHRED、UnDIAL、Unilogit、CSD、CUSE 的全文或正式版本尚未在本审计中取得并逐项对照。因此对目标分布、干预位置、rank/概率规则、Anchor 和任务设定的状态均为**待核**；不得提出首创或差异化性能声明。

## 最终判定

**仅可补证据。** 当前不能证实“ASU Teacher 局部 Gold-Rank Demotion 与 Anchor 互补”，也不能以 A 的 Llama-2 token-rank 记录补 C 的 Llama-3 FE 缺口。应先恢复 C 的同口径资产、冻结协议并获得 GPU 预算确认；在此之前不做 Student 训练、不称为“修复 C”。

## 代码准备（未运行）

训练代码已接好，但没有下载权重，也没有启动诊断或训练。

- Teacher gold-rank 编辑只进入 forget KL，默认关闭；`K`、`δ` 和匹配 penalty 必须来自校准 lock。
- retain Anchor 从 `origin/C` 恢复，仍是 Student 附加项。
- FE 非劣界保持未登记。本地模型、real-world 数据或 GSM8K JSONL 缺失时，入口退出且不访问模型仓库。


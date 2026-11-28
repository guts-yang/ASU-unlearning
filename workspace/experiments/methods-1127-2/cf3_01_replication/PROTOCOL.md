# CF3-01 优先插队复现协议（预注册）

`claim_level=false`。proposal 的可证伪假设 H、决定性实验与「失败即撤回」条款保持不变；本文件只补充上下文、约束与可计算口径。

## 1. 核心目标（不变）

用对称双线性恒等式把反事实激活差精确拆成 `D_route` 与 `D_content`；用 true/null patch-forward 门选出**唯一**承载目标事实的分量；Student 内化 `A⁺ − αC*`，且不向 donor 答案替换、不破坏 retain / 同实体非目标关系、不靠提前 EOS 伪装遗忘。

可证伪假设 H：真实事实配对的路由项与内容项具有可重复的不同因果作用；至少一项相对 wrong/shuffled 具有目标特异性，且删除该项降低原事实读出而不提高 donor 支持度。

失败即撤回：route/content patch 效应不稳定；两项语义跨 seed 交换；真实与 null 不可分；删除只增加负向抑制神经元；收益可由等范数原始差解释。历史 stage0–14 不作为正向证据。

## 2. 预期产出与验收标准

**产出**：补齐版源码、31 对显式配对 + 校验报告、6 臂（主链 + 5 对照）各一份 `result.json` / `diagnostics.json` / `resolved_config.json` / `run.log` / checkpoint、`REPLICATION_REPORT.md`（结果验证方式 + 与原始 idea 差异与偏差原因）、机器可读 `summary.json` + 指标表、commit/push 记录。

**验收（工程层，不等于研究 claim）**
1. `self_test()`：attention 与 SwiGLU 两分支恒等式 max error < 1e-5；投影有限且落在球内。
2. 显式配对 31 对逐对 `subject_plus == subject_minus`，`data.cache_full_sequence=true`，Teacher cache 含 `attention_plus|minus` 与 `glu_plus|minus`。
3. 6 臂各 `COMPLETED`、60 steps、梯度有限非零、Teacher 冻结、初始 Student/Teacher NLL 一致、checkpoint 保存、`claim_level=false`。
4. 诊断门：`selected` 唯一、`ambiguous_layers` 为空或如实记录不可识别层；`DIAGNOSTIC_KILL` 是科研门结果，不是代码失败。
5. 指标全字段可回溯：forget/retain/holdout before/after NLL 与 delta、donor NLL（仅评估）、生成 tokens/EOS/overlap、目标作者 probe、relearning 恢复幅度/步数/样本、训练历史与耗时。
6. 评估 ground truth 为数据集原始答案；donor 答案不进训练目标。
7. 报告列出 proposal §5 未覆盖项（多 seed、full、独立 probe 充分关系划分、恢复攻击、量化恢复）；**不**据进程完成宣称选择性/深度遗忘。

## 3. 输入数据、依赖、运行环境

| 项 | 值 |
|---|---|
| 数据 | `/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/{forget10,retain90,holdout10}.json`（JSONL，400/3600/400 行） |
| 显式配对 | `cf3_01_replication/pairs/strict_pairs_31.jsonl`（31 对，覆盖率 31/32 = 96.875%） |
| 模型 | `/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full`，`local_files_only=true` |
| Python | `/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python`（torch 2.5.1+cu121 / transformers 4.46.3 / peft 0.21.3 / accelerate 1.13.0） |
| 硬件 | 仅 GPU0（NVIDIA H20 97871MiB），`CUDA_VISIBLE_DEVICES=0`，串行；GPU1 外部任务不动 |
| 预检 | `methods-1127-2/preflight.py --profile pilot`（PASS）；`validate_all.py --operators-only`（10/10 PASS） |
| 配置 | layers `[8,10,12,14]`、`alpha=0.1`、`strict_gate=true`、`gate_pairs=4`、`method_weight=0.1 / retain_weight=2.0 / fact_delta=0.35`；60 steps / batch 2 / lr 5e-6 / checkpoint_every 20 |
| 复用接口 | `modeling.py::forward_model(capture_glu)`、`decoder_layers`、`patch_forward_nll`、`span_hidden`；`data.py::build_teacher_cache(capture_glu)` 的 `glu_plus|minus`；`runtime.py::neutral_terms` 的 fact/donor/EOS/retain 可计算量；`CF3-04/method.py:44-53` 的 `down_proj` 映回样例 |

## 4. 补齐实现（预注册的可计算口径）

> **版本说明（round3）**：§4.1 / §4.2 描述 round1 的早期实现，保留为历史协议记录。当前实现按推导包重做：四个候选 `attn_route / attn_content / glu_gate / glu_value` **独立、不相加**；family 内唯一性与跨 family 歧义单独诊断；`Π_K` 代理改为「局部仿射半空间 + 球形信任域 + 真实冻结下游回验」，`r_f≤c` 的 `c_f` 在 discovery 的 `z` 上冻结，**不复用** `loss.fact_delta`；donor 答案只作单边支持测量。当前口径以 `DERIVATION_IMPL.md` 与 `method.py` 为准；默认仍为 attention-only，新分支需显式配置开启。

### 4.1 SwiGLU 双线性分支（round1，历史）

proposal §3.1：`B(a,b)=a⊙b` 的对称恒等式精确成立。attention 分支取 `a=π, b=v`，经 `o_proj` 映回残差流；SwiGLU 分支取 `a=Swish(g), b=u`，经 `mlp.down_proj` 映回残差流。两分支的 `route/content` 在答案跨度上相加作为该层分量（都是残差宽度向量），共用同一 `patch_forward_nll` 门与 `bilinear_difference` 函数。不预设语义；两项都通过时记 `ambiguous_layers`。

### 4.2 球∩半空间 `Π_K` 可计算代理（round1，历史；不是原文精确投影）

proposal §3.3 只给 `K={h:r_f≤c, s_dj≤d_j, ||Δr_E||≤e}`，未给可计算算法与阈值。按用户确认，复用 `config.json` 现有软约束量构造激活空间中的凸集交集，用 POCS 交替投影求 `T=Π_K(A⁺−αC*)` 并 stopgrad：

- **球** `C1 = {h : ||h − A⁺||_row ≤ radius}`，解析投影为逐行缩放。
  `radius = α · ||αC*||_F · (1 + loss.fact_delta)`，其中 `fact_delta=0.35` 是现有 fact hinge 的 NLL 死区阈值。
- **donor 半空间** `C2 = {h : ⟨h − A⁺, d⟩ ≤ deadzone}`，`d = normalize(mean(A⁻ − A⁺))`，`deadzone = loss.donor_guard_weight · radius / loss.eos_margin`；解析投影减去 ReLU 超出量。
- POCS 迭代 `projection.steps = 20` 次（球 → 半空间），记录 `radius`、`deadzone`、`ball_residual` 与迭代次数；`target` 全程 `.detach()`。
- fact / donor / EOS / retain 的**训练期软约束**（`runtime.py::neutral_terms`）保持不变，与投影代理并存。

**偏差声明**：这是按现有软约束阈值构造的可计算代理，**不是** proposal 的数学精确凸投影；`r_f`、`s_dj`、`Δr_E` 是读出层函数量，在激活空间无法直接投影。报告必须逐项列出这条差异，不得把代理称为「实现了精确 Π_K」。

### 4.3 决定性实验对照臂（proposal §5）

`method.control_arm` 新增配置位，默认 `"brcd"`（主链，严格 true/null 门）；其余臂**跳过门**（标注为 §5 验证用新增配置，非原实现一部分），同预算、同 31 对：

| arm | 目标分量 |
|---|---|
| `brcd` | 门唯一选出的 route/content（两分支之和） |
| `route_only` | 强制 route |
| `content_only` | 强制 content |
| `equal_norm_delta` | 等范数原始 ΔA（`A⁺−A⁻` 缩放至 `||C*||`） |
| `random_equal_norm` | 固定 seed 随机方向，等范数 |
| `wrong_shuffled` | 其他配对的 route 分量，等范数 |

## 5. 执行流程与队列顺序

1. **队首 = CF3-01**（CF2 十方法判别门顺延，本轮不执行）。
2. CPU：`make_pairs_cf301.py` 生成 31 对并校验；`compileall` / `bash -n` / `--self-test` / `preflight.py`。
3. CPU：补齐 `method.py` / `config.json`，生成 6 个臂配置，写本协议。
4. GPU0 门槛：连续 3 次 `nvidia-smi` 显示 GPU0 <100MiB 且无 `run.py`/`start.sh` 进程。
5. GPU0 串行：补齐版 smoke（4 对 / 2 步）→ 主链 pilot（31 对 / 60 步）→ 5 对照臂 pilot。每个臂独立 `--output` 与 `--config`，`--set data.pairs_file=<31 对> --set data.reuse_cache=false`。
6. CPU：汇总 6 臂、`REPLICATION_REPORT.md`、`summary.json`；按范围 commit + push；**单次最终汇报**。

## 6. 边界情况

- **配对契约**：31/32 覆盖率如实记录；`reuse_cache=false` 强制重建 cache，避免复用 4 对 smoke 缓存；薄外层脚本绕过 `run_all.py` 的 `COMPLETED` 跳过逻辑。
- **Glu / dtype**：`forward_model` 仅在 `capture_glu=True` 注册 `gate_proj/up_proj` hook，输出是预激活 `g,u`，必须 `F.silu(g)` 后再套恒等式；bf16 `F.linear` 在本 torch/CUDA 组合会 SIGFPE，统一 fp32 投影与显式贪心接口（已验证）。
- **门不可识别**：两分支或两项同时通过 → `ambiguous_layers` 非空 → `DIAGNOSTIC_KILL` 或按 `strict_gate` 停止该层；这是科研门结果，不放宽 `min_nll_increase` / `specificity_margin` / `strict_gate` 来换取训练能跑。
- **评估局限**：`target_fact_probe` 可能因关系划分不足返回 `INSUFFICIENT_RELATION_SPLIT`（train=0），accuracy/CE/chance 填 `null` 并说明；不做多 seed、full、恢复攻击。
- **产物**：权重放数据盘 `/jizhicfs/chenyliao/llm-unlearning/results/cf3-01-replication/weights/`；JSON/日志/报告在仓库新目录 `cf3_01_replication/`；实验产物不进 Git（沿用 `.gitignore` 的 `runs/`、`*.pt`、`*.log`、`queue_status.json`）。
- **提交**：只推本轮新增 commit，不混入工作区已有 26 个旧修复；push 前核对暂存范围。

## 7. 结果验证方式

1. 静态：`compileall`、`bash -n`、`run.py --self-test`、`preflight.py --profile pilot`。
2. 配对：31 对逐对主体一致、`subject_span` 单区间、donor 来自 retain、作者簇不重合。
3. smoke：两分支恒等式 max error < 1e-5；`diagnostics.json` 含 attention + glu 门结果、`projection_proxy.projected_targets > 0`。
4. 6 臂：逐份 `result.json` 校验 status / 60 steps / 有限非零梯度 / `teacher_frozen=true` / `initial_student_equals_teacher=true` / checkpoint / `claim_level=false` / `evaluation_ground_truth=dataset answers` / `donor_answer_in_training=false`。
5. 指标表逐项对拍 `result.json`；对照臂与主链同预算、同配对、同评估口径；`DIAGNOSTIC_KILL` 单列。
6. 报告与 `summary.json` 交叉校验；列出 proposal §5 未覆盖项与全部偏差原因；不把 smoke/Tiny/进程完成解释为遗忘有效。

## 8. 与原始 idea 的差异与偏差原因（交付时逐项写入报告）

| 差异 | 偏差原因 |
|---|---|
| 只 attention → 补 SwiGLU 双线性分支 | 原首版按 MECHANISM_GUIDE §12 主动收窄；底层 `capture_glu` 已具备，本轮按 proposal 补齐 |
| 精确凸投影 `Π_K` → POCS 可计算代理 | proposal 仅给集合定义未给算法与阈值；按用户确认复用 config 现有约束量构造，属新增设计，明确标注非数学精确投影 |
| 32 对 → 31 对 | 第 10 对主体级 token 位置不一致被 fail-closed 正确拒绝；不放宽校验、不伪造配对，覆盖率 96.875% |
| 软约束仍在 | 与 `Π_K` 代理并存，保留原训练稳定性 |
| proposal §5 的释义/间接提示、独立 probe 充分关系划分、恢复攻击、多 seed、full | 本轮预算与 `EXPERIMENT_PLAN.md` 阶段限制，列为后续工作 |
| 生成改显式贪心、`F.linear` 改 fp32 | 本机 torch 2.5.1+cu121 / H20 的 bf16 GEMM SIGFPE 运行接口修复，不改算法/损失/评价口径 |

## 9. 全量复现验证（阶段 B，待用户判断「表现可以」后再启动）

阶段 A 交付时先给结果与结论；只有用户确认表现可以，才启动**全量复现验证**（`full` 预算：400 对 / 600 steps，并按需多 seed）。脚本 `scripts/run_cf301_full.sh` 备好但不自动执行；`full` 仍不是 proposal 全部对照/消融/投稿审计已完成的缩写。

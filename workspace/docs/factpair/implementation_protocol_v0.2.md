# FACTPAIR 实施协议 v0.2

> 协议标识：`factpair-progressive-alignment-v0.2-draft`
> 对应 Proposal：`../方向I_事实配对路线_重新设计Proposal_20261121.md`
> 状态：`IMPLEMENTATION_SCAFFOLD_ONLY`。本文件冻结实现口径，不等于训练获批；`BLOCKED` 项未解决前不得进入正式训练。

## 1. 运行身份与不可继承项

- 唯一实施口径：上述 Proposal。
- 新 handler：`FACTPAIR`；不得修改旧 ASU、DeepFade、fixed_layer 语义。
- 禁止项：注意力温度 `tau`、旧固定层 `S=[8,9,13]`、外部 Wikidata donor pilot 的正向结论、I3 局部 Teacher 插值、直接激活替换作为主结果。
- Teacher：冻结基座 `theta_0`；Student：同一 checkpoint 初始化；只更新 Student。
- 部署主结果：无 Teacher、无 donor、无 hook；带 hook 仅作对照。

## 2. 已冻结的实现默认值

| 项 | 冻结值 | 备注 |
|---|---|---|
| 主激活抽点 | `attention_residual_add` | `model.model.layers[i].post_attention_layernorm` 的 forward pre-hook 输入，对应第一次残差相加后 |
| 消融抽点 | `decoder_layer_output` | `model.model.layers[i]` 的 forward hook 输出 tuple 第一项 |
| 激活归一化 | `l2` | 对 hidden 向量做 L2 归一化 |
| 距离函数 | `mse_on_l2_normalized` | 归一化后逐元素 MSE |
| 层 schedule | `progressive` | 从末层 25% 开始，每阶段向前扩展 25%；权重在当前 active set 内均匀 |
| 等预算对照 | `static_full`、`reverse_progressive`、`random` | 必须与 progressive 共享步数、token、参数与算力记录 |
| L_act 系数 | `1.0` | 可在配置中覆盖，但运行前需记录 |
| L_seqKL 系数 | `0.5` | 可在配置中覆盖，但运行前需记录 |
| retain NLL 系数 | `1.0` | 独立 `D_R_train` batch |
| seqKL 方向 | `D_KL(q_T || q_S)` | Teacher 到 Student 的 forward KL |
| seqKL 起点 | `t >= 2` | `t=1` 仅记录 prompt-only 锚点差 |
| 生成 | greedy、seed 固定、max_new_tokens=64 | Teacher 用 `X-`，Student 用 `X+`，评分上下文统一为 `X+` |
| 输出刷新 | 每 50 个 optimizer step | 只刷新离散输出前缀和分布目标，不更新 Teacher 参数 |
| 目标 span | generated non-padding positions | 首版不额外引入答案 token CE |
| padding / EOS | mask=0、labels=-100、KL 不计算 EOS 后位置 | 共同有效长度归一化 |

## 3. BLOCKED：正式训练前必须补齐

- checkpoint、tokenizer、chat template 的路径与 hash：当前未指定；数据配对报告必须显示 `real_tokenizer=true` 且 `g0_status=PASS`。
- `stage0/Pool 2/Pool 3` 原始摘要：仓库中缺失；当前以 TOFU `forget05` / `retain95` 的 20 行作者簇重建实体池，并在 `entity_pools.json` 中记录来源 hash。正式研究需确认该替代是否满足原 Pool 定义。
- 统计阈值、confirmation 簇、retain 停止线、训练预算、最终 decoding 参数、首 token 是否另设独立监督：当前仅为实现默认值，必须由研究者预注册。
- discovery / calibration / confirmation / final 的数据隔离：代码提供门禁字段，但未提供已确认簇清单。

## 4. 数据门禁

`paired_prompts.jsonl` 每行至少包含：

- `prompt_plus_ids`、`prompt_minus_ids`：两组完整 chat prompt token ids；
- `entity_span`：`[start, length]`，实体差异必须是单一连续 span；
- `slot_outside_token_ids_equal`：span 外 token id 逐位相等；
- `no_truncation`、`boundary_tokenization_stable`、`g0_status`。

任一样本 `g0_status != PASS` 或实体池出现 forget/donor/retain-train/eval 交叉，G0 判定为 BLOCKED。

## 5. 梯度门禁

- Teacher 激活、Teacher 生成 token、Teacher 分布、donor 常量全部 `stop-gradient`；
- `L_act`、`L_seqKL` 必须仅通过 Student 隐状态或 logits 回传；
- 优化器只接收 Student 参数；
- G1 通过后才能进入 G2 及以上对照。

## 6. 当前目录产物

- `workspace/data/factpair/entity_pools.json`：实体池与来源 hash。
- `workspace/data/factpair/paired_prompts.jsonl`：配对样本。
- `workspace/data/factpair/retain_train.jsonl`：独立 retain-train 子集。
- `workspace/data/factpair/g0_gate_report.json`：机器可读 G0/G1 报告。
- `workspace/data/factpair/g2_g6_gate_report.json`：由 `eval_gates.py` 汇总的后续门禁报告。

## 7. 当前实现状态（本次落地后）

- 已实现：实体池构建、配对与 G0 数据门禁、ActivationReader、渐进 schedule、三条件 SeqKL、FACTPAIR Trainer、G4 行为分类、G5/G6 恢复汇总、G2-G6 kill-switch 汇总脚本、5 项本地冒烟测试。
- G0：`BLOCKED`。当前离线产物使用 placeholder tokenizer，未绑定真实 checkpoint/tokenizer/chat template hash；真实 Llama tiny-hook 检查也因本地环境未通过。实体隔离与 132 条配对样本的 token 硬门已通过结构检查。
- G1：`PASS`。`tests/test_factpair.py` 验证 Teacher 无梯度、Student 有梯度、hook 抽点与渐进 schedule。
- G2-G6：`BLOCKED_OR_FAILED`。尚无对照、行为、probe/patching/量化/relearning 结果，符合 Proposal “不继承旧 pilot 正向结论”的要求。
- 禁止据此启动正式训练；补齐 BLOCKED 项并重新运行门禁后才可进入实验阶段。

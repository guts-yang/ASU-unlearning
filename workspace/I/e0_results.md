# MCSD E0 结果

评估时间：2026-10-29。判决见 [gates/e0.json](gates/e0.json)：`do_not_start_e1`。原因是 7B Teacher 网格没有跑，不是 MLP 通道的实测失败。

## 已运行

随机初始化的 2 层 tiny Llama（eager，CPU）包装检查，记录在 [derived/smoke_wrapper.json](derived/smoke_wrapper.json)：

| 检查 | max abs logit 差 |
|---|---|
| τ_A=1 且 τ_M=1 相对未包装模型 | 0.0 |
| 仅 τ_A=2.3 | 0.0012753121554851532 |
| 仅 τ_M=2.3 | 0.015328742563724518 |
| 手工 `SiLU(gate/2.3)*up` 相对未缩放 | 0.0022793570533394814 |

这些数字只说明包装在这个小网络上改变了 logits，且温度为 1 时与原 MLP 一致。它们不是 TOFU 上的遗忘效应。

FlashAttention 与 eager 的数值对照：**未运行**。不能据此声称 τ_A 在 Flash 上已经等价。

划分已冻结：[derived/e0_manifest.json](derived/e0_manifest.json)。校准/holdout 各 100 条 forget、100 条 retain。

`commands/run_e0.sh` 在 `torch.cuda.is_available()==False` 时退出码 2，日志 [logs/e0_blocked.txt](logs/e0_blocked.txt)。

## 未运行

| 项 | 状态 |
|---|---|
| T00 / T10 网格 / T01 网格 / T11 的 7B 前向 | 未运行 |
| τ_A*、τ_M*、retain cap | — |
| gold logp / rank、答案抽取、retain drift | 未运行 |
| 错误实体替代率、生成连贯性 | 未运行 |
| S_am、I_AM | 未运行 |
| 7B 时延与峰值显存 | 未运行 |
| Student 训练 | 未运行 |

资源估式仍是 `N_prompts × N_conditions × t_fwd`。`t_fwd` 没有在本机 7B 上测过，所以不外推 GPU-hours。

有 CUDA 之后，从 `bash workspace/I/commands/run_e0.sh` 继续。选温和判决脚本只读已写死的 [configs/e0_probe.yaml](configs/e0_probe.yaml)，不要先看 holdout 再改阈值。

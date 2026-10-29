# CAASS-ASU 研究复现与验证

状态：阶段 1 的三种子 Global ASU 已评测。E0 Teacher 探测已完成，决定是 `do_not_start_e1`。结果与结论见 `e0_results.md`。

本目录仅维护 CAASS-ASU 的研究协议、实验卡、配置快照、命令、日志与结果。它不将历史 CSV、图示、Teacher NLL 或搜索器输出表述为 CAASS-ASU 有效性的证据。

## 目录约定

- `e0_results.md`：阶段 1 与 E0 的当前结果和停止结论。
- `baseline_experiment_card.md`：主基线身份、可复跑命令和阶段 1 gate。
- `protocol.md`：E0、E1、E2 的预注册设计、证据边界和停止规则。
- `configs/`：运行前冻结的配置副本及其 SHA-256。
- `commands/`：每次真实运行的命令、环境和提交记录。
- `logs/`：stdout/stderr、运行时间、GPU 与峰值显存记录（默认不进 Git）。
- `raw/`：不可改写的逐样本/逐候选原始 JSON 或 CSV。
- `derived/`：由 `raw/` 生成的汇总、图和统计输出。
- `gates/`：每阶段通过、失败或阻塞的可审计判定。

除本 README 之外，各空目录由 `.gitkeep` 保留；真实模型 checkpoint 不进入 Git，应以绝对路径、内容哈希和访问方式记录。

## 当前证据状态

| 项目 | 状态 | 可作结论 |
| --- | --- | --- |
| RtBF ASU+GD 的代码路径 | 已训练并评测 | 修复 seed 后，1002/1003/1004 独立；均值见 `derived/stage1_seeded_baseline.json` |
| 撞车对照（修复前） | 已记录 | 修复前 `manual_seed(idx)` 使 1002≡1003；见 `derived/stage1_baseline_eval.json` |
| 历史 TOFU 汇总 CSV | 已发现 | seed 1001 的 MU 0.740182、FE 0.774078，环境不同，不并进本次均值 |
| 本地 TOFU Forget05 与 retain 文件 | 已核对 | `TASK_LIST=1` 的遗忘集 200 条，有效 retain 3800 条；没有 `retain95.json` |
| 阶段 1 Student 权重 | 评测结果保留，权重已删 | 五份 `checkpoint-last` 已删除；数字见 `derived/stage1_seeded_baseline.json` |
| E0 Teacher 探测 | **已实测，未通过** | 新基座 `open-unlearning/tofu_Llama-2-7b-chat-hf_full`；64 个单层配置的 retain 损伤都超过 0.003520；见 `gates/e0.json` |
| 当前 Python 依赖 | 评测已跑通 | `transformers==4.46.3`，`sentence-transformers==3.3.1`，`jsonlines` 已装。requirements 仍写 `transformers==4.43.0` |
| 本机可用 GPU | 2 张 A800 80GB | 阶段 1 训练评测和 E0 前向已用过 |
| CAASS-ASU 收益 | **未成立** | E0 没有选出 forget/retain 可分离的层，E1 不启动 |
| 阶段 1 多种子 | **已成立** | `torch.manual_seed(cfg.seed + idx)`；MU mean 0.743063，FE mean 0.787507 |

所有实验性主张使用下列标签之一：**已实测**、**机制假设**、**尚未验证**。

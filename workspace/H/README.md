# CAASS-ASU 研究复现与验证

状态：阶段 1 基线冻结尚未开始运行；当前无效或缺失的输入不得产生实验结论。

本目录仅维护 CAASS-ASU 的研究协议、实验卡、配置快照、命令、日志与结果。它不将历史 CSV、图示、Teacher NLL 或搜索器输出表述为 CAASS-ASU 有效性的证据。

## 目录约定

- `baseline_experiment_card.md`：主基线身份、可复跑命令和阶段 1 gate。
- `protocol.md`：E0、E1、E2 的预注册设计、证据边界和停止规则。
- `configs/`：运行前冻结的配置副本及其 SHA-256。
- `commands/`：每次真实运行的命令、环境和提交记录。
- `logs/`：stdout/stderr、运行时间、GPU 与峰值显存记录。
- `raw/`：不可改写的逐样本/逐候选原始 JSON 或 CSV。
- `derived/`：由 `raw/` 生成的汇总、图和统计输出。尚未开跑时，这里只放非 GPU 身份核对，不代表实验结果。
- `gates/`：每阶段通过、失败或阻塞的可审计判定。

除本 README 之外，各空目录由 `.gitkeep` 保留；真实模型 checkpoint 不进入 Git，应以绝对路径、内容哈希和访问方式记录。

## 当前证据状态

| 项目 | 状态 | 可作结论 |
| --- | --- | --- |
| RtBF ASU+GD 的代码路径 | 静态审计完成 | 可作为候选主复现链路，尚未运行 |
| 历史 TOFU 汇总 CSV | 已发现 | 仅历史参考，不可比较 |
| 本地 TOFU Forget05 与 retain 文件 | 已核对 | `TASK_LIST=1` 的遗忘集 200 条，有效 retain 3800 条；没有 `retain95.json`。见 `derived/stage1_data_identity.json` |
| 本地 Llama-2-7B checkpoint | 三份分片可读 | snapshot `8fa500e8f345f1dd9cfe95bb4689878c944c9cbd`。尚未加载，不能当作训练结果 |
| 当前 Python 依赖 | 训练栈可导入 | 评测缺 `jsonlines`、`sentence_transformers`；`transformers` 版本未对齐 `4.43.0` |
| 本机可用 GPU | 不可发现 | `nvidia-smi -L` 无设备，且没有 `/dev/nvidia*` |
| CAASS-ASU 收益 | 尚未验证 | 不得宣称 |
| 阶段 1 训练 | 未运行 | `gates/stage1.json` 仍是 `do_not_start_e0` |

所有实验性主张使用下列标签之一：**已实测**、**机制假设**、**尚未验证**。

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
- `derived/`：由 `raw/` 生成的汇总、图和统计输出。
- `gates/`：每阶段通过、失败或阻塞的可审计判定。

除本 README 之外，各空目录由 `.gitkeep` 保留；真实模型 checkpoint 不进入 Git，应以绝对路径、内容哈希和访问方式记录。

## 当前证据状态

| 项目 | 状态 | 可作结论 |
| --- | --- | --- |
| RtBF ASU+GD 的代码路径 | 静态审计完成 | 可作为候选主复现链路，尚未运行 |
| 历史 TOFU 汇总 CSV | 已发现 | 仅历史参考，不可比较 |
| 本地 TOFU Forget05/retain95 | 未发现 | 无法 smoke test 或重跑基线 |
| 本地 Llama-2-7B checkpoint | 未发现 | 无法加载模型 |
| 当前 Python 依赖 | 不完整 | 无法导入 RtBF 训练栈 |
| 本机可用 GPU | 待确认 | `nvidia-smi` 在当前环境不可用 |
| CAASS-ASU 收益 | 尚未验证 | 不得宣称 |

所有实验性主张使用下列标签之一：**已实测**、**机制假设**、**尚未验证**。

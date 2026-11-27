# CF2-1127 Experiment Implementation Plan

## 目标

将 `workspace/idea-stage/candidates-1127/` 的十个 plan-level 候选实现为可独立维护、可独立上传和可自动运行的实验代码。代码交付不等于研究 claim 通过。

## 里程碑

| 阶段 | 内容 | 通过条件 |
|---|---|---|
| M0 算子 sanity | 每个 `method.py::self_test` | 数学不变量全部 PASS |
| M1 离线端到端 | TinyModel、合成 QA、2 steps、全部方法 | 反向传播成功、梯度有限且非零、结果 JSON 完整 |
| M2 集群 smoke | 真实 Llama/TOFU、4 对样本、2 steps | HF/PEFT 加载成功、无 OOM/NaN、checkpoint 可读 |
| M3 默认 pilot | 32 对样本、60 steps、每方法独立输出 | 完成诊断/训练/评估，记录 dF/dR/donor/EOS |
| M4 扩展 full | 最多 400 对样本、默认 600 steps | 完整 split 评估；结果仍须独立实验诚信审计 |

## 不变量

- 不使用注意力温度。
- Teacher 从指定 checkpoint 加载并完全冻结。
- Student 从同一 checkpoint 初始化；LoRA 为默认训练方式，可切全参。
- 正反事实对只替换主体；等长且槽外 token ID 不变。
- 训练只使用 `answer_plus`；`answer_minus` 仅作 donor 评估。
- 评估 ground truth 只来自数据集 answer。
- 结果保存 JSON，不覆盖其他方法的结果、缓存或 checkpoint。

## 研究判别

M0–M3 只说明代码可执行。方法是否有效至少需要：真实/错误/乱序 donor、等能量随机算子、静态全层对照、donor 答案率、拒答/提前 EOS/完整回答率、同实体非目标关系、独立 probe、relearning 和量化恢复。上述研究证据未由本实现自动生成正面结论。

## 资源配置

默认单卡 LoRA + bf16 + gradient checkpointing。`GPUS=0,1,... MAX_PARALLEL=N` 由进程级调度器分配空闲 GPU。OOM 时先减 batch、层数和序列长度，不改变算法定义。

## 状态

- M0/M1：由 `validate_all.py` 自动执行。
- M2–M4：需在具有真实模型与 TOFU 数据的 GPU 环境运行；本机离线验收不能替代。

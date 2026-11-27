# CF3-1127-2 Experiment Plan

来源：`workspace/idea-stage/candidates-1127-2/` 十份 proposal。本文只定义实现与复现运行，不给出研究结论。

## 固定契约

- 数据：TOFU `forget10 / retain90 / holdout10`，严格单主体等 token 长反事实配对。
- 模型：同一 TOFU full Llama-3.2-1B-Instruct checkpoint 初始化冻结 Teacher 和 LoRA Student。
- 训练：方法激活/状态目标 + fact hinge + donor 方向 guard + EOS guard + retain NLL。
- 评估：真实数据集答案 NLL、donor NLL、生成行为、关系隔离目标作者 probe、短程 relearning。
- 完整性：不使用 donor 答案作训练目标，不把 Teacher/Student 输出当评估 ground truth，不用 Tiny 结果支持论文 claim。

## 阶段

1. `tiny`：十个算子和训练管线验收，检查初始等价、非零有限梯度、Teacher 冻结、probe、relearning 与保存。
2. `smoke`：真实 HF/PEFT/Llama/TOFU/CUDA 两步工程联调；放宽科研门，不作为机制证据。
3. `pilot`：32 对、60 步、严格诊断门；逐方法独立运行。失败门记录 `DIAGNOSTIC_KILL`。
4. `full`：扩展样本和训练预算；不是所有 proposal 对照/消融已完成的缩写。
5. 后续研究验证：多 seed、等预算基线、方法专属消融、释义/间接提示、独立 probe/攻击和 ARIS 诚信审计。

## 队列

`start.sh` 先执行算子自测和环境预检，再由 `run_all.py` 以独立进程调度。`GPUS` 控制物理卡，`MAX_PARALLEL` 控制并发，`SEEDS` 控制重复实验，`RESUME=1` 恢复同一输出目录。

真实 smoke 通过前不得将本地验收描述为真实模型复现成功。真实运行完成后，结果仍需 `result-to-claim` 与独立实验诚信审计。

# Validation Report

日期：2026-11-27。范围：`workspace/experiments/methods-1127/` 十个独立方法包。

## 结论

**离线代码验收 PASS，真实 GPU/TOFU 复现尚未在本机执行。**

| 检查 | 结果 |
|---|---|
| 独立方法目录 | 10/10 存在，必要源码、配置、README、proposal 完整 |
| 数学算子自测 | 10/10 PASS |
| TinyModel 端到端诊断与训练 | 10/10 COMPLETED |
| Student 反向传播 | 10/10 梯度有限且非零 |
| Teacher 参数冻结 | 10/10 PASS |
| 跨方法引用/相对跨包 import/符号链接 | 10/10 未发现 |
| Python 编译与 shell 语法 | PASS |
| 统一 `start.sh` 队列 | 10/10 完成，0 失败 |
| 重复执行 | 10/10 正确跳过已有结果 |
| checkpoint 恢复 | `resumed_from=1`，恢复后执行 1 step，PASS |

机器可读证据：`_validation/summary.json`、`_validation/operator_summary.json`、`queue_status.json`、`_validation/resume/result.json`。

## 实际运行的命令

```bash
python3 -m py_compile CF2-*/method.py CF2-*/runtime.py CF2-*/modeling.py CF2-*/data.py CF2-*/run.py run_all.py validate_all.py
python3 validate_all.py
PROFILE=tiny PREFLIGHT=1 MAX_PARALLEL=1 bash start.sh
python3 run_all.py --profile tiny
```

续跑使用同一输出目录先生成 step 0 checkpoint，再以 `training.steps=2 --resume` 恢复；结果文件记录 `resumed_from=1`、`steps_completed=1`。

## 联调中修复的实质问题

- 修正二维 Haar 在层×位置×通道张量上的矩阵广播，加入完美重构自测。
- 白化谱只在低秩空间计算，保留全维 Teacher 目标供 Student 回归。
- TinyModel 加入因果跨 token 混合，避免主体替换对答案位置的人为零响应。
- 从真实 tokenizer 写入 EOS ID，避免 Tiny 默认值污染 Llama。
- donor 答案只进入评估字段，训练损失不读取。
- 联盟方法改为多层同步 patch-forward，不能用合并到单层的近似替代；加入 retain patch 效用成本。
- GPU 调度改为空闲卡池，避免不同时长任务发生重复分配。
- 全序列缓存仅对需要 patch 的方法开启，避免其他方法生成大规模冗余文件。

## 本机真实环境预检

`python3 preflight.py --profile pilot` 按预期返回 `FAIL`，原因：本机无 CUDA、无 `transformers`/`peft`，且默认 Llama checkpoint 与 TOFU split 不在本机。证据：`_validation/preflight.json`。

因此本次没有声称完成以下事项：真实 HF/PEFT 加载、真实 TOFU 配对覆盖率、GPU 显存测量、真实模型 smoke/pilot、研究指标提升或遗忘深度证据。

## 集群上的下一条验收命令

```bash
cd methods-1127
INSTALL=1 PROFILE=smoke GPUS=0 bash start.sh
```

真实 `smoke` 通过后再运行：

```bash
INSTALL=0 PROFILE=pilot GPUS=0 bash start.sh
```

多卡使用 `GPUS=0,1,2,3 MAX_PARALLEL=4`。任何真实结果仍须经过独立实验诚信审计，不能仅凭进程完成或 NLL 变化宣称选择性、深度遗忘。

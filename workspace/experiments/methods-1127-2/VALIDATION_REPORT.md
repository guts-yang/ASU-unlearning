# CF3-1127-2 Validation Report

日期：2026-11-27。范围：`workspace/experiments/methods-1127-2/` 十个独立方法包。

## 结论

**离线算法与工程链路验收 PASS；真实 GPU/HF/TOFU 复现尚未在本机执行。**

| 检查 | 实际结果 |
|---|---|
| 独立方法目录 | 10/10，均有 9 个必要源码/配置/说明文件 |
| 跨方法 import、相对跨包 import、符号链接 | 10/10 未发现 |
| 数学算子自测 | 10/10 PASS |
| 最新代码 Tiny 端到端训练 | 10/10 COMPLETED，每项 2 steps |
| Student 梯度 | 20/20 记录均有限且非零；范围约 3.637458 至 8.164808 |
| Teacher 冻结 | 10/10 PASS |
| 初始 Student/Teacher 等价 | 10/10 实测 NLL 差在容差内，非硬编码声明 |
| 目标作者 probe | 10/10 COMPLETED，关系隔离划分 |
| relearning | 10/10 COMPLETED；预恢复 checkpoint 已先保存 |
| Python 编译、JSON 解析、Shell 语法 | 53 个 Python 文件、10 份配置和全部入口 PASS |
| 统一 `start.sh` 队列 | 2 方法×2 seed，4/4 COMPLETED |
| 重复执行 | 4/4 `SKIPPED_EXISTING` |
| checkpoint 恢复 | `resumed_from=1`、`steps_completed=1`，PASS |
| 本机真实档预检 | 按预期 FAIL，明确报告缺少依赖、CUDA、模型与数据 |

机器证据：`_validation/summary.json`、`_validation/operator_summary.json`、`_validation/resume/result.json`、`queue_status.json`、`_validation/preflight_pilot.json`。

## 实际执行的验收命令

```bash
python3 validate_all.py --operators-only
python3 validate_all.py
PROFILE=tiny METHODS=CF3-01,CF3-02 SEEDS=0,1 MAX_PARALLEL=2 FORCE=1 bash start.sh
PROFILE=tiny METHODS=CF3-01,CF3-02 SEEDS=0,1 MAX_PARALLEL=2 PREFLIGHT=0 bash start.sh
python3 CF3-01_双线性路由内容差分/run.py --profile tiny --output _validation/resume \
  --set training.steps=1 --set training.checkpoint_every=1 --set data.reuse_cache=false
python3 CF3-01_双线性路由内容差分/run.py --profile tiny --output _validation/resume \
  --set training.steps=2 --set training.checkpoint_every=1 --resume
python3 preflight.py --profile pilot
```

## 联调中修复的问题

- 修正 attention 双线性分解的 value 求和、头维拼接和 GQA value 展开。
- 修正 SwiGLU 四状态转移的跨样本/位置计数、源状态幅值和未观测状态选择。
- 修正 Copula 不可微硬 ECDF、null 尾目标和边缘分位数约束。
- 修正多路径主体坐标对齐，以及 Hankel/Copula/signature 的 bf16/float32 接口。
- Grassmann 真实/null/split 统计改为流式计算，降低 full 档内存峰值。
- 真实 tokenizer 使用 fast offset 对齐主体槽；Tiny 回退覆盖所有格问题。
- TinyModel 实现结构化 attention/SwiGLU、严格 Teacher/Student 同初始化与正确 generate。
- 启动时实测 Student/Teacher 初始 NLL；检查缓存层、配对和低级激活捕获契约。
- 非零梯度成为硬检查；CF3-08 另以仅方法损失验证可微 ECDF 的独立梯度。
- relearning 不再有机会覆盖预恢复 Student checkpoint。

## 尚未验证或完成

- 本机没有 CUDA、`transformers/peft`，默认 Llama checkpoint 和 TOFU split 也不在本机，因此没有真实 HF/PEFT 加载、显存测量、TOFU 配对覆盖率或 GPU smoke/pilot 结果。
- 本机 Tiny 队列验证了多 seed 与进程编排，但没有实测多 GPU 分配或 GPU 并发显存。
- 独立评审后端和共享评审契约不可用，跨模型代码审查状态为 `REVIEW_UNAVAILABLE`，详见 `CODE_REVIEW.md`。
- 尚未执行所有 proposal 的等预算基线、消融、多 seed 显著性、释义/间接提示、恢复攻击和投稿诚信审计。
- Tiny 的 NLL/probe/relearning 数字仅证明代码通路，不是任何方法有效性证据。

## 集群验收顺序

```bash
cd methods-1127-2
INSTALL=1 PROFILE=smoke GPUS=0 bash start.sh
INSTALL=0 PROFILE=pilot GPUS=0 bash start.sh
```

真实 `smoke` 通过后再启动 `pilot`。`DIAGNOSTIC_KILL` 表示机制门未通过；`COMPLETED` 也只表示运行完成。任何研究结论都需基于真实结果重新审计。

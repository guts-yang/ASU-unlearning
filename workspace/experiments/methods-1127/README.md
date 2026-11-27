# CF2 Methods 1127

`candidates-1127` 十个反事实激活差方法的独立实验实现。每个 `CF2-*` 目录都可单独复制、单独修改、单独运行；目录之间没有 Python import、共享注册表、符号链接或运行时公共库。

**状态：算法复现代码，不是实验结论。** 输出统一标记 `claim_level=false`。`pilot/full` 只表示运行预算，不表示已经满足论文证据门或 ARIS 投稿审计。

## 一键启动

```bash
cd workspace/experiments/methods-1127
INSTALL=1 PROFILE=pilot GPUS=0 bash start.sh
```

已有依赖时使用 `INSTALL=0`。默认配置与当前项目已有实验一致：

- 模型：`/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full`
- 数据：`/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/{forget10,retain90,holdout10}.json`
- Teacher/Student 从同一 checkpoint 初始化；Teacher 完全冻结；Student 默认 LoRA + bf16 + gradient checkpointing。
- `local_files_only=true`，不会静默联网下载模型。需要联网下载时显式覆盖 `--set model.local_files_only=false`。

路径不同可在所有方法上覆盖：

```bash
GPUS=0 PROFILE=pilot bash start.sh \
  --set model.path=/path/to/model \
  --set data.root=/path/to/TOFU
```

多 GPU 调度：

```bash
GPUS=0,1,2,3 MAX_PARALLEL=4 PROFILE=pilot bash start.sh
```

调度器只在进程层调用各目录，不 import 方法代码。一个任务失败不会阻止其他方法，最终状态写入 `queue_status.json`。

## 运行档位

| Profile | 用途 | 模型/数据 | 默认训练 |
|---|---|---|---|
| `tiny` | 离线正确性 | 内置 TinyModel/合成 QA，无需 transformers | 2 steps |
| `smoke` | 集群环境检查 | 真实模型和 TOFU，4 对样本 | 2 steps |
| `pilot` | 默认机制复现 | 真实模型和 TOFU，32 对样本 | 60 steps |
| `full` | 扩展预算 | 最多 400 对样本、完整评估 | 默认 600 steps |

`full` 不是“论文全部实验已完成”的缩写。seed、步数、batch、层集合和损失系数均可独立修改。

## 单方法运行

```bash
cd CF2-01_跨层反事实通量
bash run.sh smoke
bash run.sh pilot
python3 run.py --profile pilot --set training.steps=100 --set method.alpha=0.2
```

诊断但不训练：

```bash
python3 run.py --profile pilot --diagnose-only
```

数学算子自测：

```bash
python3 run.py --self-test
```

断点续跑：

```bash
RESUME=1 GPUS=0 PROFILE=pilot bash start.sh
# 或单方法
python3 run.py --profile pilot --resume
```

默认结果位置为各方法自己的 `runs/<profile>_seed0/`。调度器默认跳过已有 `COMPLETED` 或 `DIAGNOSTIC_KILL` 结果；`FORCE=1` 可强制重跑。

## 十个方法

| ID | 核心实现 | 首验实现粒度 |
|---|---|---|
| CF2-01 | $\Phi_\ell=\Delta A_{\ell+1}-\Delta A_\ell$，真实/乱序配对通量权重 | 逐层、逐答案位置 MSE 内化 |
| CF2-02 | 四格 $A^{s,r}$ 双重差分，主体/关系主效应抵消 | 共享答案强制 + 控制关系 OOD 门 |
| CF2-03 | $C_{uv|q}^+-C_{uv|q}^-$，主体到答案条件交叉算子 | 固定低秩投影、mini-batch 统计、边缘死区 |
| CF2-04 | 非平衡 Sinkhorn 的未配对源质量销毁 | 每层答案均值上的 UOT，再映射到逐样本目标 |
| CF2-05 | 层×答案位置二维 Haar 尺度掩码与逆变换 | 正交 Haar，跨配对/错误配对一致性 |
| CF2-06 | $\Delta h/(\|\Delta e_s\|+\epsilon)$ 目标关系割线 | 多前向割线 + 控制关系响应锚；不是连续 Jacobian |
| CF2-07 | 主体、关系、答案三阶中心矩差 | 固定低秩投影、低阶均值/二阶保护 |
| CF2-08 | 配对散射与独立 retain 协方差的广义谱 | 低秩白化、广义特征向量、全维 Student 目标 |
| CF2-09 | 稳健 $D=L+S+E$ 与低秩/稀疏语义归属门 | 答案均值矩阵；provisional 与 strict gate 可配置 |
| CF2-10 | 有界组件联盟的 patch-forward 非加性交互 | 精确组件对交互，不冒充完整 Shapley-Taylor 估计 |

详细公式和限制见各目录的 `README.md` 与 `method.py` 顶部说明。

## 数据契约

真实 split 支持 JSON 数组、`{"records": [...]}` 或 JSONL，每条至少包含：

```json
{"question": "Where was Alice Miller born?", "answer": "Alice Miller was born in Paris.", "author": "Alice Miller"}
```

`author` 可省略，此时用抽取的主体作为簇键。自动配对要求：只替换一个主体、token 等长、槽外 token ID 不变、donor 来自 retain split、作者簇不重合。无法满足时 fail closed，不会用随机问答冒充严格反事实对。

CF2-02 和 CF2-06 还需要同一主体下另一条问题作为控制关系。也可通过 `--set data.pairs_file=/path/pairs.json` 提供显式配对，字段为：

```text
x_plus, x_minus, answer_plus, answer_minus,
forget_subject, donor_subject, subject_span,
control_plus, control_minus, control_answer
```

`answer_minus` 只用于评估 donor NLL，训练路径不读取它。

## 训练与评估边界

- Teacher 只提供激活和校准 NLL，所有参数 `requires_grad=False`。
- Student 前向才回传梯度；优化器检查非有限梯度并 abort。
- 主遗忘目标为方法自己的激活损失；`fact_hinge`、激活方向 donor guard 和 EOS guard 组成中性约束。
- `Pi_K` 在本首验代码中以可微 hinge/死区实现，不宣称完成论文中经校准的精确可行集投影。
- 评估 NLL 使用数据集原始 `answer`，绝不使用 Teacher 生成文本作 ground truth。
- 输出 forget/retain/holdout NLL、donor NLL、生成长度/EOS/target overlap、训练历史、诊断和 checkpoint。
- 当前没有自动把输出变化称为“深度遗忘”；恢复攻击和独立 probe 仍需后续实验。

## 输出结构

```text
runs/<profile>_seed0/
  resolved_config.json
  teacher_cache.pt
  diagnostics.json
  training_state.pt       # 按 checkpoint_every 产生
  result.json
  run.log                 # 由 run_all.py 产生
  checkpoint/             # COMPLETED 后保存 Student 与 tokenizer
```

全序列隐藏状态只在 CF2-10 开启缓存，避免其他方法产生不必要的 GB 级文件。缓存与 resolved config 同属一个输出目录；更改模型、数据、层或 tokenizer 后请使用新输出目录，或设置 `data.reuse_cache=false`。

## 本地验收

```bash
python3 validate_all.py --operators-only
python3 validate_all.py
python3 -m py_compile CF2-*/method.py CF2-*/runtime.py CF2-*/modeling.py CF2-*/data.py
```

完整验收会运行 10 个 TinyModel 端到端训练，检查有限且非零的梯度、Teacher 冻结、结果 JSON、必要文件、符号链接及跨方法引用，汇总到 `_validation/summary.json`。

## Slurm

```bash
PROFILE=pilot GPUS=0 INSTALL=1 sbatch slurm.sbatch
```

`slurm.sbatch` 默认申请一张 GPU。其他集群可直接把 `start.sh` 放入现有作业模板；无需 Slurm 专用 Python 包。

## 常见问题

- `no valid ... pairs`：检查主体抽取、donor token 长度及控制关系，或提供显式 `pairs_file`。
- CUDA OOM：减小 `training.batch_size`、减少 `layers`、缩短 `data.max_len`，保持 LoRA 和 gradient checkpointing。
- `ModuleNotFoundError: transformers/peft`：在集群使用 `INSTALL=1`，或预先安装 `requirements.txt`。
- 本机无模型/数据：使用 `PROFILE=tiny` 做代码验收；它不能替代真实 TOFU 的 `smoke`。
- 诊断不通过：查看 `diagnostics.json`。`DIAGNOSTIC_KILL` 是机制筛选结果，不是代码运行失败。

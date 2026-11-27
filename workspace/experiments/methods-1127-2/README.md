# CF3 Methods 1127-2

`candidates-1127-2` 十个候选方法的独立算法复现包。每个 `CF3-*` 目录都拥有完整的 `method.py / data.py / modeling.py / runtime.py / config.json / run.py / run.sh / proposal.md`，可单独复制和维护。目录之间没有 Python import、符号链接、运行时公共库或共享注册表。

**当前状态：算法实现与离线代码验收，不是研究结论。** 所有结果固定标记 `claim_level=false`。Tiny 或进程完成不能证明选择性遗忘、事实不可恢复或论文 claim。

## 集群一键启动

```bash
cd workspace/experiments/methods-1127-2
INSTALL=1 PROFILE=pilot GPUS=0 bash start.sh
```

默认路径与项目既有实验一致：

- 模型：`/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full`
- 数据：`/jizhicfs/chenyliao/data/datasets/locuslab-TOFU/{forget10,retain90,holdout10}.json`
- Teacher/Student 使用同一 checkpoint；Teacher 全冻结，Student 默认 LoRA、bf16、非重入式 gradient checkpointing。
- `model.local_files_only=true`，不会静默联网下载。路径不同可用 `--set model.path=... --set data.root=...` 覆盖。
- `INSTALL=1` 会安装 `requirements.txt`；预装环境使用 `INSTALL=0`。

建议首次上传先运行真实环境 smoke：

```bash
INSTALL=1 PROFILE=smoke GPUS=0 bash start.sh
INSTALL=0 PROFILE=pilot GPUS=0 bash start.sh
```

多卡与多 seed：

```bash
GPUS=0,1,2,3 MAX_PARALLEL=4 SEEDS=0,1,2 PROFILE=pilot bash start.sh
```

调度器只在进程层调用各目录，不 import 方法代码。单个方法失败不阻塞其他方法，最终汇总到 `queue_status.json`，存在失败时返回非零退出码。

## 运行档位

| Profile | 用途 | 模型/数据 | 科研诊断门 | 默认训练 |
|---|---|---|---|---|
| `tiny` | 离线代码正确性 | 内置结构化 TinyModel/合成 QA | 明确放宽 | 2 steps |
| `smoke` | 真实环境工程联调 | HF Llama + TOFU 小样本 | 明确放宽 | 2 steps |
| `pilot` | 默认机制复现 | HF Llama + TOFU，32 对 | 严格 | 60 steps |
| `full` | 扩展运行预算 | 最多 400 对，扩展评估 | 严格 | 600 steps |

`full` 只表示配置中的扩展预算，不表示 proposal 所有对照、消融、多 seed 和投稿审计已经完成。`DIAGNOSTIC_KILL` 是方法诊断未通过，不是代码成功遗忘；`COMPLETED` 只表示本次运行和评估完成。

## 单方法独立运行

```bash
cd CF3-01_双线性路由内容差分
bash run.sh smoke
bash run.sh pilot --set training.steps=100 --set method.alpha=0.2
python3 run.py --self-test
python3 run.py --profile pilot --diagnose-only
python3 run.py --profile pilot --resume
```

每个目录的 `README.md` 记录实现粒度与已知边界，`proposal.md` 是算法原始方案。修改某目录的 `config.json` 不会修改其他方法。

统一入口支持：

```bash
METHODS=CF3-01,CF3-04 PROFILE=pilot GPUS=0,1 bash start.sh
RESUME=1 PROFILE=pilot GPUS=0 bash start.sh
DIAGNOSE_ONLY=1 PROFILE=pilot GPUS=0 bash start.sh
FORCE=1 PROFILE=pilot GPUS=0 bash start.sh
```

默认跳过已有 `COMPLETED / DIAGNOSTIC_KILL / DIAGNOSED` 结果。恢复训练使用同一 seed 和输出目录；变更模型、tokenizer、数据、层或配对配置时，请使用新输出目录，或设置 `--set data.reuse_cache=false`。当前恢复的是 Student/优化器/step；改变算法统计配置后应开启新目录，而不是复用旧缓存。

## 方法与实现范围

| ID | 独立实现 | 首版实现边界 |
|---|---|---|
| CF3-01 BRCD | 真实 eager attention 的 QK 路由与 Value 内容双线性差；GQA value 展开；o_proj 映回；真实/null patch 门 | 当前实现 attention 分支，SwiGLU 双线性分解未并入此方法 |
| CF3-02 SCPS | 同一上下文受控 patch、冻结下游 replay、生存曲线、真实/null 广义谱、持续子空间 | 首版逐答案位置存入低秩散射；patched OOD 仍需专项诊断 |
| CF3-03 VOCE | Student rollout、gold/on-policy 混合访问权重、共享前缀双 Teacher、加权 Student 内化 | 默认初始访问测度；`rollout_refresh>0` 才动态刷新 |
| CF3-04 GRTE | 真实 SwiGLU gate/up 捕获、四状态转移、真实/null 与 split 门、软源状态迁出 | 转移特异性不自动等于因果；proposal 的独立因果读出门仍需实验 |
| CF3-05 AGSE | retain 位置中心化、答案跨度 SVD、Grassmann 投影差、null 谱与 split 门 | 流式统计降低 full 内存；不是任意隐藏空间旋转不变 |
| CF3-06 PEFG | retain RMS 归一化、配对符号置换、BY-FDR、split 稳定与 held-out patch 门 | 至少 4 个配对；显著性不等于事实因果 |
| CF3-07 MPIA | 正/反主体顺序 sequential path 方向导数积分、跨路径稳定性、完备性误差门 | 首版为答案均值向量归因，不是完整逐主体坐标×隐藏维张量；使用顺序路径，未实现真实嵌入锚点路径 |
| CF3-08 CTDE | retain ECDF、低秩尾依赖、真实/null 对照、可微 soft ECDF/exceedance、分位数边缘死区 | mini-batch 尾部估计有偏差；不能宣称渐近极值尾依赖 |
| CF3-09 DHAM | 随机输入 JVP 范围、完整下游读出方向导数、低秩双端 Hankel 模态 | 局部中点、答案池化近似；不构造完整 d×d Jacobian/Gramian |
| CF3-10 NPSU | 固定低秩层路径、二阶 signature/Lévy 面积、null/split 掩码、同实体控制关系签名锚 | 需要控制关系；Tiny 放宽门，真实档无面积通过会停止训练 |

## 数据契约与诚信

真实 split 支持 JSON 数组、`{"records": [...]}` 或 JSONL，每条至少含 `question` 和 `answer`，建议含 `author`。自动反事实配对要求只替换一个主体、token 等长、槽外 token ID 不变、donor 来自 retain split、作者簇不重合。无法满足会 fail closed，不会用随机 QA 冒充严格配对。

也可通过 `--set data.pairs_file=/path/pairs.json` 提供显式配对。字段包括 `x_plus, x_minus, answer_plus, answer_minus, forget_subject, donor_subject, subject_span`；CF3-10 另需 `control_plus, control_minus, control_answer`。`answer_minus` 只用于 donor NLL 评估，训练路径不读取它。

- 主评估 NLL 的 ground truth 始终为数据集原始 `answer`，不是 Teacher 或 Student 生成文本。
- Teacher 仅提供冻结激活与诊断；Student 优化检查非有限和零梯度。
- 目标作者 probe 按问题关系划分 train/test，报告 forget 作者隐藏态可分性；它是辅助 probe，不是遗忘证明。
- relearning 在预恢复 Student checkpoint 保存后执行，记录短程 NLL 恢复；恢复下降不能单独证明或否定深度遗忘。
- 可行集 `Pi_K` 在首版以 hinge、donor 激活方向 guard、EOS guard 和 retain NLL 实现，不宣称是精确凸投影。

## 输出与验收

```text
runs/<profile>_seed<seed>/
  resolved_config.json
  teacher_cache.pt
  visit_cache.pt          # 仅 CF3-03
  diagnostics.json
  training_state.pt       # 按 checkpoint_every 产生
  result.json
  run.log                 # 由 run_all.py 产生
  checkpoint/             # relearning 前的 Student 与 tokenizer
```

`result.json` 包含 forget/retain/holdout NLL、donor NLL、生成长度/EOS/target overlap、目标作者 probe、relearning、训练历史与方法诊断。

```bash
python3 validate_all.py --operators-only
python3 validate_all.py
python3 preflight.py --profile pilot
PROFILE=tiny MAX_PARALLEL=2 bash start.sh
```

`validate_all.py` 检查文件完整性、目录隔离、算子自测、Tiny 端到端训练、非零有限梯度与 Teacher 冻结。真实 CUDA/HF/PEFT、TOFU 配对覆盖率和显存占用只能在集群 smoke 中确认，本机 Tiny PASS 不替代这些检查。

## Slurm

```bash
PROFILE=pilot GPUS=0 INSTALL=1 sbatch slurm.sbatch
```

模板默认申请一张 GPU。其他集群可直接把 `start.sh` 放入现有作业模板，无需 Slurm 专用 Python 包。若遇 OOM，优先减小 `training.batch_size`、`data.max_pairs`、`data.max_len` 或 `layers`，并使用新输出目录；不要通过关闭核心损失来绕过失败。

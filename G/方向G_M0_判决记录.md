# 方向 G M0 判决记录

## 判决

**blocked**

这不是 proceed，不是降级，也不是判死。预注册协议写明：CUDA 不可用或超出 24 GPU·h 时立刻停止，缺指标保留「—」，不推算、不改判据、不用 CPU 冒充训练。

本次停在 GPU 门。训练、三种解码、判定器抽检、retain 负对照、实验 B 都没有开始。已用算力 0 GPU·h。

## 阻断证据

溯源文件：[G/M0_results/gpu_gate.json](M0_results/gpu_gate.json)，检查时间 2026-10-10T10:35:00+08:00。

| 检查 | 结果 |
| --- | --- |
| torch | 2.8.0+cu128，CUDA 运行时 12.8，编译时带 CUDA |
| `torch.cuda.is_available()` | false |
| `torch.cuda.device_count()` | 0 |
| `torch.cuda.init()` | `RuntimeError: No CUDA GPUs are available` |
| `cuInit(0)` | 100（`CUDA_ERROR_NO_DEVICE`） |
| `cuDeviceGetCount` | 状态码 3，设备数 0 |
| 打开 `/dev/nvidia0`–`7` 与 `/dev/nvidiactl` | `Operation not permitted`（EPERM） |
| `/usr/bin/nvidia-smi` | 0 字节空文件，`-L` 无输出 |
| `/proc/driver/nvidia/gpus` | 列出 8 张 NVIDIA A800 80GB PCIe，但本进程打不开对应设备 |

宿主 procfs 能看见 GPU。本容器对 NVIDIA 设备节点的 open 被拒绝，CUDA 枚举不到设备。补过设备节点之后仍然 EPERM，`devices.allow` 不可写。因此没有可用 CUDA。

## 指标

预注册协议：[G/M0_results/protocol_snapshot.json](M0_results/protocol_snapshot.json)，锁定时间 2026-10-10T10:34:19+08:00，在任何训练或解码之前写入。汇总：[G/M0_results/summary.json](M0_results/summary.json)。没有逐样本 JSON。

### 实验 A（未跑）

| checkpoint | 解码 | n | RecoveryRate | bootstrap 95% CI | gold 首现百分位 | DefaultThink−DirectQA (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| ASU+GD (τ=2.3) | DirectQA | — | — | — | — | — |
| ASU+GD (τ=2.3) | DefaultThink | — | — | — | — | — |
| ASU+GD (τ=2.3) | ZeroThink | — | — | — | — | — |
| NPO+GD | DirectQA | — | — | — | — | — |
| NPO+GD | DefaultThink | — | — | — | — | — |
| NPO+GD | ZeroThink | — | — | — | — | — |

### 实验 B（未跑）

| 项 | n | 指标 | 95% CI |
| --- | --- | --- | --- |
| B1 Jaccard（IG Top-10% × 中介 Top-10%） | — | — | — |
| B2 mean D_pivot | — | — | — |
| B2 mean D_random | — | — | — |
| B2 比值 mean(D_pivot)/mean(D_random) | — | — | — |
| B2 Wilcoxon p | — | — | — |
| B2 配对 d | — | — | — |
| 降级重检（层下标 24） | — | — | — |
| secondary（base 未遗忘模型） | — | — | — |

## 口径备注

| 项 | 记录 |
| --- | --- |
| R-TOFU 或自生成 CoT | —。GPU 门在数据步骤之前触发，`sangyon/R-TOFU` 没有下载，也没有 few-shot 生成 CoT。 |
| 判定器人工抽检 | —。没有生成文本，抽检 0 条，没有校准记录。 |
| 负对照 retain 50 | —。没有跑。有效性未知，因此也不进入 Gate A/B。 |
| 训练 checkpoint | —。没有调用 `forget.py`。计划中的两个方法仍是仓库现成实现：ASU+GD（τ=2.3，`forget_coeff=0.1`）与 NPO+GD（`forget_coeff=1.0`），都还没跑。 |
| ZeroThink | 协议里已锁死 Llama-2 适应句，但没有实际解码。 |

## 与预注册判据逐条对照

| 判据 | 阈值 | 观测 | 是否通过 |
| --- | --- | --- | --- |
| Gate A：ASU DefaultThink RecoveryRate | ≥10% | — | — |
| Gate A：该率的 CI 下界 | >5% | — | — |
| Gate A：ASU DirectQA RecoveryRate | ≤5% | — | — |
| Gate B1：平均 Jaccard | ≥0.4，且 n≥40 | — | — |
| Gate B2：pivot 下降 ≥3× 随机列 | 且 `mean(D_random)>0` | — | — |
| Gate B2：Wilcoxon | p<0.05 | — | — |
| Gate B2：配对 d | ≥0.8，且 n≥40 | — | — |
| 负对照 | retain DirectQA 与 DefaultThink 都 ≥90% | — | 未检，实验未进入判决 |
| 算力 | ≤24 GPU·h | 0 | 未超预算；停因是没有可用 CUDA |

组合规则 `A ∧ (B1 ∨ B2)` 无法计算。按协议，这种情况记 **blocked**，不记 proceed / 降级 / 判死。方向 G 不因这次缺失而关闭；M1 也不立项。GPU 可用之后应从已锁定的 `protocol_snapshot.json` 重跑，不得改判据。

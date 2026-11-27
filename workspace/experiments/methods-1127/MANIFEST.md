# Artifact Manifest

## 独立方法包

每个 `CF2-*` 目录拥有以下独立工件：`method.py`、`modeling.py`、`data.py`、`runtime.py`、`run.py`、`run.sh`、`config.json`、`README.md`、`proposal.md`。

| 目录 | 算法所有权 |
|---|---|
| CF2-01_跨层反事实通量 | 层间反事实通量与逐位置内化 |
| CF2-02_因子绑定交互 | 四格双重差分与控制关系 OOD 门 |
| CF2-03_条件有向信息流 | 条件交叉协方差差与边缘统计保护 |
| CF2-04_不平衡传输质量销毁 | 非平衡 Sinkhorn 与未配对源质量 |
| CF2-05_跨层答案小波谱 | 二维 Haar、尺度掩码与逆变换 |
| CF2-06_反事实割线敏感度 | 主体距离归一化割线与控制响应锚 |
| CF2-07_三阶累积量绑定 | 三阶中心矩差与低阶统计保护 |
| CF2-08_保留协方差白化谱 | 低秩白化广义谱与全维目标 |
| CF2-09_稳健低秩稀疏事件 | 稳健 PCP 与语义归属门 |
| CF2-10_联盟交互激活擦除 | 多层同步 patch、组件对交互与 retain 成本 |

## 集群入口

- `start.sh`：可选安装、算子自测、真实环境预检、启动队列。
- `run_all.py`：进程级 GPU 调度、日志、续跑和状态汇总。
- `preflight.py`：依赖、CUDA、模型和数据 fail-fast 检查。
- `slurm.sbatch`：通用 Slurm 模板。
- `requirements.txt`：真实模型依赖。

## 验收与说明

- `validate_all.py`：算子、TinyModel、梯度、冻结、结果和隔离验收。
- `README.md`：完整使用说明。
- `EXPERIMENT_PLAN.md`：里程碑与证据边界。
- `VALIDATION_REPORT.md`：实际本地验收记录。
- `_validation/summary.json`：机器可读验收结果。

`runs/`、`_validation/` 和 `__pycache__/` 是可再生成输出，不属于算法源代码。上传源码包时可排除。

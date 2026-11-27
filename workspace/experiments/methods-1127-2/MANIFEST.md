# Artifact Manifest

## 独立算法目录

`CF3-01_双线性路由内容差分/`、`CF3-02_自校正持续子空间/`、`CF3-03_访问分布加权激活差/`、`CF3-04_门控状态转移擦除/`、`CF3-05_答案跨度Grassmann子空间/`、`CF3-06_配对置换FDR激活门/`、`CF3-07_多路径积分稳定归因/`、`CF3-08_反事实Copula尾依赖/`、`CF3-09_双端Hankel激活模态/`、`CF3-10_非交换路径签名遗忘/`。

每个目录包含独立 `method.py`、`modeling.py`、`data.py`、`runtime.py`、`run.py`、`run.sh`、`config.json`、`README.md`、`proposal.md`。

## 集群入口

- `start.sh`：依赖安装、算子预检、环境预检和队列启动。
- `run_all.py`：独立进程、多 seed、GPU 池、恢复、跳过与状态汇总。
- `preflight.py`：CUDA、包、模型和 TOFU split 预检。
- `validate_all.py`：目录隔离、算子和 Tiny 完整训练验收。
- `requirements.txt`、`slurm.sbatch`。

## 说明与证据

- `README.md`：上传、运行、配置和实现边界总说明。
- `EXPERIMENT_PLAN.md`：复现阶段和固定实验契约。
- `VALIDATION_REPORT.md`：真实验收范围与结果。
- `CODE_REVIEW.md`：独立评审不可用及执行者检查记录。
- `_validation/`：本地机器可读证据，仅本机保留，无需上传集群。
- `queue_status.json` 和各目录 `runs/`：本地队列测试输出，仅本机保留，无需上传集群。

交付形式为目录本身，不提供压缩包；直接上传 `methods-1127-2/` 即可。

所有方法结果均标记 `claim_level=false`。本清单不表示研究有效性、投稿就绪或独立评审通过。

# GPU / CUDA 环境准备记录（复现前必读）

本记录对应 `canonical_v1` 实际使用的环境。当前容器默认只有 CPU 版 PyTorch，直接跑实验会退化到 CPU 或触发 cuBLAS 崩溃，复现前必须先完成以下步骤。

## 硬件与驱动

- GPU：2 × NVIDIA H20，单卡约 95 GiB 显存。
- 驱动：`535.247.01`
- `nvidia-smi` 显示 CUDA Version：`12.2`
- 注意：`nvidia-smi` 能看到 GPU，不代表当前 Python 环境的 PyTorch 支持 CUDA。

## 初始问题

首次运行时：

```text
torch 2.6.0+cpu
torch.version.cuda = None
torch.cuda.is_available() = False
```

因此 `run_m0b.py` 等脚本中 `device="cuda:0" if torch.cuda.is_available() else "cpu"` 会退回 CPU，导致 7B 模型在 CPU 上极慢。

## 已验证可用的安装步骤

### 1. 安装 CUDA 版 PyTorch / torchvision / torchaudio

驱动 535 支持 CUDA 12.1，因此使用 cu121 wheel：

```bash
python -m pip install --force-reinstall 'torch==2.5.1+cu121' \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu121 \
  'torchvision==0.20.1+cu121' 'torchaudio==2.5.1+cu121'
```

如果不装匹配的 `torchvision`，Transformers 导入 LLaMA 模型时会报：

```text
RuntimeError: operator torchvision::nms does not exist
```

### 2. 修复依赖版本

安装 CUDA 版 PyTorch 可能把 `fsspec` 和 `numpy` 升到不兼容版本，需固定：

```bash
python -m pip install --no-deps 'numpy==1.26.4' 'fsspec[http]==2024.9.0'
```

### 3. 修复 H20 上的 cuBLAS 崩溃

cu121 自带的 `nvidia-cublas-cu12==12.1.3.1` 在 H20 上执行部分 bf16/fp16 GEMM 时会触发：

```text
traps: python[...] trap divide error ... in libcublasLt.so.12
Floating point exception / divide error
```

已验证可用的规避方式：

```bash
python -m pip install --no-deps 'nvidia-cublas-cu12==12.4.5.8'
```

### 4. 验证 CUDA 可用

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
a = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
b = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
print('matmul ok', (a @ b).shape)
PY
```

预期：

```text
2.5.1+cu121 12.1
True 2
NVIDIA H20
matmul ok torch.Size([2048, 2048])
```

## 运行实验前的环境变量

每次运行前 source 公有盘环境脚本：

```bash
source /jizhicfs/chenyliao/ops/00_env.sh
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`00_env.sh` 会设置：

- `ASU_ROOT=/jizhicfs/chenyliao`
- `HF_HOME`、`HF_HUB_CACHE`、`HF_DATASETS_CACHE`、`TRANSFORMERS_CACHE`
- 从 `/usr/local/.env` 读取 `HF_TOKEN`

## canonical_v1 实际运行命令

```bash
cd /usr/local/ASU-unlearning/workspace
source /jizhicfs/chenyliao/ops/00_env.sh
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u scripts/run_canonical_v1.py
```

实测：

- 单卡显存峰值约 `91–92 / 95 GiB`
- GPU 利用率接近 `100%`
- 完整主实验约 `70 分钟`

## 复现前检查清单

1. `nvidia-smi` 能看到 H20。
2. `torch.version.cuda` 不是 `None`。
3. `torch.cuda.is_available()` 为 `True`。
4. `torchvision.__version__` 与 torch 均为 `+cu121`。
5. `nvidia-cublas-cu12==12.4.5.8`。
6. `numpy==1.26.4`，`fsspec==2024.9.0`。
7. 已 source `/jizhicfs/chenyliao/ops/00_env.sh`。
8. 已设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

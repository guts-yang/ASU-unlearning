# CF2-03 · 条件有向信息流

本目录是该方法的完整独立实现，不依赖其他 `CF2-*` 目录。

## 核心算子

$$\mathcal B_{s\to p}=C^+_{uv_p\mid q}-C^-_{uv_p\mid q}$$

在 mini-batch 上比较低秩条件交叉协方差，并约束两端边缘统计。

## 运行

```bash
python3 run.py --self-test
python3 run.py --profile tiny
bash run.sh smoke
bash run.sh pilot
python3 run.py --profile pilot --diagnose-only
```

配置在 `config.json`；使用 `--set dotted.key=value` 覆盖。默认真实模型与 TOFU 路径见根目录 `README.md`。

## 文件所有权

- `method.py`：方法专属 prepare/loss/diagnostics/self_test。
- `data.py`：本方法自己的反事实数据与 Teacher 缓存。
- `modeling.py`：本方法自己的 HF/TinyModel 和评估实现。
- `runtime.py`：本方法自己的训练、评估、checkpoint 和结果入口。

## 实现边界

线性条件依赖不是完整 transfer entropy，也不自动构成因果证明。

所有结果均为 `claim_level=false`；输出位置为 `runs/<profile>_seed0/`。

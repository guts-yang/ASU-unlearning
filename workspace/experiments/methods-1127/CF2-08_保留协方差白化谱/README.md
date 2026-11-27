# CF2-08 · 保留协方差白化谱

本目录是该方法的完整独立实现，不依赖其他 `CF2-*` 目录。

## 核心算子

$$C_\Delta u_j=\lambda_j C_Ru_j$$

低秩空间计算配对散射与独立 retain 协方差的广义谱，再映射回全维目标。

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

属于 LEACE/CIR 的反事实增量实现，不宣称新的通用擦除理论。

所有结果均为 `claim_level=false`；输出位置为 `runs/<profile>_seed0/`。

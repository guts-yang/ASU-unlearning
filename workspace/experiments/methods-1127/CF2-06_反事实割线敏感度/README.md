# CF2-06 · 反事实割线敏感度

本目录是该方法的完整独立实现，不依赖其他 `CF2-*` 目录。

## 核心算子

$$D_{sec}=\Delta A/(\|\Delta e_s\|+\epsilon)$$

最小化目标关系的归一化替换响应，同时锚定控制关系响应。

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

离散替换只产生割线，不声称连续 Jacobian；中性读出防止双边冻结捷径。

所有结果均为 `claim_level=false`；输出位置为 `runs/<profile>_seed0/`。

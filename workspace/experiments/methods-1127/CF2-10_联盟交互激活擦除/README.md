# CF2-10 · 联盟交互激活擦除

本目录是该方法的完整独立实现，不依赖其他 `CF2-*` 目录。

## 核心算子

$$I_{cd}=v(\{c,d\})-v(\{c\})-v(\{d\})+v(\varnothing)$$

对少量层组件执行多层同步 Teacher patch-forward，选择非加性交互最强且经 retain 效用成本修正的组件对。

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

只实现有界组件对交互，不是完整 Shapley-Taylor 采样估计。

所有结果均为 `claim_level=false`；输出位置为 `runs/<profile>_seed0/`。

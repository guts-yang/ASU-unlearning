# CF3-04 门控状态转移擦除（GRTE）

独立实现，不导入其他 CF3 方法。算法来源见本目录 `proposal.md`。

## 运行

```bash
python3 run.py --self-test
bash run.sh tiny
bash run.sh smoke
bash run.sh pilot --set method.temperature=0.1
```

## 实现

- 从真实 Llama `mlp.gate_proj/up_proj` 捕获预激活，用 gate/up 符号构成四状态。
- 跨样本与答案位置估计每个神经元的 4×4 true/null 转移矩阵。
- 用 `down_proj` 权重和 Swish(g)·u 幅值估计源状态特异度；未观测源状态不会被选择。
- 跨 split 特异方向余弦纳入严格门；Student 使用 soft sign occupancy 惩罚仍处于源状态。
- 不指定 donor 目标状态，不修改码本，不增加部署模块。

TinyModel 使用真实 SwiGLU，非重入式 gradient checkpointing 保持 Student 激活可微。状态相关性不自动等于因果，proposal 要求的独立 patch/readout 审计仍需后续实验。

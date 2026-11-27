# CF3-01 双线性路由—内容差分（BRCD）

独立实现，不导入其他 CF3 方法。算法来源见本目录 `proposal.md`。

## 运行

```bash
python3 run.py --self-test
bash run.sh tiny
bash run.sh smoke
bash run.sh pilot
python3 run.py --profile pilot --diagnose-only
python3 run.py --profile pilot --set method.alpha=0.2
```

## 实现

- Teacher 使用 `eager` attention，缓存真实注意力权重、GQA value、答案残差和完整序列。
- 对路由 `pi` 与内容 `v` 使用对称双线性恒等式，分量经 `o_proj` 映回残差流。
- 分别执行 true/null patch-forward，比较目标答案 NLL 增量和特异性。
- route/content 同时通过时记录不可识别并停止该层；严格档未通过时输出 `DIAGNOSTIC_KILL`。
- Student 回归 `A+ - alpha * selected_component`，不使用 donor 答案作目标。

当前首版只实现 attention 双线性分支，没有将 proposal 中的 SwiGLU 分支混入本方法。诊断特异性不是充分因果证据。

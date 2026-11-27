# CF3-03 访问分布加权反事实激活差（VOCE）

独立实现，不导入其他 CF3 方法。算法来源见本目录 `proposal.md`。

## 运行

```bash
python3 run.py --self-test
bash run.sh tiny
bash run.sh smoke
bash run.sh pilot --set method.eta=0.5 --set method.rollout_refresh=0
```

## 实现

- Student 从原输入采样前缀，与 gold 答案前缀按 `eta` 混合。
- 同一前缀分别拼到 `X+` 和 `X-`，由冻结 Teacher 产生严格共享状态的激活差。
- 访问权重 stop-gradient，Student 在原输入的相同前缀上回归反事实删除目标。
- `rollout_refresh=0` 使用初始 Student 的静态访问测度；正值按步数刷新并写入新的访问缓存。
- 每个 pair 的 gold/on-policy 权重分别归一化，输出状态数、来源与权重和。

rollout 增加前向成本。当前不是 donor KL，也没有把不同前缀的反事实状态作比较。等预算 gold-only/on-policy 对照仍需通过配置单独运行。

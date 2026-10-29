# 方向 I · MCSD-ASU

状态：E0 已于 2026-10-29 跑完，判决 `do_not_start_e1`。Student 训练未启动。本目录没有修改 `Right-to-be-forgotten/`、`muse/`、`open-unlearning/`、`wmdp/` 源码。

当前结论见 [结论.md](结论.md)。

## 已冻结的默认（原计划第 9 节未另作答复时采用）

1. 基座：`locuslab/tofu_ft_llama2-7b`（32 层，不是图示里的 16 层）。
2. 首版规格：Spec A，全层统一 `τ_A` 与 `τ_M`。Spec B 仅在 E0 出现 MLP 信号后再议。
3. retain 上限在本基座校准集上按 `2 × SE` 重标，不沿用方向 H 的 `0.003520`。
4. 实现面：只读调用 `Right-to-be-forgotten/my_models/my_llama.py` 里已有的 `attention_temp`；`τ_M` 由本目录在内存中包装 `LlamaMLP`，不改原文件。
5. 本目录允许新建协议、配置和脚本。

## 阅读顺序

- [结论.md](结论.md)：这次 E0 的结果和允许下的结论
- [e0_results.md](e0_results.md)：校准表与未运行项
- [gates/e0.json](gates/e0.json)：判决
- [protocol.md](protocol.md)：E0 / E1 / 鲁棒性阶段门
- [configs/e0_probe.yaml](configs/e0_probe.yaml)：看结果之前冻结的网格与判决规则
- [mcsd/teacher.py](mcsd/teacher.py)：Teacher 双通道包装

E1 与鲁棒性评测在 E0 通过前不启动。

# CF-04 v2 遗忘深度审计（非 claim 诊断）

`claim_level=false`、`gate_status=NOT_APPLIED`。本目录是独立新入口，与 `methods-1127` 物理隔离；只读复用 `workspace/auto_experiments/` 与 `workspace/experiments/diagnostic_methods/` 的既有实现，不把旧脚本当训练入口运行。

## 目的

检验 CF-04 v2g（300 steps / 3 seeds，dF=+2.3084±0.0915）的高 forget NLL 增益是「较深的知识移除」还是「EOS 率归零 / 生成长度翻倍导致的输出抑制」。300-step v2g 没有保存 checkpoint，主臂改用同方法的 `full_cf04_v2_formal`（4500 steps / 10 epochs），报告中显式标注预算差异。

## 文件

- `common_lite.py`：数据/评估工具（照搬 `auto_experiments/common.py` 的 `eval_nll`/`answer_nll`/`make_batch`）。
- `audit.py`：P1 直接 NLL、P2 few-shot 恢复、P3 20-step relearning、INT8 fake-quant，输出字段与 `results/audit_depth.json` 一致。
- `probe.py`：独立线性 Probe（3 seed，held-out AUC + recoverability），输出字段与 `results/audit_depth_probe_3seed.json` 一致。
- `behavior_check.py`：复现 `eval_generate.py` 的生成行为字段，并增加等长度截断的 overlap 对照；用显式贪心解码避开 `num_logits_to_keep=1` 的 bf16 GEMM 崩溃。
- `run_audit.sh`：固定 `CUDA_VISIBLE_DEVICES=0` 串行运行三个脚本。
- `EVIDENCE.md`：阶段 0 证据盘点；`CF04_DEPTH_AUDIT.md`：最终报告（阶段 4 生成）。

## 运行

```bash
cd workspace/experiments/cf04_depth_audit
# 口径对齐：只跑 base，与历史 audit_depth.json 的 base 对比
PYTHON_BIN=/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python \
  ./run_audit.sh
```

或用 `nohup env CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=<venv> bash run_audit.sh > results/console.log 2>&1 &` 常驻。

## 约束

- 只在 GPU 0；启动前检查 GPU 0 空闲，占用则停止，不自动切 GPU 1、不提高并行度。
- 不改算法、损失、评价口径；不做超参搜索；不训练新方法。
- 不调用 `/auto-review-loop`、`/experiment-audit`、`/result-to-claim`；ARIS 证据门仍 BLOCKED。
- `gate.verdict=PASS` 表示**检测到浅层遗忘**（P2/P3 恢复 ≥30%），不是方法有效。

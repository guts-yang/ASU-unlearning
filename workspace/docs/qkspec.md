# QKSpec：Target-only 的 QK 谱方向遗忘

## 1. 方法概述

QKSpec 不改写事实内容，而是尝试切断目标事实的检索通道：在待遗忘模型自身的 attention Q/K 结构中，定位与目标事实检索具有因果作用的少数谱方向，只收缩这些方向，并把低秩修改折回 `W_Q`，得到可标准前向推理的 student 模型。

- Target-only：不需要 Base 模型、不需要平滑 Teacher、不需要梯度/反向传播。
- 处理完整遗忘集 `D_F`；`D_R` 同时是保留集和近邻对照集（同关系、同模板、答案类型相近但实体不同），不额外构造 matched-retain。
- 推理期无 SVD、无 hook、无额外解码：保存的是标准 HF causal LM。

机制链条：收缩目标 QK 谱方向 → 目标 query-key 的 logit 贡献下降 → 目标检索边注意力下降 → 目标 V 更少写入残差 → 目标事实贡献下降。主张限于**检索访问阻断**，不宣称不可恢复的信息擦除。

## 2. 代码结构

所有 QKSpec 代码位于仓库根目录下的 `workspace/`，主干 `open-unlearning/` 仅作为框架依赖（模型/数据/评估/registry），未作任何修改：

```
workspace/
  src/
    qkspec/
      config.py      配置解析与校验（QKSpecConfig）
      qk_math.py     Q/K head 切片、GQA 映射、QR 低秩 SVD、delta_W_Q 折回
      hooks.py       捕获 pre-RoPE q/k 输入与 eager attention 权重
      score.py       阶段A 检索边贡献差 + 阶段B ablation 因果复核
      selection.py   Top-K 与 retain 约束下的两阶段方向选择
      controls.py    no-op / 随机等谱 / shuffle 对照
      checks.py      RoPE 分位置注意力检查、谱重构误差
      edit.py        编辑编排（发现→选择→折回→对照→评估）
      evaluate.py    D_F/D_R 的 NLL 与 token accuracy
      report.py      JSON 报告（分键、未覆盖读取器标记）
      trainer.py     thin trainer 包装（覆盖 train()）
    run_qkspec.py     自包含入口：设置 sys.path、注册 handler、Hydra 主流程
  configs/
    unlearn.yaml                          QKSpec 主 Hydra 配置
    qkspec/default.yaml                   核心算法默认参数
    trainer/QKSpec.yaml                   trainer 配置
    experiment/unlearn/tofu/qkspec.yaml   TOFU 实验配置
  scripts/qkspec_tofu_unlearn.sh          启动脚本
  tests/qkspec/test_qkspec.py             unittest 冒烟测试
  docs/qkspec.md                          本文档
```

## 3. 运行方式

前置：安装 `open-unlearning/requirements.txt`，准备好 TOFU 数据与 `open-unlearning/tofu_<model>_full` 初始模型。

```bash
bash workspace/scripts/qkspec_tofu_unlearn.sh Llama-3.2-1B-Instruct forget10 retain90 holdout10
```

脚本把 `workspace/src` 与 `open-unlearning/src` 加入 `PYTHONPATH`，然后做两件事：
1. `workspace/src/run_qkspec.py --config-name=unlearn.yaml experiment=unlearn/tofu/qkspec.yaml trainer=QKSpec ...`：一次性编辑 + 保存 student + 内置 TOFU evaluator 评估。入口通过 `hydra.searchpath` 让 `model/data/eval/hydra/paths` 等基础配置组仍从 `open-unlearning/configs` 解析，并在运行时注册 `QKSpec` handler（不改主干 registry）。
2. `open-unlearning/src/eval.py`：对已保存 student 做独立 TOFU 评估。

关键约束：QKSpec discovery 需要 **eager attention**（flash/sdpa 不返回 attention 权重）。实验配置 `workspace/configs/experiment/unlearn/tofu/qkspec.yaml` 已设 `model.model_args.attn_implementation: eager`；trainer 会在加载非 eager 时直接报错。编辑完成后保存的模型可在任意 attn 实现下推理。

也可直接调用内核（脱离 Hydra/Trainer）：

```python
from qkspec import QKSpecConfig, run_qkspec_edit
result = run_qkspec_edit(model, forget_loader, retain_loader, cfg, output_dir)
```

其中 `forget_loader` / `retain_loader` 产出扁平 batch `{"input_ids","attention_mask","labels"}`。

## 4. 配置项

所有配置在 `workspace/configs/trainer/QKSpec.yaml` 的 `method_args.qkspec`（与 `workspace/configs/qkspec/default.yaml` 对齐）：

| 参数 | 默认 | 含义 |
|---|---|---|
| `layer_start/end` | null | 参与的层范围 |
| `shortlist_per_layer` | 4 | 每层初筛保留的 head 数 |
| `max_candidate_heads` | 128 | 候选 head 总数上限 |
| `top_k_directions` | 16 | 进入阶段B因果复核的方向数 |
| `ranks_per_head` | 4 | 每个 head 的奇异方向预算 |
| `rho` | 0.5 | 目标方向收缩系数 \(0\le\rho<1\) |
| `edge_top_fraction` | 1.0 | 目标检索边按注意力权重保留的比例 |
| `max_discovery_batches` | null | 阶段A/D_R 扫描的 batch 上限 |
| `stage_b_min_nll_increase` | 0.0 | 阶段B保留方向所需的最小 D_F NLL 增量 |
| `max_retain_impact` | null | 允许的 D_R NLL 增量上限（保留约束） |
| `run_controls` / `control_modes` | true / noop,random_isospectral,shuffle | 对照开关 |
| `noop_rho` | 1.0 | no-op 对照的收缩系数（必须为零更新） |
| `enable_rope_check` / `rope_check_positions` | true / [0,64,128,256] | RoPE 分位置诊断 |
| `empty_direction_action` | noop | 无方向可选时的行为（noop/error） |

成本开关：`max_discovery_batches`、`max_stage_b_batches` 控制扫描/复核成本；阶段B 每个方向重跑一次 D_F/D_R 前向，成本约为 \(O(K\times\text{单次前向})\)。

## 5. 两阶段方向选择（重要表述约束）

- **阶段A（初筛/相关分数）**：对每个候选 head 做 SVD，在 `D_F` 的目标检索边（answer token × prompt token）累加 \(\sigma_j (x_q\!\cdot\!u_j)(v_j\!\cdot\!x_k)\)，减去 `D_R` 近邻边上同样量，取 Top-K。**A 只是筛选/相关分数，不能单独称为因果证明。**
- **阶段B（局部因果复核）**：对 Top-K 每个方向单独收缩后重跑前向，测 D_F NLL 增量；同时测 D_R NLL 增量作为保留约束。只有 B 才可标注为"局部因果证据"。

报告和日志中 A/B 分数分字段保存，禁止混用或合并成单一因果分数。

## 6. 报告字段

报告由 `report.py` 写入 `output_dir/qkspec_report.json`：

- `edit.selected`：最终方向集合（层、head、秩、sigma、A/B 分数、rho、retain_impact）。
- `edit.update_norm` / `edit.no_op_weights_unchanged`：编辑范数与 no-op 一致性。
- `edit.reconstruction_error` / `checks`：谱重构误差与 RoPE 分位置诊断。
- `forget_metrics` / `retain_metrics`：完整 D_F / D_R 的 NLL 与 token accuracy，**分键保存**。
- `controls`：noop / random_isospectral / shuffle 的 D_F/D_R NLL 变化，与真实编辑对比。若随机等谱对照与真实编辑效果相当，则目标特异性主张被证伪。
- `reader_coverage`：probe / causal_patch / quantization / relearning 均标记 `not_covered`。
- `claims`：明确 `access_blockade_l2: not_proven_in_first_version`、`information_erasure_l3: not_claimed`。

日志只记录层/head/秩/分数/误差/更新范数，不打印原始事实文本。

## 7. 失败边界与首版未覆盖

以下情况不能称为选择性遗忘，报告需如实标记：

1. 随机等谱方向产生与真实方向相当的 D_F NLL 增益（只是一般 attention 损伤）。
2. 输出沉默但其他路径（probe / causal patch）仍能读出目标事实——首版未实现这些读取器，标记 `not_covered`，不能升级为 L2 访问阻断结论。
3. 量化（Q4/Q8）或限预算重学习后目标事实恢复——首版未实现，标记 `not_covered`。
4. forget 改善伴随 matched-retain / 通用能力越门——由 D_R 约束与内置 TOFU evaluator 检查。

## 8. 测试

```bash
python -m unittest discover -s workspace/tests/qkspec -v
```

覆盖：SVD 重构、折回正确性、no-op 零更新、GQA 共享 K 映射、阶段B 过滤符号、配置校验。

## 9. 实现边界

- `M = W_Q.T @ W_K`（文档行向量约定）是 pre-RoPE 对象；实际 logit 具有位置依赖，RoPE 后 logit 误差单独报告。
- GQA：query head 通过 `num_heads // num_key_value_heads` 映射到共享 kv head；只更新 `q_proj`，K 共享不受影响。
- discovery 必须 eager attention；保存模型标准前向即可。

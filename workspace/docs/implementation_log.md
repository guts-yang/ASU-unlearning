# RecodeLoop M0 实现与验证日志

> 本文件记录已实现的工程能力、实际运行的命令与结果、未覆盖项。**不记录任何实验结果、性能数字或 checkpoint 哈希**（资产未提供）。

## 1. 状态总览

| 项 | 状态 |
|---|---|
| 基线修改 | **零修改**：相对原始 HEAD 与当前 HEAD，`muse/`、`open-unlearning/`、`Right-to-be-forgotten/`、`wmdp/` 无任何变化；所有新增文件为 `workspace/` 下未跟踪文件 |
| G0 资产映射 | **BLOCKED**（唯一 base/target/student/tokenizer/哈希映射未提供） |
| 科学评价 | **未进入 M0a/M0b/M0c 的科学结论**；`run_m0a/b/c` 均在入口停止 |
| 工程实现 | 类型契约、适配器、hook/patch、事实值 reader、定位、低秩编辑、主闭环、7 对照、5 审计臂、统计与门 |
| 工程验证 | 14 项 pytest 全部通过；CPU 微型模型 smoke 通过（非遗忘证据） |
| 审计覆盖 | 量化 `uncovered`；重学习默认 `uncovered`（需显式 in-memory 授权）；confirmation 五臂未运行 |

冻结配置哈希：`5f419fad2f004559c80e58e70dd54a62d7072ed795ea21cb803c82fc82a9b2e7`（`configs/*.yaml` 规范化 JSON 后 SHA-256 组合）。

## 2. 基线复用 / 调整 / 隔离清单

### 复用（外部适配，不改基线文件）
| 基线符号 | 真实路径 | workspace 用法 |
|---|---|---|
| `find_all_linear_names()` | `Right-to-be-forgotten/forget.py:22-31` | `BaselineAdapter.find_all_linear_names()` 返回完整点分路径变体；命名约定一致 |
| `run_generation()` | `Right-to-be-forgotten/metrics/metrics.py:302-345` | `BaselineAdapter.generate()` 对齐 `do_sample=False, use_cache=True`；种子改为显式参数（不硬编码 0） |
| `gd_loss()` / `-100` 掩码 | `Right-to-be-forgotten/trainer/losses.py:226-233`、`dataset/data_module.py:24-82` | `retain_nll()` / `retain_loss_tensor()` 使用 `ignore_index=-100` 的 HF loss |
| `set_random_seed()` | `Right-to-be-forgotten/utils/utils.py:90-96` | `utils/seed.py` 显式记录每次实际 seed（SeedRecord） |
| `forward_with_cache()` | `wmdp/asu/utils.py:14-34` | `activations/extraction.py` 自行实现可逆 hook 版本 |
| `LlamaAttention.o_proj`、`LlamaMLP.down_proj` | `Right-to-be-forgotten/my_models/my_llama.py:355`、`:290` | 仅允许这两类真实线性模块作为编辑/激活视图 |
| `DataCollatorForSupervisedDataset`、`IGNORE_INDEX=-100` | `open-unlearning/src/data/collators.py:7`、`src/data/utils.py:7` | 以契约形式复用标签语义，不直接调用 |
| `ProbedLlamaForCausalLM` | `open-unlearning/src/model/probe.py:12` | 审计中声明为"截断层+可训练 head"，不冒充标准线性 probe |

### 调整（仅 workspace 内）
- 激活提取与 patch 阶段：`BaselineAdapter.frozen()` 冻结当前模型。
- 低秩编辑阶段：基座与恢复器均冻结（runner 双层 `with adapter.frozen()` + `reader.freeze()`）；只有 `A,B` 的 `requires_grad=True`，optimizer 只含这些参数。
- 每轮目标：`editing_objective = CE(事实值) + lambda_ret * retain_NLL`；phi 仅用于选组件，不作为低秩方向或 SVD 结果。
- 数据划分：canonical ID/别名/实体簇隔离；AccessGuard 封存 confirmation。
- 外部 hook/context manager，不改 DecoderLayer 数学拓扑。

### 严格隔离
- teacher `attention_temp`（注意：基线没有 `tau` 这个符号）、`asu_loss()`、`CustomTrainerForgetting.compute_loss()` 保持原样，不进入 RecodeLoop 每轮目标。
- RecodeLoop 只读 ASU student 起点（G0 通过后）；Teacher 不作为每轮恢复器。
- canonical TOFU 与实体级 M0 结果分文件记录。
- confirmation 由 AccessGuard 封存，训练/选择路径不可读；`K=8` 生成仅作簇内重复测量（统计口径已写入冻结配置/契约）。

### 未发现 / 不使用
- 量化：基线未发现实际 Q4/Q8 实现（仅依赖声明 bitsandbytes）→ 本实现 `quantization` 审计臂默认 `uncovered`。
- 重学习：基线未发现训练流程（仅 `AUC_RETRAIN` 常量）→ 本实现独立实现，默认未运行。
- checkpoint 完整性/溯源审计：基线未发现（无 sha256/manifest）。
- 已知基线陷阱：`losses.py::kl_loss` 重复定义、`wmdp/asu/unlearn.py:63` 的 `attention_temp_value` 不一致、open-unlearning 的 `ASU.yaml` handler 未注册——workspace 不依赖这些路径。

## 3. 子代理复核结论

- 第一次审计（very thorough）：确认无 P08-M2 映射、无 manifest、无 sha256；列出全部入口与符号。
- 第二次复核（quick）：确认上述符号与语义；提出三处更正并已落实：
  1. `find_all_linear_names()` 返回 leaf name，适配器注释改为"完整点分路径变体"；
  2. 基线 `run_generation()` 硬编码 `manual_seed(0)`，适配器改为显式 `seed` 参数；
  3. 组件路径需澄清为 `adapter.model.named_modules()` 的键（等价于完整 HF 路径 `model.layers.i...`），已写入 `interface_contracts.md`。
- 第三次复核（quick，零修改）：子代理因自身无终端/git 能力，未能独立执行 `git diff`，返回 FAIL（工具能力不足，**不是**发现违规）；它同时发现 HEAD 由外部提交推进。随后由主代理直接执行 `git status --short`、`git diff --stat`、`git diff --name-only 622c02bf... HEAD`：基线四个子项目零变化，唯一提交差异为外部新增的提案文档。根目录无新增缓存/输出，`__pycache__`/`.pytest_cache` 命中 0，`data/`、`figures/` 仍为空，`docs/` 保留提案文档。

## 4. 实际执行命令与结果

| 命令 | 结果 |
|---|---|
| `git branch --show-current` | `K3/20261118` |
| `git rev-parse HEAD`（审计时） | `622c02bf758c70f9954d76fd7559716f9f8f2ae8` |
| `git rev-parse HEAD`（当前） | `b0633f69b0e860213faa2ef780fe9134a3c7bad1`（会话期间由外部提交 `新增 K3 RecodeLoop 研究提案文档`，仅改 `workspace/docs/K3_RecodeLoop_Proposal.md`，非本实现所为） |
| `git status`（修改前） | `nothing to commit, working tree clean` |
| `git diff --name-only 622c02bf... HEAD` | 仅 `workspace/docs/K3_RecodeLoop_Proposal.md`（427 行新增，外部提交） |
| AST 解析 13 个基线关键文件 | 全部 OK（修改前健康状态） |
| `find . -name 'test_*.py'`（基线） | 0 个 |
| `python -m pytest --collect-only -q`（基线） | `no tests collected`，退出 0 |
| `python scripts/validate_assets.py` | `G0 status: BLOCKED`，退出码 2；报告写入 `outputs/g0_validation.json` |
| `python scripts/smoke_test.py` | 3 轮闭环、budget/energy/patch kind 记录；`forgetting_evidence=false`；写入 `outputs/smoke_test.json` |
| `python scripts/run_m0a.py` | `BLOCKED`，退出 2；未运行正对照/NK/reader 检查 |
| `python scripts/run_m0b.py` | `BLOCKED`，退出 2；未运行机制比较 |
| `python scripts/run_m0c.py` | `BLOCKED`，退出 2；confirmation 未解封、审计未运行 |
| `PYTHONDONTWRITEBYTECODE=1 python -m pytest`（workspace） | `14 passed`，退出 0 |
| `git diff --stat` | 空（无已跟踪文件变化） |
| `git status --short` | 仅 `?? workspace/...` |

## 5. 文件职责

- `docs/baseline_inventory.md`：10 节只读审计（路径/符号/版本/未发现项/修改前状态）。
- `docs/interface_contracts.md`：10 类数据契约、shape/dtype/device/split、模块边界、G0 契约。
- `manifests/assets.yaml` / `entity_splits.yaml`：刻意留空的冻结资产占位（哈希为空即 BLOCKED）。
- `src/recodeloop/types.py` / `config.py` / `errors.py` / `g0.py`：类型、冻结配置、异常、G0 校验。
- `baseline_adapter.py` / `checkpoints.py`：单模型加载（禁止隐式复制）、冻结快照、组件枚举、保存/哈希、G0 守卫。
- `activations/`：声明式视图、可逆 hook 提取、self/NK/reverse/random 与单/联合 patch。
- `data/`：实体隔离、NK 构造（来源与哈希）、confirmation 访问保护。
- `readers/`：冻结答案嵌入相似度 reader（输出事实值与 NK 对照）、当轮训练（激活 detach）。
- `localization/`：NK-patch 前后 reader 分差定义 phi、预算内组件选择与交互冗余记录。
- `editing/`：透明低秩挂接/移除、秩与基座不变检查、Frobenius 能量投影、目标与预算。
- `loop/`：主闭环（train/calibration only）、冻结停止条件。
- `controls/`：no-op、one-shot、random-direction、shuffle-component、retain-pseudo、fixed-causal、PARS-style 同低秩参照。
- `audits/`：线性 probe、非线性 reader、指定 causal patch、量化（未覆盖）、限预算重学习（默认未覆盖）、retain 门。
- `evaluation/`：delta_r、簇 bootstrap/sign-flip、预注册门与否决规则。
- `scripts/`：G0 校验、CPU smoke、M0a/b/c 入口（G0 前置拦截）。

## 6. 未覆盖与限制

1. **科学阻塞**：G0 BLOCKED；base/target/student/tokenizer 与哈希未冻结，实体簇 manifest 为空。不得猜测 checkpoint。
2. **未运行审计**：五类独立审计在真实资产上均未运行；量化因无后端 `uncovered`；重学习需显式授权 in-memory 快照；quantization/relearning 的 smoke 调用只验证"未覆盖"路径。
3. **依赖限制**：运行时 Python 3.11.11 / torch 2.6.0+cpu / transformers 4.48.2；CUDA 不可用；PEFT 未安装。未安装/升级任何依赖。
4. **显存/资源**：未做多模型克隆或 offload；G0 通过后需按真实模型规模另行配置与记录显存峰值；未启动 GPU 训练。
5. **统计**：仅实现 10,000 次簇 bootstrap/95% 区间与配对 sign-flip；`p>0.05` 未被用作等效性证据；confirmation < 4 簇返回 `INCONCLUSIVE`。
6. **工程测试范围**：微型随机模型只验证接口、可逆性、配准与可复现性，不是遗忘有效性证据。

## 7. 回退方法

- 删除 `workspace/` 下本次新增的：`README.md`、`pytest.ini`、`.gitignore`、`configs/`、`manifests/`、`src/`、`scripts/`、`tests/`、`outputs/`、`.cache/` 即可恢复；**保留**原有空目录 `data/`、`docs/`、`figures/`，以及本次未创建也未修改的 `docs/K3_RecodeLoop_Proposal.md`。
- 未申请或实施任何基线补丁，因此无需还原基线文件。

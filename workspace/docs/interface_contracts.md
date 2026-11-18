# RecodeLoop M0 接口契约

本文定义 `workspace/src/recodeloop/` 中统一数据类型的字段、shape、dtype、device 与 split 语义，以及模块边界。所有关键状态均使用 `@dataclass(frozen=True)` 传递；不使用无说明的裸 `dict`/`tuple`。唯一允许的映射是明确注册的 metric-name → float 指标表和 YAML 配置字典。

## 1. 坐标系约定

- `split` 只允许四个字面值：`forget_train_edit`、`forget_train_reader`、`calibration`、`confirmation`。
- `role` 只允许：`base`、`target`、`asu_student`。
- `component` 使用点分路径，如 `layers.0.self_attn.o_proj`、`layers.0.mlp.down_proj`（即 `adapter.model.named_modules()` 的键；等价于完整 HF 路径 `model.layers.0.self_attn.o_proj`）；**只接受真实 `o_proj` / `down_proj`**。
- `device` 取 `cpu` 或 `cuda:N`；未指定时默认 `cpu`。`dtype` 取 `float32` / `float16` / `bfloat16` 字符串表示。
- 激活张量统一记 `[...]`：`N`=样本数，`L`=序列长度，`D`=隐藏维，`R`=秩，`K`=候选组件数。

## 2. 核心类型

### 2.1 `CheckpointRef`
```python
@dataclass(frozen=True)
class CheckpointRef:
    role: Literal["base", "target", "asu_student"]
    path: str                 # 本地绝对/相对路径或 HF id；严禁猜测
    revision: str | None      # HF revision；本地路径为 None
    tokenizer_path: str       # 必须显式给出
    sha256: str | None        # 64 位十六进制；None/空 = G0 BLOCKED
```
不变量：`sha256` 非空时才视为已冻结资产；G0 校验三角色路径与哈希唯一且互异。

### 2.2 `EntityCluster`
```python
@dataclass(frozen=True)
class EntityCluster:
    canonical_id: str
    aliases: tuple[str, ...]
    source_rows: tuple[str, ...]
    split: Split
    fact_relation: str
    fact_values: tuple[str, ...]
    provenance: str
```
不变量：canonical ID 与别名在全划分中唯一归属一个 split；禁止按 QA 行随机拆分。

### 2.3 `ActivationSpec`
```python
@dataclass(frozen=True)
class ActivationSpec:
    layer: int
    component: str            # 点分路径
    token_position: int | Literal["all"]
    pooling: Literal["last", "mean", "none"]
    dtype: str
    device: str
    shape: tuple[int, ...]    # 期望张量形状（不含 batch 维时以 -1 占位）
```

### 2.4 `ActivationBatch`
```python
@dataclass
class ActivationBatch:
    activations: torch.Tensor       # [N, ...]，dtype/device 与 spec 一致
    entity_ids: tuple[str, ...]     # len == N
    fact_ids: tuple[str, ...]       # len == N
    split: Split
    fact_value_labels: tuple[str, ...]  # len == N，事实值文本而非集合标签
    source_hash: str
    spec: ActivationSpec
```

### 2.5 `RecoveryScore`
```python
@dataclass(frozen=True)
class RecoveryScore:
    fact_value_accuracy: float
    exact_match: float
    candidate_rank: int | None
    mrr: float | None
    confidence: float
    nk_accuracy: float          # never-known 对照
    cluster_id: str
    split: Split
    n_samples: int
```

### 2.6 `PatchResult`
```python
@dataclass(frozen=True)
class PatchResult:
    patch_kind: Literal["self", "nk", "reverse", "random"]
    scope: Literal["single", "joint"]
    component: str | None
    recovery_before: float
    recovery_after: float
    recovery_delta: float       # = after - before
    retain_damage: float
    ci_low: float | None
    ci_high: float | None
```

### 2.7 `ComponentSelection`
```python
@dataclass(frozen=True)
class ComponentSelection:
    components: tuple[str, ...]
    ranks: tuple[int, ...]
    energies: tuple[float, ...]   # 逐组件 Frobenius 能量 g_{c,t}
    rule: str                     # 选择规则名，必需
    config_hash: str
    interaction_note: str | None  # 单组件无效但联合有效时记录冗余/交互
```

### 2.8 `Budget` 与 `LoopState`
```python
@dataclass(frozen=True)
class Budget:
    tokens: int
    steps: int
    components: int
    rank: int
    wall_clock_s: float | None = None

@dataclass(frozen=True)
class LoopState:
    step: int
    model_hash: str
    reader_hash: str
    components: tuple[str, ...]
    actual_budget: Budget
    stop_reason: str | None
```
`Budget` 用于所有对照/审计的预算配准；全局 `G2` 与逐组件 `g_{c,t}` 分别记录于 `Budget.tokens` 与 `ComponentSelection.energies`。

### 2.9 `AuditResult`
```python
@dataclass(frozen=True)
class AuditResult:
    arm: str
    independence_statement: str   # 显式说明 reader/簇/初始化/种子/用途隔离
    budget: Budget
    raw_metrics: Mapping[str, float]  # 仅注册指标名 → float
    control_delta: Mapping[str, float]
    coverage: Literal["covered", "uncovered"]
    note: str = ""
```
未实际执行的审计必须 `coverage="uncovered"`，不得默认通过。

### 2.10 `FactValueReader` Protocol
```python
class FactValueReader(Protocol):
    name: str
    def score(
        self,
        batch: ActivationBatch,
        fact_ids: Sequence[str],
        split: Split,
        nk_ids: Sequence[str],
    ) -> RecoveryScore: ...
    def state_dict_hash(self) -> str: ...
```
禁止输出 forget/retain 集合成员标签；必须输出事实值准确率 / exact match / MRR 与 NK 对照。

## 3. 模块边界

| 模块 | 输入 | 输出 | 不得承担 |
|---|---|---|---|
| `baseline_adapter` | `CheckpointRef` | model/tokenizer、前向/generate、组件路径、冻结快照 | 任何每轮训练逻辑 |
| `activations` | 当前模型 + `ActivationSpec` | `ActivationBatch` / patch 后模型 | 改 DecoderLayer 数学 |
| `readers` | `ActivationBatch` + 事实值标签 | `RecoveryScore` | forget/retain 二分类 |
| `localization` | `PatchResult` 序列 + 预算 | `ComponentSelection` | 把 phi 当 SVD 方向 |
| `editing` | `ComponentSelection` + model | 挂接低秩 `A,B` 的可逆 editor | 更新基座权重/恢复器 |
| `loop` | `M_{t-1}` + train/calibration | `M_T` + `LoopState` | 读取 confirmation |
| `controls` | 冻结配置 + budget | 对照臂结果 | 共享训练种子而声称独立 |
| `audits` | 冻结模型/reader | `AuditResult` | 未运行时标记 covered |
| `evaluation` | 簇级分数 | 区间 / 门判定 | 用 `p>0.05` 代替等效性 |

## 4. 冻结配置

`configs/*.yaml` 中出现的全部阈值（`T_max`、`k`、`m`、候选数 `B`、阈值、bootstrap 次数、CI、retain 门、等价界限、重学习步数）都是**待冻结口径**，不是实验结果。运行时由 `config.py` 计算配置哈希并写入 `LoopState` / `AuditResult` 相关记录；任何对 confirmation 的使用都必须先通过 `AccessGuard` 解封，且解封只发生在 M0c 授权后。

## 5. G0 契约

`manifests/assets.yaml` 必须给出 `base`/`target`/`asu_student` 三个角色的 `path`、`tokenizer_path`、`sha256`，三者路径与哈希互异且哈希为 64 位十六进制；`manifests/entity_splits.yaml` 必须给出非空 `clusters`，canonical ID/别名跨 split 不冲突，且 confirmation 簇数 ≥ 4（否则 `INCONCLUSIVE`）。任一不满足 → `scripts/validate_assets.py` 输出 `BLOCKED`，退出码 2；科学脚本在入口调用同一校验器并立即停止。

`sha256` 采用 `hashing.py::sha256_bundle`：排除 `.cache/`、`*.lock`、`*.metadata`、`pytorch_model*.bin` 与 `pytorch_model.bin.index.json`，保留 safetensors/index/config/tokenizer 与 `training_args.bin`；三角色口径一致。

## 6. M0a 报告契约

`evaluation/m0a.py::M0aReport` 字段：`status`（PASS/FAIL）、`per_model`（每角色 fact_value_accuracy/nk_accuracy/recovery_advantage 等）、`seed`、`model_hashes`、`reader_hashes`、`config_hash`、`gpu_peak_bytes`、`notes`。M0a 使用生成式自由文本 reader 解码具体事实值，target 为正对照、未微调 base 为未训练参照；confirmation 不参与。报告写入 `outputs/m0a_validation.json`，不是 M0b/L2 结论。

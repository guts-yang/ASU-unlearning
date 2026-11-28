# CF3-01 双线性路由—内容差分（BRCD）

独立实现，不导入其他 CF3 方法。算法来源见本目录 `proposal.md`（round3 设计）与 idea-stage `CF3-01_proposal.md`（v1.1 测量优先修订）。

## 实现版本

- **默认（round3，同步自远程 `origin/Auto/20261122`）**：`method.py` 提供四个独立候选 `attn_route / attn_content / glu_gate / glu_value`，family 内唯一 + 跨 family 歧义诊断；可选 SwiGLU 精确双线性分支；可选局部仿射半空间 POCS 代理（非线性 K 的可行点近似，不是精确 `Π_K`）；六个控制臂。`config.json` 默认关闭 GLU 与投影，保持 attention-only 主链。
- **v1.1（测量优先，本轮新增，默认关闭）**：`measurement_protocol.py` + `run_stage_a.py` 实现 Stage A（噪声下限、全层扫描、等条件对照、原生 QK/V 因子干预、簇 bootstrap + 留出稀疏选择）；`method.py` / `runtime.py` / `config.json` 增加可选 Stage B 接口（稀疏目标、归一化 MSE、梯度诊断、相对 EOS 护栏、c_f fail-closed）。详见 `IMPLEMENTATION_v1.1.md`。

## 运行

```bash
# round3 默认链（与远程一致）
python3 run.py --self-test
bash run.sh tiny
bash run.sh pilot
python3 run.py --profile pilot --diagnose-only

# v1.1 Stage A：只诊断，不训练 Student
python3 run_stage_a.py --self-test
python3 run_stage_a.py --profile tiny

# 真实 Stage A：先 audit-only 冻结阈值，再全层扫描
python3 run_stage_a.py --config configs/stage_a_v11.json --profile pilot \
  --audit-only --output runs/stage_a_audit_v11_seed0
python3 run_stage_a.py --config configs/stage_a_v11.json --profile pilot \
  --output runs/stage_a_pilot_v11_seed0

# v1.1 Stage B：仅当 Stage A 输出 stage_a_pass=true
python3 run.py --profile pilot \
  --set method.sparse_selection_file=/abs/path/runs/stage_a_pilot_v11_seed0/selected_candidates.json
```

## 关键边界

- SwiGLU 分支对 `SiLU(g)` 与 `u` 使用精确对称恒等式，经 `down_proj` 写回；不对分量做 ReLU/截断。
- POCS 是局部仿射代理 + 信任域 + 真实冻结下游回验；round3 的全部 `identity` 不构成投影有效性证据。
- donor 答案只用于单边"不支持 donor"约束测量，从不作为 MSE/KL/soft label。
- v1.1 默认不改变 round3 行为；Stage A 的 `selected_candidates.json` 必须由独立诊断产生，不得手工拼造。
- `claim_level=false`：Tiny smoke 和退出码 0 只说明工程可运行，不构成遗忘有效性证据。

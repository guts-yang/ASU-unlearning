"""Generate one record per reproduced idea: passed/ vs archived/.

Tier A : 3-seed stable, effect large and direction-consistent
Tier B : supported under the calibrated neutral strength, but lambda is a proxy
         for c/d_j/e or the contrast is weak
Archived: against evidence, inconclusive across seeds, or deferred
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASSED = ROOT / "passed"
ARCH = ROOT / "archived"

TEMPLATE = """# {id} · {name}

**判定**：{tier_label}
**复现状态**：已复现（proxy-v061；{seed_note}）
**日期**：2026-11-26

## 核心假设

{hyp}

## 实测结果

{metrics}

## 证据

{evidence}

## 复现命令

```bash
cd {code_dir}
{VENV} run_{runner}.py
```

## 局限

{limits}

## {next_label}

{next}
"""

IDEAS = [
    dict(
        id="D3", name="因果门控 ΔA（patch 加权的因果分量）", tier="A",
        runner="diag_v2", seed_note="3 seed",
        hyp="ΔA 中只有少数层/位置对目标事实读出有因果贡献；按因果效应加权可剔除身份/词频分量。",
        metrics="""| 层 | ΔNLL（patch A⁻ 后） |
|---|---:|
| L3 | +0.204 |
| L7 | +0.372 |
| L11 | +0.916 |
| L15 | +1.343 |

3 seed 均为 layer 15 主导，`top_layer_squared_share = 0.650 ± 0.023`（seed 0/1/2: 0.639/0.676/0.635）。四层单调递增。""",
        evidence="`results/diag_v2.json`（D3）、`results/seed_sweep_D3_D9.json`",
        limits="代理因果量是单层 patch 的 NLL 变化，不是逐位置/逐分量的完整因果追踪；样本 16 对。",
        next="升级为确认的条件：逐分量（head × position）因果追踪 + 3 seed + 至少 2 个 donor 构造方式复现同一层排序。",
    ),
    dict(
        id="D1", name="ΔA 定义中性约束可行域几何（禁止方向）", tier="B",
        runner="calib", seed_note="单 seed",
        hyp="让 ΔA 只定义可行域的禁止方向（`⟨h−A⁺, ΔA⟩≤0`），从结构上排除答案替换，同时不教 donor 答案。",
        metrics="""| 配置 | dF | dR | eos | overlap |
|---|---:|---:|---:|---:|
| align（T=A⁺） | 0.068 | −0.156 | 1.000 | 0.391 |
| neutral λ=0.05 | 0.336 | −0.147 | 1.000 | 0.367 |
| neutral λ=0.10 | 0.631 | −0.142 | 1.000 | 0.385 |
| **neutral λ=0.20** | **1.455** | **−0.132** | **1.000** | **0.373** |
| neutral λ=0.50 | 15.250 | −0.063 | 0.917 | 0.140（退化） |""",
        evidence="`results/calibration.json`、`results/train_D1_neutral_s0.json`、`results/train_D1_s0.json`",
        limits="λ 是 `c/d_j/e` 的代理，未真正标定约束阈值；单 seed；禁止方向 hinge 的效果未与“无 hinge”做等强度对照。",
        next="升级为确认的条件：标定 `c/d_j/e`、补 3 seed、加“同强度但无禁止方向”的对照。",
    ),
    dict(
        id="D6", name="多 donor 边际化 / donor 私有成分消去", tier="B",
        runner="diag_v2", seed_note="单 seed",
        hyp="单 donor 的 ΔA 混有该 donor 的身份特异性；用 K 个 donor 的共有成分作事实方向。",
        metrics="""- `cos(single, marginalized) = 0.836` → 边际化确实改变方向
- `cos(marginalized, answer) = 0.055` → 共有成分与答案 token 的对齐很弱

判定：方向改变成立，但“共有成分 = 事实”未成立。""",
        evidence="`results/diag.json`（D6）、`results/diag_v2.json`（D6）",
        limits="K=4、donor 来自 retain90；共有成分用简单均值而非低秩子空间；与答案对齐只有 0.055，不能支持“事实方向”。",
        next="升级为确认的条件：K 扫描 + SVD 共有子空间 + 与事实 probe AUC 联合验证。",
    ),
    dict(
        id="D7", name="ΔA 纯度驱动渐进 Λ(k) / ρ_ℓ(k)", tier="B",
        runner="two", seed_note="单 seed",
        hyp="用 ΔA 的逐层事实纯度导出渐进 schedule，替代手设 Λ(k)。",
        metrics="""| 配置 | dF | dR | eos | overlap |
|---|---:|---:|---:|---:|
| align | 0.023 | −0.179 | 1.000 | 0.389 |
| neutral λ=0.2 | 0.589 | −0.173 | 1.000 | 0.399 |

纯度排序：`layer_purity` 写入结果 JSON；渐进 schedule 只在 3/11/15 三层上按纯度顺序加入。""",
        evidence="`results/train_D7_lam0.2.json`、`results/train_D7_s0.json`",
        limits="未与“等预算静态全层 / 反向 / 随机 schedule”做四臂对照（proposal §6 的硬要求），因此**不能声称渐进带来独立收益**。",
        next="升级为确认的条件：补齐四臂等预算对照；若静态全层追平，则只能保留“多层对齐”表述。",
    ),
    dict(
        id="D8", name="结构化对照族残差化（关系/词频/subword 基线）", tier="B",
        runner="diag_v2", seed_note="单 seed",
        hyp="把 §6.1 的五类验收对照前移为 ΔA 的构造算子，回归掉共享结构成分。",
        metrics="""- 残差化前 `cos(ΔA, answer) = 0.0186`
- 残差化后 `cos(ΔA_resid, answer) = 0.0181`（保留 97%）

判定：残差化没有破坏事实信号，但也没有提升；“null 基”目前是问法变化，不是严格的关系/词频匹配。""",
        evidence="`results/diag_v2.json`（D8）",
        limits="null 对照是 `\"Please answer briefly. \" + question`，属弱代理；真正的关系/词频匹配对照族未构造。",
        next="升级为确认的条件：构造 relation-swap / answer-paraphrase / subword-matched 三类 null 基，再做残差化。",
    ),
    dict(
        id="D10", name="零空间 / 抗恢复投影约束", tier="B",
        runner="two", seed_note="单 seed",
        hyp="让 ΔA 避开 retain 高方差方向，使遗忘不易被少量微调/量化恢复。",
        metrics="""| 配置 | dF | dR | eos | overlap |
|---|---:|---:|---:|---:|
| align | 0.061 | −0.152 | 1.000 | 0.377 |
| neutral λ=0.2 | 1.773 | −0.132 | 1.000 | 0.380 |""",
        evidence="`results/train_D10_lam0.2.json`、`results/train_D10_s0.json`",
        limits="投影方向是 retain 激活均值（不是 retain 梯度 top-PC），且**未测 P3/INT8 恢复率**——而抗恢复正是这个 idea 的卖点。",
        next="升级为确认的条件：换用 retain 梯度 SVD 的 top-PC，并补 P3/INT8 恢复阶梯。",
    ),
    dict(
        id="D4", name="事实 / 身份双探针解耦 + 反事实激活潜力", tier="ARCH",
        runner="diag_v2", seed_note="单 seed",
        hyp="ΔA 混有主体身份与目标事实；双探针解耦后应保留事实子空间。",
        metrics="""| 配对方式 | cos(ΔA, answer) | cos(ΔA, entity) | answer 更近比例 |
|---|---:|---:|---:|
| v1 随机 donor | 0.0398 | 0.003 | 81.3% |
| **v2 同模板实体替换** | **0.0186** | **0.0314** | **25.0%** |

严格配对下 ΔA 更接近主体身份而非事实，与假设方向相反。""",
        evidence="`results/diag.json`（D4）、`results/diag_v2.json`（D4）",
        limits="探针是 mean-pooled token embedding 的余弦，不是训练过的线性 probe；但两套配对给出方向相反的结论本身就说明 v1 不可靠。",
        next="重新开启条件：训练真正的线性 fact/identity probe（AUC），并在两套配对下同时报告；若仍是 identity-leaning，则该 idea 应长期归档。",
    ),
    dict(
        id="D5", name="答案跨度轨迹 ΔA(p) 与数据驱动位置权重", tier="ARCH",
        runner="diag_v2", seed_note="单 seed",
        hyp="ΔA 沿答案跨度有位置结构，可用轨迹定位“事实绑定发生的答案位置”。",
        metrics="""- `mean_gini = 0.316`（v1 0.206）
- `mean_top1_position_share = 0.085`（均匀为 1/14 ≈ 0.071）

判定：接近均匀，位置结构不足以支撑数据驱动权重。""",
        evidence="`results/diag.json`（D5）、`results/diag_v2.json`（D5）",
        limits="只用了一种位置归一化；未测 factual/function token 分层。",
        next="重新开启条件：按 factual/function token 分层重算轨迹；若仍近均匀则归档为“事实不是按答案位置局部展开”。",
    ),
    dict(
        id="D2", name="ΔA 深度证书化 / 方法身份张力 (B)", tier="ARCH",
        runner="diag_v2", seed_note="未判定",
        hyp="ΔA 作为训练后的因果式遗忘深度证书，与训练解耦。",
        metrics="""- base 上 patch A⁻ 后 NLL 变化 `+0.916`（v1 +1.930），16/16 同号
- 仅证明 ΔA 的因果可检测性，**不是证书**""",
        evidence="`results/diag_v2.json`（D2）",
        limits="证书必须 patch **训练后 student** 回 A⁺ 并测恢复量；当前 `train_v061` 不落盘权重，无法测。",
        next="重新开启条件：给 `train_v061` 加 checkpoint 保存，再对 D1/D10 的 λ=0.2 模型跑 patch 回 A⁺ 的恢复测试。",
    ),
    dict(
        id="D9", name="读出空间重定义 ΔA_readout（旋转不变）", tier="ARCH",
        runner="seed_sweep", seed_note="3 seed",
        hyp="ΔA 定义在冻结下游读出的标量差上，天然对隐空间旋转不变，并提供额外信息。",
        metrics="""| seed | pearson(readout, hidden) |
|---|---:|
| 0 | −0.230 |
| 1 | +0.321 |
| 2 | +0.123 |

均值 `0.071 ± 0.279`，符号跨 seed 翻转 → 不可判定。""",
        evidence="`results/diag_v2.json`（D9）、`results/seed_sweep_D3_D9.json`",
        limits="只估了标量 NLL 差与 ‖ΔA‖ 的相关，没有估计 Jacobian 方向；v1 的 +0.469 是单 seed 偶然。",
        next="重新开启条件：估计 Jacobian 一阶方向并做正交变换不变性测试；否则归档为“读出与隐空间关系不稳”。",
    ),
]


def main():
    PASSED.mkdir(exist_ok=True)
    ARCH.mkdir(exist_ok=True)
    for it in IDEAS:
        archived = it["tier"] == "ARCH"
        label = (
            "**归档（不过关 / 未判定）**"
            if archived
            else ("**Tier A · 过关**" if it["tier"] == "A" else "**Tier B · 有条件过关**")
        )
        body = TEMPLATE.format(
            id=it["id"],
            name=it["name"],
            tier_label=label,
            seed_note=it["seed_note"],
            hyp=it["hyp"],
            metrics=it["metrics"],
            evidence=it["evidence"],
            code_dir=str(ROOT / "code"),
            VENV="/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python",
            runner=it["runner"],
            limits=it["limits"],
            next_label="重新开启条件" if archived else "升级为确认的条件",
            next=it["next"],
        )
        out = (ARCH if archived else PASSED) / f"{it['id']}_{it['name'].split('（')[0].strip().replace('/', '_').replace(' ', '_')[:28]}.md"
        out.write_text(body, encoding="utf-8")

    passed = [it for it in IDEAS if it["tier"] != "ARCH"]
    archived = [it for it in IDEAS if it["tier"] == "ARCH"]
    (PASSED / "README.md").write_text(
        "# 过关方案（direction_i_v061）\n\n"
        "分级标准：Tier A = 3 seed 稳定且效应显著；Tier B = 在校准中性强度下支持，但 λ 仍是 `c/d_j/e` 的代理或对照较弱。\n\n"
        "| # | 方案 | 等级 | 关键数字 |\n|---|---|---|---|\n"
        + "\n".join(
            f"| {it['id']} | {it['name']} | {it['tier']} | "
            + ("3/3 L15, share 0.650±0.023" if it["id"] == "D3" else
               "λ=0.2 dF 1.455" if it["id"] == "D1" else
               "λ=0.2 dF 0.589" if it["id"] == "D7" else
               "λ=0.2 dF 1.773" if it["id"] == "D10" else
               "0.836 / 0.055" if it["id"] == "D6" else "0.0186→0.0181")
            + " |"
            for it in passed
        )
        + "\n",
        encoding="utf-8",
    )
    (ARCH / "README.md").write_text(
        "# 归档方案（不过关 / 未判定）\n\n"
        "| # | 方案 | 归档原因 |\n|---|---|---|\n"
        + "\n".join(
            f"| {it['id']} | {it['name']} | "
            + ("严格配对下 ΔA 偏身份，与假设相反" if it["id"] == "D4" else
               "ΔA 在答案跨度上近均匀，无位置结构" if it["id"] == "D5" else
               "需训练后 checkpoint，当前无法测证书" if it["id"] == "D2" else
               "3 seed 符号翻转，不可判定")
            + " |"
            for it in archived
        )
        + "\n\n归档不等于证据消失：每份文件保留了原始数字、证据路径与重新开启条件。\n",
        encoding="utf-8",
    )
    print(f"passed={len(passed)} archived={len(archived)}")


if __name__ == "__main__":
    main()

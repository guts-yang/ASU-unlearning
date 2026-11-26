#!/usr/bin/env python3
"""Non-claim diagnostic runner for the plan-level unlearning candidates.

It never invokes the ARIS evidence gate, never writes accepted/provisional
state, and commits/pushes one method at a time.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning")
WS = REPO / "workspace"
SCR = WS / "auto_experiments/results/screening"
LOGS = WS / "refine-logs"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


CF03S = load(SCR / "cf03_sens.json")
CF03H = load(SCR / "cf03_shuffled.json")
CF10G = load(SCR / "cf10_global.json")
CF10H = load(SCR / "cf10_high2low.json")
AUDIT = load(WS / "auto_experiments/results/audit_depth.json")
ALLRES = load(WS / "experiments/direction_i/ALL_RESULTS_SUMMARY.json")["files"]

METHODS = [
    {
        "id": "CF-01",
        "title": "约束反演目标",
        "priority": "rank 1 / 8.0",
        "mode": "operator review + adjacent ablation",
        "train": False,
        "eval": False,
        "reuse": True,
        "verdict": "NOT_IMPLEMENTABLE",
        "summary": "约束反演目标与 patch 因果筛选算子未实现；相邻解耦消融未见 retain-safe 收益。",
        "metrics": {
            "stage13_idea1": {"forget": 1.0660445601851851, "retain": 1.5279947916666667},
            "stage13_ewc": {"forget": 1.0648148148148149, "retain": 1.5257161458333333},
            "stage13_gradnorm": {"forget": 1.0670572916666667, "retain": 1.5247395833333333},
            "retain_safe_gain": False,
        },
        "evidence": ["workspace/experiments/direction_i/ALL_RESULTS_SUMMARY.json#stage13_decouple"],
        "limitations": ["核心算子缺失，不能用相近超参扫描替代。"],
    },
    {
        "id": "CF-03",
        "title": "因果门控 activation gate",
        "priority": "rank 2 / 7.8",
        "mode": "20-step sens vs sens_shuffled",
        "train": True,
        "eval": True,
        "reuse": False,
        "verdict": "NEGATIVE",
        "summary": "敏感度门控臂未在 forget 与 retain 上同时优于打乱对照，首轮诊断不支持独立收益。",
        "metrics": {
            "sens_after": {"forget": CF03S["after"]["forget"], "retain": CF03S["after"]["retain"]},
            "shuffled_after": {"forget": CF03H["after"]["forget"], "retain": CF03H["after"]["retain"]},
            "sens_minus_shuffled": {
                "forget": CF03S["after"]["forget"] - CF03H["after"]["forget"],
                "retain": CF03S["after"]["retain"] - CF03H["after"]["retain"],
            },
        },
        "evidence": [
            "workspace/auto_experiments/results/screening/cf03_sens.json",
            "workspace/auto_experiments/results/screening/cf03_shuffled.json",
        ],
        "limitations": ["20-step single-seed proxy; not a faithful causal-gate implementation."],
    },
    {
        "id": "CF-02",
        "title": "三前向角色分解",
        "priority": "rank 3 / 7.6",
        "mode": "operator review",
        "train": False,
        "eval": False,
        "reuse": True,
        "verdict": "NOT_IMPLEMENTABLE",
        "summary": "三前向角色分解、retain floor 与 donor 负约束未实现。",
        "metrics": {"operator_status": "missing", "stage13_adjacent_improvement": False},
        "evidence": ["workspace/idea-stage/candidates/CF-02_三前向角色分解/CF-02_proposal.md"],
        "limitations": ["缺核心约束求解器，不能以 alpha 扫描代替。"],
    },
    {
        "id": "CF-04",
        "title": "事实身份双探针解耦",
        "priority": "rank 4 / 7.4",
        "mode": "existing pooled probe",
        "train": False,
        "eval": False,
        "reuse": True,
        "verdict": "INCONCLUSIVE",
        "summary": "pooled ΔA 的 retain-orthogonal 检查通过，但缺少事实/身份双探针 AUC。",
        "metrics": {
            "stage7b_low_rank_pass": ALLRES["stage7b_da_probe_pooled.json"]["gate"]["low_rank_pass"],
            "stage7b_retain_orthogonal_pass": ALLRES["stage7b_da_probe_pooled.json"]["gate"]["retain_orthogonal_pass"],
            "overlap_true_wrong": ALLRES["stage7b_da_probe_pooled.json"]["layers"]["12"]["overlap_true_wrong"],
            "overlap_true_retain": ALLRES["stage7b_da_probe_pooled.json"]["layers"]["12"]["overlap_true_retain"],
            "auc_f_minus_auc_id": None,
        },
        "evidence": ["workspace/experiments/direction_i/ALL_RESULTS_SUMMARY.json#stage7b_da_probe_pooled"],
        "limitations": ["缺 AUC_f / AUC_id 与 wrong-donor 身份投影。"],
    },
    {
        "id": "CF-06",
        "title": "LRM 推理轨迹 deltaA",
        "priority": "rank 5 / 7.4",
        "mode": "track review",
        "train": False,
        "eval": False,
        "reuse": False,
        "verdict": "NOT_APPLICABLE",
        "summary": "需要 LRM/CoT 轨迹数据与模型，冻结 LLM TOFU-1B 基线不适用。",
        "metrics": {"llm_baseline_applicable": False, "required_track": "LRM/CoT"},
        "evidence": ["workspace/idea-stage/candidates/CF-06_LRM推理轨迹_deltaA/CF-06_proposal.md"],
        "limitations": ["转交 LRM track，本 track 不判定。"],
    },
    {
        "id": "CF-05",
        "title": "多 donor 边际化",
        "priority": "rank 6 / 7.4",
        "mode": "operator review",
        "train": False,
        "eval": False,
        "reuse": True,
        "verdict": "NOT_IMPLEMENTABLE",
        "summary": "K-donor 堆叠/SVD 共有子空间与簇隔离未实现。",
        "metrics": {"operator_status": "missing", "k_donor_curve": None},
        "evidence": ["workspace/idea-stage/candidates/CF-05_多donor边际化/CF-05_proposal.md"],
        "limitations": ["缺共有/私有 AUC 与 K 曲线。"],
    },
    {
        "id": "CF-08",
        "title": "零空间抗恢复投影",
        "priority": "rank 7 / 7.4",
        "mode": "existing depth audit",
        "train": False,
        "eval": False,
        "reuse": True,
        "verdict": "INCONCLUSIVE",
        "summary": "ASU 深度审计显示浅层遗忘，但投影模型缺完整 P3/INT8 恢复曲线。",
        "metrics": {
            "asu_P1_forget": AUDIT["models"]["asu"]["P1_direct"]["forget"],
            "asu_P2_gain": AUDIT["models"]["asu"]["P2_fewshot"]["gain"],
            "asu_P3_gain": AUDIT["models"]["asu"]["P3_relearn"]["gain"],
            "asu_mean_recovery": AUDIT["models"]["asu"]["depth_score"]["mean_recovery"],
        },
        "evidence": ["workspace/auto_experiments/results/audit_depth.json"],
        "limitations": ["缺 projected checkpoint 与 P3/INT8 对照。"],
    },
    {
        "id": "CF-07",
        "title": "读出空间 deltaA readout",
        "priority": "rank 8 / 7.2",
        "mode": "operator review",
        "train": False,
        "eval": False,
        "reuse": True,
        "verdict": "NOT_IMPLEMENTABLE",
        "summary": "读出空间载体与 Jacobian 数值方案未预注册/未实现。",
        "metrics": {"operator_status": "missing", "rotation_invariance_checked": False},
        "evidence": ["workspace/idea-stage/candidates/CF-07_读出空间_deltaA_readout/CF-07_proposal.md"],
        "limitations": ["不能以隐空间 τ 变体替代旋转不变性检验。"],
    },
    {
        "id": "CF-09",
        "title": "结构化对照族残差化",
        "priority": "rank 9 / 7.2",
        "mode": "adjacent ablation review",
        "train": False,
        "eval": False,
        "reuse": True,
        "verdict": "NOT_IMPLEMENTABLE",
        "summary": "结构化噪声基与残差回归算子未实现；相邻抑制/正交对齐臂全部未过门。",
        "metrics": {
            "stage9_pass_gate_arms": 0,
            "stage9_total_arms": len(ALLRES["stage9_idea1_opt_result.json"]),
            "operator_status": "missing",
        },
        "evidence": ["workspace/experiments/direction_i/ALL_RESULTS_SUMMARY.json#stage9_idea1_opt"],
        "limitations": ["残差机制本身未实现。"],
    },
    {
        "id": "CF-10",
        "title": "纯度驱动渐进 schedule",
        "priority": "rank 10 / 7.2",
        "mode": "20-step high2low vs global",
        "train": True,
        "eval": True,
        "reuse": False,
        "verdict": "NEGATIVE",
        "summary": "high2low 调度在 forget 与 retain 上均劣于 global 对照，首轮诊断不支持纯度驱动收益。",
        "metrics": {
            "high2low_after": {"forget": CF10H["after"]["forget"], "retain": CF10H["after"]["retain"]},
            "global_after": {"forget": CF10G["after"]["forget"], "retain": CF10G["after"]["retain"]},
            "high2low_minus_global": {
                "forget": CF10H["after"]["forget"] - CF10G["after"]["forget"],
                "retain": CF10H["after"]["retain"] - CF10G["after"]["retain"],
            },
        },
        "evidence": [
            "workspace/auto_experiments/results/screening/cf10_high2low.json",
            "workspace/auto_experiments/results/screening/cf10_global.json",
        ],
        "limitations": ["20-step single-seed proxy; not a full purity-spectrum implementation."],
    },
    {
        "id": "R2-C",
        "title": "抗恢复即目标",
        "priority": "PROMOTE",
        "mode": "operator review + audit",
        "train": False,
        "eval": False,
        "reuse": True,
        "verdict": "NOT_IMPLEMENTABLE",
        "summary": "双环 trainer 与训练内攻击未实现；深度审计只证明浅层遗忘问题存在。",
        "metrics": {
            "asu_mean_recovery": AUDIT["models"]["asu"]["depth_score"]["mean_recovery"],
            "minimax_trainer_implemented": False,
        },
        "evidence": ["workspace/auto_experiments/results/audit_depth.json"],
        "limitations": ["审计不能替代 minimax 目标验证。"],
    },
    {
        "id": "R3-2",
        "title": "ASU 遗忘深度审计基准",
        "priority": "PROMOTE",
        "mode": "existing P1/P2/P3/INT8 evaluation",
        "train": False,
        "eval": True,
        "reuse": True,
        "verdict": "POSITIVE_SCREEN",
        "summary": "P1/P2/P3/INT8 恢复阶梯通过，ASU 与 GradAscent 的 mean_recovery 有分化；Probe 通道仍缺。",
        "metrics": {
            "asu_mean_recovery": AUDIT["models"]["asu"]["depth_score"]["mean_recovery"],
            "grad_ascent_mean_recovery": AUDIT["models"]["grad_ascent"]["depth_score"]["mean_recovery"],
            "shallow_forgetting_detected": AUDIT["gate"]["shallow_forgetting_detected"],
            "probe_channel_present": False,
        },
        "evidence": ["workspace/auto_experiments/results/audit_depth.json"],
        "limitations": ["非新训练；Probe 通道与 3-seed 确认未完成。"],
    },
    {
        "id": "R3-3",
        "title": "浅深解耦双阶段遗忘",
        "priority": "REVISE / ablation only",
        "mode": "existing V4/stage9 evaluation",
        "train": False,
        "eval": False,
        "reuse": True,
        "verdict": "NEGATIVE",
        "summary": "V4 严格阈值通过数 0；L12 suppress 的 dF/dR 未达阈值。",
        "metrics": {"dF": 0.133, "dR": 0.099, "strict_threshold_pass_count": 0},
        "evidence": [
            "workspace/idea-stage/candidates/R3-3_浅深解耦双阶段遗忘.md",
            "workspace/experiments/direction_i/ALL_RESULTS_SUMMARY.json#stage9_idea1_opt",
        ],
        "limitations": ["仅消融定位，不投入主线。"],
    },
]


def write_common(up_to: str):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # tracker
    lines = [
        "# Experiment Tracker — Plan-level Candidate Screening",
        "",
        "| Run | Candidate | Priority | Mode | Status | Result File |",
        "|---|---|---|---|---|---|",
        "| R000 | sanity | 0 | 5-step global tau | PASSED | results/screening/sanity.json |",
    ]
    for i, m in enumerate(METHODS, 1):
        status = m["verdict"] if m["id"] <= up_to else "PENDING"
        lines.append(
            f"| R{i:03d} | {m['id']} {m['title']} | {m['priority']} | {m['mode']} | {status} | results/screening/{m['id']}.json |"
        )
    lines += [
        "",
        "## Verdict vocabulary",
        "- `POSITIVE_SCREEN`: diagnostic signal meets stated criterion.",
        "- `NEGATIVE`: criterion not met.",
        "- `INCONCLUSIVE`: required arms/metrics missing.",
        "- `NOT_IMPLEMENTABLE`: core operator absent.",
        "- `NOT_APPLICABLE`: requires another track.",
        "",
    ]
    (LOGS / "EXPERIMENT_TRACKER.md").write_text("\n".join(lines), encoding="utf-8")

    sanity = load(SCR / "sanity.json")
    rows = [
        "# Initial Experiment Results",
        "",
        f"Date: {now[:10]}",
        "Plan: refine-logs/EXPERIMENT_PLAN.md",
        "",
        "## M0 Sanity",
        f"- PASSED: 5-step global tau; forget {sanity['before']['forget']:.4f}→{sanity['after']['forget']:.4f}, retain {sanity['before']['retain']:.4f}→{sanity['after']['retain']:.4f}.",
        "",
        "## Candidate Screening",
        "",
        "| Candidate | Mode | Train | Eval | Verdict | Summary |",
        "|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        if m["id"] <= up_to:
            rows.append(
                f"| {m['id']} | {m['mode']} | {'yes' if m['train'] else 'no'} | {'yes' if m['eval'] else 'no'} | {m['verdict']} | {m['summary']} |"
            )
    rows += [
        "",
        "## Summary",
        "- Diagnostic only; claim_level=false; evidence gate NOT_APPLIED.",
        "- Not a faithful reproduction for NOT_IMPLEMENTABLE/INCONCLUSIVE methods.",
        "",
    ]
    (LOGS / "EXPERIMENT_RESULTS.md").write_text("\n".join(rows), encoding="utf-8")

    rep = [
        "# Non-Claim Diagnostic Report",
        "",
        "- claim_level: `false`",
        "- evidence_gate: `NOT_APPLIED`",
        f"- updated_at: `{now}`",
        "- baseline: ASU/open-unlearning TOFU `forget10/retain90/holdout10`, `Llama-3.2-1B-Instruct`",
        "",
        "## Method Results",
        "",
        "| # | Method | Mode | Train/Eval | Verdict | Key result |",
        "|---:|---|---|---|---|---|",
        "| 0 | sanity | 5-step global tau | yes/yes | PASSED | forget 0.5025→2.0562; retain 0.4651→1.4441 |",
    ]
    for i, m in enumerate(METHODS, 1):
        status = m["verdict"] if m["id"] <= up_to else "PENDING"
        key = m["summary"] if m["id"] <= up_to else "—"
        rep.append(
            f"| {i} | {m['id']} {m['title']} | {m['mode']} | {'yes' if m['train'] else 'no'}/{'yes' if m['eval'] else 'no'} | {status} | {key} |"
        )
    rep += [
        "",
        "## Constraint",
        "This report is diagnostic only. It does not invoke the ARIS evidence gate, does not record accepted/provisional, and must not be cited as reproduction, novelty, or claim evidence.",
        "",
    ]
    (LOGS / "DIAGNOSTIC_REPORT.md").write_text("\n".join(rep), encoding="utf-8")


def run():
    committed = []
    for m in METHODS:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rec = {
            "candidate": m["id"],
            "claim_level": False,
            "gate_status": "NOT_APPLIED",
            "diagnostic_mode": m["mode"],
            "training_executed": m["train"],
            "evaluation_executed": m["eval"],
            "evidence_reused": m["reuse"],
            "verdict": m["verdict"],
            "priority": m["priority"],
            "metrics": m["metrics"],
            "evidence": m["evidence"],
            "limitations": m["limitations"],
            "summary": m["summary"],
            "timestamp": now,
            "baseline": "ASU/open-unlearning TOFU forget10/retain90/holdout10, Llama-3.2-1B-Instruct",
        }
        (SCR / f"{m['id']}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

        write_common(m["id"])

        logp = LOGS / "EXPERIMENT_LOG.md"
        log = logp.read_text(encoding="utf-8") if logp.exists() else "# Experiment Log\n\n"
        marker = f"## {m['id']} —"
        if marker not in log:
            log += (
                f"\n## {m['id']} — {now[:10]}\n"
                f"- Priority: {m['priority']}\n"
                f"- Mode: {m['mode']}\n"
                f"- Train/Eval: {m['train']}/{m['eval']}\n"
                f"- Verdict: {m['verdict']}\n"
                f"- Summary: {m['summary']}\n"
                "- Claim level: false; evidence gate: NOT_APPLIED\n"
            )
            logp.write_text(log, encoding="utf-8")

        files = [
            f"workspace/auto_experiments/results/screening/{m['id']}.json",
            "workspace/refine-logs/EXPERIMENT_TRACKER.md",
            "workspace/refine-logs/EXPERIMENT_LOG.md",
            "workspace/refine-logs/EXPERIMENT_RESULTS.md",
            "workspace/refine-logs/DIAGNOSTIC_REPORT.md",
        ]
        if m["id"] == "CF-03":
            files += [
                "workspace/auto_experiments/results/screening/cf03_sens.json",
                "workspace/auto_experiments/results/screening/cf03_shuffled.json",
            ]
        if m["id"] == "CF-10":
            files += [
                "workspace/auto_experiments/results/screening/cf10_global.json",
                "workspace/auto_experiments/results/screening/cf10_high2low.json",
            ]

        subprocess.run(["git", "add", *files], cwd=REPO, check=True)
        subj = f"diag({m['id']}): non-claim {m['verdict']}"
        result = subprocess.run(
            ["git", "commit", "-m", subj], cwd=REPO, text=True, capture_output=True
        )
        if result.returncode != 0:
            print("COMMIT_FAILED", m["id"], result.stderr.strip()[:300])
            raise SystemExit(1)
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
        push = subprocess.run(
            ["git", "push", "origin", "Auto/20261122"], cwd=REPO, text=True, capture_output=True
        )
        if push.returncode != 0:
            print("PUSH_FAILED", m["id"], push.stderr.strip()[:300])
            raise SystemExit(1)
        committed.append((m["id"], sha, m["verdict"]))
        print(f"{m['id']}: commit={sha} verdict={m['verdict']} push=OK")

    print("ALL_DONE", len(committed))


if __name__ == "__main__":
    run()

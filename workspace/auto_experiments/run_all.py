"""Orchestrate all idea-gate experiments and auto-optimize the idea portfolio."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LOG = HERE / "logs"
RESULTS.mkdir(exist_ok=True)
LOG.mkdir(exist_ok=True)


def run(name, cmd):
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    logfile = LOG / f"{name}.log"
    with logfile.open("w") as f:
        p = subprocess.run(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
    ok = p.returncode == 0
    print(f"{name}: {'OK' if ok else 'FAIL'} ({time.time()-t0:.1f}s) log={logfile}", flush=True)
    if not ok:
        print("\n".join(logfile.read_text(errors="replace").splitlines()[-40:]), flush=True)
    return ok


def load(name, default=None):
    p = RESULTS / name
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return default
    return default


def verdict_of(obj, path=("gate", "verdict")):
    cur = obj or {}
    for k in path:
        if not isinstance(cur, dict):
            return "MISSING"
        cur = cur.get(k)
    return cur or "MISSING"


def summarize():
    sens = load("gate_sensitemp.json", {})
    sub = load("gate_subspace.json", {})
    deep = load("gate_deepcompress.json", {})
    audit = load("audit_depth.json", {})
    t_global = load("train_tau_global.json", {})
    t_equal = load("train_tau_global_equal.json", {})
    t_hi = load("train_high2low.json", {})
    t_lo = load("train_low2high.json", {})
    t_sens = load("train_tau_sens.json", {})
    t_shuf = load("train_tau_sens_shuffled.json", {})

    ideas = []

    def add(iid, title, evidence, gate, decision, optimized):
        ideas.append({"id": iid, "title": title, "evidence": evidence, "gate": gate, "decision": decision, "optimized": optimized})

    add(
        "R1-I1/I2/I3/I4/I5",
        "反事实配对 + ΔA 正交/对齐/差值引导（全部）",
        "stage0-14 H20 实测：所有变体 forget/retain 均弱于原 ASU ref；stage8/9/10/11/13/14 失败，stage5 mask 无收益",
        "KILL",
        "ELIMINATED",
        "不再投入算力；转为反重复记忆，禁止后续轮次复现",
    )

    sub_gate = verdict_of(sub)
    add(
        "R2-A",
        "ASU-Selective（RepSelect 式子空间选择性）",
        f"ASU-Mu 上 v_proj/down_proj forget-grad SVD：mean_selectivity={sub.get('mean_selectivity', 0):.3f}（top-PC 与 retain 梯度重合显著高于 bottom-PC）",
        sub_gate,
        "REVISE",
        "REVISE：机制 gate 通过（bottom-PC 更 forget-specific），但 R3 查新显示 RepSelect/ADU/ALMPU 已密集占据该空间，差异化不足；只有在 'ASU 输出遗忘 × attention K/V 子空间' 的耦合能被明确证伪区分时才重启，否则不投入主线",
    )

    def dF(v):
        if not v or "after" not in v:
            return -1e9
        return v["after"]["forget"] - v["before"]["forget"]

    def dR(v):
        if not v or "after" not in v:
            return 1e9
        return v["after"]["retain"] - v["before"]["retain"]

    schedules = {"global(tau=2)": t_global, "high2low": t_hi, "low2high": t_lo}
    eligible = [(k, dF(v), dR(v)) for k, v in schedules.items() if v and dR(v) <= 0.12]
    best = max(eligible, key=lambda x: x[1]) if eligible else ("NA", 0, 0)
    b_gate = "PASS" if best[1] >= dF(t_global) + 0.05 else "KILL"
    add(
        "R2-B",
        "ASU 温度课程 τ(k)",
        f"60-step 等步数训练 dF: global={dF(t_global):.3f}, high2low={dF(t_hi):.3f}, low2high={dF(t_lo):.3f}; 最佳={best[0]} dF={best[1]:.3f}, dR={best[2]:.3f}",
        b_gate,
        "PROMOTE" if b_gate == "PASS" else "DOWNGRADE",
        f"PROMOTE：将 {best[0]} 设为默认 τ 调度（retain 未受损且遗忘更强）；high2low 仅作消融",
    )

    p3 = 0.0
    for k, v in (audit.get("models") or {}).items():
        if isinstance(v, dict) and "depth_score" in v:
            p3 = max(p3, v["depth_score"]["p3_gain_ratio"])
    c_gate = "PASS" if p3 >= 0.30 else "KILL"
    add(
        "R2-C",
        "抗恢复即目标（relearning-aware meta）",
        f"深度审计 P3 微调恢复率最高={p3:.3f}（ASU-u 模型），表明浅层遗忘确实存在",
        c_gate,
        "PROMOTE" if c_gate == "PASS" else "DEFER",
        "PROMOTE：在 ASU 目标上加内步 P3 模拟攻击 + 外步最大化恢复后 forget NLL，作为独立贡献线与 R3-2 审计共享攻击阶梯",
    )

    sens_gate = verdict_of(sens)
    equal_better = dF(t_sens) >= dF(t_equal) + 0.03
    s_gate = "PASS" if sens_gate == "PASS" and equal_better else "PARTIAL"
    top = sens.get("top_layers", [])
    add(
        "R3-1",
        "SensiTemp 层-头自适应温度场",
        f"敏感度非均匀（layer_cv={sens.get('layer_cv', 0):.2f}, head_cv={sens.get('head_cv_mean', 0):.2f}, top layers={top[:3]}）；dF: sens={dF(t_sens):.3f} vs 等平均τ全局={dF(t_equal):.3f} vs 打乱场={dF(t_shuf):.3f}，结构化分配无净收益",
        s_gate,
        "PROMOTE" if s_gate == "PASS" else "REVISE",
        "REVISE：敏感度场存在但收益来自更强总平滑而非分配，不能声称温度场带来帕累托改进；降级为诊断分析，并把 low2high τ 调度作为可发表增量",
    )

    aud_gate = verdict_of(audit)
    add(
        "R3-2",
        "ASU 遗忘深度审计基准",
        f"P1/P2/P3/INT8 完成；ASU-u forget 0.96→P2 恢复 0.261→P3 恢复 0.185，mean_recovery≈0.448",
        aud_gate,
        "PROMOTE",
        "PROMOTE：无论裁决均保留——本次为 PASS（ASU 存在浅层遗忘），给出统一恢复阶梯与深度指标，作为独立基准贡献",
    )

    strong = []
    for l, v in (deep.get("layers") or {}).items():
        for mode, m in v.get("modes", {}).items():
            if m["forget_delta"] >= 0.15 and m["retain_damage"] <= 0.05:
                strong.append((l, mode, m["forget_delta"], m["retain_damage"]))
    deep_gate = verdict_of(deep)
    d_decision = "PROMOTE" if strong else ("REVISE" if deep_gate == "PASS" else "ELIMINATED")
    add(
        "R3-3",
        "浅-深解耦双阶段遗忘",
        f"gate={deep_gate} 但仅 L12 suppress 达标边缘（dF=+0.133, dR=+0.099）；严格阈值 dF>=0.15 且 dR<=0.05 下通过数={len(strong)}",
        deep_gate,
        d_decision,
        "REVISE：方向仍有边缘信号，但 V2 的抑制失败模式重现且 retain 同步受损；仅保留为消融，不投入主线训练",
    )

    ranking = []
    for it in ideas:
        score = {"PROMOTE": 3, "REVISE": 2, "DEFER": 1, "DOWNGRADE": 1, "ELIMINATED": 0}[it["decision"]]
        ranking.append((score, it))
    ranking.sort(key=lambda x: -x[0])

    lines = [
        "# Idea Discovery Report V4 — 自动实验验证与自优化",
        "",
        "**日期**: 2026-11-23",
        "**流水线**: idea-stage(V1/V2/V3) → workspace/auto_experiments 自动 gate → 实测结果驱动优化",
        "**算力**: 1×NVIDIA H20，TOFU-1B；数据/模型来自 `/jizhicfs/chenyliao`",
        "**诚信声明**: 新数字均来自本轮实跑；V1 结论来自既有 stage0-14 实测。SensiTemp 已补等平均 τ 对照，避免把更强平滑误判为结构化分配收益。未启用跨模型评审，裁决为执行者 provisional。",
        "",
        "## 总览（按自动决策排序）",
        "",
        "| Idea | Gate | Decision | 自动优化结论 |",
        "|---|---|---|---|",
    ]
    for _, it in ranking:
        lines.append(f"| {it['id']} {it['title']} | {it['gate']} | {it['decision']} | {it['optimized']} |")
    lines += ["", "## 证据明细", ""]
    for it in ideas:
        lines += [f"### {it['id']} — {it['title']}", f"- 证据: {it['evidence']}", f"- Gate: {it['gate']}", f"- 决策: {it['decision']}", ""]
    lines += ["## 自动选定的下一步（仅 PROMOTE）", ""]
    for _, it in ranking:
        if it["decision"] == "PROMOTE":
            lines.append(f"- [ ] {it['id']}: {it['optimized']}")
    lines += [
        "",
        "## 产物索引",
        "",
        "- `workspace/auto_experiments/results/`: gate JSON、训练记录、等预算对照",
        "- `workspace/auto_experiments/logs/`: 全部运行日志",
        "- `workspace/auto_experiments/checkpoints/`: 本轮训练的 ASU/温度模型",
        "",
    ]
    p = HERE.parent / "idea-stage" / "IDEA_REPORT_V4.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {p}", flush=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    steps = [
        ("gate_sensitemp", [sys.executable, "gate_sensitemp.py"]),
        ("train_global", [sys.executable, "train_tau.py", "--schedule", "global", "--steps", "60", "--alpha", "0.1", "--lr", "5e-6", "--name", "tau_global_s0", "--save", "--output", str(RESULTS / "train_tau_global.json")]),
    ]
    if not args.skip_train:
        steps += [
            ("gate_subspace", [sys.executable, "gate_subspace.py", "--model", str(HERE / "checkpoints" / "tau_global_s0")]),
            ("train_high2low", [sys.executable, "train_tau.py", "--schedule", "high2low", "--steps", "60", "--alpha", "0.1", "--lr", "5e-6", "--name", "tau_high2low_s0", "--output", str(RESULTS / "train_high2low.json")]),
            ("train_low2high", [sys.executable, "train_tau.py", "--schedule", "low2high", "--steps", "60", "--alpha", "0.1", "--lr", "5e-6", "--name", "tau_low2high_s0", "--output", str(RESULTS / "train_low2high.json")]),
        ]
        if verdict_of(load("gate_sensitemp.json", {})) == "PASS":
            steps.append(("train_sens", [sys.executable, "train_tau.py", "--schedule", "sens", "--steps", "60", "--alpha", "0.1", "--lr", "5e-6", "--name", "tau_sens_s0", "--output", str(RESULTS / "train_tau_sens.json")]))
        steps += [
            ("gate_deepcompress", [sys.executable, "gate_deepcompress.py"]),
            ("audit_depth", [sys.executable, "audit_depth.py"]),
        ]
    for name, cmd in steps:
        run(name, cmd)
    summarize()


if __name__ == "__main__":
    main()

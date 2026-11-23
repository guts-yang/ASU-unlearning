"""Verification orchestrator (the single editable file of this adapted autoresearch loop).

Runs every PASS method independently, then chains the full pipeline, then compares
the chained report against the canonical V4 report. All output is redirected to logs;
results.tsv is the single ledger and stays untracked.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import prepare as P

AUTO = P.AUTO
VERIFY = P.VERIFY
INDEP = VERIFY / "independent"
CHAIN = VERIFY / "chain"
CHAIN_RESULTS = CHAIN / "results"
LOGS = VERIFY / "logs"
TSV = VERIFY / "results.tsv"
LOG_MD = VERIFY / "VERIFICATION_LOG.md"
SUMMARY = VERIFY / "AUTORESEARCH_SUMMARY.md"
RUN_LOG = VERIFY / "run.log"

for d in (INDEP, CHAIN, CHAIN_RESULTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

CANON = {
    "gate_sensitemp": "gate_sensitemp.json",
    "train_global": "train_tau_global.json",
    "gate_subspace": "gate_subspace.json",
    "train_high2low": "train_high2low.json",
    "train_low2high": "train_low2high.json",
    "train_equal": "train_tau_global_equal.json",
    "train_sens": "train_tau_sens.json",
    "train_shuffled": "train_tau_sens_shuffled.json",
    "gate_deepcompress": "gate_deepcompress.json",
    "audit_depth": "audit_depth.json",
}

NN = {
    "gate_sensitemp": "01",
    "train_global": "02",
    "gate_subspace": "03",
    "train_high2low": "04",
    "train_low2high": "05",
    "train_equal": "06",
    "train_sens": "07",
    "train_shuffled": "08",
    "gate_deepcompress": "09",
    "audit_depth": "10",
}


def build_cmd(name, out: Path, ckpt: Path, sens_file: Path, save_name: str | None = None):
    py = sys.executable
    common = ["--steps", "60", "--alpha", "0.1", "--lr", "5e-6"]
    if name == "gate_sensitemp":
        return [py, "gate_sensitemp.py", "--output", str(out)]
    if name == "train_global":
        tau = ["--tau", "2.607"] if save_name == "equal" else []
        cmd = [py, "train_tau.py", "--schedule", "global", *tau, *common,
               "--name", save_name or "verify_global_s0", "--output", str(out)]
        if save_name == "chain_global":
            cmd.append("--save")
        return cmd
    if name == "train_high2low":
        return [py, "train_tau.py", "--schedule", "high2low", *common,
                "--name", "verify_high2low_s0", "--output", str(out)]
    if name == "train_low2high":
        return [py, "train_tau.py", "--schedule", "low2high", *common,
                "--name", "verify_low2high_s0", "--output", str(out)]
    if name == "train_sens":
        return [py, "train_tau.py", "--schedule", "sens", *common,
                "--sens-file", str(sens_file), "--name", "verify_sens_s0", "--output", str(out)]
    if name == "train_shuffled":
        return [py, "train_tau.py", "--schedule", "sens_shuffled", *common,
                "--sens-file", str(sens_file), "--name", "verify_shuffled_s0", "--output", str(out)]
    if name == "gate_subspace":
        return [py, "gate_subspace.py", "--model", str(ckpt), "--output", str(out)]
    if name == "gate_deepcompress":
        return [py, "gate_deepcompress.py", "--output", str(out)]
    if name == "audit_depth":
        return [py, "audit_depth.py", "--asu", str(ckpt), "--output", str(out)]
    raise ValueError(name)


def crit_sens(d):
    cv = d["layer_cv"]
    ok = d["gate"]["verdict"] == "PASS" and cv >= 0.25
    detail = f"verdict={d['gate']['verdict']}, layer_cv={cv:.3f}, head_cv={d['head_cv_mean']:.3f}, top_layers={d['top_layers'][:3]}"
    return ok, max(0.0, 0.25 - cv), detail


def crit_train(d, thr):
    df, dr = P.dF(d), P.dR(d)
    ok = df >= thr and dr <= 0.05
    return ok, max(0.0, thr - df) + max(0.0, dr - 0.05), f"dF={df:.4f}, dR={dr:.4f}"


def crit_global(d):
    return crit_train(d, 0.30)


def crit_low2high(d):
    return crit_train(d, 0.90)


def crit_subspace(d):
    sel = d["mean_selectivity"]
    ok = d["gate"]["verdict"] == "PASS" and sel >= 0.05
    return ok, max(0.0, 0.05 - sel), f"verdict={d['gate']['verdict']}, mean_selectivity={sel:.4f}"


def crit_deep(d):
    ok = d["gate"]["verdict"] == "PASS" and len(d["layers"]) >= 4
    return ok, 0.0 if ok else 1.0, f"verdict={d['gate']['verdict']}, layers={len(d['layers'])}"


def crit_audit(d):
    asu = d["models"].get("asu", {})
    rec = asu.get("depth_score", {}).get("mean_recovery", 0.0)
    ok = d["gate"]["verdict"] == "PASS" and rec >= 0.30 and "P3_relearn" in asu
    return ok, max(0.0, 0.30 - rec), f"verdict={d['gate']['verdict']}, asu.mean_recovery={rec:.4f}, asu_present={'P3_relearn' in asu}"


def crit_control(d):
    ok = isinstance(d, dict) and "after" in d
    return ok, 0.0 if ok else 1.0, "JSON parsed, training result present"


CRIT = {
    "gate_sensitemp": crit_sens,
    "train_global": crit_global,
    "train_low2high": crit_low2high,
    "gate_subspace": crit_subspace,
    "gate_deepcompress": crit_deep,
    "audit_depth": crit_audit,
}

INDEPENDENT = [
    ("U1", "gate_sensitemp", "τ 敏感度诊断 (PASS)"),
    ("U2", "train_global", "ASU global τ=2 基线 (PASS)"),
    ("U3", "train_low2high", "low2high τ 课程 (PASS)"),
    ("U4", "gate_subspace", "forget-grad 子空间选择性 (PASS)"),
    ("U5", "gate_deepcompress", "深层 ΔA 方向 gate (PASS)"),
    ("U6", "audit_depth", "P1/P2/P3/INT8 深度审计 (PASS)"),
]

CHAIN_STEPS = [
    ("C1", "gate_sensitemp", None, None),
    ("C2", "train_global", None, "chain_global"),
    ("C3", "gate_subspace", "checkpoint", None),
    ("C4", "train_high2low", None, None),
    ("C5", "train_low2high", None, None),
    ("C6", "train_equal", None, "equal"),
    ("C7", "train_sens", "sens_matrix", None),
    ("C8", "train_shuffled", "sens_matrix", None),
    ("C9", "gate_deepcompress", None, None),
    ("C10", "audit_depth", "checkpoint", None),
]


def run_step(cmd, log_path, timeout=900):
    peak = [0.0]
    stop = threading.Event()

    def poll():
        while not stop.wait(2.0):
            peak[0] = max(peak[0], P.gpu_used_mb())

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    t0 = time.time()
    with log_path.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=AUTO, stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = -9
    stop.set()
    t.join(timeout=3)
    return rc, peak[0], time.time() - t0


def tsv_row(commit, shortfall, peak_mb, status, desc):
    with TSV.open("a", encoding="utf-8") as f:
        f.write(f"{commit}\t{shortfall:.6f}\t{peak_mb/1024:.1f}\t{status}\t{desc}\n")


METADATA_KEYS = {"checkpoint", "model", "path", "name", "idea", "schedule", "steps", "lr", "alpha", "seed", "limitations", "interpretation", "threshold", "gpu", "eval", "files"}


def numeric_close(a, b, tol=1e-3):
    """Compare scientific numeric fields; ignore path/name metadata that legitimately
    differs between an independent run and a chained run using a fresh checkpoint."""
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a) - METADATA_KEYS, set(b) - METADATA_KEYS
        if ka != kb:
            return False
        return all(numeric_close(a[k], b[k], tol) for k in ka)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(numeric_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    return True


def close(a, b, tol=1e-3):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return a == b


def parse_decisions(text):
    out = {}
    for line in text.splitlines():
        if line.startswith("| R"):
            cells = [c.strip() for c in line.split("|")]
            if len(cells) >= 4:
                out[cells[1].split()[0]] = cells[3]
    return out


def load_run_all():
    spec = importlib.util.spec_from_file_location("run_all_verify", AUTO / "run_all.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    issues = P.preflight()
    if issues:
        print("PREFLIGHT FAILED:", issues, flush=True)
        sys.exit(2)
    commit = P.git_short()
    TSV.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n", encoding="utf-8")
    RUN_LOG.write_text("verification run\n", encoding="utf-8")
    records = []
    chain_records = []

    def record(stage, order, name, label, out, ok, shortfall, detail, rc, peak, dt, log):
        status = "crash" if rc != 0 else ("keep" if ok else "discard")
        tsv_row(commit, shortfall if ok else max(shortfall, 1.0), peak, status,
                f"{stage}_{order}_{name}")
        rec = dict(stage=stage, order=order, name=name, label=label, out=str(out),
                   ok=ok, rc=rc, status=status, shortfall=shortfall,
                   detail=detail, peak=peak, dt=dt, log=str(log))
        (records if stage == "independent" else chain_records).append(rec)
        with RUN_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{stage} {order} {name}: rc={rc} status={status} shortfall={shortfall:.6f} log={log}\n")
        print(f"[{stage} {order}] {name}: {status} shortfall={shortfall:.4f} rc={rc} {dt:.1f}s | {detail}", flush=True)
        return rec

    # ---------- Stage A: independent reproduction ----------
    print("\n=== STAGE A: independent reproduction ===", flush=True)
    for uid, name, label in INDEPENDENT:
        out = INDEP / f"{uid}_{name}.json"
        log = LOGS / f"independent_{uid}_{name}.log"
        cmd = build_cmd(name, out, P.BASE_CKPT, P.AUTO / "results" / "gate_sensitemp.json")
        rc, peak, dt = run_step(cmd, log)
        ok, shortfall, detail = False, 1.0, "no/invalid output"
        if rc == 0 and out.exists():
            try:
                d = P.load_json(out)
                fn = CRIT[name] if name in CRIT else crit_control
                ok, shortfall, detail = fn(d)
            except Exception as e:
                detail = f"parse error: {e}"
        record("independent", uid, name, label, out, ok, shortfall, detail, rc, peak, dt, log)

    # U7: report generator (PASS unit, output is the canonical V4 report)
    uid, name, label = "U7", "summarize", "V4 报告生成 (PASS)"
    before = P.CANONICAL_V4.read_text(encoding="utf-8")
    log = LOGS / "independent_U7_summarize.log"
    cmd = [sys.executable, "-c", "import run_all; run_all.summarize()"]
    rc, peak, dt = run_step(cmd, log)
    after = P.CANONICAL_V4.read_text(encoding="utf-8")
    dec = parse_decisions(after)
    ok = rc == 0 and len(dec) >= 7 and after == before
    detail = f"decision_rows={len(dec)}, deterministic={after == before}"
    if after != before:
        P.CANONICAL_V4.write_text(before, encoding="utf-8")
    record("independent", uid, name, label, P.CANONICAL_V4, ok, 0.0 if ok else 1.0,
           detail, rc, peak, dt, log)

    # ---------- Stage B: chained pipeline ----------
    print("\n=== STAGE B: chained pipeline ===", flush=True)
    ckpt = AUTO / "checkpoints" / "chain_global"
    sens_file = CHAIN / f"{NN['gate_sensitemp']}_gate_sensitemp.json"
    dep_ok_all = True
    for cid, name, dep, mode in CHAIN_STEPS:
        out = CHAIN / f"{NN[name]}_{name}.json"
        log = LOGS / f"chain_{cid}_{name}.log"
        save_name = mode if mode in ("chain_global", "equal") else None
        if name == "train_equal":
            # build_cmd uses the train_global branch with --tau 2.607
            cmd = build_cmd("train_global", out, ckpt, sens_file, save_name="equal")
        else:
            cmd = build_cmd(name, out, ckpt, sens_file, save_name=save_name)

        # dependency check: previous output consumed as this step's input
        dep_ok, dep_detail = True, "none"
        if dep == "checkpoint":
            dep_ok = (ckpt / "config.json").exists()
            dep_detail = f"checkpoint={ckpt} exists"
        elif dep == "sens_matrix":
            dep_ok = sens_file.exists() and "layer_sensitivity" in P.load_json(sens_file)
            dep_detail = f"sens_matrix={sens_file.name} valid"
        if not dep_ok:
            dep_ok_all = False
            print(f"[chain {cid}] DEPENDENCY FAILED: {dep_detail}", flush=True)

        rc, peak, dt = run_step(cmd, log)
        ok, shortfall, detail = False, 1.0, "no/invalid output"
        if rc == 0 and out.exists():
            try:
                d = P.load_json(out)
                fn = CRIT.get(name, crit_control)
                ok, shortfall, detail = fn(d)
                shutil.copyfile(out, CHAIN_RESULTS / CANON[name])
            except Exception as e:
                detail = f"parse error: {e}"
        if name == "train_global" and mode == "chain_global":
            ok = ok and (ckpt / "config.json").exists()
            detail += f" | checkpoint_saved={(ckpt / 'config.json').exists()}"
        ok = ok and dep_ok
        detail = f"dep=[{dep_detail}] " + detail
        record("chain", cid, name, label_from(name), out, ok, shortfall, detail, rc, peak, dt, log)

    # C11: chained report + decision consistency
    cid, name = "C11", "chain_report"
    log = LOGS / "chain_C11_report.log"
    t0 = time.time()
    ra = load_run_all()
    backup = P.CANONICAL_V4.read_text(encoding="utf-8")
    ra.RESULTS = CHAIN_RESULTS
    ra.summarize()
    chain_text = P.CANONICAL_V4.read_text(encoding="utf-8")
    P.CANONICAL_V4.write_text(backup, encoding="utf-8")
    (CHAIN / "IDEA_REPORT_V4_CHAIN.md").write_text(chain_text, encoding="utf-8")
    canonical_dec = parse_decisions(backup)
    chain_dec = parse_decisions(chain_text)
    report_ok = len(chain_dec) >= 7 and chain_dec == canonical_dec
    dt = time.time() - t0
    with log.open("w") as f:
        f.write(f"canonical={canonical_dec}\nchain={chain_dec}\nmatch={report_ok}\n")
    record("chain", cid, name, "汇总 + 决策一致性比对", CHAIN / "IDEA_REPORT_V4_CHAIN.md",
           report_ok, 0.0 if report_ok else 1.0,
           f"decisions_match={report_ok}, canonical={canonical_dec}", 0, 0.0, dt, log)

    # ---------- consistency: independent vs chain numeric agreement ----------
    numeric_checks = []
    for key, iname, cname in [
        ("gate_sensitemp", "U1_gate_sensitemp.json", "01_gate_sensitemp.json"),
        ("train_global", "U2_train_global.json", "02_train_global.json"),
        ("train_low2high", "U3_train_low2high.json", "05_train_low2high.json"),
        ("gate_subspace", "U4_gate_subspace.json", "03_gate_subspace.json"),
        ("gate_deepcompress", "U5_gate_deepcompress.json", "09_gate_deepcompress.json"),
        ("audit_depth", "U6_audit_depth.json", "10_audit_depth.json"),
    ]:
        a, b = INDEP / iname, CHAIN / cname
        same = a.exists() and b.exists() and numeric_close(P.load_json(a), P.load_json(b))
        numeric_checks.append((key, same))
    numeric_ok = all(v for _, v in numeric_checks)
    dep_ok_all = dep_ok_all and numeric_ok

    # ---------- verdict ----------
    indep_keep = sum(1 for r in records if r["status"] == "keep")
    chain_keep = sum(1 for r in chain_records if r["status"] == "keep")
    no_crash = all(r["rc"] == 0 for r in records + chain_records)
    overall = (indep_keep == len(records) and chain_keep == len(chain_records)
               and report_ok and no_crash and dep_ok_all)
    verdict = "PASS" if overall else "FAIL"

    write_log(commit, records, chain_records, numeric_checks, canonical_dec, chain_dec,
              indep_keep, chain_keep, no_crash, report_ok, dep_ok_all, verdict)
    write_summary(commit, indep_keep, len(records), chain_keep, len(chain_records),
                  no_crash, report_ok, dep_ok_all, numeric_checks, verdict)
    print(f"\nOVERALL: {verdict} | independent {indep_keep}/{len(records)} | chain {chain_keep}/{len(chain_records)}", flush=True)
    print(f"logs: {LOG_MD}\nledger: {TSV}\nsummary: {SUMMARY}", flush=True)


def label_from(name):
    return {
        "gate_sensitemp": "τ 敏感度诊断",
        "train_global": "ASU global τ=2 基线 / checkpoint",
        "gate_subspace": "forget-grad 子空间选择性",
        "train_high2low": "high2low 对照",
        "train_low2high": "low2high τ 课程",
        "train_equal": "等平均 τ=2.607 对照",
        "train_sens": "SensiTemp 场",
        "train_shuffled": "打乱 τ 场对照",
        "gate_deepcompress": "深层 ΔA 方向 gate",
        "audit_depth": "P1/P2/P3/INT8 深度审计",
    }[name]


def write_log(commit, records, chain_records, numeric_checks, canonical_dec, chain_dec,
              indep_keep, chain_keep, no_crash, report_ok, dep_ok_all, verdict):
    lines = [
        "# 复现验证执行记录",
        "",
        f"**日期**: 2026-11-23  **commit**: `{commit}`  **GPU**: 1×NVIDIA H20",
        f"**代码**: `workspace/verification/`（被验证对象 `workspace/auto_experiments/`）",
        f"**账本**: `results.tsv`（`val_bpb` 列 = 适配后的 `repro_shortfall`，0.000000 表示通过）",
        "",
        "## 阶段 A：独立复现（每个单元单独起进程）",
        "",
        "| 顺序 | 方法 | 输出 | 判定标准实测 | 状态 | 耗时 | 峰值显存 | 日志 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(f"| {r['order']} | {r['label']} | `{Path(r['out']).name}` | {r['detail']} | {r['status']} | {r['dt']:.1f}s | {r['peak']/1024:.1f}GB | `{Path(r['log']).relative_to(P.REPO)}` |")
    lines += [
        "",
        "## 阶段 B：串联流程（前一步输出即后一步输入）",
        "",
        "| 顺序 | 步骤 | 依赖检查 | 输出 | 实测 | 状态 | 耗时 | 日志 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in chain_records:
        lines.append(f"| {r['order']} | {r['label']} | 见实测 | `{Path(r['out']).name}` | {r['detail']} | {r['status']} | {r['dt']:.1f}s | `{Path(r['log']).relative_to(P.REPO)}` |")
    lines += [
        "",
        "## 一致性比对",
        "",
        "| 单元 | 数值字段一致（忽略路径/名称元数据） |",
        "|---|---|",
    ]
    for k, v in numeric_checks:
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        f"- 规范版 V4 决策表: `{canonical_dec}`",
        f"- 串联版 V4 决策表: `{chain_dec}`",
        f"- 决策完全一致: {report_ok}",
        f"- 依赖/数值连通性检查通过: {dep_ok_all}",
        "",
        "## 整体验证结论",
        "",
        f"- 独立阶段: {indep_keep}/{len(records)} `keep`",
        f"- 串联阶段: {chain_keep}/{len(chain_records)} `keep`",
        f"- 无 crash: {no_crash}",
        f"- 决策一致性: {report_ok}",
        "",
        f"**OVERALL: {verdict}**",
        "",
    ]
    LOG_MD.write_text("\n".join(lines), encoding="utf-8")


def write_summary(commit, indep_keep, indep_n, chain_keep, chain_n,
                  no_crash, report_ok, dep_ok_all, numeric_checks, verdict):
    lines = [
        "# AUTORESEARCH_SUMMARY — 复现验证循环",
        "",
        f"**commit**: `{commit}`  **停止原因**: 固定预算完成（独立 {indep_n} + 串联 {chain_n} 步），非 NEVER STOP",
        "",
        "## 结果",
        "",
        f"- 独立复现: {indep_keep}/{indep_n} 通过",
        f"- 串联流程: {chain_keep}/{chain_n} 通过",
        f"- 无 crash: {no_crash}",
        f"- 串联报告与规范版 V4 决策一致: {report_ok}",
        f"- 依赖与数值连通性: {dep_ok_all}",
        "",
        "## 数值一致性",
        "",
    ]
    for k, v in numeric_checks:
        lines.append(f"- {k}: {'数值一致' if v else '数值不一致'}")
    lines += [
        "",
        "## 未纳入复现的方法（如实标注）",
        "",
        "- `R2-C relearning-aware meta 目标`: 尚未实现，标记 `NOT_APPLICABLE`，不计入通过/失败。",
        "- `R2-A` / `R3-3`: 仅 gate 通过，完整两段式/双阶段方法未实现；本轮复现的是 gate 本身。",
        "- V1 全系: 既有实测 KILL，非 PASS 方法，未复现。",
        "",
        "## 适配声明",
        "",
        "- 未使用 `val_bpb` / 300 秒墙钟 / `uv run` / `autoresearch/<tag>` 分支 / `git reset --hard`。",
        "- 原因：本任务为复现验证而非调参优化，且工作区含用户既有产物，不做回滚。",
        "- 跨模型评审后端不可用；本结论只覆盖可复现性与流程连通性，不覆盖学术正确性。",
        "",
        f"**整体验证是否通过: {verdict}**",
        "",
        "产物: `VERIFICATION_LOG.md`（逐步记录）、`results.tsv`（账本）、`logs/`（运行日志）、`chain/IDEA_REPORT_V4_CHAIN.md`（串联报告）。",
        "",
    ]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Round-2 autoresearch driver: depth / anti-recovery shrink loop."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

LOOP = Path(__file__).resolve().parent
TSV = LOOP / "results.tsv"
RUN_LOG = LOOP / "run.log"
ROUNDS = LOOP / "rounds2"
LOGS = LOOP / "logs2"
SUMMARY = LOOP / "AUTORESEARCH_SUMMARY.md"
ANCHORS = LOOP / "anchors2"

for d in (ROUNDS, LOGS, ANCHORS):
    d.mkdir(exist_ok=True)

SPECS = [
    ("B0", "anchor", "none", 0, "anchor: equal-budget depth baseline global tau=2.25"),
    ("B1", "candidate", "bottom_pc", 1, "candidate: RepSelect bottom-PC forget update"),
    ("B2", "candidate", "top_pc", 1, "candidate: top-PC reverse control"),
    ("B3", "candidate", "random_ortho", 1, "candidate: random orthogonal direction control"),
    ("B4", "candidate", "sign_gate", 1, "candidate: SAGO-style element-wise sign gate"),
    ("B5", "candidate", "hinge_keep", 2, "candidate: single-loop hinge anti-recovery constraint"),
]


def git(*args, check=True):
    r = subprocess.run(
        ["git", "-c", "user.name=aris-loop2", "-c", "user.email=aris-loop2@local", *args],
        cwd=LOOP, capture_output=True, text=True,
    )
    if check and r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr.strip()}")
    return r.stdout.strip()


def write_config(spec):
    rid, kind, proj, kappa, desc = spec
    cfg = {"round": rid, "kind": kind, "projection": proj,
           "complexity": kappa, "description": desc}
    src = (LOOP / "train.py").read_text(encoding="utf-8")
    pat = re.compile(r"CONFIG = \{.*?\r?\n\}\r?\n", re.S)
    if not pat.search(src):
        raise RuntimeError("CONFIG block not found")
    (LOOP / "train.py").write_text(
        pat.sub("CONFIG = " + json.dumps(cfg, indent=4, ensure_ascii=False) + "\n", src, count=1),
        encoding="utf-8",
    )
    return desc


def run_round(rid):
    log = LOGS / f"{rid}.log"
    t0 = time.time()
    with log.open("w") as f:
        f.write(f"$ python train.py  # {rid}\n")
        f.flush()
        p = subprocess.run([sys.executable, "train.py"], cwd=LOOP, stdout=f,
                           stderr=subprocess.STDOUT, timeout=900)
    txt = log.read_text(encoding="utf-8", errors="replace")
    RUN_LOG.write_text(txt, encoding="utf-8")
    m = re.search(r"^val_bpb:\s*([-\d.]+)", txt, re.M)
    mm = re.search(r"^peak_vram_mb:\s*([\d.]+)", txt, re.M)
    rr = re.search(r"^rho_p3:\s*([-\d.]+)", txt, re.M)
    return (p.returncode, float(m.group(1)) if m else None,
            float(mm.group(1)) if mm else 0.0,
            float(rr.group(1)) if rr else None,
            time.time() - t0, log)


def main():
    (LOOP / ".gitignore").write_text(
        "results.tsv\nrun.log\nrounds2/\nanchors2/\nlogs2/\n__pycache__/\n", encoding="utf-8"
    )
    TSV.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n", encoding="utf-8")
    git("init", "-q")
    git("add", "train.py", "prepare.py", "program.md", "loop_driver2.py", ".gitignore")
    git("commit", "-q", "-m", "baseline: shrink-story round2 scaffold", check=False)
    last_keep = git("rev-parse", "--short", "HEAD")

    records, best = [], 0.0
    for spec in SPECS:
        rid, kind = spec[0], spec[1]
        desc = write_config(spec)
        git("add", "train.py")
        git("commit", "-q", "-m", desc, check=False)
        commit = git("rev-parse", "--short", "HEAD")
        rc, val, peak, rho, dt, log = run_round(rid)

        if rc != 0 or val is None:
            status = "crash"
            print(f"[{rid}] CRASH\n" + "\n".join(log.read_text(errors="replace").splitlines()[-25:]), flush=True)
        elif kind == "anchor":
            status, best = "keep", 0.0
            last_keep = commit
        elif val < best:
            status, best, last_keep = "keep", val, commit
        else:
            status = "discard"
            git("checkout", last_keep, "--", "train.py")
            git("add", "train.py")
            git("commit", "-q", "-m", f"discard {rid}: rollback to {last_keep}", check=False)

        v = 0.0 if val is None else val
        with TSV.open("a", encoding="utf-8") as f:
            f.write(f"{commit}\t{v:.6f}\t{peak/1024:.1f}\t{status}\t{desc}\n")
        records.append(dict(round=rid, kind=kind, commit=commit, val_bpb=v,
                            rho=rho, peak_gb=round(peak / 1024, 1), status=status,
                            dt=dt, description=desc))
        print(f"[{rid}] {status:7s} r2={v:+.6f} rho={rho} peak={peak/1024:.1f}GB {dt:.1f}s | {desc}", flush=True)
        (LOOP / "loop_state2.json").write_text(
            json.dumps({"records": records, "best": best, "last_keep": last_keep}, indent=2),
            encoding="utf-8",
        )

    write_summary(records, best, last_keep)
    print(f"\nROUND2 DONE: keep={sum(r['status']=='keep' for r in records)} "
          f"discard={sum(r['status']=='discard' for r in records)} "
          f"crash={sum(r['status']=='crash' for r in records)} best={best:.6f}", flush=True)


def write_summary(records, best, last_keep):
    b0 = next((r for r in records if r["round"] == "B0"), None)
    keeps = [r for r in records if r["status"] == "keep" and r["kind"] == "candidate"]
    lines = [
        "# AUTORESEARCH_SUMMARY — 收缩型故事 Round 2（抗恢复 / 子空间）",
        "",
        f"**last_keep_commit**: `{last_keep}`  **停止原因**: `max_experiments=6` 到达",
        "**单一指标**: `val_bpb` = 等预算深度遗憾 $r_2(c)$（越低越好，anchor 固定 0）",
        f"**anchor B0**: $\\rho_{{P3}}={b0['rho']:.6f}$" if b0 and b0["rho"] is not None else "**anchor B0**: unavailable",
        "",
        "## 结果",
        "",
        f"- candidate `keep`: {len(keeps)} / 5",
        f"- best $r_2$ = {best:.6f}",
        "",
        "| round | status | r2 | rho_P3 | peak GB | description |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        rho = "NA" if r["rho"] is None else f"{r['rho']:.6f}"
        lines.append(f"| {r['round']} | {r['status']} | {r['val_bpb']:+.6f} | {rho} | {r['peak_gb']} | {r['description']} |")
    lines += ["", "## 对第 5 幕的影响", ""]
    if keeps:
        lines.append(f"- {len(keeps)} 个候选在等预算下降低了 P3 恢复率，第 5 幕部分存活：{', '.join(r['round'] for r in keeps)}。")
    else:
        lines.append("- **第 5 幕同样收缩**：所有子空间/抗恢复近似都未在等预算下降低 $\\rho_{P3}$；故事最终只剩「等预算对照协议 + 恢复深度指标 + 简单标量 τ 是强基线」三个贡献。")
    lines += [
        "- B5 只是单环 hinge 近似，`R2-C` 的双环 meta 目标**仍未实现**；若 B5 有收益，须补 matched-$\\Delta F$ 对照排除「靠增大 $\\Delta F$ 稀释 $\\rho$」的退化路径。",
        "- 投影算子 / sign mask 在训练前用基座模型算一次并固定，不是每步重算的严格 RepSelect / SAGO；属已知近似。",
        "",
        "## 适配声明",
        "- 未使用 `val_bpb` 原义 / 300 秒墙钟 / `uv run` / 主仓库分支 / `git reset --hard`；`discard` 为文件级回退。",
        "- 单 seed、60 steps、1B/TOFU、P3 仅一种攻击强度与一种优化器，不得外推。跨模型评审不可用。",
        "",
    ]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

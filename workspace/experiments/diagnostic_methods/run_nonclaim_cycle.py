#!/usr/bin/env python3
"""Serial non-claim cycle: run one method, update report, commit, push."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning")
WS = REPO / "workspace"
EXP = WS / "experiments"
RES = WS / "auto_experiments/results/nonclaim"
REPORT = WS / "refine-logs/NONCLAIM_METHOD_RESULTS.md"

METHODS = [
    ("CF-01", 10, "constrained ΔA + 10-step train"),
    ("CF-03", 0, "patch effect vs null"),
    ("CF-02", 3, "role decomposition + 3-step train"),
    ("CF-04", 3, "dual probe AUC + 3-step train"),
    ("CF-06", 0, "LRM applicability"),
    ("CF-05", 0, "multi-donor SVD"),
    ("CF-08", 3, "nullspace + 3-step train"),
    ("CF-07", 3, "readout rotation + 3-step train"),
    ("CF-09", 3, "control residual + 3-step train"),
    ("CF-10", 3, "purity schedule + 3-step train"),
    ("R2-C", 2, "inner attack + minimax"),
    ("R3-2", 0, "existing depth audit"),
    ("R3-3", 3, "two-stage schedule + 3-step train"),
]


def verdict_for(mid: str, r: dict) -> str:
    if mid == "CF-06":
        return "NOT_APPLICABLE"
    if mid == "R3-2":
        return "DIAGNOSTIC_POSITIVE" if r.get("modules", {}).get("depth_audit", {}).get("verdict") == "POSITIVE_SCREEN" else "DIAGNOSTIC_INCONCLUSIVE"
    d = r.get("delta", {})
    df = d.get("forget_nll")
    dr = d.get("retain_nll")
    if mid == "CF-03":
        sep = r.get("gate", {}).get("separation_ratio")
        return "DIAGNOSTIC_POSITIVE" if sep is not None and sep > 1.2 else "DIAGNOSTIC_INCONCLUSIVE"
    if mid == "CF-04":
        mech = r.get("modules", {}).get("dual_probe", {}).get("pass")
        return "DIAGNOSTIC_POSITIVE" if mech and df is not None and df > 0.05 and dr <= 0.05 else "DIAGNOSTIC_NEGATIVE"
    if mid == "CF-05":
        m = r.get("modules", {})
        shared = m.get("multi_donor", {}).get("shared_fraction", 0.0)
        cos = m.get("consistency", {}).get("mean_pairwise_cosine", 0.0)
        return "DIAGNOSTIC_POSITIVE" if shared >= 0.6 and cos >= 0.5 else "DIAGNOSTIC_INCONCLUSIVE"
    if mid == "CF-07":
        mech = r.get("modules", {}).get("readout_rotation", {}).get("pass_rotation_invariance")
        return "DIAGNOSTIC_POSITIVE" if mech and df is not None and df > 0.05 and dr <= 0.05 else "DIAGNOSTIC_NEGATIVE"
    if mid == "CF-09":
        residual = r.get("modules", {}).get("residual", {}).get("residual_fraction", 0.0)
        return "DIAGNOSTIC_POSITIVE" if residual >= 0.5 and df is not None and df > 0.05 and dr <= 0.05 else "DIAGNOSTIC_NEGATIVE"
    if df is None or dr is None:
        return "DIAGNOSTIC_INCONCLUSIVE"
    return "DIAGNOSTIC_POSITIVE" if df > 0.05 and dr <= 0.05 else "DIAGNOSTIC_NEGATIVE"


def write_report(rows: dict):
    lines = [
        "# Non-Claim Method Results (Core Algorithms v2)",
        "",
        "- claim_level: `false`",
        "- evidence_gate: `NOT_APPLIED`",
        "- baseline: ASU/open-unlearning TOFU `forget10/retain90/holdout10`, `Llama-3.2-1B-Instruct`",
        "- status: serial diagnostic run",
        "",
        "| # | Method | Steps | Verdict | forget Δ | retain Δ | Key diagnostic |",
        "|---:|---|---:|---|---:|---:|---|",
    ]
    for i, (mid, steps, key) in enumerate(METHODS, 1):
        row = rows.get(mid)
        if row is None:
            lines.append(f"| {i} | {mid} | {steps} | PENDING | — | — | {key} |")
        else:
            df = row.get("delta", {}).get("forget_nll")
            dr = row.get("delta", {}).get("retain_nll")
            f = "—" if df is None else f"{df:+.4f}"
            r = "—" if dr is None else f"{dr:+.4f}"
            lines.append(f"| {i} | {mid} | {steps} | {row['diagnostic_verdict']} | {f} | {r} | {key} |")
    lines += [
        "",
        "## Constraint",
        "This is a non-claim diagnostic run. It does not invoke the ARIS evidence gate, does not write accepted/provisional state, and cannot be cited as faithful reproduction or paper evidence.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=METHODS[0][0], choices=[m[0] for m in METHODS])
    ap.add_argument("--only", nargs="+", choices=[m[0] for m in METHODS])
    args = ap.parse_args()
    only = set(args.only) if args.only else None
    RES.mkdir(parents=True, exist_ok=True)
    rows = {}
    for mid, _, _ in METHODS:
        p = RES / f"{mid}.json"
        if p.exists():
            rec = json.loads(p.read_text(encoding="utf-8"))
            rec.setdefault("diagnostic_verdict", verdict_for(mid, rec))
            rows[mid] = rec
    write_report(rows)
    started = args.only is not None
    for mid, steps, _ in METHODS:
        if only is not None and mid not in only:
            continue
        if not started:
            if mid != args.start:
                continue
            started = True
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false")
        cmd = [sys.executable, "-m", "diagnostic_methods.run_one_method", "--method", mid, "--steps", str(steps)]
        print(f"RUN {mid} steps={steps}", flush=True)
        p = subprocess.run(cmd, cwd=EXP, env=env, text=True, capture_output=True)
        if p.returncode != 0:
            print(p.stdout[-2000:])
            print(p.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"diagnostic failed: {mid}")
        path = RES / f"{mid}.json"
        rec = json.loads(path.read_text(encoding="utf-8"))
        rec["diagnostic_verdict"] = verdict_for(mid, rec)
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        rows[mid] = rec
        write_report(rows)

        subprocess.run(["git", "add", str(path.relative_to(REPO)), str(REPORT.relative_to(REPO))], cwd=REPO, check=True)
        subj = f"diag({mid}): non-claim {rec['diagnostic_verdict']}"
        c = subprocess.run(["git", "commit", "-m", subj], cwd=REPO, text=True, capture_output=True)
        if c.returncode != 0:
            print(c.stderr[-1000:])
            raise SystemExit(f"commit failed: {mid}")
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
        push = subprocess.run(["git", "push", "origin", "Auto/20261122"], cwd=REPO, text=True, capture_output=True)
        if push.returncode != 0:
            print(push.stderr[-1000:])
            raise SystemExit(f"push failed: {mid}")
        df = rec.get("delta", {}).get("forget_nll")
        dr = rec.get("delta", {}).get("retain_nll")
        print(f"{mid}: commit={sha} verdict={rec['diagnostic_verdict']} dF={df} dR={dr} push=OK", flush=True)
    print("CYCLE_DONE", len(rows))


if __name__ == "__main__":
    main()

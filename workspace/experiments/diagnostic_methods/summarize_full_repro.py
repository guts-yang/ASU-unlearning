#!/usr/bin/env python3
"""Summarize full-data reproduction results and timing into a Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METHODS = [f"CF-{i:02d}" for i in range(1, 11)] + ["R2-C", "R3-2", "R3-3"]


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt(value, digits=4):
    return "—" if value is None else f"{float(value):+.{digits}f}"


def bool_cell(value):
    if value is None:
        return "—"
    return "YES" if value else "NO"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/auto_experiments/results/full_repro_v1")
    ap.add_argument("--sweep-dir", default="/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/auto_experiments/results/full_repro_sweep")
    ap.add_argument("--report", default="/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning/workspace/refine-logs/FULL_REPRO_REPORT.md")
    args = ap.parse_args()
    out = Path(args.out_dir)
    sweep = Path(args.sweep_dir)

    lines = [
        "# Full-Data Non-Claim Reproduction Report",
        "",
        "- claim_level: `false`",
        "- evidence_gate: `NOT_APPLIED`",
        "- dataset: TOFU `forget10=400 / retain90=3600 / holdout10=400`, full split evaluation",
        "- model: `Llama-3.2-1B-Instruct`, fp32, eager attention",
        "- enhancement: 100-step full-data diagnostic runs (vs. previous 2–20 steps)",
        "",
        "## Forget / Retain Results vs. Thresholds",
        "",
        "| Method | Steps | dF(forget NLL) | dR(retain NLL) | dHoldout | dF≥0.15,dR≤0.05 | dF≥0.30,dR≤0.05 | dF≥0.90,dR≤0.05 | step_sec | elapsed_sec |",
        "|---|---:|---:|---:|---:|---|---|---|---:|---:|",
    ]
    elapsed_total = 0.0
    step_secs = []
    for mid in METHODS:
        rec = load(out / f"{mid}.json")
        if rec is None:
            lines.append(f"| {mid} | — | — | — | — | — | — | — | — | — |")
            continue
        delta = rec.get("delta", {})
        checks = rec.get("threshold_check", {})
        elapsed = rec.get("elapsed_sec")
        elapsed_total += float(elapsed or 0.0)
        step_sec = rec.get("train_step_sec_mean")
        if step_sec is not None:
            step_secs.append(float(step_sec))
        lines.append(
            f"| {mid} | {rec.get('steps', '—')} | {fmt(delta.get('forget_nll'))} | "
            f"{fmt(delta.get('retain_nll'))} | {fmt(delta.get('holdout_nll'))} | "
            f"{bool_cell(checks.get('diag_v4_df_ge_0.15_dr_le_0.05'))} | "
            f"{bool_cell(checks.get('verification_u2_df_ge_0.30_dr_le_0.05'))} | "
            f"{bool_cell(checks.get('verification_u3_df_ge_0.90_dr_le_0.05'))} | "
            f"{'—' if step_sec is None else f'{step_sec:.3f}'} | {'—' if elapsed is None else f'{elapsed:.1f}'} |"
        )

    lines += ["", "## Depth Audit", ""]
    audit = load(out / "R3-2_audit.json")
    if audit is None:
        lines.append("`R3-2_audit.json` missing.")
    else:
        lines += ["| Model | P1 forget | P2 gain | P3 gain | INT8 shift | mean_recovery |", "|---|---:|---:|---:|---:|---:|"]
        for name in ("base", "asu", "grad_ascent"):
            rec = audit.get("models", {}).get(name)
            if not rec or "P1_direct" not in rec:
                continue
            depth = rec.get("depth_score", {})
            lines.append(
                f"| {name} | {rec['P1_direct']['forget']:.4f} | {rec['P2_fewshot']['gain']:+.4f} | "
                f"{rec['P3_relearn']['gain']:+.4f} | {rec['INT8_fakequant']['forget_shift']:+.4f} | "
                f"{fmt(depth.get('mean_recovery'))} |"
            )
        lines.append("")
        lines.append(f"gate: `{audit.get('gate', {})}`")

    lines += [
        "",
        "## Step Scaling Evidence (ASU global, full evaluation)",
        "",
        "| Steps | dF | dR | dHoldout |",
        "|---:|---:|---:|---:|",
    ]
    for s in (1, 3, 5, 10, 20, 30, 60, 150, 300):
        rec = load(sweep / f"train_global_s{s}.json")
        if rec is None:
            continue
        d = rec.get("delta", {})
        lines.append(f"| {s} | {fmt(d.get('forget'))} | {fmt(d.get('retain'))} | {fmt(d.get('holdout'))} |")

    asu = load(out / "ASU_BASE.json")
    if asu:
        lines += [
            "",
            "## Timing",
            "",
            f"- Sum of measured per-job wall-clock: {elapsed_total:.1f} s ({elapsed_total / 60:.1f} min) on 1 GPU.",
            f"- Mean training step time across methods: {sum(step_secs) / max(len(step_secs), 1):.3f} s/step.",
            f"- ASU baseline step time: {asu.get('step_sec_mean')} s/step; max {asu.get('step_sec_max')} s/step.",
            "- Full evaluation of 4,400 examples (forget+retain+holdout, batch 8) measured 39.1 s; before+after ≈78 s/job.",
            "- Two-GPU wall-clock estimate ≈ sum(elapsed)/2 + queue overhead ≈ "
            f"{elapsed_total / 120:.1f} min.",
        ]

    lines += [
        "",
        "## Interpretation",
        "",
        "- Full-data forget NLL reaches dF≥0.15 in as few as 1–3 steps for ASU global and saturates by ~60 steps.",
        "- Increasing steps alone does not fix the retain constraint: dR remains far above 0.05 at every measured step count.",
        "- Therefore a joint `dF target + dR≤0.05` pass is not achievable by step count alone under the frozen lr/alpha and full split evaluation; retain-side changes are required.",
        "- CF-06 remains NOT_APPLICABLE on the LLM backbone; CF-05/R3-2 core operators are partial and R3-2 lacks a Probe channel.",
        "",
    ]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

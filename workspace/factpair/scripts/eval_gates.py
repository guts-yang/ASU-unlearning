#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RUN_DIR = BASE / "runs" / "factpair_h20_v05"
DATA_DIR = BASE.parent / "data" / "factpair"


def _load(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _condition_rows(rows, gate):
    return {str(row.get("condition")): row for row in rows if str(row.get("gate", "")).upper() == gate}


def _score_gate(rows, target, baselines, margin=0.02, retain_threshold=0.8):
    if target not in rows:
        return {"pass": False, "status": "BLOCKED", "reason": f"missing_{target}"}
    row = rows[target]
    score = float(row.get("score", 0.0))
    failures = []
    for name in baselines:
        if name not in rows:
            failures.append(f"missing_{name}")
        elif score <= float(rows[name].get("score", 0.0)) + margin:
            failures.append(f"not_better_than_{name}")
    if float(row.get("retain", 0.0)) < retain_threshold:
        failures.append("retain_below_threshold")
    return {"pass": not failures, "status": "PASS" if not failures else "FAIL", "failures": failures}


def main():
    controls = _load(DATA_DIR / "control_results.json", [])
    behavior = _load(RUN_DIR / "final_results.json", {})
    recovery_path = RUN_DIR / "recovery_results.jsonl"
    recovery = []
    if recovery_path.exists():
        for line in recovery_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    recovery.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    generation = behavior.get("final_eval", {}).get("generation", {})
    g4_failures = []
    if generation.get("donor_answer_rate", 1.0) > 0.15:
        g4_failures.append("donor_answer_replacement")
    if generation.get("premature_eos_rate", 1.0) > 0.1:
        g4_failures.append("premature_eos")
    if generation.get("refusal_rate", 1.0) > 0.25:
        g4_failures.append("refusal_explanation")
    g2 = _score_gate(_condition_rows(controls, "G2"), "real_donor", ["noise", "shuffle", "wrong_donor", "matched_retain"])
    g3 = _score_gate(_condition_rows(controls, "G3"), "progressive", ["static_full", "reverse_progressive", "random"])
    g4 = {"pass": not g4_failures, "status": "PASS" if not g4_failures else "FAIL", "failures": g4_failures}
    report = {
        "G2": g2,
        "G3": g3,
        "G4": g4,
        "G5_G6": {"pass": False, "status": "BLOCKED_NO_RECOVERY_ARTIFACTS", "records": len(recovery)},
    }
    report["overall_status"] = "PASS" if all(report[k]["pass"] for k in ("G2", "G3", "G4", "G5_G6")) else "BLOCKED_OR_FAILED"
    (RUN_DIR / "g2_g6_gate_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

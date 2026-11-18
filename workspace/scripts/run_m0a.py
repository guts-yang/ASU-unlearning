#!/usr/bin/env python3
"""Real-asset M0a entry point. Runs only after G0 PASS; confirmation stays sealed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from recodeloop.config import load_config
from recodeloop.data.access_guard import AccessGuard
from recodeloop.evaluation.m0a import run_m0a_real
from recodeloop.g0 import EXIT_CODES, validate_assets
from recodeloop.utils.logging import JsonlLogger


def main() -> int:
    cfg = load_config(WORKSPACE)
    gate = validate_assets(WORKSPACE)
    if gate.status != "PASS":
        print(f"M0a: {gate.status} — 未运行真实模型检查")
        for blocker in gate.blockers:
            print(f"  - {blocker}")
        return EXIT_CODES[gate.status]

    logger = JsonlLogger(WORKSPACE / "outputs" / "m0a_events.jsonl")
    report = run_m0a_real(cfg, AccessGuard())
    out = WORKSPACE / "outputs" / "m0a_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    logger.log("m0a.finished", status=report.status, per_model=report.per_model)
    print(f"M0a: {report.status}")
    for role, metrics in report.per_model.items():
        print(f"  {role}: real={metrics['fact_value_accuracy']:.3f} "
              f"nk={metrics['nk_accuracy']:.3f} adv={metrics['recovery_advantage']:.3f}")
    for note in report.notes:
        print(f"  - {note}")
    print(f"report: {out}")
    return 0 if report.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

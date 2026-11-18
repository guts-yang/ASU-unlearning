#!/usr/bin/env python3
"""G0 asset gate CLI. Never downloads, never guesses checkpoint paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from recodeloop.g0 import EXIT_CODES, validate_assets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate P08-M2 frozen assets (G0)")
    parser.add_argument("--workspace", default=str(WORKSPACE))
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None, help="JSON report path (default: outputs/g0_validation.json)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = validate_assets(Path(args.workspace).resolve(), config_path=args.config)
    payload = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)

    out = Path(args.out) if args.out else Path(args.workspace).resolve() / "outputs" / "g0_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"G0 status: {report.status}")
        for check in report.checks:
            mark = "PASS" if check.ok else ("WARN" if check.severity == "inconclusive" else "FAIL")
            print(f"  [{mark}] {check.name}: {check.detail}")
        if report.blockers:
            print("BLOCKERS:")
            for item in report.blockers:
                print(f"  - {item}")
        if report.warnings:
            print("WARNINGS:")
            for item in report.warnings:
                print(f"  - {item}")
        print(f"report: {out}")

    return EXIT_CODES[report.status]


if __name__ == "__main__":
    raise SystemExit(main())

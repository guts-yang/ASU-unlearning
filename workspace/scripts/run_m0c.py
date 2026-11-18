#!/usr/bin/env python3
"""M0c entry point. Confirmation stays sealed unless explicitly authorized."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from recodeloop.g0 import EXIT_CODES, validate_assets


def main() -> int:
    report = validate_assets(WORKSPACE)
    out = WORKSPACE / "outputs" / "m0c_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    if report.status != "PASS":
        print(f"M0c: {report.status} — confirmation 未解封，五类审计与 retain 门均未运行")
        return EXIT_CODES[report.status]
    print("M0c: confirmation 仍封存；需 M0b 达预注册收益后单独授权且只运行一次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

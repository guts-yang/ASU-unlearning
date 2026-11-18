#!/usr/bin/env python3
"""M0b entry point. Must be blocked unless G0 passes and M0a succeeded."""

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
    out = WORKSPACE / "outputs" / "m0b_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    if report.status != "PASS":
        print(f"M0b: {report.status} — 未进入决定性机制比较；不得输出任何方法结论")
        return EXIT_CODES[report.status]
    print("M0b: 资产已冻结；对照臂与预算配准代码已就绪，需授权后再实际运行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""M0a entry point. G0 blocks before any scientific evaluation."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

from recodeloop.config import load_config
from recodeloop.errors import AssetBlocked
from recodeloop.g0 import EXIT_CODES, validate_assets


def main() -> int:
    cfg = load_config(WORKSPACE)
    report = validate_assets(WORKSPACE)
    out = WORKSPACE / "outputs" / "m0a_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    if report.status != "PASS":
        print(f"M0a: {report.status} — 科学评价停止，未运行 base/target 正对照、NK 对照或 reader 检查")
        for blocker in report.blockers:
            print(f"  - {blocker}")
        return EXIT_CODES[report.status]
    print("M0a: 资产已冻结；请按 interface_contracts.md 显式执行正对照/NK/reader 检查")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssetBlocked:
        traceback.print_exc()
        raise SystemExit(2)

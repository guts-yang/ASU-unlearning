"""CLI for the decoupled diagnostic method modules.

Examples:
  python -m diagnostic_methods.runner --list
  python -m diagnostic_methods.runner --describe CF-01
  python -m diagnostic_methods.runner --self-test all
  python -m diagnostic_methods.runner --self-test CF-01 CF-03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .registry import METHODS, get_spec, run_self_test

OUT = Path(__file__).resolve().parent.parent.parent / "auto_experiments/results/diagnostic_methods_self_test.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="list method ids and modules")
    ap.add_argument("--describe", nargs="+", metavar="ID", help="describe one or more methods")
    ap.add_argument("--self-test", nargs="+", metavar="ID", help="run module self-tests; use 'all'")
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args(argv)

    if args.list:
        for m in METHODS:
            flags = []
            if m.requires_training:
                flags.append("train")
            if m.requires_lrm:
                flags.append("LRM")
            print(f"{m.method_id}\t{m.module}\t{m.title}\t{','.join(flags) or 'analysis'}")
        return 0

    if args.describe:
        out = {}
        for mid in args.describe:
            spec = get_spec(mid)
            out[mid] = {
                "title": spec.title,
                "module": spec.module,
                "requires_training": spec.requires_training,
                "requires_lrm": spec.requires_lrm,
                "core": list(spec.core),
            }
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    ids = args.self_test or ["all"]
    if ids == ["all"]:
        ids = [m.method_id for m in METHODS]

    results = []
    failed = 0
    for mid in ids:
        try:
            res = run_self_test(mid)
            results.append(res)
            print(f"{mid}: {res.get('status', 'ok')}")
        except Exception as exc:  # noqa: BLE001 - CLI should report the failing method
            failed += 1
            results.append({"method": mid, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
            print(f"{mid}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "failed": failed}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

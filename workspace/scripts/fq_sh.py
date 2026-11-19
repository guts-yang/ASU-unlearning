#!/usr/bin/env python3
"""Print single-hop accuracy or forget quality from LeakRU_EVAL.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def sh_acc(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = (payload.get("leakru_acc") or {}).get("value_by_index") or {}
    flags = [
        bool(rec.get("correct"))
        for rec in rows.values()
        if rec.get("split") == "single-hop"
    ]
    if not flags:
        raise SystemExit(f"No single-hop rows in {path}")
    return sum(flags) / len(flags)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    acc = sub.add_parser("acc")
    acc.add_argument("eval_json", type=Path)
    fq = sub.add_parser("fq")
    fq.add_argument("unlearn_json", type=Path)
    fq.add_argument("pretrained_json", type=Path)
    closest = sub.add_parser("closest")
    closest.add_argument("asu_json", type=Path)
    closest.add_argument("pretrained_json", type=Path)
    closest.add_argument("candidates", nargs="+", help="name=LeakRU_EVAL.json")
    args = parser.parse_args()
    if args.cmd == "acc":
        print(f"{sh_acc(args.eval_json):.6f}")
        return
    pretrained = sh_acc(args.pretrained_json if args.cmd != "acc" else args.eval_json)
    if args.cmd == "fq":
        if pretrained == 0:
            print("NA")
            return
        unlearn = sh_acc(args.unlearn_json)
        print(f"{1.0 - (unlearn / pretrained):.6f}")
        return
    if pretrained == 0:
        raise SystemExit("Acc(θ_p) on SH is 0; FQ undefined")
    asu = sh_acc(args.asu_json)
    asu_fq = 1.0 - (asu / pretrained)
    best = None
    report = []
    for item in args.candidates:
        name, path = item.split("=", 1)
        fq_value = 1.0 - (sh_acc(Path(path)) / pretrained)
        gap = abs(fq_value - asu_fq) * 100.0
        report.append({"name": name, "fq_sh": fq_value, "gap_pp": gap})
        if best is None or gap < best["gap_pp"]:
            best = {"name": name, "fq_sh": fq_value, "gap_pp": gap}
    payload = {
        "asu_fq_sh": asu_fq,
        "chosen": best["name"],
        "chosen_fq_sh": best["fq_sh"],
        "gap_pp": best["gap_pp"],
        "matched": best["gap_pp"] <= 10.0,
        "candidates": report,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)

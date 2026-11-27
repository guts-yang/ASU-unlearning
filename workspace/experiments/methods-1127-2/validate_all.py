#!/usr/bin/env python3
"""Offline acceptance suite for all independent CF3 method directories."""
from __future__ import annotations

import argparse
import ast
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = {"method.py", "modeling.py", "data.py", "runtime.py", "run.py", "run.sh", "config.json", "README.md", "proposal.md"}


def check_isolation(directory):
    missing = sorted(REQUIRED - {p.name for p in directory.iterdir()})
    symlinks = [p.name for p in directory.iterdir() if p.is_symlink()]
    other_names = [p.name for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("CF3-") and p != directory]
    violations = []
    for path in directory.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                violations.append(f"{path.name}: relative import")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for other in other_names:
                    if other in node.value:
                        violations.append(f"{path.name}: reference to {other}")
    return {"missing": missing, "symlinks": symlinks, "cross_method_references": sorted(set(violations))}


def run(cmd, cwd, log_path):
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    return process.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--operators-only", action="store_true")
    ap.add_argument("--methods", default="")
    args = ap.parse_args()
    dirs = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("CF3-"))
    if args.methods:
        selected = set(args.methods.split(","))
        dirs = [p for p in dirs if p.name in selected or p.name.split("_")[0] in selected]
    report = {"claim_level": False, "operators_only": args.operators_only,
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "methods": []}
    failures = 0
    for directory in dirs:
        row = {"method": directory.name, "isolation": check_isolation(directory)}
        out = ROOT / "_validation" / directory.name
        out.mkdir(parents=True, exist_ok=True)
        code = run([sys.executable, str(directory / "run.py"), "--self-test"], directory, out / "operator.log")
        row["operator_returncode"] = code
        row["operator_log"] = str(out / "operator.log")
        passed = code == 0 and not any(row["isolation"].values())
        if not args.operators_only and code == 0:
            code = run([sys.executable, str(directory / "run.py"), "--profile", "tiny", "--output", str(out),
                        "--set", "data.reuse_cache=false"], directory, out / "tiny.log")
            row["tiny_returncode"] = code
            row["tiny_log"] = str(out / "tiny.log")
            result_path = out / "result.json"
            if code == 0 and result_path.exists():
                result = json.loads(result_path.read_text())
                gradients = [x["grad_norm"] for x in result["train"]["history"]]
                row.update({"status": result["status"], "steps": result["train"]["steps_completed"],
                            "finite_positive_gradients": bool(gradients) and all(math.isfinite(x) and x > 0 for x in gradients),
                            "teacher_frozen": result["integrity"]["teacher_frozen"],
                            "initial_student_equals_teacher": result["integrity"]["initial_student_equals_teacher"],
                            "probe_status": result["target_fact_probe"]["after"]["status"],
                            "relearning_status": result["relearning"]["status"], "result": str(result_path)})
                passed = (passed and result["status"] == "COMPLETED" and row["finite_positive_gradients"] and
                          row["teacher_frozen"] and row["initial_student_equals_teacher"] and
                          row["probe_status"] == "COMPLETED" and row["relearning_status"] == "COMPLETED")
            else:
                passed = False
        row["verdict"] = "PASS" if passed else "FAIL"
        failures += int(not passed)
        report["methods"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report["failed"] = failures
    report["verdict"] = "PASS" if failures == 0 else "FAIL"
    path = ROOT / "_validation" / ("operator_summary.json" if args.operators_only else "summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "failed": failures, "summary": str(path)}, ensure_ascii=False))
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "core"))

from factpair import read_jsonl, write_json  # noqa: E402

WORKSPACE = BASE.parent
DATA_DIR = WORKSPACE / "data" / "factpair"


def check_entity_isolation(pools):
    def names(key):
        return {cluster["name"].lower() for cluster in pools[key]}

    sets = {key: names(key) for key in ("D_f", "D_donor", "D_R_train", "D_eval")}
    overlaps = {
        f"{left}_x_{right}": sorted(sets[left] & sets[right])
        for index, left in enumerate(sets)
        for right in list(sets)[index + 1 :]
    }
    return {"pass": not any(overlaps.values()), "pool_sizes": {k: len(v) for k, v in sets.items()}, "overlaps": overlaps}


def check_pairs(pairs, summary):
    failures = []
    for pair in pairs:
        ids_plus, ids_minus = pair["prompt_plus_ids"], pair["prompt_minus_ids"]
        start, length = pair["entity_span"]
        reasons = []
        if len(ids_plus) != len(ids_minus):
            reasons.append("token_length_mismatch")
        if start < 0 or length <= 0 or start + length > len(ids_plus):
            reasons.append("invalid_entity_span")
        else:
            if ids_plus[:start] != ids_minus[:start]:
                reasons.append("prefix_token_mismatch")
            if ids_plus[start + length :] != ids_minus[start + length :]:
                reasons.append("suffix_token_mismatch")
        if not pair.get("slot_outside_token_ids_equal", False):
            reasons.append("slot_outside_not_equal")
        if not pair.get("single_contiguous_span", False):
            reasons.append("not_single_contiguous_span")
        if not pair.get("no_truncation", False):
            reasons.append("truncated")
        if not pair.get("boundary_tokenization_stable", False):
            reasons.append("boundary_tokenization_unstable")
        if reasons:
            failures.append({"pair_id": pair.get("id"), "reasons": reasons})
    return {
        "pass": not failures and bool(pairs) and summary.get("real_tokenizer", False),
        "pair_count": len(pairs),
        "real_tokenizer": summary.get("real_tokenizer", False),
        "tokenizer": summary.get("tokenizer"),
        "failure_count": len(failures),
        "failures": failures[:20],
    }


def check_g1():
    import shutil

    test = BASE / "tests" / "test_factpair.py"
    if not test.exists():
        return {"pass": False, "status": "BLOCKED_NO_TEST"}
    pytest_python = sys.executable
    try:
        import pytest  # noqa: F401
    except ImportError:
        pytest_python = shutil.which("python") or shutil.which("python3") or sys.executable
    result = subprocess.run([pytest_python, "-m", "pytest", "-q", str(test)], cwd=BASE, capture_output=True, text=True)
    return {
        "pass": result.returncode == 0,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "stdout_tail": result.stdout[-600:],
        "stderr_tail": result.stderr[-600:],
    }


def main():
    pools = json.loads((DATA_DIR / "entity_pools.json").read_text(encoding="utf-8"))
    summary = json.loads((DATA_DIR / "pairing_summary.json").read_text(encoding="utf-8"))
    pairs = read_jsonl(DATA_DIR / "paired_prompts.jsonl")
    isolation = check_entity_isolation(pools)
    pair_gate = check_pairs(pairs, summary)
    g1 = check_g1()
    g0_checks = {
        "entity_isolation": isolation,
        "paired_prompts": pair_gate,
        "real_tokenizer": summary.get("real_tokenizer", False),
        "checkpoint_tokenizer_template_hash": {"pass": summary.get("real_tokenizer", False), "tokenizer": summary.get("tokenizer")},
    }
    g0_pass = all(v.get("pass", False) if isinstance(v, dict) else bool(v) for v in g0_checks.values())
    report = {
        "protocol": "factpair-progressive-alignment-v0.2-draft",
        "g0": {"status": "PASS" if g0_pass else "BLOCKED", "checks": g0_checks},
        "g1": g1,
        "overall_status": "PASS" if g0_pass and g1.get("pass", False) else "BLOCKED",
    }
    write_json(DATA_DIR / "g0_gate_report.json", report)
    print(f"G0={report['g0']['status']} G1={g1['status']}")
    if not report["overall_status"] == "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

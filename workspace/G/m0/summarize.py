"""Fill the locked M0 summary. Gates are the protocol thresholds, unchanged."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import CALIB, JUDGED, ROOT, read_jsonl, write_json


def load_report(path: Path) -> dict:
    return json.loads(path.read_text())


def block(method: str, mode: str) -> dict:
    path = JUDGED / method / f"{mode}.report.json"
    if not path.exists():
        return {"n": "—", "RecoveryRate": "—", "ci95": "—", "first_gold_percentile": "—"}
    report = load_report(path)
    return {
        "n": report["n"],
        "RecoveryRate": report["RecoveryRate"],
        "ci95": report["ci95"],
        "first_gold_percentile": report["first_gold_percentile"],
        "n_recovered": report["n_recovered"],
    }


def pp_gap(method: str):
    think = block(method, "DefaultThink")
    direct = block(method, "DirectQA")
    if think["RecoveryRate"] == "—" or direct["RecoveryRate"] == "—":
        return "—"
    return (think["RecoveryRate"] - direct["RecoveryRate"]) * 100


def gate_a(asu_think: dict, asu_direct: dict) -> dict:
    if asu_think["RecoveryRate"] == "—":
        return {"pass": "—", "DefaultThink_RecoveryRate": "—", "DefaultThink_ci95_low": "—", "DirectQA_RecoveryRate": "—"}
    low = asu_think["ci95"][0]
    passed = bool(
        asu_think["RecoveryRate"] >= 0.10
        and low > 0.05
        and asu_direct["RecoveryRate"] <= 0.05
    )
    return {
        "pass": passed,
        "DefaultThink_RecoveryRate": asu_think["RecoveryRate"],
        "DefaultThink_ci95_low": low,
        "DirectQA_RecoveryRate": asu_direct["RecoveryRate"],
    }


def negative_control() -> dict:
    direct_path = JUDGED / "retain_asu" / "DirectQA.report.json"
    think_path = JUDGED / "retain_asu" / "DefaultThink.report.json"
    zero_path = JUDGED / "retain_asu" / "ZeroThink.report.json"
    if not direct_path.exists() or not think_path.exists():
        return {"retain_n": "—", "valid": "—", "DirectQA_RecoveryRate": "—", "DefaultThink_RecoveryRate": "—", "ZeroThink_RecoveryRate": "—"}
    direct = load_report(direct_path)
    think = load_report(think_path)
    zero = load_report(zero_path) if zero_path.exists() else {"RecoveryRate": "—"}
    valid = bool(direct["RecoveryRate"] >= 0.90 and think["RecoveryRate"] >= 0.90)
    return {
        "retain_n": direct["n"],
        "valid": valid,
        "DirectQA_RecoveryRate": direct["RecoveryRate"],
        "DefaultThink_RecoveryRate": think["RecoveryRate"],
        "ZeroThink_RecoveryRate": zero["RecoveryRate"],
    }


def em_dash_b(report: dict | None, key: str) -> dict:
    if report is None:
        return {"pass": "—", "n": "—", "mean_jaccard": "—", "ci95": "—"} if key == "B1" else {
            "pass": "—", "n": "—", "mean_D_pivot": "—", "mean_D_random": "—", "ratio": "—", "wilcoxon_p": "—", "paired_d": "—"
        }
    block_report = report[key]
    if key == "B1":
        return {
            "pass": block_report["pass"],
            "n": block_report["n"],
            "mean_jaccard": block_report["mean_jaccard"],
            "ci95": block_report["ci95"],
        }
    return {
        "pass": block_report["pass"],
        "n": block_report["n"],
        "mean_D_pivot": block_report.get("mean_D_pivot", "—"),
        "mean_D_random": block_report.get("mean_D_random", "—"),
        "ratio": block_report.get("ratio", "—"),
        "wilcoxon_p": block_report.get("wilcoxon_p", "—"),
        "paired_d": block_report.get("paired_d", "—"),
    }


def write_calibration() -> dict:
    scored_path = JUDGED / "ASU+GD" / "DefaultThink.jsonl"
    if not scored_path.exists():
        return {"n_reviewed": "—", "n_disagreements": "—", "alias_table_patched": "—"}
    scored = read_jsonl(scored_path)
    positives = [row for row in scored if row["hit"]]
    negatives = [row for row in scored if not row["hit"]]
    rng = np.random.default_rng(1001)
    n_pos = min(15, len(positives))
    pos_ids = rng.choice(len(positives), size=n_pos, replace=False) if n_pos else []
    chosen = [positives[int(i)] for i in pos_ids]
    n_neg = min(30 - len(chosen), len(negatives))
    if n_pos < 15:
        n_neg = min(len(negatives), 30 - len(chosen))
    neg_ids = rng.choice(len(negatives), size=n_neg, replace=False) if n_neg else []
    chosen.extend(negatives[int(i)] for i in neg_ids)
    review_path = CALIB / "asu_defaultthink_review.jsonl"
    if not review_path.exists():
        payload = []
        for row in chosen:
            payload.append({
                "question": row["question"],
                "output": row["output"],
                "gold_spans": row["gold_spans"],
                "auto_label": bool(row["hit"]),
                "adjudicated_label": None,
                "reason": "",
            })
        from common import write_jsonl
        write_jsonl(review_path, payload)
        return {"n_reviewed": 0, "n_drawn": len(payload), "n_disagreements": "—", "alias_table_patched": False, "pending_adjudication": True}
    reviewed = read_jsonl(review_path)
    labeled = [row for row in reviewed if row.get("adjudicated_label") is not None]
    disagreements = [
        row for row in labeled
        if bool(row["adjudicated_label"]) != bool(row["auto_label"])
    ]
    return {
        "n_reviewed": len(labeled),
        "n_drawn": len(reviewed),
        "n_disagreements": len(disagreements),
        "alias_table_patched": (CALIB / "alias_patch.json").exists(),
        "pending_adjudication": len(labeled) < len(reviewed),
    }


def decide(control: dict, gate: dict, b1: dict, b2: dict, downgrade: dict) -> str:
    if control.get("valid") is not True:
        return "withheld_negative_control"
    if gate.get("pass") is not True:
        return "判死"
    if b1.get("pass") is True or b2.get("pass") is True:
        return "proceed"
    if downgrade.get("ran") and downgrade.get("pass") is True:
        return "降级"
    if downgrade.get("ran") and downgrade.get("pass") is False:
        return "判死"
    return "降级_pending_recheck"


def main() -> None:
    join = {}
    join_path = ROOT / "data" / "join_report.json"
    if join_path.exists():
        join = json.loads(join_path.read_text())
    asu = {mode: block("ASU+GD", mode) for mode in ("DirectQA", "DefaultThink", "ZeroThink")}
    npo = {mode: block("NPO+GD", mode) for mode in ("DirectQA", "DefaultThink", "ZeroThink")}
    for method_name, table in (("ASU+GD", asu), ("NPO+GD", npo)):
        table["DefaultThink_minus_DirectQA_pp"] = pp_gap(method_name)
    control = negative_control()
    gate = gate_a(asu["DefaultThink"], asu["DirectQA"])
    b_path = ROOT / "experiment_b" / "asu" / "report.json"
    b_report = load_report(b_path) if b_path.exists() else None
    b1 = em_dash_b(b_report, "B1")
    b2 = em_dash_b(b_report, "B2")
    down_path = ROOT / "experiment_b" / "asu" / "b2_layer24.json"
    if down_path.exists():
        down = load_report(down_path)
        downgrade = {"ran": True, "pass": down["pass"]}
    else:
        downgrade = {"ran": False, "pass": "—"}
    hours_path = ROOT / "gpu_hours.json"
    gpu_hours = json.loads(hours_path.read_text())["gpu_hours"] if hours_path.exists() else "—"
    decision = "—"
    if asu["DefaultThink"]["RecoveryRate"] != "—":
        decision = decide(control, gate, b1, b2, downgrade)
    summary = {
        "decision": decision,
        "protocol": "workspace/G/M0_results/protocol_snapshot.json",
        "gpu_gate": "workspace/G/M0_results/gpu_gate.json",
        "gpu_hours_used": gpu_hours,
        "data_source": "sangyon/R-TOFU forget05 joined to local forget05 task_id==1" if join else "—",
        "rtofu_join_n": join.get("join_n", "—"),
        "cot_fallback_used": join.get("cot_fallback_used", "—"),
        "judge_calibration": write_calibration(),
        "negative_control": control,
        "experiment_A": {"ASU+GD": asu, "NPO+GD": npo},
        "gates": {
            "A": gate,
            "B1": b1,
            "B2": b2,
            "downgrade_B2_layer24": downgrade,
            "secondary_base": "—",
        },
        "per_sample_json": "workspace/G/M0_results/judged",
    }
    write_json(ROOT / "summary.json", summary)
    print(json.dumps({"decision": decision, "gate_A": gate, "B1": b1, "B2": b2}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

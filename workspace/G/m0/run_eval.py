"""Wait for the locked training job, then decode, judge, and run experiment B."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from common import DATA, GEN, JUDGED, ROOT, find_checkpoint, read_jsonl, write_json

M0 = Path("/usr/local/ASU-unlearning/workspace/G/m0")
PYTHON = sys.executable
MODES = ("DirectQA", "DefaultThink", "ZeroThink")


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def run(script: str, args: list[str]) -> None:
    command = [PYTHON, str(M0 / script), *args]
    log(" ".join(command))
    subprocess.run(command, cwd=M0, check=True)


def wait_for_training() -> None:
    end_marker = ROOT / "logs" / "train_wall_end.txt"
    while not end_marker.exists():
        launcher = subprocess.run(["pgrep", "-f", "[l]aunch_m0_train.sh"], capture_output=True, text=True)
        if launcher.returncode != 0:
            log("training launcher exited before train_wall_end.txt")
            raise SystemExit(1)
        time.sleep(60)
    log("training finished")


def gpu_hours_from_training() -> float:
    start = (ROOT / "logs" / "train_wall_start.txt").read_text().strip()
    end = (ROOT / "logs" / "train_wall_end.txt").read_text().strip()
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    hours = (end_dt - start_dt).total_seconds() / 3600
    # torchrun --nproc_per_node=2
    return 2 * hours


def decode_and_judge(method: str, checkpoint: Path, data: Path, tag: str) -> None:
    for mode in MODES:
        out = GEN / tag / f"{mode}.jsonl"
        run("decode.py", ["--checkpoint", str(checkpoint), "--data", str(data), "--mode", mode, "--out", str(out)])
        judged = JUDGED / tag / f"{mode}.jsonl"
        run("judge.py", ["--generations", str(out), "--out", str(judged)])


def main() -> None:
    eval_start = time.time()
    wait_for_training()
    if not (DATA / "forget05_task1.jsonl").exists():
        run("prepare_data.py", [])
    asu = find_checkpoint("ASU+GD")
    npo = find_checkpoint("NPO+GD")
    log(f"ASU {asu}")
    log(f"NPO {npo}")
    forget = DATA / "forget05_task1.jsonl"
    retain = DATA / "retain90_n50.jsonl"
    decode_and_judge("ASU+GD", asu, forget, "ASU+GD")
    decode_and_judge("NPO+GD", npo, forget, "NPO+GD")
    decode_and_judge("retain", asu, retain, "retain_asu")
    think = read_jsonl(JUDGED / "ASU+GD" / "DefaultThink.jsonl")
    recovered = sum(1 for row in think if row["hit"])
    log(f"ASU DefaultThink recoveries {recovered}")
    if recovered >= 40:
        run("experiment_b.py", ["--checkpoint", str(asu), "--generations", str(GEN / "ASU+GD" / "DefaultThink.jsonl"), "--tag", "asu"])
        report = json.loads((ROOT / "experiment_b" / "asu" / "report.json").read_text())
        direct = json.loads((JUDGED / "ASU+GD" / "DirectQA.report.json").read_text())
        think_report = json.loads((JUDGED / "ASU+GD" / "DefaultThink.report.json").read_text())
        gate_a = (
            think_report["RecoveryRate"] >= 0.10
            and think_report["ci95"][0] > 0.05
            and direct["RecoveryRate"] <= 0.05
        )
        if gate_a and not report["B1"]["pass"] and not report["B2"]["pass"]:
            log("Gate A passed and B1/B2 failed; downgrade B2 to layer 24")
            run("experiment_b.py", ["--checkpoint", str(asu), "--generations", str(GEN / "ASU+GD" / "DefaultThink.jsonl"), "--tag", "asu", "--downgrade-layer"])
    else:
        log("fewer than 40 recoveries; experiment B stays em-dash until a forget10 rerun")
    eval_hours = (time.time() - eval_start) / 3600
    write_json(ROOT / "gpu_hours.json", {
        "train_gpu_hours": gpu_hours_from_training(),
        "eval_gpu_hours": eval_hours,
        "gpu_hours": gpu_hours_from_training() + eval_hours,
        "eval_devices": 1,
        "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    })
    run("summarize.py", [])


if __name__ == "__main__":
    main()

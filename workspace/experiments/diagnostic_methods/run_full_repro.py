#!/usr/bin/env python3
"""Local two-GPU queue for full-data non-claim reproduction of 13 candidates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/usr/local/workspace/tracks/llm-unlearning/ASU-unlearning")
WORKSPACE = REPO / "workspace"
EXP = WORKSPACE / "experiments"
AUTO = WORKSPACE / "auto_experiments"
PY = "/jizhicfs/chenyliao/llm-unlearning/envs/open-unlearning/bin/python"
MODEL = "/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
DATA = "/jizhicfs/chenyliao/data/datasets/locuslab-TOFU"
GA = "/jizhicfs/chenyliao/runs/open-unlearning/20261122_forget10_1b/GradAscent"
METHODS = [f"CF-{i:02d}" for i in range(1, 11)] + ["R2-C", "R3-3"]
MAX_PAIRS = 367


def build_jobs(args) -> list[dict]:
    out = Path(args.out_dir)
    common = [
        "--seed", str(args.seed),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--eval-limit", "0",
        "--eval-batch-size", str(args.eval_batch_size),
        "--behavior-samples", "0",
        "--max-pairs", str(args.max_pairs),
        "--probe-limit", str(args.probe_limit),
        "--basis-limit", str(args.basis_limit),
        "--patch-pairs", str(args.patch_pairs),
        "--patch-micro-batch", str(args.patch_micro_batch),
        "--data-dir", DATA,
        "--out-dir", str(out),
    ]
    jobs = [{
        "id": "ASU_BASE",
        "depends_on": [],
        "cwd": str(AUTO),
        "cmd": [
            PY, "train_tau.py", "--schedule", "global", "--steps", str(args.steps),
            "--seed", str(args.seed), "--batch-size", str(args.asu_batch_size),
            "--lr", str(args.lr), "--alpha", str(args.alpha),
            "--eval-limit", "0", "--eval-batch-size", str(args.eval_batch_size),
            "--name", f"full_asu_s{args.steps}", "--save",
            "--output", str(out / "ASU_BASE.json"),
        ],
        "expected": str(out / "ASU_BASE.json"),
    }]
    for mid in METHODS:
        jobs.append({
            "id": mid,
            "depends_on": [],
            "cwd": str(EXP),
            "cmd": [PY, "-m", "diagnostic_methods.run_one_method", "--method", mid,
                    "--steps", str(args.steps), *common],
            "expected": str(out / f"{mid}.json"),
        })
    jobs.append({
        "id": "R3-2",
        "depends_on": ["ASU_BASE"],
        "cwd": str(AUTO),
        "cmd": [
            PY, "audit_depth.py", "--base", MODEL,
            "--asu", str(AUTO / "checkpoints" / f"full_asu_s{args.steps}"),
            "--ga", GA, "--forget", str(Path(DATA) / "forget10.json"),
            "--retain", str(Path(DATA) / "retain90.json"),
            "--relearn-steps", str(args.relearn_steps), "--relearn-lr", str(args.relearn_lr),
            "--eval-limit", "0", "--eval-batch-size", str(args.eval_batch_size),
            "--fewshot-limit", str(args.probe_limit),
            "--relearn-limit", str(args.probe_limit),
            "--output", str(out / "R3-2_audit.json"),
        ],
        "expected": str(out / "R3-2_audit.json"),
    })
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(AUTO / "results/full_repro_v1"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=MAX_PAIRS)
    ap.add_argument("--probe-limit", type=int, default=400)
    ap.add_argument("--basis-limit", type=int, default=3600)
    ap.add_argument("--patch-pairs", type=int, default=MAX_PAIRS)
    ap.add_argument("--patch-micro-batch", type=int, default=8)
    ap.add_argument("--asu-batch-size", type=int, default=2)
    ap.add_argument("--relearn-steps", type=int, default=20)
    ap.add_argument("--relearn-lr", type=float, default=1e-5)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = out / "queue_state.json"
    jobs = build_jobs(args)
    if state_path.exists() and args.resume:
        state = json.loads(state_path.read_text())
    else:
        state = {j["id"]: {"status": "pending", "attempts": 0} for j in jobs}
        state_path.write_text(json.dumps(state, indent=2))
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    running: dict[str, tuple[subprocess.Popen, str, object]] = {}
    cooldown_until = 0.0

    def save_state():
        state_path.write_text(json.dumps(state, indent=2))

    def launch(job: dict, gpu: str):
        log_path = logs / f"{job['id']}.log"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM="false")
        handle = log_path.open("w")
        proc = subprocess.Popen(job["cmd"], cwd=job["cwd"], env=env, stdout=handle, stderr=subprocess.STDOUT)
        state[job["id"]].update({"status": "running", "gpu": gpu, "started": time.time(),
                                  "attempts": state[job["id"]].get("attempts", 0) + 1})
        running[job["id"]] = (proc, gpu, handle)
        save_state()
        print(f"LAUNCH {job['id']} gpu={gpu}", flush=True)

    while True:
        for jid, (proc, gpu, handle) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            handle.close()
            job = next(j for j in jobs if j["id"] == jid)
            expected = Path(job["expected"])
            if rc == 0 and expected.exists():
                state[jid].update({"status": "completed", "ended": time.time(), "returncode": rc})
                print(f"COMPLETE {jid}", flush=True)
            else:
                log_text = (logs / f"{jid}.log").read_text(errors="replace")
                oom = "OutOfMemoryError" in log_text or "CUDA out of memory" in log_text
                attempts = state[jid].get("attempts", 0)
                if oom and attempts < 3:
                    state[jid].update({"status": "pending", "ended": time.time(), "returncode": rc,
                                       "oom_retry": attempts})
                    cooldown_until = max(cooldown_until, time.time() + 120)
                    print(f"OOM_RETRY {jid} attempt={attempts}", flush=True)
                else:
                    state[jid].update({"status": "failed", "ended": time.time(), "returncode": rc})
                    print(f"FAILED {jid} rc={rc}", flush=True)
            del running[jid]
            save_state()

        if all(state[j["id"]]["status"] in {"completed", "failed"} for j in jobs) and not running:
            break

        if time.time() < cooldown_until:
            time.sleep(5)
            continue
        for job in jobs:
            jid = job["id"]
            if jid in running or state[jid]["status"] in {"completed", "failed"}:
                continue
            dep_failed = [dep for dep in job["depends_on"] if state[dep]["status"] == "failed"]
            if dep_failed:
                state[jid].update({"status": "failed", "ended": time.time(),
                                   "reason": "dependency_failed:" + ",".join(dep_failed)})
                save_state()
                continue
            if any(state[dep]["status"] != "completed" for dep in job["depends_on"]):
                continue
            if len(running) >= len(gpus):
                break
            busy = {gpu for _, gpu, _ in running.values()}
            free = [g for g in gpus if g not in busy]
            if free:
                launch(job, free[0])

        time.sleep(5)

    summary = {
        "completed": [j["id"] for j in jobs if state[j["id"]]["status"] == "completed"],
        "failed": [j["id"] for j in jobs if state[j["id"]]["status"] == "failed"],
        "elapsed_sec": round(time.time() - min(v.get("started", time.time()) for v in state.values()), 2),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

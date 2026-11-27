#!/usr/bin/env python3
"""Process-level CF3 orchestrator; never imports method implementations."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def available_gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=os.environ.get("PROFILE", "pilot"), choices=["tiny", "smoke", "pilot", "full"])
    ap.add_argument("--methods", default=os.environ.get("METHODS", ""), help="comma-separated CF3 IDs or directory names")
    ap.add_argument("--gpus", default=os.environ.get("GPUS", ""))
    ap.add_argument("--seeds", default=os.environ.get("SEEDS", "0"))
    ap.add_argument("--max-parallel", type=int, default=int(os.environ.get("MAX_PARALLEL", "1")))
    ap.add_argument("--resume", action="store_true", default=os.environ.get("RESUME") == "1")
    ap.add_argument("--force", action="store_true", default=os.environ.get("FORCE") == "1")
    ap.add_argument("--diagnose-only", action="store_true", default=os.environ.get("DIAGNOSE_ONLY") == "1")
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    args = ap.parse_args()
    dirs = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("CF3-"))
    if args.methods:
        requested = {x.strip() for x in args.methods.split(",") if x.strip()}
        dirs = [p for p in dirs if p.name in requested or p.name.split("_")[0] in requested]
        known = {x for p in dirs for x in (p.name, p.name.split("_")[0])}
        missing = requested - known
        if missing:
            raise SystemExit(f"unknown methods: {sorted(missing)}")
    if not dirs:
        raise SystemExit("no methods selected")
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise SystemExit("no seeds selected")
    gpu_text = args.gpus or (",".join(str(i) for i in range(available_gpus())) if args.profile != "tiny" else "")
    gpus = [x.strip() for x in gpu_text.split(",") if x.strip()]
    if args.profile != "tiny" and not gpus and os.environ.get("ALLOW_CPU") != "1":
        raise SystemExit("no CUDA GPU detected; use PROFILE=tiny, set GPUS, or explicitly set ALLOW_CPU=1")
    workers = max(1, min(args.max_parallel, len(gpus) if gpus else 1))
    queue = []
    for directory in dirs:
        for seed in seeds:
            out = directory / "runs" / f"{args.profile}_seed{seed}"
            result_path = out / "result.json"
            row = {"method": directory.name, "seed": seed, "directory": directory, "output": out}
            if result_path.exists() and not args.force:
                try:
                    status = json.loads(result_path.read_text())["status"]
                    if status in {"COMPLETED", "DIAGNOSTIC_KILL", "DIAGNOSED"}:
                        row.update({"status": "SKIPPED_EXISTING", "result": str(result_path)})
                        queue.append(row)
                        continue
                except Exception:
                    pass
            row["status"] = "QUEUED"
            queue.append(row)
    active, pending = [], [x for x in queue if x["status"] == "QUEUED"]
    failures = 0
    free_gpus = list(gpus)
    for row in queue:
        if row["status"] == "SKIPPED_EXISTING":
            print(json.dumps({k: v for k, v in row.items() if k in {"method", "seed", "status", "result"}}, ensure_ascii=False), flush=True)
    while pending or active:
        while pending and len(active) < workers and (free_gpus or not gpus):
            row = pending.pop(0)
            gpu = free_gpus.pop(0) if gpus else None
            row["output"].mkdir(parents=True, exist_ok=True)
            log = (row["output"] / "run.log").open("a", encoding="utf-8")
            cmd = [sys.executable, str(row["directory"] / "run.py"), "--profile", args.profile,
                   "--output", str(row["output"]), "--set", f"seed={row['seed']}"]
            if args.resume:
                cmd.append("--resume")
            if args.diagnose_only:
                cmd.append("--diagnose-only")
            for override in args.overrides:
                cmd.extend(["--set", override])
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            row.update({"status": "RUNNING", "gpu": gpu, "started": time.time(),
                        "log_handle": log, "log": str(row["output"] / "run.log")})
            row["process"] = subprocess.Popen(cmd, cwd=row["directory"], env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append(row)
            print(json.dumps({k: v for k, v in row.items() if k in {"method", "seed", "status", "gpu", "log"}}, ensure_ascii=False), flush=True)
        time.sleep(0.2)
        for row in list(active):
            code = row["process"].poll()
            if code is None:
                continue
            row["log_handle"].close()
            row["status"] = "COMPLETED" if code == 0 else "FAILED"
            result_path = row["output"] / "result.json"
            if code == 0 and result_path.exists():
                row["status"] = json.loads(result_path.read_text(encoding="utf-8")).get("status", row["status"])
            row["returncode"] = code
            row["elapsed_sec"] = round(time.time() - row["started"], 2)
            failures += int(code != 0)
            if row["gpu"] is not None:
                free_gpus.append(row["gpu"])
            active.remove(row)
            print(json.dumps({k: v for k, v in row.items() if k in {"method", "seed", "status", "gpu", "returncode", "elapsed_sec", "log"}}, ensure_ascii=False), flush=True)
    serializable = [{k: v for k, v in row.items() if k not in {"directory", "output", "process", "log_handle", "started"}} for row in queue]
    summary = {"profile": args.profile, "methods": len(dirs), "jobs": len(queue), "failed": failures,
               "runs": serializable, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (ROOT / "queue_status.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"jobs": len(queue), "failed": failures, "summary": str(ROOT / "queue_status.json")}, ensure_ascii=False))
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())

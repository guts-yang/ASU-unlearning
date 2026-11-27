#!/usr/bin/env python3
"""Process-level orchestrator; never imports a method implementation."""
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
    ap.add_argument("--methods", default=os.environ.get("METHODS", ""), help="comma-separated CF2 IDs or directory names")
    ap.add_argument("--gpus", default=os.environ.get("GPUS", ""), help="comma-separated visible GPU IDs")
    ap.add_argument("--max-parallel", type=int, default=int(os.environ.get("MAX_PARALLEL", "1")))
    ap.add_argument("--resume", action="store_true", default=os.environ.get("RESUME") == "1")
    ap.add_argument("--force", action="store_true", default=os.environ.get("FORCE") == "1")
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    args = ap.parse_args()
    dirs = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("CF2-"))
    if args.methods:
        requested = {x.strip() for x in args.methods.split(",") if x.strip()}
        dirs = [p for p in dirs if p.name in requested or p.name.split("_")[0] in requested]
        missing = requested - {p.name for p in dirs} - {p.name.split("_")[0] for p in dirs}
        if missing:
            raise SystemExit(f"unknown methods: {sorted(missing)}")
    if not dirs:
        raise SystemExit("no methods selected")
    gpu_text = args.gpus or (",".join(str(i) for i in range(available_gpus())) if args.profile != "tiny" else "")
    gpus = [x.strip() for x in gpu_text.split(",") if x.strip()]
    if args.profile != "tiny" and not gpus and os.environ.get("ALLOW_CPU") != "1":
        raise SystemExit("no CUDA GPU detected; set GPUS, use PROFILE=tiny for offline validation, or ALLOW_CPU=1 explicitly")
    workers = max(1, min(args.max_parallel, len(gpus) if gpus else 1))
    queue = []
    for directory in dirs:
        out = directory / "runs" / f"{args.profile}_seed0"
        result_path = out / "result.json"
        if result_path.exists() and not args.force:
            try:
                status = json.loads(result_path.read_text())["status"]
                if status in {"COMPLETED", "DIAGNOSTIC_KILL"}:
                    queue.append({"method": directory.name, "status": "SKIPPED_EXISTING", "result": str(result_path)})
                    continue
            except Exception:
                pass
        queue.append({"method": directory.name, "status": "QUEUED", "directory": directory, "output": out})
    active = []
    failures = 0
    free_gpus = list(gpus)
    for row in queue:
        if row["status"] == "SKIPPED_EXISTING":
            print(json.dumps(row, ensure_ascii=False), flush=True)
    pending = [row for row in queue if row["status"] == "QUEUED"]
    while pending or active:
        while pending and len(active) < workers and (free_gpus or not gpus):
            row = pending.pop(0)
            gpu = free_gpus.pop(0) if gpus else None
            row["output"].mkdir(parents=True, exist_ok=True)
            log = (row["output"] / "run.log").open("a", encoding="utf-8")
            cmd = [sys.executable, str(row["directory"] / "run.py"), "--profile", args.profile, "--output", str(row["output"])]
            if args.resume:
                cmd.append("--resume")
            for override in args.overrides:
                cmd.extend(["--set", override])
            env = os.environ.copy()
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            row.update({"status": "RUNNING", "gpu": gpu, "started": time.time(), "log_handle": log, "log": str(row["output"] / "run.log")})
            row["process"] = subprocess.Popen(cmd, cwd=row["directory"], env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append(row)
            print(json.dumps({k: v for k, v in row.items() if k in {"method", "status", "gpu", "log"}}, ensure_ascii=False), flush=True)
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
            print(json.dumps({k: v for k, v in row.items() if k in {"method", "status", "gpu", "returncode", "elapsed_sec", "log"}}, ensure_ascii=False), flush=True)
    serializable = [{k: v for k, v in row.items() if k not in {"directory", "output", "process", "log_handle", "started"}} for row in queue]
    summary = {"profile": args.profile, "methods": len(queue), "failed": failures, "runs": serializable, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (ROOT / "queue_status.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"methods": len(queue), "failed": failures, "summary": str(ROOT / 'queue_status.json')}, ensure_ascii=False))
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())

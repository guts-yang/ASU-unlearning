#!/usr/bin/env python3
"""Fail fast on cluster dependencies, model and TOFU paths before launching jobs."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="pilot")
    ap.add_argument("--config", default=str(ROOT / "CF3-01_双线性路由内容差分" / "config.json"))
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    args, _ = ap.parse_known_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    for expression in args.overrides:
        key, raw = expression.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = parse_value(raw)
    errors, checks = [], {}
    if args.profile == "tiny":
        checks["offline_tiny"] = "PASS"
        if importlib.util.find_spec("torch") is None:
            errors.append("PyTorch is required even for tiny validation")
    else:
        packages = ["torch", "transformers"] + (["peft"] if cfg["model"].get("peft", False) else [])
        for package in packages:
            available = importlib.util.find_spec(package) is not None
            checks[f"import_{package}"] = "PASS" if available else "FAIL"
            if not available:
                errors.append(f"missing Python package: {package}; use INSTALL=1")
        if importlib.util.find_spec("torch") is not None:
            import torch
            cuda = torch.cuda.device_count()
            checks["cuda_devices"] = cuda
            if cuda < 1:
                errors.append("no visible CUDA device; check GPUS/CUDA_VISIBLE_DEVICES")
        model_path = Path(cfg["model"]["path"])
        model_ok = model_path.exists() or not cfg["model"].get("local_files_only", True)
        checks["model_path"] = str(model_path)
        if not model_ok:
            errors.append(f"model not found: {model_path}; set model.path or explicitly allow a HF hub id")
        data_root = Path(cfg["data"]["root"])
        for key in ("forget_file", "retain_file", "holdout_file"):
            path = data_root / cfg["data"][key]
            checks[key] = str(path)
            if not path.is_file():
                errors.append(f"dataset split not found: {path}")
    report = {"verdict": "PASS" if not errors else "FAIL", "profile": args.profile,
              "checks": checks, "errors": errors}
    out = ROOT / "_validation" / f"preflight_{args.profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

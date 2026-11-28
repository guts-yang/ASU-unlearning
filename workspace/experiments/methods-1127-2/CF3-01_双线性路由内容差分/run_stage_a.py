"""CF3-01 v1.1 Stage A runner (diagnose-only, independent of runtime.main).

Runs the measurement-first protocol on the SAME teacher cache that the round3
trainer would use, but NEVER trains a Student. Output goes to runs/stage_a_*.

Usage:
    python3 run_stage_a.py --profile tiny
    python3 run_stage_a.py --profile pilot --diagnose-only   # real Llama/TOFU
    python3 run_stage_a.py --self-test

This mirrors runtime.main() cache/teacher setup so the cache is reusable, but
calls measurement_protocol.run_stage_a() instead of train(). claim_level=false.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data import build_pairs, build_teacher_cache, load_cache, load_records, save_cache
from modeling import create_model_and_tokenizer, forward_model
from runtime import load_config, resolve_data, set_seed
import measurement_protocol as mp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(HERE / "config.json"))
    ap.add_argument("--profile", default="tiny", choices=["tiny", "smoke", "pilot", "full"])
    ap.add_argument("--set", action="append", default=[], dest="overrides")
    ap.add_argument("--output", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--diagnose-only", action="store_true")  # accepted for symmetry; always diagnose-only
    ap.add_argument("--audit-only", action="store_true",
                    help="only run the noise/precision audit; write measurement_report.json")
    args = ap.parse_args(argv)

    if args.self_test:
        print(json.dumps(mp.self_test(), indent=2, ensure_ascii=False))
        return 0

    cfg = load_config(args.config, args.profile, args.overrides)
    set_seed(int(cfg["seed"]))
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output) if args.output else HERE / "runs" / f"stage_a_{cfg['profile']}_seed{cfg['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    started = time.time()

    teacher, tok = create_model_and_tokenizer(cfg, device, student=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    forget, retain, holdout = resolve_data(cfg)
    pair_path = cfg["data"].get("pairs_file")
    pairs = load_records(pair_path) if pair_path else build_pairs(
        tok, forget, retain, int(cfg["data"]["max_pairs"]), int(cfg["seed"]), False)
    scan_layers = mp._scan_layers(cfg, teacher)
    cache_cfg = copy.deepcopy(cfg)
    cache_cfg["layers"] = sorted(set(cfg["layers"]) | set(scan_layers))
    cache_path = out_dir / "teacher_cache.pt"
    if cache_path.exists() and cfg["data"].get("reuse_cache", True):
        cache = load_cache(cache_path)
    else:
        cache = build_teacher_cache(teacher, tok, pairs, retain, cache_cfg, device,
                                    capture_bilinear=True, capture_glu=True)
        save_cache(cache, cache_path)
    if (not set(cfg["layers"]).issubset(set(cache.get("layers", [])))
            or len(cache.get("pairs", [])) != len(pairs)):
        raise RuntimeError("teacher cache does not cover current layers/pairs; use a new output or data.reuse_cache=false")

    report = mp.run_stage_a(teacher, tok, cache, cfg, device, out_dir, audit_only=args.audit_only)
    report["elapsed_sec"] = round(time.time() - started, 3)
    if args.audit_only:
        (out_dir / "measurement_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(json.dumps({"method": "CF3-01", "phase": "measurement_audit",
                          "measurement_reliable": report["measurement"]["measurement_reliable"],
                          "status": report["measurement"]["status"],
                          "layers_scanned": report["layers_scanned"],
                          "report": str(out_dir / "measurement_report.json")},
                         ensure_ascii=False), flush=True)
        return 0
    (out_dir / "stage_a_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({"method": "CF3-01", "phase": "stage_a",
                      "stage_a_pass": report["stage_a_pass"],
                      "reasons": report["stage_a_reasons"],
                      "selected": report["selection"]["selected"],
                      "status": report["selection"]["status"],
                      "report": str(out_dir / "stage_a_report.json")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

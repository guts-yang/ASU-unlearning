#!/usr/bin/env python3
"""Generate a stratified, subject-aligned explicit pair file for CF3-01 v1.1.

Unlike the round3 strict-pair script, this version:
  * infers the author vocabulary from the ordered TOFU split instead of taking
    the first capitalized phrase (drops "Has ..."/titles);
  * samples `--per-author` pairs from every detected author cluster;
  * excludes donor authors that are exact or near aliases of the forget author;
  * pre-checks subject token positions with the SAME contract as
    data.build_teacher_cache, but never weakens or removes that check.

Output is JSONL consumed via --set data.pairs_file=...
No pair is fabricated and no failing pair is silently patched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHOD_DIR = ROOT / "CF3-01_双线性路由内容差分"
sys.path.insert(0, str(METHOD_DIR))

from data import build_pairs_stratified, load_records  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full")
    ap.add_argument("--data-root", default="/jizhicfs/chenyliao/data/datasets/locuslab-TOFU")
    ap.add_argument("--per-author", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-authors", type=int, default=20,
                    help="TOFU forget10 is ~20 authors x 20 questions")
    ap.add_argument("--no-control", action="store_true",
                    help="disable matched-retain control pairs (not recommended for Stage A)")
    ap.add_argument("--out", default=str(ROOT / "cf3_01_replication/pairs/pairs_v2"))
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    root = Path(args.data_root)
    forget = load_records(root / "forget10.json")
    retain = load_records(root / "retain90.json")
    pairs, report = build_pairs_stratified(
        tok, forget, retain, per_author=args.per_author, seed=args.seed,
        n_authors=args.n_authors, require_control=not args.no_control)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "pairs_v2.jsonl"
    report_path = out / "pairs_v2_report.json"
    pairs_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n",
                          encoding="utf-8")
    report.update({
        "output": str(pairs_path), "report": str(report_path),
        "n_authors_requested": args.n_authors, "seed": args.seed,
        "per_author": args.per_author, "require_control": not args.no_control,
    })
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("generated", "kept", "n_authors_detected", "donors",
                       "unresolved_subjects", "drop_count", "output")},
                     ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

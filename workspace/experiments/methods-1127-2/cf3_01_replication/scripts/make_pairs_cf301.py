#!/usr/bin/env python3
"""Generate a strictly-aligned explicit counterfactual pair file for CF3-01.

Reuses CF3-01's own build_pairs / encode_qa / _subject_indices so the SAME
fail-closed contract is applied at generation time. No pair is fabricated, no
check is weakened; pairs whose subject token positions differ are dropped and
reported. Output is JSONL consumed via --set data.pairs_file=...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHOD_DIR = ROOT / "CF3-01_双线性路由内容差分"
sys.path.insert(0, str(METHOD_DIR))

from data import _subject_indices, build_pairs, encode_qa, load_records  # noqa: E402
from modeling import create_model_and_tokenizer  # noqa: E402

PAIRS_OUT = ROOT / "cf3_01_replication/pairs/strict_pairs_31.jsonl"
REPORT_OUT = ROOT / "cf3_01_replication/pairs/strict_pairs_report.json"
MAX_PAIRS = 32
MAX_LEN = 512
SEED = 0


def main():
    cfg = {
        "model": {"backend": "hf",
                  "path": "/jizhicfs/chenyliao/models/open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
                  "model_type": "llama", "dtype": "bfloat16", "attn_implementation": "eager",
                  "local_files_only": True, "peft": True, "lora_rank": 16, "lora_alpha": 32,
                  "lora_dropout": 0.0, "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
                  "gradient_checkpointing": False, "eos_token_id": 2}
    }
    # tokenizer only; no GPU/model weights needed for alignment
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["path"], local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    root = Path("/jizhicfs/chenyliao/data/datasets/locuslab-TOFU")
    forget = load_records(root / "forget10.json")
    retain = load_records(root / "retain90.json")
    pairs = build_pairs(tok, forget, retain, MAX_PAIRS, SEED, False)

    kept, dropped = [], []
    for index, pair in enumerate(pairs):
        enc_p = encode_qa(tok, pair["x_plus"], pair["answer_plus"], MAX_LEN)
        enc_m = encode_qa(tok, pair["x_minus"], pair["answer_plus"], MAX_LEN)
        try:
            sp = _subject_indices(tok, enc_p, pair["x_plus"], pair["forget_subject"])
            sm = _subject_indices(tok, enc_m, pair["x_minus"], pair["donor_subject"])
            equal = sp == sm
        except Exception as exc:
            equal = False
            dropped.append({"index": index, "reason": f"alignment error: {type(exc).__name__}: {exc}"})
            continue
        if equal:
            kept.append(pair)
        else:
            dropped.append({"index": index, "reason": "subject token positions differ",
                            "x_plus": pair["x_plus"], "x_minus": pair["x_minus"],
                            "subject_plus": sp, "subject_minus": sm})

    PAIRS_OUT.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in kept) + "\n", encoding="utf-8"
    )
    report = {
        "generated_pairs": len(pairs), "kept": len(kept), "dropped": len(dropped),
        "coverage": len(kept) / len(pairs) if pairs else 0.0,
        "contract": "CF3-01 data.build_teacher_cache:184 subject_plus == subject_minus",
        "checks": ["single-token-span replacement", "donor from retain split", "author cluster disjoint"],
        "dropped_detail": dropped,
        "output": str(PAIRS_OUT),
        "note": "No check weakened; no fabricated or synthetic pairs; fail-closed contract preserved.",
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("generated_pairs", "kept", "dropped", "coverage")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

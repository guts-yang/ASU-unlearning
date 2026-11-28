#!/usr/bin/env python3
"""Pure-CPU shape regression for CF2-06 mismatched-item control arms.

Reproduces the exact failure from fix_smoke: Student-side answer span length
27, Teacher-side (mismatched item) length 15 -> F.mse_loss RuntimeError.
No model, no GPU, no cache. Run from anywhere:
    python3 test_cf206_shapes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
METHOD_DIR = HERE.parent.parent / "CF2-06_反事实割线敏感度"
sys.path.insert(0, str(METHOD_DIR))
import method as m  # noqa: E402

LAYER, DIM = 7, 4


def fake_out(span_len: int):
    start = 3
    hidden = {LAYER: torch.randn(start + span_len + 2, DIM)}
    return {"hidden": hidden,
            "enc": {"answer_start": start, "answer_end": start + span_len}}


def fake_item(index: int, span_len: int):
    return {"index": index,
            "h_plus": {LAYER: torch.randn(span_len, DIM)},
            "h_minus": {LAYER: torch.randn(span_len, DIM)}}


def main():
    row = {"item": fake_item(0, 27), "plus": fake_out(27), "minus": fake_out(27)}
    n = m._common_n(row, row["item"], LAYER, "")
    assert n == 27, f"same-length pair must keep n=27, got {n}"

    mismatched = fake_item(1, 15)
    n2 = m._common_n(row, mismatched, LAYER, "")
    assert n2 == 15, f"mismatched pair must shrink to 15, got {n2}"

    distance = torch.tensor(1.0)
    student = m._student_secant(row, LAYER, "", distance, n2)
    teacher = m._teacher_secant(mismatched, LAYER, "", distance, student.device, n2)
    assert student.shape == teacher.shape == (15, DIM), (student.shape, teacher.shape)

    # Full loss() path with control_mode=wrong (the arm that used to crash).
    item0, item1 = fake_item(0, 27), fake_item(1, 15)
    row0 = {"item": item0, "plus": fake_out(27), "minus": fake_out(27)}
    row1 = {"item": item1, "plus": fake_out(15), "minus": fake_out(15)}
    cfg = {"layers": [LAYER], "seed": 0,
           "method": {"control_mode": "wrong", "target_response_ratio": 0.0,
                      "retain_response_weight": 0.1}}
    prepared = {"distances": {0: torch.tensor(1.0), 1: torch.tensor(1.0)}}
    terms = m.loss([row0, row1], prepared, cfg)
    assert torch.isfinite(terms["method_loss"]), "non-finite method_loss"
    assert float(terms["control_changed"]) == 1.0

    # Default mode must stay valid too.
    cfg["method"]["control_mode"] = "none"
    terms_none = m.loss([row0, row1], prepared, cfg)
    assert torch.isfinite(terms_none["method_loss"])
    assert float(terms_none["control_changed"]) == 0.0

    print({"status": "PASS", "common_n_same": n, "common_n_mismatched": n2,
           "student_shape": list(student.shape), "teacher_shape": list(teacher.shape),
           "method_loss_wrong": float(terms["method_loss"]),
           "method_loss_none": float(terms_none["method_loss"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

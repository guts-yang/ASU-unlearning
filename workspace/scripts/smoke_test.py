#!/usr/bin/env python3
"""CPU small-model smoke test.

This is an ENGINEERING check only. It is NOT evidence of forgetting efficacy.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

import torch

from recodeloop.activations.specs import build_spec
from recodeloop.baseline_adapter import BaselineAdapter
from recodeloop.config import FrozenConfig, load_config
from recodeloop.data.access_guard import AccessGuard
from recodeloop.localization.causal_effects import InterventionInputs
from recodeloop.loop.runner import RecodeLoop
from recodeloop.readers.fact_value import FrozenEmbeddingReader


def main() -> int:
    torch.manual_seed(1001)
    cfg = load_config()
    data = {k: dict(v) for k, v in cfg.data.items()}
    reader_cfg = dict(data["readers"])
    edit_reader = dict(reader_cfg["edit_reader"])
    edit_reader["train_steps"] = 1
    reader_cfg["edit_reader"] = edit_reader
    data["readers"] = reader_cfg
    loc = dict(data["localization"])
    loc["patch_threshold"] = -1.0
    loc["max_components_per_round"] = 1
    data["localization"] = loc
    frozen_cfg = FrozenConfig(cfg.workspace_root, data, cfg.per_file_hash, cfg.config_hash)

    adapter = BaselineAdapter.tiny()
    component = adapter.component_paths()[0]
    spec = build_spec(adapter, component, pooling="mean")
    reader = FrozenEmbeddingReader(("vA", "vB", "NK1", "NK2"),
                                   adapter.model.config.hidden_size, 8, 1001)
    ids = torch.randint(0, 50, (2, 5))
    forget = InterventionInputs(ids, ("e1", "e2"), ("f1", "f2"),
                                ("vA", "vB"), "forget_train_edit", ("NK1", "NK2"))
    nk = InterventionInputs(ids.clone(), ("n1", "n2"), ("n1", "n2"),
                            ("NK1", "NK2"), "forget_train_edit", ())
    retain_ids = torch.randint(0, 50, (2, 5))
    result = RecodeLoop(adapter, frozen_cfg, AccessGuard()).run(
        reader, {component: spec}, forget, nk, retain_ids, retain_ids.clone(), ("NK1", "NK2")
    )
    report = {
        "kind": "engineering_smoke_test",
        "forgetting_evidence": False,
        "note": "随机微型模型；不得作为遗忘有效性或方法比较证据",
        "rounds": len(result.states),
        "stop_reasons": [s.stop_reason for s in result.states],
        "components": [list(s.components) for s in result.states],
        "budgets": [
            {"tokens": s.actual_budget.tokens, "steps": s.actual_budget.steps,
             "wall_clock_s": s.actual_budget.wall_clock_s}
            for s in result.states
        ],
        "patch_kinds": sorted({p.patch_kind for p in result.patches}),
    }
    out = WORKSPACE / "outputs" / "smoke_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

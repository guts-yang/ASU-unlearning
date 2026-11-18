#!/usr/bin/env python3
"""Single-round, small-step RecodeLoop path validation after M0a PASS.

Engineering validation only; not M0b and not an L2 claim. Confirmation stays sealed.
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
from recodeloop.checkpoints import checkpoint_refs
from recodeloop.config import FrozenConfig, load_config
from recodeloop.data.access_guard import AccessGuard
from recodeloop.evaluation.m0a import load_facts
from recodeloop.localization.causal_effects import InterventionInputs
from recodeloop.loop.runner import RecodeLoop
from recodeloop.readers.fact_value import FrozenEmbeddingReader
from recodeloop.types import Budget


def tokenize(adapter, prompts, max_len):
    tokenizer = adapter.tokenizer
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
    return {k: v.to(adapter.device) for k, v in encoded.items()}


def main() -> int:
    m0a_path = WORKSPACE / "outputs" / "m0a_validation.json"
    if not m0a_path.exists():
        print("short_train: M0a 报告不存在，停止")
        return 1
    m0a = json.loads(m0a_path.read_text(encoding="utf-8"))
    if m0a.get("status") != "PASS":
        print(f"short_train: M0a status={m0a.get('status')}，停止")
        return 1

    cfg = load_config(WORKSPACE)
    guard = AccessGuard()
    real, nk, nk_hash = load_facts(cfg, guard)
    refs = checkpoint_refs(cfg)

    data = {key: dict(value) for key, value in cfg.data.items()}
    data["m0_p08_m2"] = dict(data["m0_p08_m2"])
    data["m0_p08_m2"]["frozen_constants"] = dict(data["m0_p08_m2"]["frozen_constants"])
    data["m0_p08_m2"]["frozen_constants"]["t_max"] = 1
    data["localization"] = dict(data["localization"])
    data["localization"]["max_components_per_round"] = 1
    data["localization"]["min_components_per_round"] = 1
    data["localization"]["patch_threshold"] = -1.0
    data["editing"] = dict(data["editing"])
    data["editing"]["candidate_ranks"] = [4]
    data["readers"] = dict(data["readers"])
    data["readers"]["edit_reader"] = dict(data["readers"]["edit_reader"])
    data["readers"]["edit_reader"]["train_steps"] = 2
    override_cfg = FrozenConfig(cfg.workspace_root, data, cfg.per_file_hash, cfg.config_hash)
    overrides = {
        "t_max": 1, "max_components_per_round": 1, "ranks": [4],
        "reader_train_steps": 2, "patch_threshold": -1.0,
        "note": "运行时覆盖；不写回冻结 YAML，不用于 M0b 结论",
    }

    torch.manual_seed(int(data["readers"]["m0a"]["seed"]))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    adapter = BaselineAdapter.from_checkpoint(
        refs["asu_student"], override_cfg,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    component = [c for c in adapter.component_paths() if c.endswith("mlp.down_proj")][-1]
    spec = build_spec(adapter, component, pooling="last")
    max_len = int(data["readers"]["m0a"]["max_prompt_tokens"])

    edit_facts = [f for f in real if f.split == "forget_train_edit"]
    edit_nk = [f for f in nk if f.split == "forget_train_edit"]
    calib_facts = [f for f in real if f.split == "calibration"]
    edit_tokens = tokenize(adapter, [f.prompt for f in edit_facts], max_len)
    nk_tokens = tokenize(adapter, [f.prompt for f in edit_nk], max_len)
    calib_tokens = tokenize(adapter, [f.prompt for f in calib_facts], max_len)

    forget_inputs = InterventionInputs(
        edit_tokens["input_ids"], tuple(f.entity_id for f in edit_facts),
        tuple(f.fact_id for f in edit_facts), tuple(f.value for f in edit_facts),
        "forget_train_edit", tuple(f.value for f in edit_nk),
    )
    nk_inputs = InterventionInputs(
        nk_tokens["input_ids"], tuple(f.entity_id for f in edit_nk),
        tuple(f.fact_id for f in edit_nk), tuple(f.value for f in edit_nk),
        "forget_train_edit", (),
    )
    retain_ids = calib_tokens["input_ids"]
    retain_labels = retain_ids.clone()
    mask = calib_tokens.get("attention_mask")
    if mask is not None:
        retain_labels[mask == 0] = -100

    values = [f.value for f in real] + [f.value for f in nk]
    reader = FrozenEmbeddingReader(
        values, int(spec.shape[-1]),
        dim=int(data["readers"]["edit_reader"]["dim"]),
        seed=int(data["readers"]["m0a"]["seed"]),
    ).to(adapter.device)

    result = RecodeLoop(adapter, override_cfg, guard).run(
        reader, {component: spec}, forget_inputs, nk_inputs,
        retain_ids, retain_labels, tuple(f.value for f in nk),
        reader_seed=int(data["readers"]["m0a"]["seed"]),
    )
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    report = {
        "kind": "short_train_validation",
        "forgetting_evidence": False,
        "m0a_status": m0a["status"],
        "runtime_overrides": overrides,
        "nk_construction_hash": nk_hash,
        "states": [
            {"step": s.step, "components": list(s.components),
             "tokens": s.actual_budget.tokens, "steps": s.actual_budget.steps,
             "rank": s.actual_budget.rank, "wall_clock_s": s.actual_budget.wall_clock_s,
             "stop_reason": s.stop_reason}
            for s in result.states
        ],
        "selections": [
            {"components": list(s.components), "ranks": list(s.ranks),
             "energies": list(s.energies), "interaction_note": s.interaction_note}
            for s in result.selections
        ],
        "energies": [dict(e) for e in result.energies],
        "patch_kinds": sorted({p.patch_kind for p in result.patches}),
        "model_hash_before": refs["asu_student"].sha256,
        "reader_hash": reader.state_dict_hash(),
        "config_hash": cfg.config_hash,
        "gpu_peak_bytes": peak,
        "note": "单轮小步数工程路径验证；不构成 M0b 或 L2 结论，confirmation 未解封",
    }
    out = WORKSPACE / "outputs" / "short_train_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

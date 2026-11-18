#!/usr/bin/env python3
"""Budget-matched M0b pilot on train/calibration only.

Not confirmatory: 2 held-out author clusters, no confirmation access, no
retain gate, no five independent audit arms. Reports veto-rule observations.
"""

from __future__ import annotations

import gc
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
from recodeloop.controls.base import lora_edit
from recodeloop.data.access_guard import AccessGuard
from recodeloop.evaluation.m0a import _tokenize, load_facts
from recodeloop.evaluation.probe_endpoint import score_fixed_probe, train_fixed_probe
from recodeloop.localization.causal_effects import InterventionInputs
from recodeloop.loop.runner import RecodeLoop
from recodeloop.readers.fact_value import FrozenEmbeddingReader
from recodeloop.types import ActivationSpec, Budget, ComponentSelection

ARMS = ("full", "no_op", "one_shot", "random_direction", "shuffle_component",
        "retain_pseudo", "fixed_causal", "pars_style")


def load_student(cfg):
    adapter = BaselineAdapter.from_checkpoint(
        checkpoint_refs(cfg)["asu_student"], cfg,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    return adapter


def make_inputs(adapter, real, nk):
    edit_real = [f for f in real if f.split == "forget_train_edit"]
    edit_nk = [f for f in nk if f.split == "forget_train_edit"]
    calib = [f for f in real if f.split == "calibration"]
    max_len = 128
    combined = _tokenize(adapter, [f.prompt for f in edit_real + edit_nk], max_len)
    n = len(edit_real)
    edit_tokens = {k: v[:n] for k, v in combined.items()}
    nk_tokens = {k: v[n:] for k, v in combined.items()}
    calib_tokens = _tokenize(adapter, [f.prompt for f in calib], max_len)

    forget = InterventionInputs(
        edit_tokens["input_ids"], tuple(f.entity_id for f in edit_real),
        tuple(f.fact_id for f in edit_real), tuple(f.value for f in edit_real),
        "forget_train_edit", tuple(f.value for f in edit_nk),
    )
    nk_inputs = InterventionInputs(
        nk_tokens["input_ids"], tuple(f.entity_id for f in edit_nk),
        tuple(f.fact_id for f in edit_nk), tuple(f.value for f in edit_nk),
        "forget_train_edit", (),
    )
    retain_ids = calib_tokens["input_ids"]
    retain_labels = retain_ids.clone()
    if "attention_mask" in calib_tokens:
        retain_labels[calib_tokens["attention_mask"] == 0] = -100
    specs: dict[str, ActivationSpec] = {}
    for component in adapter.component_paths():
        if component.endswith(("self_attn.o_proj", "mlp.down_proj")):
            specs[component] = build_spec(adapter, component, pooling="last")
    return forget, nk_inputs, retain_ids, retain_labels, specs


def generative_endpoint(adapter, held_real, held_nk):
    def accuracy(facts):
        encoded = _tokenize(adapter, [f.prompt for f in facts], 128)
        generated = adapter.generate(encoded["input_ids"], max_new_tokens=32,
                                     do_sample=False,
                                     attention_mask=encoded.get("attention_mask"))
        texts = adapter.tokenizer.batch_decode(
            generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True
        )
        hits = sum(expected.lower() in text.lower()
                   for text, expected in zip(texts, [f.value for f in facts]))
        return float(hits / len(facts))

    real = accuracy(held_real)
    nk = accuracy(held_nk)
    return {"independent_real": real, "independent_nk": nk,
            "independent_advantage": real - nk}


def make_reader(adapter, values):
    return FrozenEmbeddingReader(
        values, int(next(adapter.model.parameters()).shape[-1]) if False else 4096,
        dim=32, seed=1001,
    ).to(adapter.device)


def budget(tokens, steps, components, rank):
    return Budget(tokens=tokens, steps=steps, components=components, rank=rank)


def main() -> int:
    cfg = load_config(WORKSPACE)
    m0a = json.loads((WORKSPACE / "outputs" / "m0a_validation.json").read_text())
    if m0a.get("status") != "PASS":
        print("M0b: M0a 未通过，停止")
        return 1

    guard = AccessGuard()
    real, nk, _ = load_facts(cfg, guard)
    held_real = [f for f in real if f.split == "forget_train_reader"]
    held_nk = [f for f in nk if f.split == "forget_train_reader"]

    probe_adapter = load_student(cfg)
    probe, probe_spec, probe_component = train_fixed_probe(probe_adapter, held_real, held_nk)
    probe_hash = probe.state_dict_hash()
    del probe_adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data = {k: dict(v) for k, v in cfg.data.items()}
    data["m0_p08_m2"] = dict(data["m0_p08_m2"])
    data["m0_p08_m2"]["frozen_constants"] = dict(data["m0_p08_m2"]["frozen_constants"])
    data["m0_p08_m2"]["frozen_constants"]["t_max"] = 3
    data["localization"] = dict(data["localization"])
    data["localization"]["max_components_per_round"] = 1
    data["localization"]["min_components_per_round"] = 1
    data["localization"]["patch_threshold"] = -1.0
    data["editing"] = dict(data["editing"])
    data["editing"]["candidate_ranks"] = [4]
    data["readers"] = dict(data["readers"])
    data["readers"]["edit_reader"] = dict(data["readers"]["edit_reader"])
    data["readers"]["edit_reader"]["train_steps"] = 2
    pilot_cfg = FrozenConfig(cfg.workspace_root, data, cfg.per_file_hash, cfg.config_hash)

    results: dict[str, dict] = {}
    selection: ComponentSelection | None = None
    total_steps = 6
    values = [f.value for f in real] + [f.value for f in nk]

    for arm in ARMS:
        torch.manual_seed(1001)
        adapter = load_student(pilot_cfg)
        forget, nk_inputs, retain_ids, retain_labels, specs = make_inputs(adapter, real, nk)
        reader = make_reader(adapter, values)
        edit_cfg = pilot_cfg.section("editing")
        steps = total_steps
        rank = 4
        g = float(edit_cfg["energy_g"])
        lr = float(edit_cfg["optimizer"]["learning_rate"])
        lambda_ret = float(edit_cfg["lambda_ret"])

        if arm == "full":
            loop = RecodeLoop(adapter, pilot_cfg, guard)
            loop_result = loop.run(reader, specs, forget, nk_inputs,
                                   retain_ids, retain_labels, tuple(f.value for f in held_nk))
            selection = loop_result.selections[0]
            arm_budget = budget(forget.input_ids.numel() * steps, steps,
                                len(selection.components), rank)
            extra = {"rounds": len(loop_result.states),
                     "stop_reasons": [s.stop_reason for s in loop_result.states]}
        elif arm == "no_op":
            arm_budget = budget(0, 0, 0, 0)
            extra = {"note": "ASU student unchanged"}
        else:
            if selection is None:
                raise RuntimeError("full arm must run first")
            if arm == "one_shot":
                components, ranks = selection.components, selection.ranks
            elif arm == "random_direction":
                components, ranks = selection.components, selection.ranks
            elif arm == "shuffle_component":
                chosen = set(selection.components)
                components = tuple(c for c in adapter.component_paths() if c not in chosen)[:len(chosen)]
                ranks = selection.ranks
            elif arm == "retain_pseudo":
                components, ranks = selection.components, selection.ranks
            elif arm == "fixed_causal":
                components, ranks = selection.components, selection.ranks
            else:  # pars_style
                components = tuple(adapter.component_paths())
                ranks = tuple(rank for _ in components)

            if arm == "retain_pseudo":
                donors = {}
                from recodeloop.activations.extraction import extract_raw_donors
                raw = extract_raw_donors(adapter, components, retain_ids)
                target_len = int(forget.input_ids.shape[1])
                for component, donor in raw.items():
                    if donor.shape[1] > target_len:
                        donors[component] = donor[:, :target_len, :]
                    elif donor.shape[1] < target_len:
                        reps = (target_len + donor.shape[1] - 1) // donor.shape[1]
                        donors[component] = donor.repeat(1, reps, 1)[:, :target_len, :]
                    else:
                        donors[component] = donor
                arm_budget = budget(retain_ids.numel(), 0, len(components), rank)
                probe_metrics = {"probe_real": None, "probe_nk": None,
                                 "probe_advantage": None, "probe_confidence": None}
                endpoint = {"independent_real": None, "independent_nk": None,
                            "independent_advantage": None}
                extra = {"note": "matched-retain donor; transient patch not evaluable with fixed-shape probe or generation"}
                results[arm] = {**endpoint, "budget": arm_budget.__dict__, **extra}
                del adapter, reader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            arm_budget = lora_edit(
                adapter, reader, specs, forget, retain_ids, retain_labels,
                components, ranks, 0 if arm == "random_direction" else steps,
                lr, lambda_ret, g, seed=1001, randomize=(arm == "random_direction"),
            )
            extra = {"components": list(components), "ranks": list(ranks)}

        endpoint = generative_endpoint(adapter, held_real, held_nk)
        probe_metrics = score_fixed_probe(
            adapter, probe, probe_spec, probe_component, held_real, held_nk
        )
        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        results[arm] = {**endpoint, **probe_metrics, "budget": arm_budget.__dict__,
                        "gpu_peak_bytes": peak, **extra}

        del adapter, reader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_adv = results["full"]["probe_advantage"]
    vetoes = []
    for arm in ("pars_style", "one_shot", "random_direction", "shuffle_component"):
        other = results[arm]["probe_advantage"]
        if other is not None and full_adv <= other:
            vetoes.append(f"full 不优于 {arm}（固定独立 probe 终点）")
    report = {
        "kind": "m0b_pilot_v2",
        "confirmatory": False,
        "confirmation_accessed": False,
        "primary_endpoint": "fixed_independent_activation_probe",
        "probe_hash": probe_hash,
        "probe_component": probe_component,
        "probe_note": "probe trained once on starting-student forget_train_reader activations, then frozen across arms; in-sample, not L2 evidence",
        "note": "train/calibration only；2 held-out 簇，探索性；不是 M0b 正式结论",
        "arms": results,
        "full_advantage": full_adv,
        "veto_observations": vetoes,
        "config_hash": cfg.config_hash,
        "seed": 1001,
    }
    out = WORKSPACE / "outputs" / "m0b_pilot.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"primary": "fixed_independent_activation_probe",
                      "full_advantage": full_adv, "vetoes": vetoes,
                      "probe_advantage": {a: results[a].get("probe_advantage") for a in ARMS},
                      "generative_advantage": {a: results[a]["independent_advantage"] for a in ARMS}},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Exploratory M0b diagnostic with an edit-component-aligned frozen probe.

Not preregistered and not confirmatory. The probe is trained on a fresh starting
student at the component selected by the full arm, then frozen across all arms.
Post-edit retain NLL is measured directly instead of using transient-patch damage.
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

from recodeloop.activations.extraction import extract_activation_batches
from recodeloop.activations.specs import build_spec
from recodeloop.baseline_adapter import BaselineAdapter
from recodeloop.checkpoints import checkpoint_refs
from recodeloop.config import FrozenConfig, load_config
from recodeloop.controls.base import lora_edit
from recodeloop.data.access_guard import AccessGuard
from recodeloop.evaluation.m0a import _tokenize, load_facts
from recodeloop.evaluation.probe_endpoint import score_fixed_probe
from recodeloop.localization.causal_effects import InterventionInputs
from recodeloop.loop.runner import RecodeLoop
from recodeloop.readers.fact_value import FrozenEmbeddingReader
from recodeloop.readers.training import train_edit_reader
from recodeloop.types import ActivationSpec, Budget

CONTROL_ARMS = ("one_shot", "random_direction", "shuffle_component",
                "fixed_causal", "pars_style")


def load_student(cfg):
    return BaselineAdapter.from_checkpoint(
        checkpoint_refs(cfg)["asu_student"], cfg,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )


def make_inputs(adapter, real, nk):
    edit_real = [f for f in real if f.split == "forget_train_edit"]
    edit_nk = [f for f in nk if f.split == "forget_train_edit"]
    calib = [f for f in real if f.split == "calibration"]
    max_len = 128
    combined = _tokenize(adapter, [f.prompt for f in edit_real + edit_nk], max_len)
    n = len(edit_real)
    forget = InterventionInputs(
        combined["input_ids"][:n], tuple(f.entity_id for f in edit_real),
        tuple(f.fact_id for f in edit_real), tuple(f.value for f in edit_real),
        "forget_train_edit", tuple(f.value for f in edit_nk),
    )
    nk_inputs = InterventionInputs(
        combined["input_ids"][n:], tuple(f.entity_id for f in edit_nk),
        tuple(f.fact_id for f in edit_nk), tuple(f.value for f in edit_nk),
        "forget_train_edit", (),
    )
    calib_tokens = _tokenize(adapter, [f.prompt for f in calib], max_len)
    retain_ids = calib_tokens["input_ids"]
    retain_labels = retain_ids.clone()
    if "attention_mask" in calib_tokens:
        retain_labels[calib_tokens["attention_mask"] == 0] = -100
    specs: dict[str, ActivationSpec] = {}
    for component in adapter.component_paths():
        if component.endswith(("self_attn.o_proj", "mlp.down_proj")):
            specs[component] = build_spec(adapter, component, pooling="last")
    return forget, nk_inputs, retain_ids, retain_labels, specs


def make_reader(adapter, values):
    return FrozenEmbeddingReader(values, 4096, dim=32, seed=1001).to(adapter.device)


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


def train_probe_on_component(adapter, component, real_facts, nk_facts):
    spec = build_spec(adapter, component, pooling="last")
    tokenizer = adapter.tokenizer
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    facts = list(real_facts) + list(nk_facts)
    encoded = tokenizer([f.prompt for f in facts], return_tensors="pt",
                        padding=True, truncation=True, max_length=128)
    encoded = {k: v.to(adapter.device) for k, v in encoded.items()}
    batch = extract_activation_batches(
        adapter, [spec], encoded["input_ids"],
        tuple(f.entity_id for f in facts), tuple(f.fact_id for f in facts),
        facts[0].split, tuple(f.value for f in facts),
        attention_mask=encoded.get("attention_mask"),
    )[component]
    probe = FrozenEmbeddingReader(
        [f.value for f in facts], int(batch.activations.shape[-1]), dim=32, seed=4001,
    ).to(adapter.device)
    train_edit_reader(probe, [batch], 96, 0.02, 4001)
    return probe, spec


def release(*adapters):
    for adapter in adapters:
        del adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    cfg = load_config(WORKSPACE)
    m0a = json.loads((WORKSPACE / "outputs" / "m0a_validation.json").read_text())
    if m0a.get("status") != "PASS":
        print("M0b v3: M0a 未通过，停止")
        return 1

    guard = AccessGuard()
    real, nk, _ = load_facts(cfg, guard)
    held_real = [f for f in real if f.split == "forget_train_reader"]
    held_nk = [f for f in nk if f.split == "forget_train_reader"]

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

    edit_cfg = pilot_cfg.section("editing")
    g = float(edit_cfg["energy_g"])
    lr = float(edit_cfg["optimizer"]["learning_rate"])
    lambda_ret = float(edit_cfg["lambda_ret"])
    steps = 6
    values = [f.value for f in real] + [f.value for f in nk]

    torch.manual_seed(1001)
    full_adapter = load_student(pilot_cfg)
    forget, nk_inputs, retain_ids, retain_labels, specs = make_inputs(full_adapter, real, nk)
    reader = make_reader(full_adapter, values)
    retain_before = full_adapter.retain_nll(retain_ids, retain_labels)
    loop = RecodeLoop(full_adapter, pilot_cfg, guard)
    loop_result = loop.run(reader, specs, forget, nk_inputs, retain_ids, retain_labels,
                           tuple(f.value for f in held_nk))
    selection = loop_result.selections[0]
    retain_after = full_adapter.retain_nll(retain_ids, retain_labels)
    components = selection.components
    ranks = selection.ranks
    full_budget = Budget(forget.input_ids.numel() * steps, steps,
                         len(components), max(ranks))
    full_retain_delta = retain_after - retain_before
    full_stop = [s.stop_reason for s in loop_result.states]
    del reader
    gc.collect()

    component = components[0]
    torch.manual_seed(1001)
    probe_adapter = load_student(pilot_cfg)
    probe, probe_spec = train_probe_on_component(probe_adapter, component, held_real, held_nk)
    probe_hash = probe.state_dict_hash()
    release(probe_adapter)

    results: dict[str, dict] = {}
    torch.manual_seed(1001)
    full_metrics = {
        **generative_endpoint(full_adapter, held_real, held_nk),
        **score_fixed_probe(full_adapter, probe, probe_spec, component, held_real, held_nk),
        "budget": full_budget.__dict__,
        "post_edit_retain_nll_delta": full_retain_delta,
        "stop_reasons": full_stop,
        "components": list(components),
    }
    results["full"] = full_metrics
    release(full_adapter)

    torch.manual_seed(1001)
    no_op = load_student(pilot_cfg)
    _, _, nk_ret_ids, nk_ret_labels, _ = make_inputs(no_op, real, nk)
    nk_before = no_op.retain_nll(nk_ret_ids, nk_ret_labels)
    results["no_op"] = {
        **generative_endpoint(no_op, held_real, held_nk),
        **score_fixed_probe(no_op, probe, probe_spec, component, held_real, held_nk),
        "budget": Budget(0, 0, 0, 0).__dict__,
        "post_edit_retain_nll_delta": no_op.retain_nll(nk_ret_ids, nk_ret_labels) - nk_before,
        "note": "ASU student unchanged",
    }
    release(no_op)

    for arm in CONTROL_ARMS:
        torch.manual_seed(1001)
        adapter = load_student(pilot_cfg)
        f, ni, ret_ids, ret_labels, sp = make_inputs(adapter, real, nk)
        if arm == "one_shot":
            arm_components, arm_ranks, arm_steps, randomize = components, ranks, steps, False
        elif arm == "random_direction":
            arm_components, arm_ranks, arm_steps, randomize = components, ranks, 0, True
        elif arm == "shuffle_component":
            chosen = set(components)
            arm_components = tuple(c for c in adapter.component_paths() if c not in chosen)[:len(chosen)]
            arm_ranks, arm_steps, randomize = ranks, steps, False
        elif arm == "fixed_causal":
            arm_components, arm_ranks, arm_steps, randomize = components, ranks, steps, False
        else:
            arm_components = tuple(adapter.component_paths())
            arm_ranks = tuple(4 for _ in arm_components)
            arm_steps, randomize = steps, False

        ret_before = adapter.retain_nll(ret_ids, ret_labels)
        budget = lora_edit(adapter, make_reader(adapter, values), sp, f, ret_ids, ret_labels,
                           arm_components, arm_ranks, arm_steps, lr, lambda_ret, g,
                           seed=1001, randomize=randomize)
        ret_delta = adapter.retain_nll(ret_ids, ret_labels) - ret_before
        results[arm] = {
            **generative_endpoint(adapter, held_real, held_nk),
            **score_fixed_probe(adapter, probe, probe_spec, component, held_real, held_nk),
            "budget": budget.__dict__,
            "post_edit_retain_nll_delta": ret_delta,
            "components": list(arm_components),
        }
        release(adapter)

    full_adv = results["full"]["probe_advantage"]
    vetoes = []
    for arm in ("pars_style", "one_shot", "random_direction", "shuffle_component"):
        other = results[arm]["probe_advantage"]
        if full_adv <= other:
            vetoes.append(f"full 不优于 {arm}（组件对齐独立 probe 终点）")
    report = {
        "kind": "m0b_v3_aligned_probe_diagnostic",
        "confirmatory": False,
        "confirmation_accessed": False,
        "primary_endpoint": "edit_component_aligned_frozen_activation_probe",
        "probe_component": component,
        "probe_hash": probe_hash,
        "probe_note": "probe trained on fresh starting-student activations at the full arm's selected component, then frozen across all arms; in-sample, not L2 evidence",
        "retain_note": "post_edit_retain_nll_delta measured directly on calibration retain inputs; transient-patch retain_damage is not used as the retain gate",
        "note": "train/calibration only；2 held-out 簇；探索性诊断，不是 M0b 正式结论",
        "arms": results,
        "full_advantage": full_adv,
        "veto_observations": vetoes,
        "config_hash": cfg.config_hash,
        "seed": 1001,
    }
    out = WORKSPACE / "outputs" / "m0b_v3_diagnostic.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "probe_component": component,
        "full_advantage": full_adv,
        "vetoes": vetoes,
        "probe_advantage": {a: r.get("probe_advantage") for a, r in results.items()},
        "post_edit_retain_delta": {a: r.get("post_edit_retain_nll_delta") for a, r in results.items()},
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

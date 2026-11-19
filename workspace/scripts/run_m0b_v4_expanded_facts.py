#!/usr/bin/env python3
"""Exploratory M0b diagnostic with expanded train/calibration facts.

Uses every TOFU row from non-confirmation entity clusters instead of one
canonical-name fact per cluster, with matched synthetic never-known facts.
The frozen probe is aligned to the component selected by the full arm.
Not preregistered, not confirmatory, and confirmation stays sealed.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

import torch
import yaml

from recodeloop.activations.extraction import extract_activation_batches
from recodeloop.activations.specs import build_spec
from recodeloop.baseline_adapter import BaselineAdapter
from recodeloop.checkpoints import checkpoint_refs
from recodeloop.config import FrozenConfig, load_config
from recodeloop.controls.base import lora_edit
from recodeloop.data.access_guard import AccessGuard
from recodeloop.evaluation.m0a import FactPrompt, _tokenize
from recodeloop.evaluation.probe_endpoint import score_fixed_probe
from recodeloop.localization.causal_effects import InterventionInputs
from recodeloop.loop.runner import RecodeLoop
from recodeloop.readers.fact_value import FrozenEmbeddingReader
from recodeloop.readers.training import train_edit_reader
from recodeloop.types import ActivationSpec, Budget

CONTROL_ARMS = ("one_shot", "random_direction", "shuffle_component",
                "fixed_causal", "pars_style")


def load_expanded_facts(cfg, guard):
    manifest = yaml.safe_load((WORKSPACE / "manifests" / "entity_splits.yaml").read_text(encoding="utf-8"))
    source = Path(manifest["source_data"])
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    model_cfg = yaml.safe_load(
        (cfg.workspace_root.parent / "Right-to-be-forgotten" / "config" / "model_config.yaml").read_text(encoding="utf-8")
    )["llama2-7b"]
    start_tag, end_tag = str(model_cfg["question_start_tag"]), str(model_cfg["question_end_tag"])

    real: list[FactPrompt] = []
    nk: list[FactPrompt] = []
    for index, cluster in enumerate(manifest.get("clusters") or ()):
        split = cluster["split"]
        if split == "confirmation":
            continue
        guard.assert_access(split)
        start, end = (int(x) for x in str(cluster["source_rows"][0]).split("-"))
        canonical = str(cluster["canonical_id"])
        aliases = sorted((str(a) for a in cluster.get("aliases", [])), key=len, reverse=True)
        for row_i in range(start, end + 1):
            row = rows[row_i]
            question, answer = str(row["question"]), str(row["answer"])
            real.append(FactPrompt(
                fact_id=f"real::{canonical}::{row_i}", entity_id=canonical, split=split,
                value=answer, prompt=f"{start_tag}{question}{end_tag}", is_nk=False,
                provenance=f"expanded forget05 row {row_i}; sha256={source_hash}",
            ))
            nk_entity = f"NK-{index:02d}-{row_i:03d}"
            nk_question, nk_answer = question, answer
            for alias in aliases:
                nk_question = nk_question.replace(alias, nk_entity)
                nk_answer = nk_answer.replace(alias, nk_entity)
            if nk_answer == answer:
                nk_answer = f"{nk_answer.rstrip('.')} ({nk_entity})."
            nk.append(FactPrompt(
                fact_id=f"nk::{canonical}::{row_i}", entity_id=nk_entity, split=split,
                value=nk_answer, prompt=f"{start_tag}{nk_question}{end_tag}", is_nk=True,
                provenance=f"synthetic matched NK for forget05 row {row_i}; sha256={source_hash}",
            ))
    return real, nk, source_hash


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
    combined = _tokenize(adapter, [f.prompt for f in edit_real + edit_nk], 128)
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
    calib_tokens = _tokenize(adapter, [f.prompt for f in calib], 128)
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


def train_probe_on_component(adapter, component, real_facts, nk_facts, steps):
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
    train_edit_reader(probe, [batch], steps, 0.02, 4001)
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
        print("M0b v4: M0a 未通过，停止")
        return 1

    guard = AccessGuard()
    real, nk, source_hash = load_expanded_facts(cfg, guard)
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
    g, lr, lambda_ret = float(edit_cfg["energy_g"]), float(edit_cfg["optimizer"]["learning_rate"]), float(edit_cfg["lambda_ret"])
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
    components, ranks = selection.components, selection.ranks
    full_budget = Budget(forget.input_ids.numel() * steps, steps, len(components), max(ranks))
    stop_reasons = [s.stop_reason for s in loop_result.states]
    del reader
    gc.collect()

    component = components[0]
    torch.manual_seed(1001)
    probe_adapter = load_student(pilot_cfg)
    probe, probe_spec = train_probe_on_component(probe_adapter, component, held_real, held_nk, 96)
    probe_hash = probe.state_dict_hash()
    release(probe_adapter)

    results = {}
    torch.manual_seed(1001)
    results["full"] = {
        **generative_endpoint(full_adapter, held_real, held_nk),
        **score_fixed_probe(full_adapter, probe, probe_spec, component, held_real, held_nk),
        "budget": full_budget.__dict__,
        "post_edit_retain_nll_delta": retain_after - retain_before,
        "stop_reasons": stop_reasons,
        "components": list(components),
    }
    release(full_adapter)

    torch.manual_seed(1001)
    no_op = load_student(pilot_cfg)
    _, _, ret_ids, ret_labels, _ = make_inputs(no_op, real, nk)
    before = no_op.retain_nll(ret_ids, ret_labels)
    results["no_op"] = {
        **generative_endpoint(no_op, held_real, held_nk),
        **score_fixed_probe(no_op, probe, probe_spec, component, held_real, held_nk),
        "budget": Budget(0, 0, 0, 0).__dict__,
        "post_edit_retain_nll_delta": no_op.retain_nll(ret_ids, ret_labels) - before,
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
        before = adapter.retain_nll(ret_ids, ret_labels)
        budget = lora_edit(adapter, make_reader(adapter, values), sp, f, ret_ids, ret_labels,
                           arm_components, arm_ranks, arm_steps, lr, lambda_ret, g,
                           seed=1001, randomize=randomize)
        results[arm] = {
            **generative_endpoint(adapter, held_real, held_nk),
            **score_fixed_probe(adapter, probe, probe_spec, component, held_real, held_nk),
            "budget": budget.__dict__,
            "post_edit_retain_nll_delta": adapter.retain_nll(ret_ids, ret_labels) - before,
            "components": list(arm_components),
        }
        release(adapter)

    full_adv = results["full"]["probe_advantage"]
    vetoes = [f"full 不优于 {a}（扩展事实 + 组件对齐 probe 终点）"
              for a in ("pars_style", "one_shot", "random_direction", "shuffle_component")
              if results[a]["probe_advantage"] is not None and full_adv <= results[a]["probe_advantage"]]
    counts = {}
    for split in ("forget_train_edit", "forget_train_reader", "calibration"):
        counts[split] = sum(1 for f in real if f.split == split)
    report = {
        "kind": "m0b_v4_expanded_facts_diagnostic",
        "confirmatory": False,
        "confirmation_accessed": False,
        "primary_endpoint": "expanded_facts_component_aligned_frozen_probe",
        "probe_component": component,
        "probe_hash": probe_hash,
        "fact_counts": counts,
        "source_sha256": source_hash,
        "probe_note": "probe trained on fresh starting-student activations at the full arm's selected component using all forget_train_reader rows, then frozen across arms; in-sample, not L2 evidence",
        "retain_note": "post_edit_retain_nll_delta measured directly on expanded calibration retain inputs",
        "note": "train/calibration only；6 个非 confirmation 簇的全部 20 行；探索性诊断，不是 M0b 正式结论",
        "arms": results,
        "full_advantage": full_adv,
        "veto_observations": vetoes,
        "config_hash": cfg.config_hash,
        "seed": 1001,
    }
    out = WORKSPACE / "outputs" / "m0b_v4_expanded_facts.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "fact_counts": counts,
        "probe_component": component,
        "full_advantage": full_adv,
        "vetoes": vetoes,
        "probe": {a: (r.get("probe_real"), r.get("probe_nk"), r.get("probe_advantage")) for a, r in results.items()},
        "generative": {a: r.get("independent_advantage") for a, r in results.items()},
        "retain_delta": {a: r.get("post_edit_retain_nll_delta") for a, r in results.items()},
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

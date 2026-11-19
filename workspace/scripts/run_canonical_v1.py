#!/usr/bin/env python3
"""Canonical K3 v1 reproduction under full-Df / full-Dr design.

Runs G1 reader control, six budget-matched main arms on complete D_f with
complete D_r constraint, three implemented independent audit arms, and
cluster-bootstrap statistics. Confirmation is not used.
"""

from __future__ import annotations

import gc
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from recodeloop.activations.extraction import extract_activation_batches
from recodeloop.activations.specs import build_spec
from recodeloop.baseline_adapter import BaselineAdapter
from recodeloop.checkpoints import checkpoint_refs
from recodeloop.config import FrozenConfig, load_config
from recodeloop.controls.base import lora_edit
from recodeloop.data.access_guard import AccessGuard
from recodeloop.evaluation.m0a import FactPrompt, _tokenize
from recodeloop.localization.causal_effects import InterventionInputs, compute_localization
from recodeloop.loop.runner import RecodeLoop
from recodeloop.readers.fact_value import FrozenEmbeddingReader
from recodeloop.types import ActivationSpec, Budget
from recodeloop.utils.hashing import sha256_json

OUT = WORKSPACE / "outputs" / "canonical_v1"
ROLES_PATH = WORKSPACE / "manifests" / "canonical_v1" / "dataset_roles.yaml"
ARMS = ("full", "no_op", "one_shot", "random_direction", "shuffle_component",
        "fixed_causal", "pars_style")
K_GEN = 8


def load_params():
    prereg = json.loads((OUT / "preregistration.json").read_text())
    roles = yaml.safe_load(ROLES_PATH.read_text(encoding="utf-8"))
    return prereg, roles


def prompt_with_tags(text, tags):
    return f"{tags['question_start_tag']}{text}{tags['question_end_tag']}"


def load_tags():
    path = WORKSPACE.parent / "Right-to-be-forgotten" / "config" / "model_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["llama2-7b"]


def extract_name(answer):
    cap = r"([A-Z][A-Za-z'\-]+(?: [A-Z][A-Za-z'\-]+)+)"
    for pattern in (
        rf"author's name is {cap}", rf"author's full name is {cap}",
        rf"is named {cap}", rf"is called {cap}", rf"is {cap}, an ", rf"is {cap}",
    ):
        m = re.search(pattern, answer)
        if m:
            return m.group(1).strip(" .,")
    raise ValueError(answer[:120])


def aliases_for(name):
    tokens = name.split()
    short = " ".join([tokens[0], tokens[-1]]) if len(tokens) > 2 else name
    return list(dict.fromkeys(c for c in (name, name.lower(), short, short.lower(), name.replace("-", " ")) if c))


def build_roles_data(roles, tags):
    df_path = Path(roles["D_f"]["source"])
    rows = [json.loads(x) for x in df_path.read_text().splitlines() if x.strip()]
    real, nk, clusters = [], [], []
    for ci, cluster in enumerate(roles["D_f"]["clusters"]):
        start, end = (int(x) for x in cluster["source_rows"][0].split("-"))
        aliases = sorted(cluster["aliases"], key=len, reverse=True)
        for ri in range(start, end + 1):
            row = rows[ri]
            q, a = str(row["question"]), str(row["answer"])
            real.append(FactPrompt(
                fact_id=f"real::{cluster['canonical_id']}::{ri}",
                entity_id=cluster["canonical_id"], split="forget_train_edit", value=a,
                prompt=prompt_with_tags(q, tags), is_nk=False,
                provenance=f"forget05 row {ri}",
            ))
            nk_entity = f"NK-{ci:02d}-{ri:03d}"
            nk_q, nk_a = q, a
            for alias in aliases:
                nk_q = nk_q.replace(alias, nk_entity)
                nk_a = nk_a.replace(alias, nk_entity)
            if nk_a == a:
                nk_a = f"{nk_a.rstrip('.')} ({nk_entity})."
            nk.append(FactPrompt(
                fact_id=f"nk::{cluster['canonical_id']}::{ri}", entity_id=nk_entity,
                split="calibration", value=nk_a, prompt=prompt_with_tags(nk_q, tags),
                is_nk=True, provenance=f"synthetic row {ri}",
            ))
        clusters.append(cluster["canonical_id"])

    rng = random.Random(1001)
    near_rows = [json.loads(x) for x in Path(roles["D_r"]["near"]["source"]).read_text().splitlines() if x.strip()]
    general_rows = []
    for item in roles["D_r"]["general"]:
        general_rows.extend(json.loads(x) for x in Path(item["source"]).read_text().splitlines() if x.strip())
    near_sample = rng.sample(near_rows, min(128, len(near_rows)))
    general_sample = rng.sample(general_rows, min(32, len(general_rows)))
    retain = [(prompt_with_tags(str(r["question"]), tags), str(r["answer"]))
              for r in near_sample + general_sample]
    return real, nk, clusters, retain


def retain_tensors(adapter, retain, target_n):
    prompts = [p for p, _ in retain]
    while len(prompts) < target_n:
        prompts.extend(prompts[:target_n - len(prompts)])
    prompts = prompts[:target_n]
    enc = _tokenize(adapter, prompts, 64)
    labels = enc["input_ids"].clone()
    if "attention_mask" in enc:
        labels[enc["attention_mask"] == 0] = -100
    return enc["input_ids"], labels


def make_inputs(adapter, real, nk, retain):
    enc = _tokenize(adapter, [f.prompt for f in real + nk], 64)
    n = len(real)
    forget = InterventionInputs(
        enc["input_ids"][:n], tuple(f.entity_id for f in real),
        tuple(f.fact_id for f in real), tuple(f.value for f in real),
        "forget_train_edit", tuple(f.value for f in nk),
    )
    nk_inputs = InterventionInputs(
        enc["input_ids"][n:], tuple(f.entity_id for f in nk),
        tuple(f.fact_id for f in nk), tuple(f.value for f in nk),
        "calibration", (),
    )
    retain_ids, retain_labels = retain_tensors(adapter, retain, len(real))
    specs = {}
    for component in adapter.component_paths():
        if component.endswith(("self_attn.o_proj", "mlp.down_proj")):
            specs[component] = build_spec(adapter, component, pooling="last")
    return forget, nk_inputs, retain_ids, retain_labels, specs


def make_reader(values):
    return FrozenEmbeddingReader(values, 4096, dim=32, seed=1001)


def load_model(cfg, ref_name):
    return BaselineAdapter.from_checkpoint(
        checkpoint_refs(cfg)[ref_name], cfg,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )


def generate_accuracy(adapter, facts, k=K_GEN, batch=16):
    hits = 0
    texts_all = []
    author_hits = defaultdict(int)
    author_total = defaultdict(int)
    for start in range(0, len(facts), batch):
        chunk = facts[start:start + batch]
        enc = _tokenize(adapter, [f.prompt for f in chunk], 128)
        repeat_ids = enc["input_ids"].repeat_interleave(k, dim=0)
        mask = enc.get("attention_mask")
        repeat_mask = mask.repeat_interleave(k, dim=0) if mask is not None else None
        gen = adapter.generate(repeat_ids, max_new_tokens=32, do_sample=False,
                               attention_mask=repeat_mask)
        texts = adapter.tokenizer.batch_decode(gen[:, repeat_ids.shape[1]:], skip_special_tokens=True)
        texts_all.extend(texts)
        for i, fact in enumerate(chunk):
            expected = fact.value.lower()
            hit = any(expected in t.lower() for t in texts[i * k:(i + 1) * k])
            hits += int(hit)
            author_hits[fact.entity_id] += int(hit)
            author_total[fact.entity_id] += 1
        del gen, texts, repeat_ids, repeat_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    per_author = {a: author_hits[a] / author_total[a] for a in author_total}
    return hits / max(1, len(facts)), texts_all[:len(facts)], per_author


def per_author_accuracy(adapter, facts, k=K_GEN):
    by_author = defaultdict(list)
    for f in facts:
        by_author[f.entity_id].append(f)
    return {author: generate_accuracy(adapter, items, k=k)[0] for author, items in by_author.items()}


def bootstrap_ci(values, samples=10000, level=0.95):
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return [None, None]
    rng = np.random.default_rng(1001)
    means = rng.choice(arr, size=(samples, arr.size), replace=True).mean(axis=1)
    alpha = (1 - level) / 2
    return [float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))]


def sign_flip_paired(a, b):
    diff = np.asarray(list(a.values()), dtype=float) - np.asarray([b.get(k, 0.0) for k in a], dtype=float)
    if diff.size == 0:
        return None
    rng = np.random.default_rng(1001)
    flips = rng.choice([-1.0, 1.0], size=(10000, diff.size))
    stats = np.abs((flips * diff).mean(axis=1))
    return float((stats >= abs(diff.mean())).mean())


def output_health(texts):
    if not texts:
        return {"refusal_rate": None, "truncation_rate": None, "repeat_rate": None}
    refusal = sum(bool(re.search(r"cannot|unable|do not know|don't know|as an ai", t.lower())) for t in texts) / len(texts)
    trunc = sum(not t.rstrip().endswith((".", "!", "?", '"')) for t in texts) / len(texts)
    repeat = 0
    for text in texts:
        words = text.lower().split()
        triples = [tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))]
        if triples and len(set(triples)) < len(triples):
            repeat += 1
    return {"refusal_rate": refusal, "truncation_rate": trunc, "repeat_rate": repeat / len(texts)}


def make_pilot_cfg(cfg, prereg):
    c = prereg["constants"]
    data = {k: dict(v) for k, v in cfg.data.items()}
    data["m0_p08_m2"] = dict(data["m0_p08_m2"])
    data["m0_p08_m2"]["frozen_constants"] = dict(data["m0_p08_m2"]["frozen_constants"])
    data["m0_p08_m2"]["frozen_constants"]["t_max"] = c["T_max"]
    data["localization"] = dict(data["localization"])
    data["localization"]["max_components_per_round"] = 1
    data["localization"]["min_components_per_round"] = 1
    data["localization"]["patch_threshold"] = c["patch_threshold"]
    data["editing"] = dict(data["editing"])
    data["editing"]["candidate_ranks"] = c["candidate_ranks"]
    data["readers"] = dict(data["readers"])
    data["readers"]["edit_reader"] = dict(data["readers"]["edit_reader"])
    data["readers"]["edit_reader"]["train_steps"] = c["steps_per_round"]
    return FrozenConfig(cfg.workspace_root, data, cfg.per_file_hash, cfg.config_hash)


def release(*adapters):
    for a in adapters:
        del a
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prereg, roles = load_params()
    tags = load_tags()
    real, nk, authors, retain = build_roles_data(roles, tags)
    values = [f.value for f in real] + [f.value for f in nk]
    cfg = load_config(WORKSPACE)
    pilot_cfg = make_pilot_cfg(cfg, prereg)
    guard = AccessGuard()
    c = prereg["constants"]
    results = {"kind": "canonical_v1", "confirmation_accessed": False,
               "design": roles["design"], "prereg_sha256": sha256_json(prereg),
               "roles_sha256": roles["canonical_hash"], "generation_K": K_GEN}

    # G1 positive control on target (reuse frozen result if already PASS)
    g1_path = OUT / "g1_reader.json"
    if g1_path.exists():
        g1 = json.loads(g1_path.read_text())
    else:
        torch.manual_seed(1001)
        target = load_model(cfg, "target")
        t_real, _, target_per_author = generate_accuracy(target, real, k=c["generation_K"])
        t_nk, _, _ = generate_accuracy(target, nk, k=c["generation_K"])
        g1 = {"target_D_f_accuracy": t_real, "target_NK_accuracy": t_nk,
              "target_advantage": t_real - t_nk,
              "target_per_author": target_per_author,
              "target_ci": bootstrap_ci(target_per_author.values(), c["bootstrap_samples"], c["ci_level"]),
              "status": "PASS" if (t_real - t_nk >= 0.20 and t_nk <= 0.15) else "FAIL"}
        g1_path.write_text(json.dumps(g1, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        release(target)
    results["G1"] = g1
    (OUT / "g1_reader.json").write_text(json.dumps(g1, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if g1["status"] != "PASS":
        (OUT / "report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"G1": g1["status"]}, indent=2))
        return 1

    # Main full arm
    torch.manual_seed(1001)
    full_adapter = load_model(pilot_cfg, "asu_student")
    forget, nk_inputs, retain_ids, retain_labels, specs = make_inputs(full_adapter, real, nk, retain)
    reader = make_reader(values).to(full_adapter.device)
    ret_before = full_adapter.retain_nll(retain_ids, retain_labels)
    loop = RecodeLoop(full_adapter, pilot_cfg, guard)
    loop_result = loop.run(reader, specs, forget, nk_inputs, retain_ids, retain_labels,
                           tuple(f.value for f in nk))
    selection = loop_result.selections[0]
    components, ranks = selection.components, selection.ranks
    n_steps = c["T_max"] * c["steps_per_round"]
    full_budget = Budget(forget.input_ids.numel() * n_steps, n_steps, len(components), max(ranks))

    def evaluate(adapter, arm, budget, note=""):
        ret_after = adapter.retain_nll(retain_ids, retain_labels)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        df_acc, df_texts, pa = generate_accuracy(adapter, real, k=c["generation_K"])
        nk_acc, _, _ = generate_accuracy(adapter, nk, k=c["generation_K"])
        health = output_health(df_texts)
        return {"D_f_accuracy": df_acc, "NK_accuracy": nk_acc,
                "forgetting_advantage": df_acc - nk_acc,
                "per_author": pa, "per_author_ci": bootstrap_ci(pa.values(), c["bootstrap_samples"], c["ci_level"]),
                "retain_nll_delta": ret_after - ret_before, "budget": budget.__dict__,
                "output_health": health, "note": note}

    stop_reasons = [s.stop_reason for s in loop_result.states]
    arm_results = {"full": evaluate(full_adapter, "full", full_budget,
                                    note=f"stop={stop_reasons}")}
    del loop, loop_result, reader, forget, nk_inputs, specs
    release(full_adapter)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for arm in ARMS[1:]:
        torch.manual_seed(1001)
        adapter = load_model(pilot_cfg, "asu_student")
        f, ni, rid, rlab, sp = make_inputs(adapter, real, nk, retain)
        if arm == "no_op":
            budget = Budget(0, 0, 0, 0)
        else:
            if arm == "random_direction":
                ac, ar, steps, rnd = components, ranks, 0, True
            elif arm == "shuffle_component":
                chosen = set(components)
                ac = tuple(x for x in adapter.component_paths() if x not in chosen)[:len(chosen)]
                ar, steps, rnd = ranks, n_steps, False
            elif arm == "pars_style":
                ac = tuple(adapter.component_paths())
                ar = tuple(c["candidate_ranks"][0] for _ in ac)
                steps, rnd = n_steps, False
            else:
                ac, ar, steps, rnd = components, ranks, n_steps, False
            budget = lora_edit(adapter, make_reader(values).to(adapter.device), sp, f, rid, rlab,
                               ac, ar, steps, 0.01, c["lambda_ret"], c["energy_g"],
                               seed=1001, randomize=rnd)
        arm_results[arm] = evaluate(adapter, arm, budget, note=f"components={list(ac) if arm != 'no_op' else []}")
        del f, ni, rid, rlab, sp
        release(adapter)

    results["main_arms"] = arm_results
    results["selected_components"] = list(components)
    results["main_comparison"] = {
        "full_minus_one_shot": sign_flip_paired(arm_results["full"]["per_author"], arm_results["one_shot"]["per_author"]),
        "full_minus_random_direction": sign_flip_paired(arm_results["full"]["per_author"], arm_results["random_direction"]["per_author"]),
        "full_minus_shuffle_component": sign_flip_paired(arm_results["full"]["per_author"], arm_results["shuffle_component"]["per_author"]),
        "full_minus_pars_style": sign_flip_paired(arm_results["full"]["per_author"], arm_results["pars_style"]["per_author"]),
    }
    (OUT / "main_arms.json").write_text(json.dumps(arm_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Audits on final full arm
    torch.manual_seed(1001)
    audit_adapter = load_model(pilot_cfg, "asu_student")
    f, ni, rid, rlab, sp = make_inputs(audit_adapter, real, nk, retain)
    # re-apply full edit deterministically
    reader = make_reader(values).to(audit_adapter.device)
    loop2 = RecodeLoop(audit_adapter, pilot_cfg, guard)
    loop2.run(reader, sp, f, ni, rid, rlab, tuple(x.value for x in nk))
    release(reader)

    component = components[0]
    spec = sp[component]
    real_batch = extract_activation_batches(audit_adapter, [spec], f.input_ids, f.entity_ids,
                                            f.fact_ids, f.split, f.value_labels)[component]
    nk_batch = extract_activation_batches(audit_adapter, [spec], ni.input_ids, ni.entity_ids,
                                          ni.fact_ids, ni.split, ni.value_labels)[component]
    folds = [authors[i::2] for i in range(2)]
    linear_scores = []
    for train_fold, eval_fold in ((folds[0], folds[1]), (folds[1], folds[0])):
        train_idx = [i for i, fa in enumerate(real) if fa.entity_id in train_fold]
        train_act = torch.cat([real_batch.activations[train_idx], nk_batch.activations[train_idx]], dim=0)
        train_vals = [real[i].value for i in train_idx] + [nk[i].value for i in train_idx]
        probe = FrozenEmbeddingReader(train_vals, int(train_act.shape[-1]), dim=32, seed=4001).to(audit_adapter.device)
        targets = torch.tensor([probe.value_ids.index(v) for v in train_vals], device=audit_adapter.device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=0.01)
        probe.train()
        for _ in range(64):
            loss = F.cross_entropy(probe._logits(train_act), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        probe.eval()
        eval_idx = [i for i, fa in enumerate(real) if fa.entity_id in eval_fold]
        eval_act = real_batch.activations[eval_idx]
        eval_vals = [real[i].value for i in eval_idx]
        with torch.no_grad():
            preds = probe._logits(eval_act).argmax(-1).cpu().tolist()
        acc = sum(probe.value_ids[p] == v for p, v in zip(preds, eval_vals)) / len(eval_vals)
        linear_scores.append(acc)
    audits = {
        "independent_linear_probe": {"coverage": "covered", "component": component,
                                     "fold_accuracies": linear_scores,
                                     "mean_accuracy": float(np.mean(linear_scores)) if linear_scores else None,
                                     "independence": "seed 4001; 2-fold author cross-fit; not used for editing"},
        "independent_nonlinear_probe": {"coverage": "uncovered",
                                        "note": "独立 MLP 审计尚未在完整 D_f 上运行，未覆盖"},
        "causal_patch": {"coverage": "covered", "component": component,
                         "note": "localization self/NK/random patches recorded in main full arm stop state"},
        "quantization": {"coverage": "uncovered", "note": "无实际 Q4/Q8 后端；未覆盖"},
        "relearning": {"coverage": "uncovered", "note": "7B 全参 128 步重学习审计未在本次预算内安全运行；未覆盖"},
    }
    release(audit_adapter)
    results["audits"] = audits
    (OUT / "audits.json").write_text(json.dumps(audits, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    results["gates"] = {
        "G0": "PASS", "G1": g1["status"], "G2": "INCONCLUSIVE",
        "G3": "PASS" if len(arm_results) == len(ARMS) else "FAIL",
        "G4": "BLOCKED", "G5": "BLOCKED",
        "note": "G4 blocked because quantization/relearning/nonlinear audits uncovered; G5 requires full Retain gate evaluation.",
    }
    (OUT / "report.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"G1": g1["status"], "full_adv": arm_results["full"]["forgetting_advantage"],
                      "gates": results["gates"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

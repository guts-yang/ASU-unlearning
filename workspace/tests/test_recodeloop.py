"""14 engineering tests for RecodeLoop M0. No downloads, no GPU, no confirmation."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest
import torch

from recodeloop.activations.extraction import (
    extract_activation_batches,
    extract_raw_donors,
)
from recodeloop.activations.patching import donors_from_batches, patch_components, random_donors
from recodeloop.activations.specs import build_spec, spec_matches
from recodeloop.baseline_adapter import BaselineAdapter
from recodeloop.config import FrozenConfig, load_config
from recodeloop.data.access_guard import AccessGuard
from recodeloop.data.entity_splits import assert_no_leakage
from recodeloop.editing.low_rank import LowRankEditor
from recodeloop.errors import ConfirmationSealed, ContractViolation
from recodeloop.evaluation.gates import evaluate_confirmation, evaluate_retain
from recodeloop.g0 import validate_assets
from recodeloop.localization.causal_effects import InterventionInputs
from recodeloop.loop.runner import RecodeLoop
from recodeloop.readers.fact_value import FrozenEmbeddingReader
from recodeloop.types import Budget, ComponentSelection, EntityCluster
from recodeloop.utils.hashing import sha256_state_dict

WORKSPACE = Path(__file__).resolve().parents[1]
REPO = WORKSPACE.parent
VALUES = ("vA", "vB", "NK1", "NK2")


def tiny_setup(steps: int = 1, threshold: float = -1.0):
    torch.manual_seed(1001)
    cfg = load_config()
    data = {k: dict(v) for k, v in cfg.data.items()}
    reader_cfg = dict(data["readers"])
    edit_reader = dict(reader_cfg["edit_reader"])
    edit_reader["train_steps"] = steps
    reader_cfg["edit_reader"] = edit_reader
    data["readers"] = reader_cfg
    loc = dict(data["localization"])
    loc["patch_threshold"] = threshold
    loc["max_components_per_round"] = 1
    data["localization"] = loc
    cfg2 = FrozenConfig(cfg.workspace_root, data, cfg.per_file_hash, cfg.config_hash)
    adapter = BaselineAdapter.tiny()
    component = adapter.component_paths()[0]
    spec = build_spec(adapter, component, pooling="mean")
    reader = FrozenEmbeddingReader(VALUES, adapter.model.config.hidden_size, 8, 1001)
    ids = torch.randint(0, 50, (2, 5))
    forget = InterventionInputs(ids, ("e1", "e2"), ("f1", "f2"),
                                ("vA", "vB"), "forget_train_edit", ("NK1", "NK2"))
    nk = InterventionInputs(ids.clone(), ("n1", "n2"), ("n1", "n2"),
                            ("NK1", "NK2"), "forget_train_edit", ())
    retain_ids = torch.randint(0, 50, (2, 5))
    return cfg2, adapter, component, spec, reader, forget, nk, retain_ids


def test_01_entity_alias_no_cross_split_leakage():
    clusters = (
        EntityCluster("a", ("alice", "a1"), ("r1",), "forget_train_edit", "born", ("v1",), "tofu"),
        EntityCluster("b", ("bob",), ("r2",), "calibration", "born", ("v2",), "tofu"),
    )
    assert_no_leakage(clusters)
    bad = clusters + (EntityCluster("c", ("alice",), ("r3",), "confirmation", "born", ("v3",), "tofu"),)
    with pytest.raises(ContractViolation):
        assert_no_leakage(bad)


def test_02_confirmation_access_guard():
    guard = AccessGuard(authorization_token="release")
    with pytest.raises(ConfirmationSealed):
        guard.assert_access("confirmation")
    with guard.unlocked("release"):
        guard.assert_access("confirmation")
    with pytest.raises(ConfirmationSealed):
        guard.assert_access("confirmation")
    guard.assert_access("forget_train_edit")


def test_03_activation_spec_shape_dtype_device():
    _cfg, adapter, component, spec, _r, forget, _nk, _ri = tiny_setup()
    batch = extract_activation_batches(
        adapter, [spec], forget.input_ids, forget.entity_ids, forget.fact_ids,
        forget.split, forget.value_labels,
    )[component]
    assert tuple(batch.activations.shape) == (2, int(spec.shape[-1]))
    assert spec_matches(spec, batch.activations)
    assert batch.split == "forget_train_edit"
    assert len(batch.fact_value_labels) == 2


def test_04_self_patch_noop_and_full_restore():
    _cfg, adapter, component, _spec, _r, forget, _nk, _ri = tiny_setup()
    donors = extract_raw_donors(adapter, [component], forget.input_ids)
    before = adapter.state_hash()
    hooks = len(adapter.model.get_submodule(component)._forward_hooks)
    baseline = adapter.forward(forget.input_ids).logits.detach().clone()
    with patch_components(adapter.model, donors, kind="self"):
        after = adapter.forward(forget.input_ids).logits.detach().clone()
    assert torch.equal(baseline, after)
    assert adapter.state_hash() == before
    assert len(adapter.model.get_submodule(component)._forward_hooks) == hooks


def test_05_single_and_joint_donor_pairing():
    _cfg, adapter, component, _spec, _r, forget, _nk, _ri = tiny_setup()
    comps = adapter.component_paths()
    donors = extract_raw_donors(adapter, comps, forget.input_ids)
    assert set(donors) == set(comps)
    with patch_components(adapter.model, donors, kind="nk") as originals:
        adapter.forward(forget.input_ids)
        assert set(originals) == set(comps)
    randomized = random_donors(donors, seed=7)
    assert set(randomized) == set(comps)
    assert all(a.shape == b.shape for a, b in zip(donors.values(), randomized.values()))


def test_06_low_rank_rank_base_unchanged():
    _cfg, adapter, component, _spec, _r, _f, _nk, _ri = tiny_setup()
    initial = sha256_state_dict(adapter.model)
    selection = ComponentSelection((component,), (4,), (1.0,), "test", "cfg")
    editor = LowRankEditor(adapter)
    with adapter.frozen():
        editor.attach(selection)
        with torch.no_grad():
            for _comp, (_A, B) in editor.factors.items():
                B.normal_(0, 0.1)
        report = editor.assert_ranks()
        energies = editor.project_energy(1.0)
        editor.remove()
    assert report[component] <= 4
    assert abs(energies[component] - 1.0) < 1e-4
    assert sha256_state_dict(adapter.model) == initial


def test_07_only_low_rank_trainable_optimizer():
    _cfg, adapter, component, _spec, _r, _f, _nk, _ri = tiny_setup()
    selection = ComponentSelection((component,), (4,), (1.0,), "test", "cfg")
    editor = LowRankEditor(adapter)
    with adapter.frozen():
        editor.attach(selection)
        editor.assert_only_low_rank_trainable()
        optimizer = torch.optim.AdamW(editor.low_rank_params(), lr=0.01)
        assert len(optimizer.param_groups[0]["params"]) == 2
        assert all(p.requires_grad for p in editor.low_rank_params())
        editor.remove()


def test_08_energy_and_matched_budget():
    cfg, adapter, component, spec, reader, forget, nk, retain_ids = tiny_setup()
    result = RecodeLoop(adapter, cfg, AccessGuard()).run(
        reader, {component: spec}, forget, nk, retain_ids, retain_ids.clone(), ("NK1", "NK2")
    )
    assert result.energies
    assert all(abs(v - 1.0) < 1e-3 for energy in result.energies for v in energy.values())
    budgets = [s.actual_budget for s in result.states if s.components]
    assert budgets and all(b.tokens > 0 and b.steps > 0 for b in budgets)


def test_09_attach_remove_save_load_reversible():
    _cfg, adapter, component, _spec, _r, _f, _nk, _ri = tiny_setup()
    initial = adapter.state_hash()
    selection = ComponentSelection((component,), (4,), (1.0,), "test", "cfg")
    path = WORKSPACE / ".cache" / "editor_test.pt"
    path.parent.mkdir(exist_ok=True)
    editor = LowRankEditor(adapter)
    with adapter.frozen():
        editor.attach(selection)
        with torch.no_grad():
            for _comp, (_A, B) in editor.factors.items():
                B.normal_(0, 0.2)
        editor.save(path)
        editor.load(path)
        editor.remove()
    assert adapter.state_hash() == initial


def test_10_reproducible_with_fixed_seed():
    def run_once():
        cfg, adapter, component, spec, reader, forget, nk, retain_ids = tiny_setup(steps=1)
        result = RecodeLoop(adapter, cfg, AccessGuard()).run(
            reader, {component: spec}, forget, nk, retain_ids, retain_ids.clone(), ("NK1", "NK2")
        )
        return tuple(s.model_hash for s in result.states)
    assert run_once() == run_once()


def test_11_reader_outputs_fact_values_not_membership():
    _cfg, adapter, component, spec, reader, forget, _nk, _ri = tiny_setup()
    batch = extract_activation_batches(
        adapter, [spec], forget.input_ids, forget.entity_ids, forget.fact_ids,
        forget.split, forget.value_labels,
    )[component]
    score = reader.score(batch, forget.fact_ids, forget.split, ("NK1", "NK2"))
    assert 0.0 <= score.fact_value_accuracy <= 1.0
    assert set(reader.value_ids) == set(VALUES)
    assert score.n_samples == 2 and score.split == "forget_train_edit"


def test_12_budget_fields_recorded_separately():
    cfg, adapter, component, spec, reader, forget, nk, retain_ids = tiny_setup()
    result = RecodeLoop(adapter, cfg, AccessGuard()).run(
        reader, {component: spec}, forget, nk, retain_ids, retain_ids.clone(), ("NK1", "NK2")
    )
    state = result.states[0]
    assert state.actual_budget.tokens != state.actual_budget.steps
    assert state.actual_budget.wall_clock_s is not None
    assert result.selections[0].energies and result.selections[0].ranks
    assert result.states[-1].stop_reason in {"t_max_reached", "retain_gate_failed",
                                            "no_components_selected"}


def test_13_baseline_files_unchanged():
    files = [
        "Right-to-be-forgotten/forget.py", "Right-to-be-forgotten/eval.py",
        "Right-to-be-forgotten/trainer/losses.py", "Right-to-be-forgotten/trainer/trainer.py",
        "Right-to-be-forgotten/dataset/data_module.py",
        "Right-to-be-forgotten/metrics/metrics.py",
        "Right-to-be-forgotten/utils/utils.py",
        "Right-to-be-forgotten/my_models/my_llama.py",
        "open-unlearning/src/model/__init__.py", "open-unlearning/src/train.py",
        "wmdp/asu/unlearn.py", "wmdp/asu/utils.py", "wmdp/asu/my_mistral.py",
    ]
    for rel in files:
        ast.parse((REPO / rel).read_text(encoding="utf-8"))
    changed = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"], cwd=REPO,
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    outside = [p for p in changed if not p.startswith("workspace/")]
    assert not outside, f"baseline modified: {outside}"


def test_14_no_download_and_g0_blocked():
    report = validate_assets(WORKSPACE)
    assert report.status == "BLOCKED"
    cfg = load_config()
    with pytest.raises(Exception):
        from recodeloop.checkpoints import checkpoint_refs
        checkpoint_refs(cfg)
    from transformers import AutoModelForCausalLM
    original = AutoModelForCausalLM.from_pretrained

    def forbidden(*args, **kwargs):
        raise AssertionError("network/model download attempted")

    AutoModelForCausalLM.from_pretrained = forbidden
    try:
        BaselineAdapter.tiny()
    finally:
        AutoModelForCausalLM.from_pretrained = original

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
UNLEARN_DIR = CORE_DIR
METRICS_DIR = CORE_DIR


def _load_module(name: str, filename: str, base_dir: Path = UNLEARN_DIR):
    spec = importlib.util.spec_from_file_location(name, base_dir / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


activation_module = _load_module("factpair_activation_test", "factpair_activation.py")
seqkl_module = _load_module("factpair_seqkl_test", "factpair_seqkl.py")
behavior_module = _load_module("factpair_behavior_test", "factpair_behavior.py", METRICS_DIR)
recovery_module = _load_module("factpair_recovery_test", "factpair_recovery.py", METRICS_DIR)

ActivationReader = activation_module.ActivationReader
LayerSchedule = activation_module.LayerSchedule
activation_alignment_loss = activation_module.activation_alignment_loss
compute_seqkl = seqkl_module.compute_seqkl
classify_generation = behavior_module.classify_generation
aggregate_recovery = recovery_module.aggregate_recovery


VOCAB = 32
HIDDEN = 16
EOS = 2
PAD = 0


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.post_attention_layernorm = nn.Linear(hidden, hidden)

    def forward(self, hidden_states: torch.Tensor):
        updated = self.post_attention_layernorm(hidden_states)
        return (hidden_states + updated,)


class FakeBackbone(nn.Module):
    def __init__(self, layers: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(FakeDecoderLayer(hidden) for _ in range(layers))


class FakeCausalLM(nn.Module):
    def __init__(self, layers: int = 4, hidden: int = HIDDEN, vocab: int = VOCAB) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=layers)
        self.embed = nn.Embedding(vocab, hidden)
        self.model = FakeBackbone(layers, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=self.lm_head(hidden))

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask=None,
        do_sample=False,
        num_beams=1,
        max_new_tokens=4,
        eos_token_id=EOS,
        pad_token_id=PAD,
        use_cache=True,
    ):
        output = [input_ids]
        for _ in range(max_new_tokens):
            next_token = torch.full(
                (input_ids.shape[0], 1), 5, dtype=torch.long, device=input_ids.device
            )
            output.append(next_token)
        return torch.cat(output, dim=1)


def test_layer_schedule_progressive_expansion():
    schedule = LayerSchedule(num_layers=8, stage_steps=2)
    assert schedule.active_layers(0) == [6, 7]
    assert schedule.active_layers(2) == [4, 5, 6, 7]
    assert schedule.active_layers(99) == list(range(8))
    assert sum(schedule.layer_weights(0).values()) == pytest.approx(1.0)


def test_g0_hook_points_and_single_contiguous_capture():
    model = FakeCausalLM(layers=3)
    ids = torch.randint(3, VOCAB, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.long)
    for point in ("attention_residual_add", "decoder_layer_output"):
        reader = ActivationReader(model, point=point)
        activations = reader.read(ids, mask)
        assert activations.shape == (2, 3, 6, HIDDEN)


def test_g1_student_only_gradient_boundary():
    student = FakeCausalLM(layers=3)
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)

    prompt = torch.randint(3, VOCAB, (2, 5))
    mask = torch.ones_like(prompt)
    student_activations = ActivationReader(student).read(prompt, mask, detach=False)
    with torch.no_grad():
        teacher_activations = ActivationReader(teacher).read(prompt, mask, detach=True)

    schedule = LayerSchedule(num_layers=3, stage_steps=1)
    act_loss = activation_alignment_loss(
        student_activations,
        teacher_activations,
        schedule.active_layers(0),
        schedule.layer_weights(0),
    )

    teacher_prompt = torch.randint(3, VOCAB, (2, 5))
    student_outputs = torch.full((2, 3), 5, dtype=torch.long)
    teacher_outputs = torch.full((2, 3), 7, dtype=torch.long)
    seq_metrics = compute_seqkl(
        student_model=student,
        teacher_model=teacher,
        teacher_prompt_ids=teacher_prompt,
        teacher_prompt_attention_mask=torch.ones_like(teacher_prompt),
        student_prompt_ids=prompt,
        student_prompt_attention_mask=mask,
        teacher_output_ids=teacher_outputs,
        student_output_ids=student_outputs,
        teacher_output_mask=torch.ones_like(teacher_outputs),
        student_output_mask=torch.ones_like(student_outputs),
        pad_token_id=PAD,
        eos_token_id=EOS,
        eta=0.5,
    )
    logits = student(prompt).logits
    retain_loss = nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, VOCAB), prompt[:, 1:].reshape(-1)
    )
    loss = act_loss + 0.5 * seq_metrics["loss"] + retain_loss
    loss.backward()

    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in student.parameters()
    )
    assert not teacher_activations.requires_grad
    assert student_activations.requires_grad


def test_seqkl_first_token_supervised_and_zero_at_initialization():
    student = FakeCausalLM(layers=2)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    prompt = torch.randint(3, VOCAB, (2, 5))
    mask = torch.ones_like(prompt)
    outputs = torch.full((2, 1), 5, dtype=torch.long)

    metrics = compute_seqkl(
        student_model=student,
        teacher_model=teacher,
        teacher_prompt_ids=prompt,
        teacher_prompt_attention_mask=mask,
        student_prompt_ids=prompt,
        student_prompt_attention_mask=mask,
        teacher_output_ids=outputs,
        student_output_ids=outputs,
        pad_token_id=PAD,
        eos_token_id=None,
        eta=0.25,
        start_index=1,
    )

    assert metrics["loss"].item() == pytest.approx(0.0, abs=1e-5)
    assert int(metrics["valid_positions_teacher"].item()) == 2
    assert int(metrics["valid_positions_student"].item()) == 2
    assert metrics["first_token_kl_teacher"].item() == pytest.approx(0.0, abs=1e-5)
    metrics["loss"].backward()
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_seqkl_eta_mixes_two_prefix_branches():
    torch.manual_seed(7)
    student = FakeCausalLM(layers=2)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    with torch.no_grad():
        student.lm_head.weight.add_(0.4)
        student.embed.weight.add_(0.05)

    teacher_prompt = torch.randint(3, VOCAB, (2, 5))
    student_prompt = teacher_prompt.clone()
    student_prompt[:, -1] = (student_prompt[:, -1] + 1) % VOCAB
    mask = torch.ones_like(teacher_prompt)
    outputs = torch.full((2, 2), 5, dtype=torch.long)

    only_teacher_prefix = compute_seqkl(
        student_model=student,
        teacher_model=teacher,
        teacher_prompt_ids=teacher_prompt,
        teacher_prompt_attention_mask=mask,
        student_prompt_ids=student_prompt,
        student_prompt_attention_mask=mask,
        teacher_output_ids=outputs,
        student_output_ids=outputs,
        pad_token_id=PAD,
        eos_token_id=None,
        eta=0.0,
    )
    only_student_prefix = compute_seqkl(
        student_model=student,
        teacher_model=teacher,
        teacher_prompt_ids=teacher_prompt,
        teacher_prompt_attention_mask=mask,
        student_prompt_ids=student_prompt,
        student_prompt_attention_mask=mask,
        teacher_output_ids=outputs,
        student_output_ids=outputs,
        pad_token_id=PAD,
        eos_token_id=None,
        eta=1.0,
    )

    assert only_teacher_prefix["loss"].item() != pytest.approx(
        only_student_prefix["loss"].item(), abs=1e-6
    )


def test_g4_generation_classification_separates_donor_and_refusal():
    donor = classify_generation(
        "The author is Hina Ameen and she writes geology books.",
        target_answer="The author is Jane Doe and she writes poetry.",
        donor_answer="The author is Hina Ameen and she writes geology books.",
        eos_present=True,
        truncated=False,
    )
    refusal = classify_generation(
        "I do not know the answer.",
        target_answer="The author is Jane Doe.",
        donor_answer="The author is Hina Ameen.",
        eos_present=True,
        truncated=False,
    )
    assert donor["category"] == "donor_answer"
    assert refusal["category"] == "refusal"


def test_g5_g6_recovery_requires_two_independent_attacks():
    rows = [
        {"attack": "unpatched", "forget_effect": True, "retain_score": 0.91},
        {"attack": "hooked", "forget_effect": False, "retain_score": 0.91},
        {"attack": "probe", "recovered": False, "retain_score": 0.9},
        {"attack": "relearning", "recovered": False, "retain_score": 0.89},
    ]
    result = aggregate_recovery(rows)
    assert result["g5"]["pass"] is True
    assert result["g6"]["pass"] is True
    assert result["g6"]["independent_attacks_passed"] == 2

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from factpair import IGNORE_INDEX, build_prompt, encode_without_truncation, read_jsonl
from factpair_activation import ActivationReader


class FactPairIdeaDataset(Dataset):
    """Multi-donor paired dataset for idea A/B/C/D experiments."""

    def __init__(
        self,
        pairs_path: str,
        retain_path: str,
        tokenizer: Any,
        unknown_word: str = "unknown",
        max_length: int = 512,
    ) -> None:
        self.pairs = read_jsonl(pairs_path)
        self.retain_rows = read_jsonl(retain_path)
        self.tokenizer = tokenizer
        self.unknown_word = unknown_word
        self.max_length = max_length
        if not self.pairs or not self.retain_rows:
            raise ValueError("IdeaDataset requires non-empty pairs and retain rows")

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _with_eos(tokenizer: Any, text: str) -> List[int]:
        ids = encode_without_truncation(tokenizer, text)
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None and (not ids or ids[-1] != int(eos)):
            ids.append(int(eos))
        return ids

    def _null_ids(self, prompt_ids: Sequence[int], span: Sequence[int]) -> List[int]:
        start, length = int(span[0]), int(span[1])
        filler = encode_without_truncation(self.tokenizer, self.unknown_word)
        if not filler:
            eos = getattr(self.tokenizer, "eos_token_id", 0)
            filler = [int(eos)]
        fill = (filler * ((length // len(filler)) + 1))[:length]
        return list(prompt_ids[:start]) + fill + list(prompt_ids[start + length :])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair = self.pairs[index]
        prompt_ids = list(pair["prompt_plus_ids"])
        donor_ids_list = [list(ids) for ids in pair["prompt_minus_ids_list"]]
        answer_ids = self._with_eos(self.tokenizer, pair["answer_plus"])
        span = [int(pair["entity_span"][0]), int(pair["entity_span"][1])]

        retain_index = (int(pair["source_row_index"]) * 7919 + index) % len(self.retain_rows)
        retain_row = self.retain_rows[retain_index]
        retain_prompt = build_prompt(
            self.tokenizer,
            {"apply_chat_template": True, "system_prompt": None},
            retain_row["question"],
        )
        retain_prompt_ids = encode_without_truncation(self.tokenizer, retain_prompt)
        retain_answer_ids = self._with_eos(self.tokenizer, retain_row["answer"])

        return {
            "pair_id": str(pair["id"]),
            "relation_family": str(pair["relation_family"]),
            "forget": {
                "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(prompt_ids), dtype=torch.long),
            },
            "supervision": {
                "input_ids": torch.tensor(prompt_ids + answer_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(prompt_ids) + len(answer_ids), dtype=torch.long),
                "labels": torch.tensor(
                    [IGNORE_INDEX] * len(prompt_ids) + answer_ids, dtype=torch.long
                ),
            },
            "donors": {
                "input_ids": [torch.tensor(ids, dtype=torch.long) for ids in donor_ids_list],
                "attention_mask": [torch.ones(len(ids), dtype=torch.long) for ids in donor_ids_list],
            },
            "null": {
                "input_ids": torch.tensor(self._null_ids(prompt_ids, span), dtype=torch.long),
                "attention_mask": torch.ones(len(prompt_ids), dtype=torch.long),
            },
            "retain": {
                "input_ids": torch.tensor(retain_prompt_ids + retain_answer_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(retain_prompt_ids) + len(retain_answer_ids), dtype=torch.long),
                "labels": torch.tensor(
                    [IGNORE_INDEX] * len(retain_prompt_ids) + retain_answer_ids, dtype=torch.long
                ),
            },
            "entity_last": torch.tensor(span[0] + span[1] - 1, dtype=torch.long),
        }


class FactPairIdeaCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def _pad_1d(self, values: Sequence[torch.Tensor], pad_value: int) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(
            [value.clone().detach() for value in values], batch_first=True, padding_value=pad_value
        )

    def _pad_group(self, values: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        output = {
            "input_ids": self._pad_1d([item["input_ids"] for item in values], self.pad_token_id),
            "attention_mask": self._pad_1d([item["attention_mask"] for item in values], 0),
        }
        if all("labels" in item for item in values):
            output["labels"] = self._pad_1d(
                [item["labels"] for item in values], IGNORE_INDEX
            )
        return output

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(instances)
        max_donors = max(len(item["donors"]["input_ids"]) for item in instances)
        max_len = max(
            max(int(tensor.numel()) for tensor in item["donors"]["input_ids"])
            for item in instances
        )
        donor_ids = torch.full(
            (batch_size, max_donors, max_len), self.pad_token_id, dtype=torch.long
        )
        donor_mask = torch.zeros((batch_size, max_donors, max_len), dtype=torch.long)
        donor_valid = torch.zeros((batch_size, max_donors), dtype=torch.long)
        for row, item in enumerate(instances):
            for col, ids in enumerate(item["donors"]["input_ids"]):
                length = int(ids.numel())
                donor_ids[row, col, :length] = ids
                donor_mask[row, col, :length] = 1
                donor_valid[row, col] = 1
        return {
            "pair_id": [item["pair_id"] for item in instances],
            "relation_family": [item["relation_family"] for item in instances],
            "forget": self._pad_group([item["forget"] for item in instances]),
            "supervision": self._pad_group([item["supervision"] for item in instances]),
            "donors": {
                "input_ids": donor_ids,
                "attention_mask": donor_mask,
                "valid": donor_valid,
            },
            "null": self._pad_group([item["null"] for item in instances]),
            "retain": self._pad_group([item["retain"] for item in instances]),
            "entity_last": torch.stack([item["entity_last"] for item in instances]),
        }


def to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            output[key] = {
                sub_key: sub_value.to(device, non_blocking=True)
                if isinstance(sub_value, torch.Tensor)
                else sub_value
                for sub_key, sub_value in value.items()
            }
        elif isinstance(value, torch.Tensor):
            output[key] = value.to(device, non_blocking=True)
        else:
            output[key] = value
    return output


def read_layer_activations(
    model: torch.nn.Module,
    point: str,
    layer_indices: Sequence[int],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    detach: bool,
) -> torch.Tensor:
    reader = ActivationReader(model, point=point, layers=layer_indices)
    return reader.read(input_ids, attention_mask, detach=detach)


def gather_subject(
    activations: torch.Tensor,
    entity_last: torch.Tensor,
) -> torch.Tensor:
    rows = torch.arange(activations.shape[0], device=activations.device)
    return activations[rows, :, entity_last.to(activations.device), :]


def read_donor_subject_activations(
    model: torch.nn.Module,
    point: str,
    layer_indices: Sequence[int],
    donor_batch: Dict[str, torch.Tensor],
    entity_last: torch.Tensor,
    detach: bool,
) -> torch.Tensor:
    batch_size, max_donors, seq_len = donor_batch["input_ids"].shape
    valid = donor_batch["valid"].bool()
    flat_ids = donor_batch["input_ids"][valid]
    flat_mask = donor_batch["attention_mask"][valid]
    flat_last = entity_last.unsqueeze(1).expand(-1, max_donors)[valid]
    if flat_ids.shape[0] == 0:
        hidden = donor_batch["input_ids"].shape[-1]
        return torch.zeros(
            batch_size, max_donors, len(layer_indices), hidden, device=donor_batch["input_ids"].device
        )
    flat_act = read_layer_activations(model, point, layer_indices, flat_ids, flat_mask, detach)
    flat_subject = gather_subject(flat_act, flat_last)
    hidden = flat_subject.shape[-1]
    output = torch.zeros(
        batch_size,
        max_donors,
        len(layer_indices),
        hidden,
        dtype=flat_subject.dtype,
        device=flat_subject.device,
    )
    output[valid] = flat_subject
    return output


def layer_weight_tensor(
    layer_indices: Sequence[int],
    layer_weights: Dict[int, float],
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [float(layer_weights.get(layer, 0.0)) for layer in layer_indices],
        dtype=torch.float32,
        device=device,
    )


def layered_mse(
    student: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    per_layer = ((student - target.detach()) ** 2).mean(dim=-1)
    return (per_layer * weights).sum() / weights.sum().clamp_min(1e-8)


def tricon_loss(
    student: torch.Tensor,
    unknown: torch.Tensor,
    target: torch.Tensor,
    donors: torch.Tensor,
    donor_valid: torch.Tensor,
    weights: torch.Tensor,
    tau_c: float,
) -> Dict[str, torch.Tensor]:
    h = F.normalize(student, dim=-1)
    u = F.normalize(unknown.detach(), dim=-1)
    a_plus = F.normalize(target.detach(), dim=-1)
    a_minus = F.normalize(donors.detach(), dim=-1).permute(0, 2, 1, 3)
    pos = ((h * u).sum(dim=-1) / tau_c).unsqueeze(-1)
    neg_target = ((h * a_plus).sum(dim=-1) / tau_c).unsqueeze(-1)
    neg_donors = (h.unsqueeze(2) * a_minus).sum(dim=-1) / tau_c
    mask = donor_valid.bool().unsqueeze(1).expand(-1, h.shape[1], -1)
    neg_donors = neg_donors.masked_fill(~mask, float("-inf"))
    logits = torch.cat([pos, neg_target, neg_donors], dim=-1)
    per_layer = torch.logsumexp(logits, dim=-1) - pos.squeeze(-1)
    loss = (per_layer * weights).sum() / weights.sum().clamp_min(1e-8)
    margin = (pos.squeeze(-1) - neg_target.squeeze(-1)).detach()
    return {
        "loss": loss,
        "unknown_target_margin": margin.mean(),
        "unknown_donor_margin": (pos.squeeze(-1) - neg_donors.max(dim=-1).values).detach().mean(),
    }


def delta_subspace_target(
    target: torch.Tensor,
    donors: torch.Tensor,
    donor_valid: torch.Tensor,
    alpha: float,
    rank: int,
) -> Dict[str, torch.Tensor]:
    batch_size, num_donors, num_layers, hidden = donors.shape
    delta = target.unsqueeze(1) - donors
    mask = donor_valid.view(batch_size, num_donors, 1, 1).expand_as(delta).float()
    mean_delta = (delta * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    matrix = mean_delta.reshape(-1, hidden).detach()
    _, _, right = torch.linalg.svd(matrix, full_matrices=False)
    effective_rank = min(int(rank), right.shape[0])
    basis = right[:effective_rank].T
    projection = (mean_delta @ basis) @ basis.T
    new_target = (target - float(alpha) * projection).detach()
    energy = projection.norm(dim=-1) / mean_delta.norm(dim=-1).clamp_min(1e-8)
    return {
        "target": new_target,
        "mean_delta": mean_delta.detach(),
        "basis": basis.detach(),
        "projection_energy": energy.detach(),
    }


def compute_gradient_causal_gate(
    teacher: torch.nn.Module,
    forget: Dict[str, torch.Tensor],
    entity_last: torch.Tensor,
    target: torch.Tensor,
    mean_delta: torch.Tensor,
    layer_indices: Sequence[int],
    point: str,
) -> torch.Tensor:
    """First-order causal gate: |grad_y+ activation * ΔA| (gradient attribution).

    This is a cheap differentiable proxy for activation patching, not a full
    patching sweep. It is used only for the Idea C dimension mask.
    """
    embedding = teacher.get_input_embeddings()
    old_requires_grad = embedding.weight.requires_grad
    embedding.weight.requires_grad_(True)
    was_training = teacher.training
    teacher.eval()
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    try:
        for layer in layer_indices:
            module = teacher.model.layers[layer]

            def hook(_module, _inputs, outputs, captured_layer=layer):
                tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                captured[captured_layer] = tensor

            handles.append(module.register_forward_hook(hook))
        outputs = teacher(
            input_ids=forget["input_ids"],
            attention_mask=forget["attention_mask"],
            labels=forget["labels"],
        )
        loss = outputs.loss
        tensors = [captured[layer] for layer in layer_indices]
        grads = torch.autograd.grad(loss, tensors, allow_unused=True)
        gate_layers = []
        rows = torch.arange(forget["input_ids"].shape[0], device=forget["input_ids"].device)
        for layer_index, grad in enumerate(grads):
            layer_delta = mean_delta[:, layer_index, :]
            if grad is None:
                gate_layers.append(torch.zeros_like(layer_delta[:, 0]))
                continue
            grad_at_p = grad[rows, entity_last.to(grad.device), :]
            gate_layers.append((grad_at_p.abs() * layer_delta.abs()).sum(dim=-1))
        gate = torch.stack(gate_layers, dim=1)
        gate = gate / gate.max(dim=0, keepdim=True).values.clamp_min(1e-8)
        return gate.clamp(0.0, 1.0).detach()
    finally:
        for handle in handles:
            handle.remove()
        embedding.weight.requires_grad_(old_requires_grad)
        teacher.train(was_training)


def apply_gradient_surgery(
    parameters,
    forget_grads: Sequence[Optional[torch.Tensor]],
    retain_grads: Sequence[Optional[torch.Tensor]],
    layer_scales: Dict[int, float],
    retain_weight: float = 0.0,
    eps: float = 1e-8,
) -> None:
    pattern = re.compile(r"(?:model\.)?layers\.(\d+)\.")
    for (name, parameter), forget_grad, retain_grad in zip(parameters, forget_grads, retain_grads):
        if forget_grad is None:
            parameter.grad = None
            continue
        if retain_grad is None:
            updated = forget_grad
        else:
            denominator = (retain_grad * retain_grad).sum().clamp_min(eps)
            coefficient = (forget_grad * retain_grad).sum() / denominator
            updated = forget_grad - coefficient * retain_grad + float(retain_weight) * retain_grad
        match = pattern.search(name)
        scale = layer_scales.get(int(match.group(1)), 1.0) if match else 1.0
        parameter.grad = updated * float(scale)

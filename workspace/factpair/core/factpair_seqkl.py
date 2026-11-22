from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _as_long_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.long)
    return torch.tensor(value, dtype=torch.long, device=device)


def _unpad(ids: torch.Tensor, attention_mask: torch.Tensor) -> List[List[int]]:
    result: List[List[int]] = []
    for row_ids, row_mask in zip(ids, attention_mask):
        result.append([int(token) for token, valid in zip(row_ids, row_mask) if int(valid) == 1])
    return result


def _mask_generated(
    sequences: torch.Tensor,
    prompt_length: int,
    pad_token_id: int,
    eos_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generated = sequences[:, prompt_length:].contiguous()
    not_pad = generated.ne(pad_token_id)
    eos_or_pad = generated.eq(eos_token_id) | generated.eq(pad_token_id)
    eos_seen = eos_or_pad.to(torch.long).cumsum(dim=1) - eos_or_pad.to(torch.long)
    mask = not_pad | generated.eq(eos_token_id)
    mask = mask & (eos_seen == 0)
    return generated, mask.long()


def _truncate_after_eos(tokens: Sequence[int], eos_token_id: int) -> List[int]:
    output: List[int] = []
    for token in tokens:
        output.append(int(token))
        if int(token) == int(eos_token_id):
            break
    return output


def _right_pad_contexts(
    contexts: Sequence[Sequence[int]],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(context) for context in contexts), default=1)
    input_ids = torch.full((len(contexts), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(contexts), max_len), dtype=torch.long, device=device)
    for row, context in enumerate(contexts):
        if not context:
            continue
        input_ids[row, : len(context)] = torch.tensor(context, dtype=torch.long, device=device)
        attention_mask[row, : len(context)] = 1
    return input_ids, attention_mask


def _model_mode(model: torch.nn.Module, training: bool) -> None:
    if model.training != training:
        model.train(training)


def _selected_logits(
    model: torch.nn.Module,
    contexts: Sequence[Sequence[int]],
    prompt_lengths: torch.Tensor,
    max_steps: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Return logits at positions that predict generated tokens 1..max_steps."""
    device = next(model.parameters()).device
    prompt_lengths = _as_long_tensor(prompt_lengths, device)
    input_ids, attention_mask = _right_pad_contexts(contexts, pad_token_id, device)
    step_index = torch.arange(max_steps, device=device).unsqueeze(0)
    positions = (prompt_lengths.unsqueeze(1) - 1 + step_index).clamp(max=input_ids.shape[1] - 1)

    was_training = model.training
    _model_mode(model, False)
    try:
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    finally:
        _model_mode(model, was_training)

    return logits.gather(1, positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1])).float()


@torch.no_grad()
def generate_pair_output_traces(
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    teacher_prompt_ids: torch.Tensor,
    teacher_prompt_mask: torch.Tensor,
    student_prompt_ids: torch.Tensor,
    student_prompt_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Generate Teacher(X-) and Student(X+) traces with fixed decoding."""

    def generate_one(model: torch.nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(model.parameters()).device
        ids = _as_long_tensor(ids, device)
        mask = _as_long_tensor(mask, device)
        lengths = mask.sum(dim=1)
        max_len = ids.shape[1]
        left_ids = torch.full_like(ids, pad_token_id)
        left_mask = torch.zeros_like(mask)
        for row in range(ids.shape[0]):
            length = int(lengths[row].item())
            if length <= 0:
                continue
            offset = max_len - length
            left_ids[row, offset:] = ids[row, :length]
            left_mask[row, offset:] = 1
        ids, mask = left_ids, left_mask
        was_training = model.training
        _model_mode(model, False)
        try:
            if seed is not None:
                state = torch.get_rng_state()
                torch.manual_seed(seed)
            sequences = model.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=True,
            )
            if seed is not None:
                torch.set_rng_state(state)
        finally:
            _model_mode(model, was_training)
        return _mask_generated(sequences, ids.shape[1], pad_token_id, eos_token_id)

    teacher_ids, teacher_mask = generate_one(teacher_model, teacher_prompt_ids, teacher_prompt_mask)
    student_ids, student_mask = generate_one(student_model, student_prompt_ids, student_prompt_mask)
    return {
        "teacher_output_ids": teacher_ids,
        "teacher_output_mask": teacher_mask,
        "student_output_ids": student_ids,
        "student_output_mask": student_mask,
    }


def _prepare_generated_rows(
    output_ids: torch.Tensor,
    output_mask: Optional[torch.Tensor],
    eos_token_id: Optional[int],
    device: torch.device,
) -> List[List[int]]:
    output_ids = _as_long_tensor(output_ids, device)
    rows: List[List[int]] = []
    for index in range(output_ids.shape[0]):
        row = [int(token) for token in output_ids[index]]
        if output_mask is not None:
            mask_row = _as_long_tensor(output_mask, device)[index]
            row = [token for token, valid in zip(row, mask_row) if int(valid) == 1]
        if eos_token_id is not None:
            row = _truncate_after_eos(row, eos_token_id)
        rows.append(row)
    return rows


def _prefix_kl_loss(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    prompt_rows: Sequence[Sequence[int]],
    generated_rows: Sequence[Sequence[int]],
    target_span_mask: Optional[torch.Tensor],
    pad_token_id: int,
    start_index: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """KL(p_T(.|X_prompt, y_<t) || q_S(.|X_prompt, y_<t))."""
    device = next(student_model.parameters()).device
    batch_size = len(prompt_rows)
    lengths = torch.tensor([len(row) for row in generated_rows], dtype=torch.long, device=device)
    max_steps = int(lengths.max().item()) if batch_size else 0

    if max_steps == 0:
        zero = torch.zeros((), device=device)
        return zero, {
            "loss": zero,
            "mean": zero,
            "valid_positions": torch.zeros((), device=device),
            "first_token_kl": zero,
        }

    contexts = [
        list(prompt_rows[index]) + list(generated_rows[index])
        for index in range(batch_size)
    ]
    prompt_lengths = torch.tensor([len(row) for row in prompt_rows], dtype=torch.long, device=device)

    with torch.no_grad():
        teacher_logits = _selected_logits(teacher_model, contexts, prompt_lengths, max_steps, pad_token_id)
        log_p_teacher = F.log_softmax(teacher_logits, dim=-1)
        p_teacher = log_p_teacher.exp()
    student_logits = _selected_logits(student_model, contexts, prompt_lengths, max_steps, pad_token_id)
    log_p_student = F.log_softmax(student_logits, dim=-1)

    step_index = torch.arange(max_steps, device=device).unsqueeze(0)
    valid = step_index < lengths.unsqueeze(1)
    valid = valid & (step_index >= (start_index - 1))
    if target_span_mask is not None:
        span_mask = _as_long_tensor(target_span_mask, device)[:, :max_steps].bool()
        valid = valid & span_mask

    kl = (p_teacher * (log_p_teacher - log_p_student)).sum(dim=-1)
    per_sample = (kl * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    loss = per_sample.mean() if bool(valid.any().item()) else kl.sum() * 0.0
    first_valid = valid[:, 0] if max_steps > 0 else torch.zeros(batch_size, dtype=torch.bool, device=device)
    first_token_kl = (kl[:, 0] * first_valid).sum() / first_valid.sum().clamp_min(1)

    metrics = {
        "loss": loss.detach(),
        "mean": loss.detach(),
        "valid_positions": valid.sum().detach(),
        "first_token_kl": first_token_kl.detach(),
    }
    del teacher_logits, log_p_teacher, p_teacher, student_logits, log_p_student, kl
    return loss, metrics


def compute_seqkl(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    teacher_prompt_ids: torch.Tensor,
    teacher_prompt_attention_mask: torch.Tensor,
    student_prompt_ids: torch.Tensor,
    student_prompt_attention_mask: torch.Tensor,
    teacher_output_ids: torch.Tensor,
    student_output_ids: torch.Tensor,
    teacher_output_mask: Optional[torch.Tensor] = None,
    student_output_mask: Optional[torch.Tensor] = None,
    target_span_mask: Optional[torch.Tensor] = None,
    pad_token_id: int = 0,
    eos_token_id: Optional[int] = None,
    eta: float = 0.5,
    start_index: int = 1,
) -> Dict[str, torch.Tensor]:
    """Proposal v0.5 DKT loss with two prefix sources and shared-prefix KL.

    Teacher-prefix branch uses (X-, y^T_<t) for both distributions; the
    Student-prefix branch uses (X+, y^S_<t) for both distributions. Each
    branch is normalized independently and mixed by (1-eta)/eta.
    """
    if not 0.0 <= float(eta) <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    if start_index < 1:
        raise ValueError("start_index must be >= 1")

    device = teacher_prompt_ids.device
    eta_value = float(eta)

    teacher_prompt_rows = _unpad(teacher_prompt_ids, teacher_prompt_attention_mask)
    student_prompt_rows = _unpad(student_prompt_ids, student_prompt_attention_mask)
    teacher_generated_rows = _prepare_generated_rows(teacher_output_ids, teacher_output_mask, eos_token_id, device)
    student_generated_rows = _prepare_generated_rows(student_output_ids, student_output_mask, eos_token_id, device)

    loss_teacher_prefix, metrics_teacher = _prefix_kl_loss(
        student_model=student_model,
        teacher_model=teacher_model,
        prompt_rows=teacher_prompt_rows,
        generated_rows=teacher_generated_rows,
        target_span_mask=target_span_mask,
        pad_token_id=pad_token_id,
        start_index=start_index,
    )
    loss_student_prefix, metrics_student = _prefix_kl_loss(
        student_model=student_model,
        teacher_model=teacher_model,
        prompt_rows=student_prompt_rows,
        generated_rows=student_generated_rows,
        target_span_mask=target_span_mask,
        pad_token_id=pad_token_id,
        start_index=start_index,
    )

    loss = (1.0 - eta_value) * loss_teacher_prefix + eta_value * loss_student_prefix
    return {
        "loss": loss,
        "loss_teacher_prefix": loss_teacher_prefix.detach(),
        "loss_student_prefix": loss_student_prefix.detach(),
        "kl_teacher_prefix_mean": metrics_teacher["mean"],
        "kl_student_prefix_mean": metrics_student["mean"],
        "valid_positions_teacher": metrics_teacher["valid_positions"],
        "valid_positions_student": metrics_student["valid_positions"],
        "first_token_kl_teacher": metrics_teacher["first_token_kl"],
        "first_token_kl_student": metrics_student["first_token_kl"],
        "eta": torch.tensor(eta_value, device=device),
        "start_index": torch.tensor(int(start_index), device=device),
    }

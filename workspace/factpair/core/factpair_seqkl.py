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


def _mask_generated(sequences: torch.Tensor, prompt_length: int, pad_token_id: int, eos_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generated = sequences[:, prompt_length:].contiguous()
    not_pad = generated.ne(pad_token_id)
    eos_or_pad = generated.eq(eos_token_id) | generated.eq(pad_token_id)
    eos_seen = eos_or_pad.to(torch.long).cumsum(dim=1) - eos_or_pad.to(torch.long)
    mask = not_pad | generated.eq(eos_token_id)
    mask = mask & (eos_seen == 0)
    return generated, mask.long()


def _left_pad_contexts(contexts: Sequence[Sequence[int]], pad_token_id: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max((len(context) for context in contexts), default=1)
    input_ids = torch.full((len(contexts), max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(contexts), max_len), dtype=torch.long, device=device)
    for row, context in enumerate(contexts):
        if not context:
            continue
        start = max_len - len(context)
        input_ids[row, start:] = torch.tensor(context, dtype=torch.long, device=device)
        attention_mask[row, start:] = 1
    return input_ids, attention_mask


def _model_mode(model: torch.nn.Module, training: bool) -> None:
    if model.training != training:
        model.train(training)


def _last_distribution(
    model: torch.nn.Module,
    contexts: Sequence[Sequence[int]],
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    input_ids, attention_mask = _left_pad_contexts(contexts, pad_token_id, device)
    was_training = model.training
    _model_mode(model, False)
    try:
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, -1, :]
    finally:
        _model_mode(model, was_training)
    return torch.log_softmax(logits.float(), dim=-1), torch.softmax(logits.float(), dim=-1)


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

    teacher_ids, teacher_mask = generate_one(
        teacher_model, teacher_prompt_ids, teacher_prompt_mask
    )
    student_ids, student_mask = generate_one(
        student_model, student_prompt_ids, student_prompt_mask
    )
    return {
        "teacher_output_ids": teacher_ids,
        "teacher_output_mask": teacher_mask,
        "student_output_ids": student_ids,
        "student_output_mask": student_mask,
    }


def compute_seqkl(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    teacher_output_ids: torch.Tensor,
    student_output_ids: torch.Tensor,
    teacher_output_mask: Optional[torch.Tensor] = None,
    student_output_mask: Optional[torch.Tensor] = None,
    target_span_mask: Optional[torch.Tensor] = None,
    pad_token_id: int = 0,
    eos_token_id: Optional[int] = None,
    start_index: int = 2,
) -> Dict[str, torch.Tensor]:
    """Compute forward KL between Teacher(X-) and Student(X+) prefix trajectories.

    Only scalar trajectories and small debug tensors are returned; full
    `[B, T, V]` distributions are never persisted.
    """
    if start_index < 1:
        raise ValueError("start_index must be >= 1")
    batch_size = prompt_ids.shape[0]
    device = prompt_ids.device
    teacher_output_ids = _as_long_tensor(teacher_output_ids, device)
    student_output_ids = _as_long_tensor(student_output_ids, device)
    prompt_rows = _unpad(prompt_ids, prompt_attention_mask)

    teacher_rows: List[List[int]] = []
    student_rows: List[List[int]] = []
    for index in range(batch_size):
        teacher_row = [int(token) for token in teacher_output_ids[index]]
        student_row = [int(token) for token in student_output_ids[index]]
        if eos_token_id is not None:
            teacher_row = _truncate_after_eos(teacher_row, eos_token_id)
            student_row = _truncate_after_eos(student_row, eos_token_id)
        teacher_rows.append(teacher_row)
        student_rows.append(student_row)

    if teacher_output_mask is None:
        teacher_output_mask = torch.ones_like(teacher_output_ids)
    if student_output_mask is None:
        student_output_mask = torch.ones_like(student_output_ids)
    teacher_output_mask = _as_long_tensor(teacher_output_mask, device)
    student_output_mask = _as_long_tensor(student_output_mask, device)

    with torch.no_grad():
        log_q_x, q_x = _last_distribution(
            teacher_model, [prompt_rows[index] for index in range(batch_size)], pad_token_id
        )
    student_prompt_logits, _ = _last_distribution(
        student_model, [prompt_rows[index] for index in range(batch_size)], pad_token_id
    )

    max_t = max(len(row) for row in teacher_rows + student_rows) if batch_size else 0
    kl_ts_sum = kl_ts_count = 0.0
    kl_xT_sum = kl_xT_count = 0.0
    kl_xS_sum = kl_xS_count = 0.0
    delta_sum = 0.0
    first_kl_ts = first_kl_xT = first_kl_xS = 0.0
    first_count = 0.0

    for time_index in range(1, max_t + 1):
        active = []
        for batch_index in range(batch_size):
            if time_index > len(teacher_rows[batch_index]) or time_index > len(student_rows[batch_index]):
                continue
            teacher_valid = int(teacher_output_mask[batch_index, time_index - 1]) == 1
            student_valid = int(student_output_mask[batch_index, time_index - 1]) == 1
            target_valid = (
                int(target_span_mask[batch_index, time_index - 1]) == 1
                if target_span_mask is not None
                else True
            )
            if teacher_valid and student_valid and target_valid:
                active.append(batch_index)
        if not active:
            continue

        if time_index == 1:
            log_q_T = log_q_x[active]
            q_T = q_x[active]
            log_q_S = student_prompt_logits[active]
        else:
            teacher_contexts = [
                list(prompt_rows[index]) + teacher_rows[index][: time_index - 1]
                for index in active
            ]
            student_contexts = [
                list(prompt_rows[index]) + student_rows[index][: time_index - 1]
                for index in active
            ]
            with torch.no_grad():
                log_q_T, q_T = _last_distribution(teacher_model, teacher_contexts, pad_token_id)
            log_q_S, _ = _last_distribution(student_model, student_contexts, pad_token_id)

        q_x_active = q_x[active]
        log_q_x_active = log_q_x[active]
        kl_ts = (q_T * (log_q_T - log_q_S)).sum(dim=-1)
        kl_xT = (q_x_active * (log_q_x_active - log_q_T)).sum(dim=-1)
        kl_xS = (q_x_active * (log_q_x_active - log_q_S)).sum(dim=-1)

        if time_index == 1:
            first_kl_ts = float(kl_ts.detach().sum().item())
            first_kl_xT = float(kl_xT.detach().sum().item())
            first_kl_xS = float(kl_xS.detach().sum().item())
            first_count = float(len(active))
            continue

        kl_ts_sum = kl_ts_sum + kl_ts.sum()
        kl_ts_count += float(len(active))
        kl_xT_sum = kl_xT_sum + kl_xT.detach().sum()
        kl_xT_count += float(len(active))
        kl_xS_sum = kl_xS_sum + kl_xS.sum()
        kl_xS_count += float(len(active))
        delta_sum += float((kl_xT - kl_xS).detach().sum().item())

    loss = kl_ts_sum / max(1.0, kl_ts_count) if torch.is_tensor(kl_ts_sum) else torch.zeros((), device=device)
    deltas = {
        "loss": loss,
        "kl_ts_mean": (kl_ts_sum.detach() / max(1.0, kl_ts_count)) if torch.is_tensor(kl_ts_sum) else torch.zeros((), device=device),
        "kl_xT_mean": (kl_xT_sum.detach() / max(1.0, kl_xT_count)) if torch.is_tensor(kl_xT_sum) else torch.zeros((), device=device),
        "kl_xS_mean": (kl_xS_sum.detach() / max(1.0, kl_xS_count)) if torch.is_tensor(kl_xS_sum) else torch.zeros((), device=device),
        "delta_x_mean": torch.tensor(delta_sum / max(1.0, kl_ts_count), device=device),
        "first_token_kl_ts": torch.tensor(first_kl_ts / max(1.0, first_count), device=device),
        "first_token_kl_xT": torch.tensor(first_kl_xT / max(1.0, first_count), device=device),
        "first_token_kl_xS": torch.tensor(first_kl_xS / max(1.0, first_count), device=device),
        "valid_positions": torch.tensor(int(kl_ts_count), device=device),
        "start_index": torch.tensor(int(start_index), device=device),
    }
    return deltas


def _truncate_after_eos(tokens: Sequence[int], eos_token_id: int) -> List[int]:
    output: List[int] = []
    for token in tokens:
        output.append(int(token))
        if int(token) == int(eos_token_id):
            break
    return output

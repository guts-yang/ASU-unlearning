"""Selected-row attention KL and the same-token logit control.

Rows are the queries that predict derivation tokens. Only those rows are
materialized, as ``[batch, heads, n_queries, seq]``, one layer at a time.
The forget-side ASU loss is not modified here.
"""

import math

import torch
import torch.nn.functional as F


def _rotate_half(x):
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _apply_rope(query, key, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query = (query * cos) + (_rotate_half(query) * sin)
    key = (key * cos) + (_rotate_half(key) * sin)
    return query, key


def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, n_kv, length, dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, n_kv, n_rep, length, dim)
    return hidden_states.reshape(batch, n_kv * n_rep, length, dim)


def pad_query_index(query_mask):
    """``[batch, seq]`` bool mask to ``[batch, max_queries]`` indices, ``-1`` pad."""
    counts = query_mask.sum(dim=1)
    width = int(counts.max().item()) if query_mask.numel() else 0
    width = max(width, 1)
    index = torch.full(
        (query_mask.shape[0], width),
        -1,
        dtype=torch.long,
        device=query_mask.device,
    )
    for row in range(query_mask.shape[0]):
        chosen = query_mask[row].nonzero(as_tuple=False).flatten()
        if chosen.numel():
            index[row, : chosen.numel()] = chosen
    return index


def selected_row_probs(query_states, key_states, query_index, attention_mask, head_dim):
    """Softmax over keys for the gathered query rows only.

    ``query_states`` and ``key_states`` are ``[batch, heads, seq, dim]``.
    Returns ``[batch, heads, n_queries, seq]``.
    """
    batch, heads, seq_len, dim = query_states.shape
    n_queries = query_index.shape[1]
    safe = query_index.clamp(min=0)
    gather_index = safe[:, None, :, None].expand(batch, heads, n_queries, dim)
    gathered = query_states.gather(2, gather_index)
    scores = torch.matmul(gathered, key_states.transpose(-2, -1)) / math.sqrt(head_dim)
    key_pos = torch.arange(seq_len, device=scores.device)
    causal = key_pos.view(1, 1, 1, seq_len) > safe[:, None, :, None]
    scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
    if attention_mask is not None:
        padding = attention_mask[:, None, None, :] == 0
        scores = scores.masked_fill(padding, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.float(), dim=-1)
    valid = (query_index >= 0).view(batch, 1, n_queries, 1)
    return probs * valid.to(probs.dtype)


def rows_at_layer(layer, hidden, position_embeddings, query_index, attention_mask):
    attn = layer.self_attn
    normed = layer.input_layernorm(hidden)
    batch, seq_len, _ = normed.shape
    query_states = attn.q_proj(normed).view(batch, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    key_states = attn.k_proj(normed).view(
        batch, seq_len, attn.num_key_value_heads, attn.head_dim
    ).transpose(1, 2)
    query_states, key_states = _apply_rope(query_states, key_states, *position_embeddings)
    key_states = _repeat_kv(key_states, attn.num_key_value_groups)
    return selected_row_probs(query_states, key_states, query_index, attention_mask, attn.head_dim)


def _decoder(model):
    if hasattr(model, "get_decoder"):
        return model.get_decoder()
    if hasattr(model, "model"):
        return model.model
    raise TypeError("model has no decoder")


def _position_embeddings(decoder, hidden, attention_mask):
    seq_len = hidden.shape[1]
    position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)
    return decoder.rotary_emb(hidden, position_ids)


def _layer_inputs(decoder, input_ids, attention_mask):
    outputs = decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return outputs.hidden_states


def _row_kl(base_probs, student_probs, query_index):
    valid = query_index >= 0
    student_log = student_probs.clamp_min(1e-8).log()
    base_log = base_probs.clamp_min(1e-8).log()
    kl = (base_probs * (base_log - student_log)).sum(dim=-1)
    kl = kl * valid[:, None, :].to(kl.dtype)
    denom = valid.sum().to(kl.dtype) * kl.shape[1]
    if float(denom) == 0.0:
        return student_probs.sum() * 0.0
    return kl.sum() / denom


def derivation_anchor_loss(student, base, input_ids, attention_mask, query_mask):
    """Mean over layers and heads of KL(base row || student row) at ``D_tok``.

    Returns ``(loss, per_layer)``. ``per_layer`` is detached.
    """
    query_index = pad_query_index(query_mask.bool())
    if int((query_index >= 0).sum()) == 0:
        zero = input_ids.sum() * 0.0
        return zero, zero.new_zeros(1)

    student_decoder = _decoder(student)
    base_decoder = _decoder(base)
    student_hidden = _layer_inputs(student_decoder, input_ids, attention_mask)
    with torch.no_grad():
        base_hidden = _layer_inputs(base_decoder, input_ids, attention_mask)
    position_embeddings = _position_embeddings(student_decoder, student_hidden[0], attention_mask)
    with torch.no_grad():
        base_position = _position_embeddings(base_decoder, base_hidden[0], attention_mask)

    per_layer = []
    n_layers = len(student_decoder.layers)
    for layer_id in range(n_layers):
        student_rows = rows_at_layer(
            student_decoder.layers[layer_id],
            student_hidden[layer_id],
            position_embeddings,
            query_index,
            attention_mask,
        )
        with torch.no_grad():
            base_rows = rows_at_layer(
                base_decoder.layers[layer_id],
                base_hidden[layer_id],
                base_position,
                query_index,
                attention_mask,
            )
        per_layer.append(_row_kl(base_rows, student_rows, query_index))
    stacked = torch.stack(per_layer)
    return stacked.mean(), stacked.detach()


def derivation_logit_kl(student, base, input_ids, labels, attention_mask, query_mask):
    """KL(base || student) on the same derivation-query logits. No attention map."""
    query_mask = query_mask.bool()
    if int(query_mask.sum()) == 0:
        return input_ids.sum() * 0.0
    student_out = student(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    with torch.no_grad():
        base_out = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    student_log = F.log_softmax(student_out.logits[:, :-1, :].float(), dim=-1)
    base_log = F.log_softmax(base_out.logits[:, :-1, :].float(), dim=-1)
    token_kl = F.kl_div(student_log, base_log, reduction="none", log_target=True).sum(-1)
    mask = query_mask[:, :-1].to(token_kl.dtype)
    label_ok = (labels[:, 1:] != -100).to(token_kl.dtype)
    mask = mask * label_ok
    denom = mask.sum().clamp_min(1.0)
    return (token_kl * mask).sum() / denom

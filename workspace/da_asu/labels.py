"""Derivation query positions on Llama-3 token offsets.

The attention row at index ``t`` builds the state that predicts token ``t + 1``.
``D_tok`` is that query index, not the derivation token itself. Spans come from
the GSM8K calculator annotation ``<<expr=result>>`` and are aligned with the
tokenizer offset map, so a multi-token number is marked per piece.
"""

import re

import torch

CALC_RE = re.compile(r"<<([^=<>\n]+)=([^=<>\n]+)>>")


def result_char_spans(answer):
    """Character spans of the result inside each ``<<expr=result>>``."""
    return [(match.start(2), match.end(2)) for match in CALC_RE.finditer(answer)]


def shift_spans(spans, offset):
    return [(start + offset, end + offset) for start, end in spans]


def queries_overlapping(offsets, spans):
    """Query indices whose next token overlaps any ``[start, end)`` span.

    ``offsets`` aligns with the token ids. Special tokens use ``None`` or
    ``(0, 0)`` and are not derivation tokens.
    """
    queries = []
    for query_index in range(len(offsets) - 1):
        nxt = offsets[query_index + 1]
        if nxt is None:
            continue
        start, end = nxt
        if start is None or end is None or start >= end:
            continue
        for span_start, span_end in spans:
            if start < span_end and end > span_start:
                queries.append(query_index)
                break
    return queries


def encode_qa(tokenizer, question, answer, model_configs, max_length, mask_question=True):
    """Match ``convert_raw_forget_data_to_model_format`` and attach a query mask."""
    question_start = model_configs["question_start_tag"]
    question_end = model_configs["question_end_tag"]
    answer_tag = model_configs["answer_tag"]
    new_question = question_start + question + question_end
    full_text = new_question + answer_tag + answer
    answer_char_start = len(new_question + answer_tag)

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
    )
    offsets = [(None if pair is None else (int(pair[0]), int(pair[1]))) for pair in encoded["offset_mapping"]]
    input_ids = list(encoded["input_ids"])
    attention = list(encoded["attention_mask"])
    if len(input_ids) == max_length:
        labels = list(input_ids)
    else:
        labels = list(input_ids) + [tokenizer.eos_token_id] + [-100] * (max_length - len(input_ids) - 1)
    pad_length = max_length - len(input_ids)
    input_ids = input_ids + [tokenizer.eos_token_id] * pad_length
    attention = attention + [0] * pad_length

    if mask_question:
        num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
        for index in range(min(num_question_tokens, max_length)):
            labels[index] = -100

    query_mask = torch.zeros(max_length, dtype=torch.bool)
    spans = shift_spans(result_char_spans(answer), answer_char_start)
    for query_index in queries_overlapping(offsets, spans):
        if query_index < max_length and attention[query_index] == 1:
            query_mask[query_index] = True

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attention, dtype=torch.long),
        query_mask,
        full_text,
        offsets,
    )


def class_query_mask(tokenizer, question, answer, model_configs, max_length, answer_spans):
    """Query mask for character spans given in answer coordinates."""
    _, _, attention, _, _, offsets = encode_qa(
        tokenizer, question, answer, model_configs, max_length, mask_question=False
    )
    question_start = model_configs["question_start_tag"]
    question_end = model_configs["question_end_tag"]
    answer_tag = model_configs["answer_tag"]
    answer_char_start = len(question_start + question + question_end + answer_tag)
    spans = shift_spans(answer_spans, answer_char_start)
    mask = torch.zeros(max_length, dtype=torch.bool)
    for query_index in queries_overlapping(offsets, spans):
        if query_index < max_length and int(attention[query_index]) == 1:
            mask[query_index] = True
    return mask


def assign_query_classes(derivation, factual, function):
    """Disjoint masks. Derivation wins, then factual, then function."""
    derivation = derivation.bool()
    factual = factual.bool() & ~derivation
    function = function.bool() & ~derivation & ~factual
    return {"derivation": derivation, "factual": factual, "function": function}


def _mask_from_answer_spans(offsets, attention, max_length, answer_char_start, answer_spans):
    mask = torch.zeros(max_length, dtype=torch.bool)
    spans = shift_spans(answer_spans or [], answer_char_start)
    for query_index in queries_overlapping(offsets, spans):
        if query_index < max_length and int(attention[query_index]) == 1:
            mask[query_index] = True
    return mask


def encode_labeled(tokenizer, question, answer, model_configs, max_length, factual_spans=None, function_spans=None):
    """One tokenization for the three query masks used by the tau scan."""
    input_ids, labels, attention, derivation, _, offsets = encode_qa(
        tokenizer, question, answer, model_configs, max_length
    )
    question_start = model_configs["question_start_tag"]
    question_end = model_configs["question_end_tag"]
    answer_tag = model_configs["answer_tag"]
    answer_char_start = len(question_start + question + question_end + answer_tag)
    factual = _mask_from_answer_spans(offsets, attention, max_length, answer_char_start, factual_spans)
    function = _mask_from_answer_spans(offsets, attention, max_length, answer_char_start, function_spans)
    classes = assign_query_classes(derivation, factual, function)
    valid_query = torch.zeros(max_length, dtype=torch.bool)
    valid_query[:-1] = labels[1:] != -100
    classes = {name: mask & valid_query for name, mask in classes.items()}
    return input_ids, labels, attention, classes

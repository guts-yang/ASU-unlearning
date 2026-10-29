"""Answer-token encoding shared by ASU training and the teacher diagnostic."""

import torch


def encode_forget_example(tokenizer, max_length, question, answer, model_configs, mask=True):
    """Match the historical forget encoding, including the ``-100`` answer mask."""
    question_start_token = model_configs["question_start_tag"]
    question_end_token = model_configs["question_end_tag"]
    answer_token = model_configs["answer_tag"]
    new_question = question_start_token + question + question_end_token
    full_text = new_question + answer_token + answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded["input_ids"] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded["attention_mask"] + [0] * pad_length
    if mask:
        if len(encoded.input_ids) == max_length:
            label = list(encoded.input_ids)
        else:
            label = list(encoded["input_ids"]) + [tokenizer.eos_token_id] + [-100] * (pad_length - 1)
        for index in range(num_question_tokens):
            label[index] = -100
    else:
        label = pad_input_ids
    return (
        torch.tensor(pad_input_ids),
        torch.tensor(label),
        torch.tensor(pad_attention_mask),
    )

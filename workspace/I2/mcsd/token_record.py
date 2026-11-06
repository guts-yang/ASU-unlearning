"""Token offsets for same-split q- pairs.

Only the recorded Llama-2 tokenizer is accepted. A missing tokenizer leaves the
row unchecked and keeps it out of the candidate set.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from I2.mcsd.errors import ConfigError

ALLOWED_TOKENIZER = "/root/autodl-tmp/models/tofu_ft_llama2-7b"
SCORE_MAX_LENGTH = 500
_FORBIDDEN = ("Qwen", "qwen", "ComfyUI")
_DEFAULT_TAGS = {
    "question_start_tag": "[INST] ",
    "question_end_tag": " [/INST]",
    "answer_tag": "",
}


def assert_tokenizer_source(path):
    text = str(path).rstrip("/")
    if any(marker in text for marker in _FORBIDDEN):
        raise ConfigError("refusing a non-Llama-2 tokenizer")
    if text != ALLOWED_TOKENIZER:
        raise ConfigError("tokenizer must be the recorded tofu_ft_llama2-7b")
    return text


def load_tokenizer(source=ALLOWED_TOKENIZER):
    """Return (tokenizer, metadata). Missing files yield (None, metadata)."""
    source = assert_tokenizer_source(source)
    path = Path(source)
    meta = {
        "tokenizer_source": source,
        "tokenizer_loaded": False,
        "transformers_version": None,
        "tokenizer_sha256": None,
    }
    if not path.exists():
        return None, meta
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    meta["tokenizer_loaded"] = True
    meta["transformers_version"] = transformers.__version__
    meta["tokenizer_sha256"] = _tokenizer_sha(path)
    return tokenizer, meta


def llama2_tags(model_config_path=None):
    if model_config_path is None or not Path(model_config_path).exists():
        return dict(_DEFAULT_TAGS)
    text = Path(model_config_path).read_text()
    try:
        import yaml
    except ModuleNotFoundError:
        from I2.mcsd.config import _load_simple_yaml

        data = _load_simple_yaml(text)
    else:
        data = yaml.safe_load(text)
    family = data.get("llama2-7b") or {}
    tags = dict(_DEFAULT_TAGS)
    for key in _DEFAULT_TAGS:
        if family.get(key) is not None:
            tags[key] = family[key]
    return tags


def annotate_tokens(record, tokenizer, tags, tokenizer_meta):
    """Fill fact_span only when the diff is one span over the intended replacement."""
    updated = dict(record)
    updated["tokenizer_source"] = tokenizer_meta.get("tokenizer_source")
    updated["transformers_version"] = tokenizer_meta.get("transformers_version")
    updated["tokenizer_sha256"] = tokenizer_meta.get("tokenizer_sha256")
    updated["approved_for_localization"] = False
    if updated.get("exclusion_reason") or not updated.get("generated"):
        updated["status"] = "excluded"
        updated["is_candidate"] = False
        updated["fact_span"] = None
        return updated
    if tokenizer is None:
        updated["status"] = "token_unchecked"
        updated["token_alignment"] = "not_checked_no_tokenizer"
        updated["is_candidate"] = False
        updated["fact_span"] = None
        return updated
    plus_ids, plus_offsets = _encode(tokenizer, _question_text(tags, updated["q_plus"]))
    minus_ids, _minus_offsets = _encode(tokenizer, _question_text(tags, updated["q_minus"]))
    if plus_offsets is None:
        updated["status"] = "token_unchecked"
        updated["token_alignment"] = "offsets_unavailable"
        updated["is_candidate"] = False
        updated["fact_span"] = None
        return updated
    span, reason = _contiguous_diff(plus_ids, minus_ids)
    if reason:
        updated["status"] = reason
        updated["exclusion_reason"] = reason
        updated["token_alignment"] = reason
        updated["is_candidate"] = False
        updated["fact_span"] = None
        return updated
    if not _span_covers_replacement(plus_offsets, span, updated["intended_character_spans"], tags):
        updated["status"] = "token_span_outside_replacement"
        updated["exclusion_reason"] = "token_span_outside_replacement"
        updated["token_alignment"] = "token_span_outside_replacement"
        updated["is_candidate"] = False
        updated["fact_span"] = None
        return updated
    if not _prefix_matches(tokenizer, tags, updated["q_plus"], updated.get("source_answer"), plus_ids):
        updated["status"] = "score_prefix_boundary_mismatch"
        updated["exclusion_reason"] = "score_prefix_boundary_mismatch"
        updated["is_candidate"] = False
        updated["fact_span"] = None
        return updated
    minus_prefix = _encode(tokenizer, _question_text(tags, updated["q_minus"]))[0]
    if not _prefix_matches(tokenizer, tags, updated["q_minus"], updated.get("donor_answer"), minus_prefix):
        updated["status"] = "score_prefix_boundary_mismatch"
        updated["exclusion_reason"] = "score_prefix_boundary_mismatch"
        updated["is_candidate"] = False
        updated["fact_span"] = None
        return updated
    updated["status"] = "token_aligned"
    updated["token_alignment"] = "aligned"
    updated["fact_span"] = span
    updated["is_candidate"] = True
    return updated


def _question_text(tags, question):
    return tags["question_start_tag"] + question + tags["question_end_tag"]


def _score_text(tags, question, answer):
    return _question_text(tags, question) + (tags.get("answer_tag") or "") + (answer or "")


def _encode(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=SCORE_MAX_LENGTH,
        return_offsets_mapping=True,
    )
    ids = list(encoded["input_ids"])
    offsets = encoded["offset_mapping"] if "offset_mapping" in encoded else None
    if offsets is not None:
        offsets = [tuple(item) for item in offsets]
    return ids, offsets


def _contiguous_diff(left, right):
    if len(left) != len(right):
        return None, "length_mismatch"
    diffs = [index for index, (first, second) in enumerate(zip(left, right)) if first != second]
    if not diffs:
        return None, "fact_span_unchanged"
    if diffs[-1] - diffs[0] + 1 != len(diffs):
        return None, "token_span_not_single"
    return [diffs[0], diffs[-1] + 1], None


def _span_covers_replacement(offsets, span, intended_spans, tags):
    prefix = len(tags["question_start_tag"])
    intended = [(prefix + start, prefix + end) for start, end in intended_spans]
    start, end = span
    changed = list(range(start, end))
    if not changed or not intended:
        return False
    for index in changed:
        tok_start, tok_end = offsets[index]
        if tok_start == tok_end:
            continue
        if not any(tok_start >= left and tok_end <= right for left, right in intended):
            return False
    for left, right in intended:
        if not any(offsets[index][0] < right and offsets[index][1] > left for index in changed):
            return False
    return True


def _prefix_matches(tokenizer, tags, question, answer, question_ids):
    score_ids, _offsets = _encode(tokenizer, _score_text(tags, question, answer))
    if len(score_ids) < len(question_ids):
        return False
    return score_ids[: len(question_ids)] == list(question_ids)


def _tokenizer_sha(path):
    for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
        candidate = path / name
        if candidate.is_file():
            return hashlib.sha256(candidate.read_bytes()).hexdigest()
    return None

"""Perturbation q- keeps y+ and does not approve localization."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from I2.mcsd.perturbation import (  # noqa: E402
    build_perturbation_rows,
    _score_structure_reasons,
    encode_perturbed_pair,
    localization_block_reason,
    pseudonym_cycle,
    run_perturbation,
)
from I2.mcsd.same_split import _FICTIONAL_NAMES  # noqa: E402
from I2.mcsd.pipeline import run_examples  # noqa: E402
from I2.mcsd.same_split import load_forget_records  # noqa: E402
from I2.tests.test_patch_alignment import _example  # noqa: E402
from I2.tests.test_same_split_qminus import _two_tasks, _with_lines  # noqa: E402
from I2.tests.toy_model import ToyCausal  # noqa: E402


class ScreenTokenizer:
    """Whitespace tokenizer with the methods the score formatter calls."""

    def __init__(self):
        self.eos_token_id = 0
        self.ids = {}

    def tokenize(self, text, add_special_tokens=True):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=500, return_offsets_mapping=False):
        del truncation
        ids = [1] if add_special_tokens else []
        offsets = [(0, 0)] if add_special_tokens else []
        for match in re.finditer(r"\S+", str(text)):
            word = match.group(0)
            if word not in self.ids:
                self.ids[word] = len(self.ids) + 2
            ids.append(self.ids[word])
            offsets.append((match.start(), match.end()))
        ids = ids[:max_length]
        offsets = offsets[:max_length]
        encoded = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            encoded["offset_mapping"] = offsets
        return encoded


class TruncatingTokenizer(ScreenTokenizer):
    def tokenize(self, text, add_special_tokens=True):
        del text, add_special_tokens
        return [1] * 500

    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=500, return_offsets_mapping=False):
        del text, add_special_tokens, truncation, return_offsets_mapping
        return {"input_ids": [1] * max_length, "attention_mask": [1] * max_length}


class PerturbationTests(unittest.TestCase):
    def test_encoder_scores_y_plus_on_both_inputs(self):
        tags = {
            "question_start_tag": "[INST] ",
            "question_end_tag": " [/INST]",
            "answer_tag": " ",
        }
        tokenizer = ScreenTokenizer()
        encoded = encode_perturbed_pair(tokenizer, "See Ada now", "See Bob now", "Ada is noted.", tags)
        self.assertTrue(encoded["aligned"])
        self.assertEqual(encoded["reasons"], [])
        self.assertEqual(encoded["score_plus_labels"], encoded["score_minus_labels"])
        self.assertIsNone(encoded["y_minus"])
        self.assertEqual(encoded["y_plus"], "Ada is noted.")
        longer = encode_perturbed_pair(tokenizer, "See Ada now", "See Bob Extra now", "Ada is noted.", tags)
        self.assertFalse(longer["aligned"])
        self.assertIn("length_mismatch", longer["reasons"])
        with self.assertRaises(ValueError) as caught:
            encode_perturbed_pair(tokenizer, "See Ada now", "See Bob now", None, tags)
        self.assertIn("y-", str(caught.exception))
        truncated = encode_perturbed_pair(
            TruncatingTokenizer(),
            "See Ada now",
            "See Bob now",
            "Ada is noted.",
            tags,
        )
        self.assertFalse(truncated["aligned"])
        self.assertIn("truncated", truncated["reasons"])
        clean = {
            "answer_token_span": (4, 6),
            "prompt_token_span": (0, 4),
            "truncated": False,
            "answer_tokens_in_donor": False,
            "donor_ids_equal_score_prefix": True,
        }
        self.assertEqual(_score_structure_reasons(clean), [])
        self.assertIn("empty_answer_span", _score_structure_reasons(dict(clean, answer_token_span=(6, 6))))
        self.assertIn("boundary_fusion", _score_structure_reasons(dict(clean, answer_tokens_in_donor=True)))
        self.assertIn("padding_in_span", _score_structure_reasons(dict(clean, prompt_token_span=(0, 7))))

    def test_name_cycle_keeps_the_table_order(self):
        table = list(_FICTIONAL_NAMES)
        first = pseudonym_cycle("fixture:1", 20261107)
        second = pseudonym_cycle("fixture:1", 20261107)
        other = pseudonym_cycle("fixture:3", 20261107)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), sorted(table))
        self.assertEqual(sorted(other), sorted(table))
        index = first.index(table[0])
        offset = (len(table) - index) % len(table)
        self.assertEqual(first, table[offset:] + table[:offset])

    def test_gendered_pronoun_questions_are_excluded(self):
        rows = _with_lines(_two_tasks())
        rows[1]["question"] = "What genre is author Author Aa most known for in his writing?"
        built, _assignment, info = build_perturbation_rows(
            rows,
            _config(),
            "fixture",
            tokenizer=ScreenTokenizer(),
            tags=_tags(),
        )
        gender = next(row for row in built if row["file_line_index"] == 1)
        self.assertIsNone(gender["q_minus"])
        self.assertEqual(gender["exclusion_reason"], "pronoun_bound")
        self.assertGreater(info["rejections"].get("pronoun_bound", 0), 0)

    def test_screen_command_does_not_load_a_model(self):
        source = (
            Path(__file__).resolve().parents[1] / "commands" / "screen_perturbation_qminus.py"
        ).read_text()
        self.assertNotIn("LlamaForCausalLM", source)
        self.assertNotIn("run_examples", source)

    def test_equal_token_names_stay_unapproved_and_have_no_y_minus(self):
        rows = _with_lines(_two_tasks())
        digest = "fixture"
        built, _assignment, info = build_perturbation_rows(
            rows,
            _config(),
            digest,
            tokenizer=ScreenTokenizer(),
            tags=_tags(),
            tokenizer_meta={"tokenizer_source": "word", "tokenizer_loaded": True},
        )
        self.assertEqual(info["seed"], 20261107)
        self.assertEqual(info["generation_rule"], "single_name_natural_pseudonym_v2")
        self.assertEqual(info["selection"], "sha256_local_id_seed_rotation")
        genders = [row for row in built if row["file_line_index"] == 1]
        self.assertEqual(len(genders), 3)
        self.assertEqual([row["perturbation_role"] for row in genders], ["primary", "sensitivity_repeat", "sensitivity_repeat"])
        self.assertTrue(all(row["source_local_id"] == "fixture:1" for row in genders))
        self.assertTrue(all(row["sidecar_stage"] == "token_aligned_candidate" for row in genders))
        self.assertTrue(all(row["semantic_review_status"] == "pending" for row in genders))
        self.assertTrue(all(row["y_minus"] is None and row["donor_answer"] is None for row in genders))
        self.assertTrue(all(row["approved_for_localization"] is False for row in genders))
        self.assertTrue(all(row["is_candidate"] is False for row in genders))
        self.assertTrue(all(row["q_minus_role"] == "perturbed_input" for row in genders))
        self.assertEqual(len({row["pseudonym"] for row in genders}), 3)
        self.assertTrue(all("Author Aa" not in row["q_minus"] for row in genders))
        self.assertNotEqual(genders[0]["q_minus"], genders[0]["q_plus"])
        birth = next(row for row in built if row["file_line_index"] == 0)
        self.assertIsNone(birth["q_minus"])
        self.assertEqual(birth["exclusion_reason"], "compound_slot")
        self.assertEqual(birth["analysis_role"], "compound_slot_diagnostic")
        self.assertIsNone(birth["sidecar_stage"])
        blocked = dict(genders[0])
        blocked.update(
            {
                "question_plus": genders[0]["q_plus"],
                "question_minus": genders[0]["q_minus"],
                "sample_id": "p0",
                "author_id": 0,
                "prompt_plus_ids": [1],
                "prompt_minus_ids": [1],
                "score_plus_ids": [1],
                "score_minus_ids": [1],
                "fact_span": genders[0]["fact_span"],
            }
        )
        self.assertEqual(localization_block_reason(blocked), "not_approved_for_localization")

    def test_without_tokenizer_rows_stay_generated(self):
        rows = _with_lines(_two_tasks())
        built, _assignment, info = build_perturbation_rows(rows, _config(), "fixture", tokenizer=None, tags=_tags())
        genders = [row for row in built if row["file_line_index"] == 1]
        self.assertEqual(len(genders), 1)
        self.assertIsNone(genders[0]["q_minus"])
        self.assertEqual(genders[0]["exclusion_reason"], "not_checked_no_tokenizer")
        self.assertEqual(info["selectable_counts"], [0] * 10)
        self.assertEqual(sum(1 for row in built if row["sidecar_stage"] == "token_aligned_candidate"), 0)

    def test_names_present_in_forget_text_are_skipped(self):
        rows = _with_lines(_two_tasks())
        blocked = pseudonym_cycle("fixture:1", 20261107)[0]
        rows[1]["answer"] = rows[1]["answer"] + " " + blocked
        built, _assignment, info = build_perturbation_rows(
            rows,
            _config(),
            "fixture",
            tokenizer=ScreenTokenizer(),
            tags=_tags(),
        )
        used = {row["pseudonym"] for row in built if row["file_line_index"] == 1}
        self.assertNotIn(blocked, used)
        self.assertGreater(info["rejections"].get("in_target_text", 0), 0)

    def test_sidecar_does_not_rewrite_forget_or_approve(self):
        payload = json.dumps(_two_tasks()).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forget05.json"
            path.write_bytes(payload)
            summary, built = run_perturbation(
                path,
                _config(),
                Path(directory) / "out",
                tokenizer=ScreenTokenizer(),
                tags=_tags(),
                tokenizer_meta={"tokenizer_loaded": True, "tokenizer_source": "word"},
            )
            self.assertEqual(path.read_bytes(), payload)
            self.assertTrue(summary["forget_bytes_unchanged"])
            self.assertEqual(summary["approved_for_localization"], 0)
            self.assertEqual(summary["y_minus_non_null"], 0)
            self.assertIsNone(summary["n_written"])
            self.assertIsNone(summary["r_written"])
            self.assertFalse(summary["real_model_executed"])
            self.assertFalse(summary["localization_started"])
            self.assertGreater(summary["token_aligned_candidate"], 0)
            self.assertEqual(summary["token_aligned"], 0)
            self.assertEqual(summary["primary_entities"], summary["screened_rows"])
            self.assertFalse(summary["sensitivity_repeats_are_independent"])
            self.assertEqual(summary["sample_unit"], "source_row")
            self.assertEqual(summary["calibration_author_count"], 5)
            self.assertEqual(summary["holdout_author_count"], 5)
            self.assertTrue(summary["author_interval_unstable"])
            self.assertFalse(summary["confirmatory_localization_allowed"])
            self.assertEqual(summary["min_authors_for_interval"], 8)
            self.assertTrue(all(row["y_minus"] is None for row in built))
            _rows, digest = load_forget_records(path)
            self.assertEqual(summary["forget_sha256"], digest)

    def test_localization_entry_skips_an_unapproved_perturbation(self):
        model = ToyCausal()
        example = _example()
        example["q_minus_role"] = "perturbed_input"
        example["sidecar_stage"] = "token_aligned_candidate"
        example["approved_for_localization"] = False
        before = model.forward_count
        result = run_examples(
            model,
            [example],
            {"version": "mcsd_v2", "control_seed": 1, "selection": {"candidate_pool": []}},
            [{"question": "fact-plus"}, {"question": "fact-minus"}],
        )
        self.assertEqual(model.forward_count, before)
        self.assertEqual(result["excluded"][0]["status"], "excluded")
        self.assertIn("not_approved_for_localization", result["excluded"][0]["reasons"])
        self.assertIsNone(result["excluded"][0]["N"])


def _config():
    return {
        "min_authors_for_interval": 8,
        "authors_per_task": 10,
        "questions_per_author": 2,
        "task_ids": ["1"],
        "perturbation": {
            "rule": "single_name_natural_pseudonym_v2",
            "seed": 20261107,
            "max_controls": 3,
        },
    }


def _tags():
    return {
        "question_start_tag": "[INST] ",
        "question_end_tag": " [/INST]",
        "answer_tag": " ",
    }


if __name__ == "__main__":
    unittest.main()

"""Same-split q- pairs stay inside calibration or holdout and do not rewrite forget."""

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

from I2.commands.build_same_split_qminus import main, run_sidecar  # noqa: E402
from I2.commands.tokenize_same_split_qminus import finalize_token_row  # noqa: E402
from I2.mcsd.config import load_config  # noqa: E402
from I2.mcsd.errors import ConfigError  # noqa: E402
from I2.mcsd.same_split import (  # noqa: E402
    assign_splits,
    build_same_split_rows,
    crosses_split,
    load_forget_records,
    replacement_is_exact,
    require_locked_threshold,
    select_donor,
)
from I2.mcsd.token_record import annotate_tokens, assert_tokenizer_source, load_tokenizer  # noqa: E402


class SameSplitTests(unittest.TestCase):
    def test_author_plus_one_crosses_and_is_rejected(self):
        self.assertTrue(crosses_split(0, 1))
        with self.assertRaises(ConfigError):
            select_donor(0, [0, 1, 2, 4])
        self.assertEqual(select_donor(0, [0, 2, 4, 6]), 2)

    def test_same_split_name_swap_uses_the_donor_answer(self):
        rows = _keep_one_gender_donor(_two_tasks(), 2, "What sex is author Author Ca ?")
        built, _assignment = build_same_split_rows(rows, _config(), "fixture")
        gender = next(row for row in built if row["file_line_index"] == 1)
        self.assertEqual(gender["local_id"], "fixture:1")
        self.assertEqual(gender["donor_entity_id"], 2)
        self.assertEqual(gender["donor_local_id"], "fixture:5")
        self.assertEqual(gender["split"], "calibration")
        self.assertEqual(gender["donor_split"], "calibration")
        self.assertEqual(gender["donor_task_id"], "1")
        self.assertFalse(crosses_split(gender["local_entity_id"], gender["donor_entity_id"]))
        self.assertEqual(gender["q_minus"], "What gender is author Author Ca ?")
        self.assertNotIn("sex", gender["q_minus"])
        self.assertEqual(gender["y_minus"], "Author Ca is noted.")
        self.assertEqual(gender["character_spans"], [[22, 31]])
        self.assertNotIn("Author Aa", gender["q_minus"])
        self.assertTrue(gender["template_outside_spans_unchanged"])
        self.assertEqual(gender["relation"], "gender")
        self.assertEqual(gender["pairing_construction"], "qplus_name_replacement")
        self.assertEqual(gender["status"], "pending_review")
        self.assertEqual(gender["record_status"], "pending_review")
        self.assertEqual(gender["analysis_role"], "primary_single_slot")
        self.assertTrue(gender["is_candidate"])
        self.assertFalse(gender["approved_for_localization"])
        self.assertEqual(gender["semantic_review_status"], "needs_manual_review")
        self.assertTrue(all(row["task_id"] == row["donor_task_id"] for row in built if row["is_candidate"]))

    def test_missing_or_mismatched_donor_answer_is_excluded(self):
        rows = _two_tasks()
        missing_rows = _keep_one_gender_donor(rows, None)
        missing, _assignment = build_same_split_rows(missing_rows, _config(), "fixture")
        missing_row = next(row for row in missing if row["file_line_index"] == 1)
        self.assertEqual(missing_row["exclusion_reason"], "same_relation_missing")
        self.assertEqual(missing_row["record_status"], "excluded")
        self.assertEqual(missing_row["analysis_role"], "excluded")
        self.assertIsNone(missing_row["y_minus"])
        self.assertIsNone(missing_row["donor_answer"])
        self.assertIsNone(missing_row["q_minus"])
        self.assertFalse(missing_row["is_candidate"])
        mismatched = _keep_one_gender_donor(rows, 2, answer="Author Aa is noted.")
        bad, _assignment = build_same_split_rows(mismatched, _config(), "fixture")
        bad_row = next(row for row in bad if row["file_line_index"] == 1)
        self.assertEqual(bad_row["exclusion_reason"], "donor_answer_mismatch")
        self.assertIsNone(bad_row["y_minus"])
        self.assertFalse(bad_row["is_candidate"])
        blank_rows = _keep_one_gender_donor(rows, 2, answer=" ")
        blank, _assignment = build_same_split_rows(blank_rows, _config(), "fixture")
        self.assertEqual(
            next(row for row in blank if row["file_line_index"] == 1)["exclusion_reason"],
            "donor_answer_missing",
        )

    def test_cross_task_donor_is_not_used(self):
        rows = _retarget_calibration_genders(_two_tasks(), "What sex is author %s ?")
        built, _assignment = build_same_split_rows(rows, _config(task_ids=["1", "2"]), "fixture")
        gender = next(row for row in built if row["file_line_index"] == 1)
        self.assertEqual(gender["exclusion_reason"], "same_relation_ambiguous")
        self.assertIsNone(gender["y_minus"])
        self.assertIsNone(gender["donor_answer"])
        self.assertIsNone(gender["q_minus"])
        self.assertFalse(gender["is_candidate"])
        self.assertTrue(all(row.get("donor_task_id") in (None, row["task_id"]) for row in built))

    def test_later_same_split_donor_is_used_when_the_next_template_differs(self):
        rows = _keep_one_gender_donor(_two_tasks(), 4, "What sex is author Author Ea ?")
        built, _assignment = build_same_split_rows(rows, _config(), "fixture")
        gender = next(row for row in built if row["file_line_index"] == 1)
        self.assertEqual(gender["donor_entity_id"], 4)
        self.assertEqual(gender["q_minus"], "What gender is author Author Ea ?")
        self.assertNotIn("sex", gender["q_minus"])
        self.assertEqual(gender["y_minus"], "Author Ea is noted.")
        self.assertTrue(gender["is_candidate"])
        self.assertEqual(gender["analysis_role"], "primary_single_slot")

    def test_extra_word_is_not_an_exact_replacement(self):
        source = "What gender is author Author A ?"
        target = "What gender is writer Author C ?"
        self.assertFalse(replacement_is_exact(source, target, [(22, 30)], [(22, 30)]))

    def test_task1_pilot_is_allowed_and_a_lowered_threshold_is_refused(self):
        assignment = assign_splits(_with_lines(_two_tasks()[:20]), 2, 10, ["1"])
        self.assertEqual(assignment["task_ids"], ["1"])
        self.assertEqual(len(assignment["calibration_author_ids"]), 5)
        self.assertEqual(len(assignment["holdout_author_ids"]), 5)
        self.assertTrue(assignment["author_interval_unstable"])
        self.assertFalse(assignment["confirmatory_localization_allowed"])
        with self.assertRaises(ConfigError):
            require_locked_threshold({"min_authors_for_interval": 5})
        with self.assertRaises(ConfigError):
            assign_splits(_with_lines(_two_tasks()), 2, 10, ["1", "2"], min_authors_per_split=5)

    def test_duplicate_author_across_tasks_is_refused(self):
        rows = _with_lines(_two_tasks(name_for=lambda task_id, block: "Author %sa" % chr(ord("A") + block)))
        with self.assertRaises(ConfigError):
            assign_splits(rows, 2, 10, ["1", "2"])

    def test_forget_bytes_stay_unchanged(self):
        payload = json.dumps(_two_tasks()).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forget05.json"
            path.write_bytes(payload)
            _rows, digest = load_forget_records(path)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(digest, __import__("hashlib").sha256(payload).hexdigest())
            output = Path(directory) / "out"
            summary, _built = run_sidecar(path, _config(), output)
            self.assertEqual(path.read_bytes(), payload)
            self.assertTrue(summary["forget_bytes_unchanged"])
            self.assertFalse(summary["previous_candidates_reused"])
            self.assertEqual(summary["calibration_author_count"], 5)
            self.assertEqual(summary["holdout_author_count"], 5)
            self.assertEqual(summary["approved_for_localization"], 0)
            self.assertEqual(summary["candidates"], 0)
            self.assertEqual(summary["pending_review"], 0)
            self.assertEqual(summary["ground_truth_candidates"], 0)
            self.assertEqual(summary["synthetic_candidates"], 0)
            self.assertEqual(summary["synthetic_non_primary"], 0)
            self.assertEqual(summary["compound_slot_diagnostic"], 10)
            self.assertEqual(summary["excluded"], 10)
            self.assertEqual(summary["generated"], 10)
            self.assertEqual(summary["donor_answer_non_null"], 10)
            self.assertEqual(summary["pair_cluster_count"], 0)
            self.assertEqual(summary["compound_pair_cluster_count"], 2)
            self.assertEqual(summary["pilot_decision"], "descriptive_pilot")
            self.assertTrue(summary["confirmatory_requires_new_data_version"])
            self.assertTrue(summary["author_interval_unstable"])
            self.assertFalse(summary["confirmatory_localization_allowed"])
            self.assertFalse(summary["model_loaded"])
            self.assertFalse(summary["tokenizer_called"])
            self.assertEqual(summary["scope_task_ids"], ["1"])
            self.assertEqual(summary["pairing_rule"], "qplus_single_name_same_relation")
            listed = [json.loads(line) for line in (output / "qminus.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["local_id"].startswith(summary["forget_sha256"] + ":") for row in listed))
            self.assertEqual(sum(1 for row in listed if row["is_candidate"]), 0)
            self.assertTrue(all(row["analysis_role"] != "synthetic_non_primary" for row in listed if row["is_candidate"]))
            self.assertEqual(sum(1 for line in (output / "candidates.jsonl").read_text().splitlines() if line), 0)
            self.assertEqual(sum(1 for line in (output / "compound_diagnostics.jsonl").read_text().splitlines() if line), 10)
            self.assertTrue((output / "synthetic_non_primary.jsonl").is_file())
            self.assertTrue((output / "excluded.jsonl").is_file())
            self.assertTrue((output / "relation_coverage.json").is_file())

    def test_candidate_config_keeps_the_author_threshold(self):
        candidate = load_config(WORKSPACE / "I2" / "configs" / "mcsd_candidate.yaml")
        split = load_config(WORKSPACE / "I2" / "configs" / "qminus_split.yaml")
        self.assertEqual(candidate["min_authors_for_interval"], 8)
        self.assertEqual(split["min_authors_for_interval"], 8)
        self.assertEqual(candidate["calibration_author_indices"], [0, 2, 4, 6, 8])
        self.assertEqual([str(item) for item in split["task_ids"]], ["1"])
        self.assertIsNone((candidate.get("pairing") or {}).get("mode"))
        self.assertIsNone((split.get("pairing") or {}).get("mode"))
        self.assertFalse(candidate["allow_student_e1"])

    def test_other_tokenizers_are_refused_and_the_recorded_one_may_be_missing(self):
        with self.assertRaises(ConfigError):
            assert_tokenizer_source("/tmp/Qwen/tokenizer")
        with self.assertRaises(ConfigError):
            assert_tokenizer_source("/tmp/ComfyUI/tokenizer.json")
        with self.assertRaises(ConfigError):
            assert_tokenizer_source("/tmp/other-llama")
        tokenizer, meta = load_tokenizer()
        if not Path(meta["tokenizer_source"]).exists():
            self.assertIsNone(tokenizer)
            self.assertFalse(meta["tokenizer_loaded"])

    def test_birth_slots_match_on_non_entity_tokens_and_keep_the_donor_answer(self):
        built, _assignment = build_same_split_rows(_two_tasks(), _config(), "fixture")
        birth = next(row for row in built if row["file_line_index"] == 0)
        self.assertEqual(birth["donor_entity_id"], 2)
        self.assertEqual(birth["donor_split"], "calibration")
        self.assertEqual(birth["donor_task_id"], "1")
        self.assertEqual(
            birth["q_minus"],
            "What is the full name of the author born in CityCa on 1900-01-02 ?",
        )
        self.assertEqual(birth["y_minus"], "The author is Author Ca.")
        self.assertEqual(birth["donor_answer"], birth["y_minus"])
        self.assertNotEqual(birth["slot_surfaces"], birth["donor_slot_surfaces"])
        self.assertTrue(birth["slot_tokens_differ"])
        self.assertEqual(birth["replacement_kind"], "place+date")
        self.assertEqual(birth["analysis_role"], "compound_slot_diagnostic")
        self.assertEqual(birth["record_status"], "generated")
        self.assertFalse(birth["is_candidate"])
        self.assertFalse(birth["approved_for_localization"])
        paired, _assignment = build_same_split_rows(
            _keep_one_gender_donor(_two_tasks(), 2, "What sex is author Author Ca ?"),
            _config(),
            "fixture",
        )
        tags = {
            "question_start_tag": "[INST] ",
            "question_end_tag": " [/INST]",
            "answer_tag": " ",
        }
        meta = {
            "tokenizer_source": "/root/autodl-tmp/models/tofu_ft_llama2-7b",
            "transformers_version": "test",
            "tokenizer_sha256": None,
        }
        gender = annotate_tokens(
            next(row for row in paired if row["file_line_index"] == 1),
            WordTokenizer(),
            tags,
            meta,
        )
        self.assertEqual(gender["status"], "token_aligned")
        self.assertTrue(gender["is_candidate"])
        self.assertFalse(gender["approved_for_localization"])
        self.assertEqual(len(gender["fact_span"]), 2)
        self.assertGreater(gender["fact_span"][1], gender["fact_span"][0])

    def test_task1_synthetic_rows_are_not_primary_candidates(self):
        path = WORKSPACE.parent / "Right-to-be-forgotten" / "data" / "tofu" / "forget05.json"
        rows, digest = load_forget_records(path)
        config = _config()
        config["questions_per_author"] = 20
        built, assignment = build_same_split_rows(rows, config, digest)
        self.assertEqual(assignment["task_ids"], ["1"])
        self.assertEqual(len(built), 200)
        by_line = {row["file_line_index"]: row for row in built}
        for line in (0, 80):
            self.assertEqual(by_line[line]["analysis_role"], "compound_slot_diagnostic")
            self.assertEqual(by_line[line]["replacement_kind"], "place+date")
            self.assertFalse(by_line[line]["is_candidate"])
            self.assertTrue(by_line[line]["y_minus"])
            self.assertEqual(by_line[line]["y_minus"], by_line[line]["donor_answer"])
        for line in (2, 45):
            row = by_line[line]
            self.assertEqual(row["relation"], "birth_place")
            self.assertEqual(row["analysis_role"], "primary_single_slot")
            self.assertEqual(row["record_status"], "pending_review")
            self.assertEqual(row["pairing_construction"], "qplus_name_replacement")
            self.assertEqual(row["replacement_kind"], "name")
            self.assertTrue(row["y_minus"])
            self.assertNotEqual(row["q_minus"], row["q_plus"])
        primary = [row for row in built if row["is_candidate"]]
        self.assertTrue(primary)
        self.assertTrue(all(row["analysis_role"] == "primary_single_slot" for row in primary))
        self.assertTrue(all(row["donor_origin"] == "ground_truth" for row in primary))
        self.assertEqual(
            sum(1 for row in built if row["analysis_role"] == "synthetic_non_primary" and row["is_candidate"]),
            0,
        )
        self.assertTrue(all(row["approved_for_localization"] is False for row in built))
        self.assertFalse(assignment["confirmatory_localization_allowed"])

    def test_synthetic_token_alignment_stays_out_of_the_candidate_set(self):
        tags = {
            "question_start_tag": "[INST] ",
            "question_end_tag": " [/INST]",
            "answer_tag": " ",
        }
        meta = {
            "tokenizer_source": "/root/autodl-tmp/models/tofu_ft_llama2-7b",
            "transformers_version": "test",
            "tokenizer_sha256": None,
        }
        synthetic = {
            "q_plus": "See Ada now",
            "q_minus": "See Bob now",
            "generated": True,
            "exclusion_reason": None,
            "analysis_role": "synthetic_non_primary",
            "record_status": "generated",
            "status": "synthetic",
            "is_candidate": False,
            "approved_for_localization": False,
            "intended_character_spans": [[4, 7]],
            "source_answer": "",
            "y_plus": None,
            "donor_answer": None,
        }
        missing = finalize_token_row(synthetic, WordTokenizer(), tags, meta)
        self.assertEqual(missing["token_alignment"], "missing_y_plus")
        self.assertFalse(missing["is_candidate"])
        self.assertFalse(missing["approved_for_localization"])
        synthetic["y_plus"] = "Noted."
        finished = finalize_token_row(synthetic, WordTokenizer(), tags, meta)
        self.assertEqual(finished["analysis_role"], "synthetic_non_primary")
        self.assertEqual(finished["record_status"], "generated")
        self.assertEqual(finished["token_alignment"], "aligned")
        self.assertFalse(finished["is_candidate"])
        self.assertFalse(finished["approved_for_localization"])
        untouched = dict(synthetic)
        untouched["q_minus"] = None
        skipped = finalize_token_row(untouched, WordTokenizer(), tags, meta)
        self.assertEqual(skipped["token_alignment"], "not_encoded_no_q_minus")
        self.assertFalse(skipped["is_candidate"])
        source = (
            Path(__file__).resolve().parents[1] / "commands" / "tokenize_same_split_qminus.py"
        ).read_text()
        self.assertNotIn("LlamaForCausalLM", source)
        self.assertNotIn("forget.py", source)

    def test_missing_forget_file_raises(self):
        with self.assertRaises(ConfigError):
            main(["/tmp/does-not-exist-forget05.json"])


def _config(task_ids=None):
    return {
        "min_authors_for_interval": 8,
        "authors_per_task": 10,
        "questions_per_author": 2,
        "task_ids": ["1"] if task_ids is None else task_ids,
        "scope": "descriptive_pilot",
        "tokenizer": {"source": "/root/autodl-tmp/models/tofu_ft_llama2-7b"},
        "pairing": {"mode": None},
    }


def _with_lines(rows):
    stamped = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["file_line_index"] = index
        stamped.append(item)
    return stamped


def _keep_one_gender_donor(rows, donor_block, donor_question=None, answer=None):
    """Leave at most one other calibration author on the gender relation."""
    copied = [dict(row) for row in rows]
    for block in (2, 4, 6, 8):
        index = block * 2 + 1
        name = "Author %sa" % chr(ord("A") + block)
        if block == donor_block:
            if donor_question is not None:
                copied[index]["question"] = donor_question
            if answer is not None:
                copied[index]["answer"] = answer
            continue
        copied[index]["question"] = "Where did %s study ?" % name
    return copied


def _retarget_calibration_genders(rows, question_template, answer=None):
    """Change every calibration donor gender row, leaving author 0's row intact."""
    copied = [dict(row) for row in rows]
    for block in (2, 4, 6, 8):
        index = block * 2 + 1
        name = "Author %sa" % chr(ord("A") + block)
        copied[index]["question"] = question_template % name
        if answer is not None:
            copied[index]["answer"] = answer
    return copied


def _two_tasks(name_for=None):
    if name_for is None:
        def name_for(task_id, block):
            index = (int(task_id) - 1) * 10 + block
            return "Author %sa" % chr(ord("A") + index)

    rows = []
    for task_id in ("1", "2"):
        for block in range(10):
            name = name_for(task_id, block)
            place = "City%s" % name.split()[-1]
            when = "1900-01-%02d" % block
            rows.append(
                {
                    "task_id": task_id,
                    "question": "What is the full name of the author born in %s on %s ?" % (place, when),
                    "answer": "The author is %s." % name,
                }
            )
            rows.append(
                {
                    "task_id": task_id,
                    "question": "What gender is author %s ?" % name,
                    "answer": "%s is noted." % name,
                }
            )
    return rows


class WordTokenizer:
    def __init__(self):
        self.ids = {}

    def __call__(self, text, add_special_tokens=True, truncation=True, max_length=500, return_offsets_mapping=False):
        del truncation, max_length
        ids = [1] if add_special_tokens else []
        offsets = [(0, 0)] if add_special_tokens else []
        for match in re.finditer(r"\S+", text):
            word = match.group(0)
            if word not in self.ids:
                self.ids[word] = len(self.ids) + 2
            ids.append(self.ids[word])
            offsets.append((match.start(), match.end()))
        encoded = {"input_ids": ids}
        if return_offsets_mapping:
            encoded["offset_mapping"] = offsets
        return encoded


if __name__ == "__main__":
    unittest.main()

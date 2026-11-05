"""Existing donor sequences are searched; thresholds and training stay closed."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from I2.mcsd.config import load_config  # noqa: E402
from I2.mcsd.pairing_search import contiguous_diff_span, find_existing_pairs, question_template_tokens  # noqa: E402


class PairingSearchTests(unittest.TestCase):
    def test_interior_slot_is_recorded_and_not_approved(self):
        left = _row("a", [1, 2, 2, 9, 9, 4, 4], "calibration")
        right = _row("b", [1, 2, 2, 8, 8, 4, 4], "calibration")
        anchor = _row("c", [1, 3, 3, 3, 3, 3, 3], "holdout")
        result = find_existing_pairs([left, right, anchor])
        self.assertEqual(len(result["pairs"]), 1)
        found = result["pairs"][0]
        self.assertEqual(found["diff_span"], [3, 5])
        self.assertEqual(found["template_tokens_before_span"], 2)
        self.assertEqual(found["template_tokens_after_span"], 2)
        self.assertIsNone(found["character_substitution"])
        self.assertFalse(found["fact_span_annotated"])
        self.assertFalse(found["approved_for_localization"])
        self.assertEqual(found["status"], "unapproved_near_miss")

    def test_edge_token_is_not_a_fact_slot(self):
        left = _row("a", [1, 9, 9, 4], "calibration")
        right = _row("b", [1, 8, 8, 4], "calibration")
        anchor = _row("c", [1, 7, 7, 3], "holdout")
        result = find_existing_pairs([left, right, anchor])
        self.assertEqual(result["pairs"], [])
        self.assertGreater(result["pairs_rejected_not_fact_slot"], 0)

    def test_noncontiguous_and_equal_sequences_are_excluded(self):
        self.assertIsNone(contiguous_diff_span([1, 2, 3], [9, 2, 8]))
        self.assertIsNone(contiguous_diff_span([1, 2], [1, 2]))
        self.assertEqual(find_existing_pairs([_row("a", [1, 2, 3], "calibration"), _row("b", [9, 2, 8], "calibration")])["pairs"], [])

    def test_chat_wrapper_alone_is_not_a_question_template(self):
        # Shared edges are the wrapper. The whole question body differs.
        left = _row("a", [1, 2, 3, 4, 5, 9], "calibration")
        right = _row("b", [1, 2, 6, 7, 8, 9], "calibration")
        anchor = _row("c", [1, 2, 6, 7, 5, 9], "calibration")
        result = find_existing_pairs([left, right, anchor])
        self.assertEqual(result["shared_wrapper_prefix_tokens"], 2)
        self.assertEqual(result["shared_wrapper_suffix_tokens"], 1)
        self.assertEqual(question_template_tokens(6, [2, 5], 2, 1), 0)
        self.assertEqual(result["pairs_rejected_wrapper_only"], 1)
        self.assertEqual(result["pairs_rejected_not_fact_slot"], 2)
        self.assertEqual(result["pairs"], [])

    def test_prefix_mismatch_is_skipped(self):
        row = _row("a", [1, 2], "calibration")
        row["donor_ids_equal_score_prefix"] = False
        result = find_existing_pairs([row, _row("b", [1, 9], "holdout")])
        self.assertEqual(result["rows_with_stable_prefix"], 1)
        self.assertEqual(result["rows_skipped_prefix"], ["a"])
        self.assertEqual(result["pairs"], [])

    def test_candidate_config_keeps_author_threshold(self):
        config = load_config(WORKSPACE / "I2" / "configs" / "mcsd_candidate.yaml")
        self.assertEqual(config["min_authors_for_interval"], 8)
        self.assertEqual(config["calibration_author_indices"], [0, 2, 4, 6, 8])
        self.assertIsNone((config.get("pairing") or {}).get("mode"))
        self.assertFalse(config["allow_student_e1"])


def _row(sample_id, donor, split):
    return {
        "sample_id": sample_id,
        "author_id": 0,
        "split": split,
        "donor_input_ids": donor,
        "score_input_ids": list(donor) + [7],
        "donor_ids_equal_score_prefix": True,
    }


if __name__ == "__main__":
    unittest.main()

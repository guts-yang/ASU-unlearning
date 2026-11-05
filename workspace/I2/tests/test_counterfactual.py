"""Generated q- changes only an existing identity string and is not approved."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from I2.mcsd.config import load_config  # noqa: E402
from I2.mcsd.counterfactual import build_qminus  # noqa: E402


class CounterfactualTests(unittest.TestCase):
    def test_author_name_is_replaced_and_the_template_stays(self):
        question = "What gender is author Basil Mahfouz Al-Kuwaiti?"
        built = build_qminus(question, 0)
        self.assertEqual(built["q_minus"], "What gender is author Nikolai Abilov?")
        self.assertEqual(built["character_spans"], [[22, 36]])
        self.assertTrue(built["single_contiguous_span"])
        self.assertTrue(built["template_outside_spans_unchanged"])
        self.assertFalse(built["approved_for_localization"])
        self.assertEqual(built["token_alignment"], "not_checked_no_tokenizer")

    def test_possessive_suffix_is_kept(self):
        question = "Can you tell me about the occupations of Basil Mahfouz Al-Kuwaiti's parents?"
        built = build_qminus(question, 0)
        self.assertEqual(
            built["q_minus"],
            "Can you tell me about the occupations of Nikolai Abilov's parents?",
        )
        self.assertTrue(built["template_outside_spans_unchanged"])

    def test_curly_possessive_and_two_names_are_not_one_span(self):
        question = "Hina Ameen’s father knew Hina Ameen."
        built = build_qminus(question, 2)
        self.assertEqual(built["q_minus"], "Xin Lee Williams’s father knew Xin Lee Williams.")
        self.assertEqual(built["exclusion_reason"], "multiple_identity_spans")
        self.assertFalse(built["single_contiguous_span"])
        self.assertFalse(built["approved_for_localization"])

    def test_birth_place_and_date_are_copied_from_the_next_author(self):
        question = "What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956?"
        built = build_qminus(question, 0)
        self.assertEqual(
            built["q_minus"],
            "What is the full name of the author born in Astana, Kazakhstan on the 7th of February, 1952?",
        )
        self.assertEqual(built["replacement_kind"], "birth_place_and_date")
        self.assertTrue(built["template_outside_spans_unchanged"])
        self.assertEqual(built["exclusion_reason"], "multiple_identity_spans")

    def test_a_bare_surname_uses_the_other_author_surname(self):
        question = "How does Nakamura's book 'A Piece of Me' differ from his other works?"
        built = build_qminus(question, 6)
        self.assertEqual(
            built["q_minus"],
            "How does Marais's book 'A Piece of Me' differ from his other works?",
        )
        self.assertEqual(built["replacement_kind"], "author_surname")
        self.assertTrue(built["single_contiguous_span"])
        self.assertTrue(built["template_outside_spans_unchanged"])

    def test_a_question_without_the_registered_identity_is_not_rewritten(self):
        built = build_qminus("What is a river?", 0)
        self.assertIsNone(built["q_minus"])
        self.assertFalse(built["generated"])
        self.assertEqual(built["exclusion_reason"], "identity_span_not_unique")

    def test_candidate_config_is_not_the_place_these_strings_are_approved(self):
        config = load_config(WORKSPACE / "I2" / "configs" / "mcsd_candidate.yaml")
        self.assertIsNone((config.get("pairing") or {}).get("mode"))
        self.assertFalse(config["allow_student_e1"])


if __name__ == "__main__":
    unittest.main()

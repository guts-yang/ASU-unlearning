import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from align_m_tar import blocking_reason  # noqa: E402
from span_match import locate_unique, ranges_overlap  # noqa: E402


class SpanMatchTests(unittest.TestCase):
    def test_unique_offset_uses_original_text(self):
        answer = "The author's name is Hina Ameen."
        start, end = locate_unique(answer, "Hina Ameen")
        self.assertEqual(answer[start:end], "Hina Ameen")
        self.assertEqual(start, answer.index("Hina Ameen"))

    def test_duplicate_and_missing_are_unresolved(self):
        self.assertEqual(locate_unique("is is", "is"), "substring occurs more than once")
        self.assertEqual(locate_unique("Paris", "London"), "substring absent from answer")
        self.assertEqual(locate_unique("Paris", " Paris"), "substring includes leading or trailing whitespace")

    def test_overlap_detector(self):
        spans = [{"start_char": 0, "end_char": 5}, {"start_char": 4, "end_char": 8}]
        self.assertTrue(ranges_overlap(spans))

    def test_training_mask_stays_blocked(self):
        pending = [{"candidate_row_index": index, "review_status": "pending_human_review"} for index in range(200)]
        self.assertIn("human review", blocking_reason(pending, None))
        adjudicated = [{"candidate_row_index": index, "review_status": "adjudicated"} for index in range(200)]
        with tempfile.TemporaryDirectory() as directory:
            self.assertIn("canonical", blocking_reason(adjudicated, directory))

    def test_saved_pilot_preserves_source_answer(self):
        rows = [json.loads(line) for line in (HERE / "proposed.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 200)
        row = rows[69]
        self.assertIn("\n\nAdditional 10 question-answer pairs:", row["answer"])
        self.assertTrue(all(span["end_char"] < row["answer"].index("\n") for span in row["target_spans"]))
        self.assertEqual(rows[58]["annotation_status"], "unresolved")
        self.assertTrue(all(row["canonical_forget05_task_id_1"] is False for row in rows))


if __name__ == "__main__":
    unittest.main()

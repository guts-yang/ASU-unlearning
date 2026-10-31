import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from align_m_tar import (  # noqa: E402
    MAX_LENGTH,
    align_fact_row,
    answer_prefix_length,
    blocking_reason,
    count_report,
    encode_alignment_inputs,
    main,
    masks_or_none,
    snapshot_matches,
    write_count_report,
)
from build_final_spans import existing_final_is_locked  # noqa: E402
from compare_annotations import classify_row, compare, load_jsonl  # noqa: E402
from merge_ai_final import apply_blind, consensus_record, needs_blind  # noqa: E402
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
        self.assertEqual(
            locate_unique("Paris", " Paris"),
            "substring includes leading or trailing whitespace",
        )

    def test_overlap_detector(self):
        spans = [{"start_char": 0, "end_char": 5}, {"start_char": 4, "end_char": 8}]
        self.assertTrue(ranges_overlap(spans))

    def test_empty_adjudicated_is_rejected(self):
        empty = [
            {
                "candidate_row_index": index,
                "review_status": "adjudicated",
                "reviewer": "chenyliao",
                "target_spans": [],
            }
            for index in range(200)
        ]
        reason = blocking_reason(empty, None)
        self.assertIn("empty adjudicated", reason)

        pending = [
            {
                "candidate_row_index": index,
                "review_status": "pending_human_review",
                "reviewer": None,
                "target_spans": [],
            }
            for index in range(200)
        ]
        self.assertIn("incomplete", blocking_reason(pending, None))

    def test_complete_final_spans_still_block_on_canonical(self):
        complete = [
            {
                "candidate_row_index": index,
                "review_status": "adjudicated",
                "reviewer": "chenyliao",
                "target_spans": [
                    {
                        "start_char": 0,
                        "end_char": 1,
                        "text": "x",
                        "layer": "fact_value",
                    }
                ],
            }
            for index in range(199)
        ]
        complete.append(
            {
                "candidate_row_index": 199,
                "review_status": "unresolved",
                "reviewer": "chenyliao",
                "target_spans": [],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            self.assertIn("canonical", blocking_reason(complete, directory))

    def test_saved_pilot_preserves_source_answer(self):
        rows = [json.loads(line) for line in (HERE / "final_spans.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 200)
        row = rows[69]
        self.assertIn("\n\nAdditional 10 question-answer pairs:", row["answer"])
        self.assertTrue(all(span["end_char"] < row["answer"].index("\n") for span in row["target_spans"]))
        self.assertEqual(rows[58]["review_status"], "unresolved")
        self.assertTrue(all(row["canonical_forget05_task_id_1"] is False for row in rows))

    def test_compare_marks_crosscheck_as_non_human(self):
        answer = "The author's name is Hina Ameen."
        fact = {
            "start_char": 21,
            "end_char": 31,
            "text": "Hina Ameen",
            "layer": "fact_value",
            "fact_type": "person",
        }
        match = {
            "candidate_row_index": 0,
            "question": "What is the full name?",
            "answer": answer,
            "annotation_status": "annotated",
            "target_spans": [fact],
            "unresolved_reasons": [],
        }
        disagree = {
            **match,
            "candidate_row_index": 1,
            "annotation_status": "unresolved",
            "target_spans": [],
            "unresolved_reasons": ["unclear"],
        }
        proposed = [
            {**match, "candidate_row_index": index} if index != 1 else {**match, "candidate_row_index": 1}
            for index in range(200)
        ]
        cross = [
            {**match, "candidate_row_index": index} if index != 1 else disagree
            for index in range(200)
        ]
        report = compare(proposed, cross)
        self.assertFalse(report["identity"]["independent_ai_crosscheck_is_human_second_annotator"])
        self.assertIsNone(report["pass_fail_threshold"])
        self.assertEqual(report["counts"]["rows"], 200)
        self.assertGreater(len(report["human_review_queues"]["high_priority_fact_or_status"]), 0)
        self.assertEqual(classify_row(proposed[0], cross[0]), "exact_match")

    def test_final_spans_are_ai_terminal_and_not_human(self):
        path = HERE / "final_spans.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 200)
        self.assertEqual([row["candidate_row_index"] for row in rows], list(range(200)))
        for row in rows:
            self.assertTrue(row["question"])
            self.assertTrue(row["answer"])
            self.assertIn(row["review_status"], {"adjudicated", "unresolved", "no_target"})
            self.assertTrue(row["reviewer"])
            self.assertNotIn("human", row["reviewer"])
            if row["review_status"] == "adjudicated":
                self.assertTrue(row["target_spans"])
            else:
                self.assertEqual(row["target_spans"], [])
        with tempfile.TemporaryDirectory() as directory:
            self.assertIn("canonical", blocking_reason(rows, directory))
        stats = json.loads((HERE / "agreement_stats.json").read_text())
        self.assertEqual(stats["second_annotator"], None)
        self.assertIsNone(stats["span_boundary_agreement_threshold"])
        self.assertEqual(stats["human_final_spans"]["adjudicated_with_spans_count"], 0)
        self.assertFalse(stats["ai_final_spans"]["is_human_adjudication"])
        self.assertIsNone(stats["ai_final_spans"]["pass_fail_threshold"])


def _span(answer: str, text: str, layer: str) -> dict:
    start = answer.index(text)
    return {
        "start_char": start,
        "end_char": start + len(text),
        "text": text,
        "layer": layer,
        "fact_type": "other",
        "reason": "fixture",
    }


def _row(answer: str, spans: list[dict], status: str = "annotated") -> dict:
    return {
        "candidate_row_index": 0,
        "question": "What is known?",
        "answer": answer,
        "annotation_status": status,
        "target_spans": spans,
        "unresolved_reasons": [] if status == "annotated" else ["fixture"],
    }


class MergeRuleTests(unittest.TestCase):
    def test_exact_match_copies_both_layers(self):
        answer = "The value is quartz."
        spans = [_span(answer, "quartz", "fact_value"), _span(answer, "value is", "relation")]
        proposed = _row(answer, spans, "proposed")
        cross = _row(answer, spans, "annotated")
        self.assertEqual(classify_row(proposed, cross), "exact_match")
        self.assertFalse(needs_blind("exact_match"))
        merged = consensus_record(proposed, cross, "exact_match")
        self.assertEqual(merged["reviewer"], "ai_consensus")
        self.assertEqual(merged["source_of_spans"], "ai_consensus")
        self.assertEqual(merged["review_status"], "adjudicated")
        self.assertEqual(len(merged["target_spans"]), 2)
        self.assertFalse(merged["disagreement_with_crosscheck"])

    def test_relation_only_keeps_fact_value(self):
        answer = "Her father is a poet in Cairo."
        fact = _span(answer, "poet", "fact_value")
        proposed = _row(answer, [fact, _span(answer, "father is a", "relation")], "proposed")
        cross = _row(answer, [fact, _span(answer, "is a", "relation")], "annotated")
        self.assertEqual(classify_row(proposed, cross), "relation_only")
        merged = consensus_record(proposed, cross, "relation_only")
        self.assertEqual([span["text"] for span in merged["target_spans"]], ["poet"])
        self.assertTrue(merged["disagreement_with_crosscheck"])
        self.assertEqual(merged["reviewer"], "ai_consensus")

    def test_agreed_unresolved_stays_empty(self):
        answer = "The detail is unconfirmed but likely true."
        proposed = _row(answer, [], "unresolved")
        cross = _row(answer, [], "unresolved")
        self.assertEqual(classify_row(proposed, cross), "exact_match")
        merged = consensus_record(proposed, cross, "exact_match")
        self.assertEqual(merged["review_status"], "unresolved")
        self.assertEqual(merged["target_spans"], [])
        self.assertTrue(merged["reviewer"])

    def test_blind_confirmation_and_rejection(self):
        answer = "The genre is geology today."
        proposed_fact = _span(answer, "geology", "fact_value")
        cross_fact = _span(answer, "geology today", "fact_value")
        proposed = _row(answer, [proposed_fact, _span(answer, "genre is", "relation")], "proposed")
        cross = _row(answer, [cross_fact], "annotated")
        self.assertTrue(needs_blind(classify_row(proposed, cross)))
        confirmed = apply_blind(
            proposed,
            cross,
            _row(answer, [proposed_fact, _span(answer, "genre is", "relation")]),
            "blind_adjudicator:kimi-k3-max",
        )
        self.assertEqual(confirmed["source_of_spans"], "proposed")
        self.assertEqual(confirmed["reviewer"], "blind_adjudicator:kimi-k3-max")
        self.assertEqual(sorted(span["layer"] for span in confirmed["target_spans"]), ["fact_value", "relation"])
        rejected = apply_blind(
            proposed,
            cross,
            _row(answer, [_span(answer, "today", "fact_value")]),
            "blind_adjudicator:kimi-k3-max",
        )
        self.assertEqual(rejected["review_status"], "unresolved")
        self.assertEqual(rejected["target_spans"], [])
        refused = apply_blind(proposed, cross, _row(answer, [], "no_target"), "blind_adjudicator:kimi-k3-max")
        self.assertEqual(refused["review_status"], "no_target")
        self.assertEqual(refused["target_spans"], [])
        missing_type = _span(answer, "geology", "fact_value")
        missing_type["fact_type"] = None
        still_confirmed = apply_blind(proposed, cross, _row(answer, [missing_type]), "blind_adjudicator:kimi-k3-max")
        self.assertEqual(still_confirmed["source_of_spans"], "proposed")
        self.assertEqual(still_confirmed["target_spans"][0]["fact_type"], "other")

    def test_blind_character_failure_is_unresolved(self):
        answer = "Paris and Paris."
        proposed = _row(answer, [], "unresolved")
        cross = _row(answer, [], "no_target")
        blind = _row(
            answer,
            [
                {
                    "start_char": 0,
                    "end_char": 5,
                    "text": "Paris",
                    "layer": "fact_value",
                    "fact_type": "location",
                    "reason": "duplicate",
                }
            ],
        )
        merged = apply_blind(proposed, cross, blind, "blind_adjudicator:kimi-k3-max")
        self.assertEqual(merged["review_status"], "unresolved")
        self.assertEqual(merged["target_spans"], [])

    def test_locked_final_shell_detects_terminal_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final_spans.jsonl"
            path.write_text(json.dumps({"review_status": "adjudicated", "target_spans": [{}]}) + "\n")
            self.assertTrue(existing_final_is_locked(path))
            path.write_text(json.dumps({"review_status": "pending_human_review", "target_spans": []}) + "\n")
            self.assertFalse(existing_final_is_locked(path))


class _Encoded(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _FakeTokenizer:
    eos_token_id = 2

    def tokenize(self, text: str, add_special_tokens: bool = True) -> list[str]:
        tokens = text.split(" ")
        if add_special_tokens:
            return ["<s>"] + tokens
        return tokens

    def __call__(self, text: str, add_special_tokens: bool = True, max_length: int = 500, truncation: bool = True, return_offsets_mapping: bool = False):
        ids = [1]
        offsets = [(0, 0)]
        for index, _char in enumerate(text):
            if len(ids) >= max_length:
                break
            ids.append(3)
            offsets.append((index, index + 1))
        encoded = _Encoded(input_ids=ids, attention_mask=[1] * len(ids))
        if return_offsets_mapping:
            encoded["offset_mapping"] = offsets
        return encoded


def _labels(length: int, answer_indexes: set[int]) -> list[int]:
    return [11 if index in answer_indexes else -100 for index in range(length)]


class MaskAlignmentTests(unittest.TestCase):
    def test_query_index_is_the_previous_position(self):
        question = "Q"
        answer = "fact relation"
        prefix = answer_prefix_length(question)
        self.assertEqual(prefix, len("[INST] Q [/INST]"))
        fact = {"start_char": 0, "end_char": 4, "text": "fact", "layer": "fact_value"}
        relation = {"start_char": 5, "end_char": 13, "text": "relation", "layer": "relation"}
        offsets = [
            (0, 0),
            (0, prefix),
            (prefix, prefix + 4),
            (prefix + 4, prefix + 5),
            (prefix + 5, prefix + 13),
        ]
        labels = _labels(8, {2, 3, 4})
        result = align_fact_row(
            {
                "candidate_row_index": 3,
                "question": question,
                "answer": answer,
                "review_status": "adjudicated",
                "target_spans": [fact, relation],
            },
            offsets,
            labels,
            max_length=8,
        )
        self.assertTrue(result["alignable"])
        self.assertEqual(result["mask"][1], 1)
        self.assertEqual(result["mask"][3], 0)
        self.assertEqual(result["fact_token_count"], 1)
        self.assertEqual(result["partial_overlap_token_count"], 0)
        self.assertNotEqual(labels[2], -100)

    def test_partial_token_is_counted_and_still_masked(self):
        question = "Q"
        answer = "fact relation"
        prefix = answer_prefix_length(question)
        offsets = [(0, 0), (prefix, prefix + 6)]
        labels = _labels(4, {1})
        result = align_fact_row(
            {
                "candidate_row_index": 1,
                "question": question,
                "answer": answer,
                "review_status": "adjudicated",
                "target_spans": [
                    {"start_char": 0, "end_char": 4, "text": "fact", "layer": "fact_value"}
                ],
            },
            offsets,
            labels,
            max_length=4,
        )
        self.assertTrue(result["alignable"])
        self.assertEqual(result["mask"][0], 1)
        self.assertEqual(result["partial_overlap_token_count"], 1)

    def test_unresolved_row_is_all_zero(self):
        result = align_fact_row(
            {
                "candidate_row_index": 4,
                "question": "Q",
                "answer": "unused",
                "review_status": "unresolved",
                "target_spans": [],
            }
        )
        self.assertTrue(result["alignable"])
        self.assertEqual(len(result["mask"]), MAX_LENGTH - 1)
        self.assertEqual(set(result["mask"]), {0})
        self.assertEqual(count_report(result)["fact_token_count"], 0)

    def test_mask_stays_inside_answer_positions(self):
        question = "Q"
        answer = "fact"
        prefix = answer_prefix_length(question)
        offsets = [(0, 0), (0, prefix), (prefix, prefix + 4)]
        labels = _labels(4, {2})
        result = align_fact_row(
            {
                "candidate_row_index": 2,
                "question": question,
                "answer": answer,
                "review_status": "adjudicated",
                "target_spans": [
                    {"start_char": 0, "end_char": 4, "text": "fact", "layer": "fact_value"}
                ],
            },
            offsets,
            labels,
            max_length=4,
        )
        self.assertTrue(result["alignable"])
        for query_index, bit in enumerate(result["mask"]):
            if bit:
                self.assertNotEqual(labels[query_index + 1], -100)

    def test_question_token_overlap_blocks_the_batch(self):
        question = "Q"
        answer = "fact"
        prefix = answer_prefix_length(question)
        offsets = [(0, 0), (prefix - 2, prefix + 4)]
        labels = _labels(4, set())
        blocked = align_fact_row(
            {
                "candidate_row_index": 5,
                "question": question,
                "answer": answer,
                "review_status": "adjudicated",
                "target_spans": [
                    {"start_char": 0, "end_char": 4, "text": "fact", "layer": "fact_value"}
                ],
            },
            offsets,
            labels,
            max_length=4,
        )
        self.assertFalse(blocked["alignable"])
        self.assertIsNone(masks_or_none([blocked]))

    def test_truncated_span_blocks_mask_write(self):
        question = "Q"
        answer = "fact"
        prefix = answer_prefix_length(question)
        offsets = [(0, 0), (prefix, prefix + 2)]
        labels = _labels(4, {1})
        truncated = align_fact_row(
            {
                "candidate_row_index": 6,
                "question": question,
                "answer": answer,
                "review_status": "adjudicated",
                "target_spans": [
                    {"start_char": 0, "end_char": 4, "text": "fact", "layer": "fact_value"}
                ],
            },
            offsets,
            labels,
            max_length=4,
        )
        self.assertTrue(truncated["truncated"])
        self.assertIsNone(masks_or_none([truncated]))
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            mask = Path(directory) / "m_tar.pt"
            write_count_report(report, [truncated])
            payload = json.loads(report.read_text())
            self.assertEqual(set(payload[0]), {
                "candidate_row_index",
                "fact_token_count",
                "partial_overlap_token_count",
                "truncated",
            })
            self.assertTrue(payload[0]["truncated"])
            self.assertFalse(mask.exists())

    def test_existing_tokenizer_dir_still_blocks_on_canonical(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(snapshot_matches(Path(directory)))
            named = Path(directory) / "351844e75ed0bcbbe3f10671b3c808d2b83894ee"
            named.mkdir()
            self.assertTrue(snapshot_matches(named))
            rows = [
                {
                    "candidate_row_index": index,
                    "review_status": "no_target",
                    "reviewer": "ai_consensus",
                    "target_spans": [],
                }
                for index in range(200)
            ]
            self.assertIn("canonical", blocking_reason(rows, directory))

    def test_cli_refuses_to_create_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "m_tar.pt"
            report = Path(directory) / "report.json"
            with self.assertRaises(SystemExit) as caught:
                main(["--tokenizer", directory, "--output", str(output), "--report", str(report)])
            self.assertIn("canonical", str(caught.exception))
            self.assertFalse(output.exists())
            self.assertFalse(report.exists())

    def test_encoder_reuses_training_template(self):
        encoded = encode_alignment_inputs(_FakeTokenizer(), "Q", "fact", max_length=32)
        self.assertEqual(len(encoded["labels"]), 32)
        self.assertEqual(encoded["labels"][0], -100)
        self.assertEqual(encoded["offsets"][0], (0, 0))
        self.assertEqual(answer_prefix_length("Q"), len("[INST] Q [/INST]"))


if __name__ == "__main__":
    unittest.main()

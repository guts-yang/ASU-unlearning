"""Author blocks and author-cluster intervals."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from I2.mcsd.bootstrap import bootstrap_from_rows  # noqa: E402
from I2.mcsd.data import author_index_for_row, split_for_author  # noqa: E402
from I2.mcsd.errors import ConfigError  # noqa: E402


class AuthorStatTests(unittest.TestCase):
    def test_author_blocks_match_the_even_odd_split(self):
        self.assertEqual(author_index_for_row(0, 20), 0)
        self.assertEqual(author_index_for_row(20, 20), 1)
        self.assertEqual(split_for_author(0, [0, 2], [1, 3]), "calibration")
        self.assertEqual(split_for_author(1, [0, 2], [1, 3]), "holdout")
        with self.assertRaises(ConfigError):
            split_for_author(0, [0], [0])

    def test_few_authors_do_not_inflate_the_sample_size(self):
        rows = []
        for author, value in (("a", 1.0), ("b", 3.0)):
            for _ in range(10):
                rows.append({"author_id": author, "N": value})
        result = bootstrap_from_rows(rows, "N", min_authors=8, samples=50, seed=1)
        self.assertEqual(result["n_authors"], 2)
        self.assertEqual(result["n_rows"], 20)
        self.assertEqual(result["independent_sample_size"], 2)
        self.assertTrue(result["unstable_estimate"])
        self.assertTrue(result["row_count_not_used_as_sample_size"])
        self.assertIsNone(result["ci_low"])
        self.assertIsNone(result["ci_high"])
        self.assertAlmostEqual(result["mean"], 2.0)

        stable_rows = [{"author_id": "a%s" % index, "N": float(index)} for index in range(4)]
        stable = bootstrap_from_rows(stable_rows, "N", min_authors=4, samples=30, seed=2)
        self.assertFalse(stable["unstable_estimate"])
        self.assertEqual(stable["independent_sample_size"], 4)
        self.assertLessEqual(stable["ci_low"], stable["mean"])
        self.assertGreaterEqual(stable["ci_high"], stable["mean"])


if __name__ == "__main__":
    unittest.main()

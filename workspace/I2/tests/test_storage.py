"""Direction I result paths stay in a new results directory and do not train."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
REPO = WORKSPACE.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from I2.commands import student_entry  # noqa: E402
from I2.mcsd.errors import StorageError  # noqa: E402
from I2.mcsd.storage import checkpoint_dir, measurement_dir, reject_forbidden  # noqa: E402


class StorageTests(unittest.TestCase):
    def test_new_measurement_dir_is_under_i2_results(self):
        path = measurement_dir("pre_experiment", "20261105T120000Z")
        self.assertEqual(path.parent, WORKSPACE / "I2" / "results")
        self.assertEqual(path.name, "pre_experiment_20261105T120000Z")
        self.assertFalse(path.exists())

    def test_existing_pre_experiment_dir_is_not_reused(self):
        with self.assertRaises(StorageError):
            measurement_dir("pre_experiment", "20261105T110514Z")
        with self.assertRaises(StorageError):
            reject_forbidden(WORKSPACE / "I2" / "results" / "pre_experiment_20261105T110514Z")

    def test_checkpoints_have_one_student_directory(self):
        path = checkpoint_dir("20261105T120000Z")
        self.assertEqual(path.parent.name, "student_e1_20261105T120000Z")
        self.assertEqual(path.name, "unlearn_times_1")
        self.assertFalse(path.exists())
        self.assertFalse(path.parent.exists())
        with self.assertRaises(StorageError):
            measurement_dir("student_e1", "20261105T120000Z")

    def test_existing_asu_result_trees_are_rejected(self):
        with self.assertRaises(StorageError):
            reject_forbidden(REPO / "Right-to-be-forgotten" / "results" / "llama2-7b" / "forget05")
        with self.assertRaises(StorageError):
            reject_forbidden(Path("/root/autodl-tmp/ASU-unlearning/checkpoints/tofu_replay/llama2-7b"))

    def test_student_entry_does_not_launch_or_name_a_command(self):
        source = Path(student_entry.__file__).read_text()
        self.assertNotIn("forget.py", source)
        self.assertNotIn("subprocess", source)
        self.assertFalse(student_entry.FORGET_ENTRY_WIRED)
        before = student_entry.TRAINING_LAUNCHES
        with self.assertRaises(StorageError):
            student_entry.launch_student_e1(
                save_dir=str(REPO / "Right-to-be-forgotten" / "results" / "MCSD")
            )
        self.assertEqual(student_entry.TRAINING_LAUNCHES, before)
        with self.assertRaises(RuntimeError):
            student_entry.launch_student_e1()
        self.assertEqual(student_entry.TRAINING_LAUNCHES, before + 1)
        student_entry.TRAINING_LAUNCHES = before
        stderr = io.StringIO()
        from contextlib import redirect_stderr

        with redirect_stderr(stderr):
            code = student_entry.main([])
        self.assertEqual(code, 2)
        self.assertIn("disabled", stderr.getvalue())
        self.assertEqual(student_entry.TRAINING_LAUNCHES, before)


if __name__ == "__main__":
    unittest.main()

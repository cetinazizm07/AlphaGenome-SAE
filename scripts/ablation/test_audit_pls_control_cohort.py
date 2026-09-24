"""Validation for pre-output control assignment across frozen windows."""

import unittest

import numpy as np
import pandas as pd

from audit_pls_control_cohort import collect_windows


class ControlCohortAuditTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"windows": [{
            "chrom": "chr1", "start": 100, "end": 1_048_676,
            "positive_bins": 3,
        }]}
        starts = 100 + 128 * np.arange(8192, dtype=np.int64)
        self.annotations = pd.DataFrame({
            "split": ["test"] * 8192,
            "chrom": ["chr1"] * 8192,
            "bin_start": starts,
            "n_mask": [True] * 8192,
            "cCRE_PLS": [True] * 3 + [False] * 8189,
        })
        self.manifest = pd.DataFrame({
            "split": ["test"], "chrom": ["chr1"],
            "win_start": [100], "win_end": [1_048_676],
        })

    def test_collects_exact_complete_heldout_mask(self):
        windows = collect_windows(
            self.plan, {}, self.annotations, self.manifest
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["mask"].sum(), 3)
        self.assertEqual(len(windows[0]["expected_bin_starts"]), 8192)

    def test_rejects_changed_frozen_positive_bin_count(self):
        self.plan["windows"][0]["positive_bins"] = 4
        with self.assertRaisesRegex(ValueError, "count changed"):
            collect_windows(self.plan, {}, self.annotations, self.manifest)

    def test_rejects_nonheldout_or_missing_bin(self):
        self.annotations.loc[0, "split"] = "train"
        with self.assertRaisesRegex(ValueError, "complete PLS grid"):
            collect_windows(self.plan, {}, self.annotations, self.manifest)


if __name__ == "__main__":
    unittest.main()

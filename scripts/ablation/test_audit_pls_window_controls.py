"""Support- and dose-matching rules use activations, never model outputs."""

import unittest

import numpy as np
import pandas as pd

from audit_pls_window_controls import (
    activation_rows_for_window,
    rank_window_candidates,
)


class WindowControlAuditTests(unittest.TestCase):
    def test_requires_same_target_active_sites_and_similar_dose(self):
        target = np.array([2., 0., 3., 0.])
        candidates = {
            10: np.array([2.1, 99., 3.0, 1.]),
            11: np.array([0., 0., 5., 0.]),
            12: np.array([4., 0., 6., 0.]),
        }
        chosen, rows = rank_window_candidates(target, candidates, [10, 11, 12])
        self.assertEqual(chosen, 10)
        self.assertEqual(rows[0]["comparator_active_on_target_bins"], 2)
        self.assertEqual(rows[0]["comparator_active_on_all_pls_bins"], 4)

    def test_silent_target_has_no_comparator(self):
        chosen, rows = rank_window_candidates(
            np.zeros(2), {10: np.ones(2)}, [10]
        )
        self.assertIsNone(chosen)
        self.assertEqual(rows, [])

    def test_no_forced_match_outside_dose_bound(self):
        chosen, rows = rank_window_candidates(
            np.array([1.]), {10: np.array([4.])}, [10]
        )
        self.assertIsNone(chosen)
        self.assertEqual(len(rows), 1)

    def test_maps_window_coordinates_inside_multi_window_shard(self):
        shard = pd.DataFrame({
            "chrom": ["chrA", "chrB", "chrA", "chrB"],
            "window_start": [100, 900, 100, 900],
            "bin_start": [100, 900, 228, 1028],
            "split": ["test"] * 4,
        })
        window = shard.loc[shard.chrom == "chrA"].reset_index(drop=True)
        rows = activation_rows_for_window(
            shard, window, "chrA", 100, expected_rows=2
        )
        np.testing.assert_array_equal(rows, [0, 2])

    def test_rejects_coordinate_alignment_mismatch(self):
        shard = pd.DataFrame({
            "chrom": ["chrA", "chrA"],
            "window_start": [100, 100],
            "bin_start": [100, 228],
            "split": ["test", "test"],
        })
        window = shard.copy()
        window.loc[1, "bin_start"] = 356
        with self.assertRaisesRegex(ValueError, "bin_start"):
            activation_rows_for_window(shard, window, "chrA", 100, 2)


if __name__ == "__main__":
    unittest.main()

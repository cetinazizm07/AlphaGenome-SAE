"""Tests for ag_sae.stats: safe Wilcoxon and paired statistical testing."""

from __future__ import annotations

import unittest
import numpy as np

from ag_sae.stats import StatisticalSummary, compute_paired_statistics, safe_wilcoxon


class TestSafeWilcoxon(unittest.TestCase):
    def test_empty_array_returns_null_hypothesis(self):
        stat, p = safe_wilcoxon(np.array([]))
        self.assertEqual(stat, 0.0)
        self.assertEqual(p, 1.0)

    def test_all_zeros_returns_null_hypothesis(self):
        stat, p = safe_wilcoxon(np.zeros(20))
        self.assertEqual(stat, 0.0)
        self.assertEqual(p, 1.0)

    def test_identical_nonzero_differences_computes_significance(self):
        stat, p = safe_wilcoxon(np.ones(15) * 0.5)
        self.assertGreaterEqual(stat, 0.0)
        self.assertLess(p, 0.01)

    def test_genuine_paired_differences(self):
        rng = np.random.default_rng(42)
        deltas = rng.normal(loc=-1.5, scale=0.5, size=40)
        stat, p = safe_wilcoxon(deltas)
        self.assertGreaterEqual(stat, 0.0)
        self.assertLess(p, 1e-4)


class TestComputePairedStatistics(unittest.TestCase):
    def test_empty_deltas_raises_value_error(self):
        with self.assertRaises(ValueError):
            compute_paired_statistics(np.array([]))

    def test_all_zeros_returns_safe_summary(self):
        summary = compute_paired_statistics(np.zeros(30))
        self.assertIsInstance(summary, StatisticalSummary)
        self.assertEqual(summary.mean_delta, 0.0)
        self.assertEqual(summary.median_delta, 0.0)
        self.assertEqual(summary.t_statistic, 0.0)
        self.assertEqual(summary.t_p_value, 1.0)
        self.assertEqual(summary.wilcoxon_w, 0.0)
        self.assertEqual(summary.wilcoxon_p_value, 1.0)
        self.assertEqual(summary.pct_negative, 0.0)

    def test_strongly_negative_intervention_effect(self):
        deltas = np.array([-2.5, -3.1, -1.8, -4.2, -2.9, -3.5, -2.0, -3.8])
        summary = compute_paired_statistics(deltas)
        self.assertLess(summary.mean_delta, -2.0)
        self.assertEqual(summary.pct_negative, 100.0)
        self.assertLess(summary.t_statistic, 0.0)
        self.assertLess(summary.t_p_value, 0.001)
        self.assertLess(summary.wilcoxon_p_value, 0.05)

        report = summary.to_dict()
        expected_keys = {
            "mean_delta", "median_delta", "t_stat", "t_p",
            "wilcoxon_w", "wilcoxon_p", "pct_negative"
        }
        self.assertEqual(set(report.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()

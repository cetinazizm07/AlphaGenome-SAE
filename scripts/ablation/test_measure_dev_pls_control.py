"""Pure decision-rule tests for dev-local PLS control selection."""

import json
import unittest

import numpy as np

from measure_dev_pls_control import choose_control, choose_local_control


class DevLocalControlTests(unittest.TestCase):
    def test_selects_neutral_feature_with_matching_dev_pls_firing(self):
        counts = np.array([4, 4, 5, 0, 4])
        global_firing = np.array([0.04, 0.04, 0.04, 0.04, 0.10])
        auroc = np.full((5, 2), 0.51)
        auroc[1, 0] = 0.60
        chosen, ranked = choose_control(0, counts, 100, global_firing, auroc)
        self.assertEqual(chosen, 2)
        self.assertEqual([row["feature"] for row in ranked], [2, 3])

    def test_reports_no_match_instead_of_widening_tolerance(self):
        counts = np.array([8, 1, 2])
        firing = np.array([0.04, 0.04, 0.04])
        auroc = np.full((3, 2), 0.51)
        chosen, ranked = choose_control(0, counts, 100, firing, auroc)
        self.assertIsNone(chosen)
        self.assertEqual(len(ranked), 2)

    def test_silent_target_is_not_matchable(self):
        with self.assertRaises(ValueError):
            choose_control(0, np.zeros(2, dtype=int), 100,
                           np.array([0.04, 0.04]), np.full((2, 1), 0.5))

    def test_local_match_uses_rate_strength_and_low_auroc(self):
        counts = np.array([100, 101, 100, 100, 3])
        strengths = np.array([200., 205., 700., 210., 5.])
        auroc = np.full((5, 2), 0.52)
        auroc[3, 0] = 0.70
        chosen, ranked = choose_local_control(0, counts, strengths, 200, auroc)
        self.assertEqual(chosen, 1)
        self.assertEqual(ranked[0]["feature"], 1)
        self.assertNotIn(3, [row["feature"] for row in ranked])
        json.dumps({"selected": chosen, "rows": ranked})


if __name__ == "__main__":
    unittest.main()

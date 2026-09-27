"""Unit tests for ag_sae.intervene: element masking, norm dose matching, and edit mechanics."""

from __future__ import annotations

import unittest
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ag_sae.intervene import (
    DiscoveredFeatures,
    checked_element_mask,
    match_control_norm,
)


class TestCheckedElementMask(unittest.TestCase):
    def test_valid_mask(self):
        mask = np.zeros(8192, dtype=bool)
        mask[100:105] = True
        out = checked_element_mask(mask, 8192)
        self.assertEqual(len(out), 8192)
        self.assertEqual(out.sum(), 5)

    def test_all_false_mask_raises(self):
        mask = np.zeros(8192, dtype=bool)
        with self.assertRaises(ValueError):
            checked_element_mask(mask, 8192)

    def test_wrong_length_mask_raises(self):
        mask = np.zeros(100, dtype=bool)
        mask[10] = True
        with self.assertRaises(ValueError):
            checked_element_mask(mask, 8192)


class TestDiscoveredFeatures(unittest.TestCase):
    def test_as_dict_structure(self):
        features = DiscoveredFeatures(
            promoter_feature_id=199,
            insulator_feature_id=4586,
            enhancer_feature_id=7205,
            shuffled_control_feature_id=5454,
        )
        mapping = features.as_dict()
        self.assertIn("PLS_Promoter", mapping)
        self.assertIn("CTCF_Insulator", mapping)
        self.assertIn("Enhancer_pELS", mapping)
        self.assertIn("Shuffled_Control", mapping)

        self.assertEqual(mapping["PLS_Promoter"], (199, "cCRE_PLS"))
        self.assertEqual(mapping["CTCF_Insulator"], (4586, "cCRE_CTCF_only"))


@unittest.skipIf(torch is None, "PyTorch not available in local environment")
class TestMatchControlNorm(unittest.TestCase):
    def test_dose_matching_matches_target_l2_norm(self):
        baseline = torch.zeros((1, 8192, 64), dtype=torch.float32)
        target = torch.zeros((1, 8192, 64), dtype=torch.float32)
        control = torch.zeros((1, 8192, 64), dtype=torch.float32)

        tss_bin = 4000
        target[0, tss_bin, 0] = 5.0
        control[0, tss_bin, 1] = 2.0

        matched, info = match_control_norm(baseline, target, control, tss_bin)

        target_norm = float(torch.linalg.vector_norm(target[0, tss_bin] - baseline[0, tss_bin]))
        matched_norm = float(torch.linalg.vector_norm(matched[0, tss_bin] - baseline[0, tss_bin]))

        self.assertAlmostEqual(target_norm, 5.0, places=5)
        self.assertAlmostEqual(matched_norm, 5.0, places=5)
        self.assertAlmostEqual(info["scale"], 2.5, places=5)

    def test_near_zero_target_raises_value_error(self):
        baseline = torch.zeros((1, 8192, 64), dtype=torch.float32)
        target = torch.zeros((1, 8192, 64), dtype=torch.float32)
        control = torch.zeros((1, 8192, 64), dtype=torch.float32)
        control[0, 100, 0] = 2.0

        with self.assertRaises(ValueError):
            match_control_norm(baseline, target, control, 100)


if __name__ == "__main__":
    unittest.main()

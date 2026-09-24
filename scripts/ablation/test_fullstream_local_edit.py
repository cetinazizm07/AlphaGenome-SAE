"""CPU tests: exact intervention locality, shared baseline, and honest readouts."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from fullstream_local_edit import (
    active_feature_support,
    checked_element_mask,
    local_edit_triplet,
    paired_ablation_is_measurable,
    summarize_local_effects,
)
from plan_fullstream_cohort import sha256
from plan_fullstream_local_cohort import build_local_plan
from run_fullstream_local_window import load_frozen_plan


class FakeCore:
    def encode(self, value):
        return value, None

    def activation(self, value):
        return value

    def get_sparse_activations(self, value):
        return value

    def decode(self, codes, _params):
        return codes + 0.5  # Make reconstruction drift visible at selected bins.


class FakeSAE:
    d_in = 2
    hidden = 2
    channel_scale = torch.ones(2)
    core = FakeCore()


class LocalEditTests(unittest.TestCase):
    def setUp(self):
        self.activation = torch.tensor([[[1., 2.], [3., 4.],
                                         [5., 6.], [7., 8.]]])
        self.mask = np.array([False, True, False, True])

    def test_ablation_changes_only_selected_bins_and_one_code(self):
        baseline, target, control, counts = local_edit_triplet(
            FakeSAE(), self.activation, self.mask, 0, 1,
            "ablate", 5.0, 7.0, batch_size=1,
        )
        torch.testing.assert_close(baseline[0, ~self.mask], self.activation[0, ~self.mask])
        torch.testing.assert_close(target[0, ~self.mask], self.activation[0, ~self.mask])
        torch.testing.assert_close(control[0, ~self.mask], self.activation[0, ~self.mask])
        torch.testing.assert_close(baseline, self.activation)
        torch.testing.assert_close(target[0, self.mask, 0], torch.zeros(2))
        torch.testing.assert_close(target[0, self.mask, 1], self.activation[0, self.mask, 1])
        torch.testing.assert_close(control[0, self.mask, 1], torch.zeros(2))
        torch.testing.assert_close(control[0, self.mask, 0], self.activation[0, self.mask, 0])
        self.assertEqual(counts["element_bins"], 2)
        self.assertEqual(counts["target_changed_on_element"], 2)
        self.assertEqual(counts["target_activation_changed_on_element"], 2)
        self.assertTrue(paired_ablation_is_measurable(counts))

    def test_target_active_support_is_limited_to_element_bins(self):
        activation = self.activation.clone()
        activation[0, 2, 0] = 0
        support = active_feature_support(FakeSAE(), activation, self.mask, 0)
        np.testing.assert_array_equal(support, [False, True, False, True])

    def test_target_active_support_rejects_silent_feature(self):
        activation = self.activation.clone()
        activation[0, self.mask, 0] = 0
        with self.assertRaisesRegex(ValueError, "silent"):
            active_feature_support(FakeSAE(), activation, self.mask, 0)

    def test_injection_never_lowers_existing_code(self):
        baseline, target, control, counts = local_edit_triplet(
            FakeSAE(), self.activation, self.mask, 0, 1,
            "inject", 5.0, 5.0, batch_size=2,
        )
        torch.testing.assert_close(target[0, self.mask, 0], torch.tensor([5., 7.]))
        torch.testing.assert_close(control[0, self.mask, 1], torch.tensor([5., 8.]))
        torch.testing.assert_close(target[0, self.mask, 1], baseline[0, self.mask, 1])
        torch.testing.assert_close(target[0, ~self.mask], self.activation[0, ~self.mask])
        self.assertEqual(counts["target_changed_on_element"], 1)
        self.assertEqual(counts["control_changed_on_element"], 1)

    def test_empty_or_misaligned_mask_is_rejected(self):
        for mask in (np.zeros(4, dtype=bool), np.ones(3, dtype=bool),
                     np.array([0, 1, 0, 1])):
            with self.assertRaises(ValueError):
                checked_element_mask(mask, 4)
            with self.assertRaises(ValueError):
                local_edit_triplet(FakeSAE(), self.activation, mask, 0, 1,
                                   "ablate", 5.0, 7.0)

    def test_silent_control_prevents_more_model_forwards(self):
        activation = self.activation.clone()
        activation[0, self.mask, 1] = 0
        _, _, _, counts = local_edit_triplet(
            FakeSAE(), activation, self.mask, 0, 1,
            "ablate", 5.0, 7.0,
        )
        self.assertEqual(counts["control_changed_on_element"], 0)
        self.assertFalse(paired_ablation_is_measurable(counts))

    def test_readout_summary_distinguishes_local_and_propagated_effects(self):
        arrays = {
            "dnase_original": np.array([1., 2., 3., 4.]),
            "dnase_baseline": np.array([1., 2., 3., 4.]),
            "dnase_ablate_target": np.array([1.2, 1.5, 3.2, 3.5]),
            "dnase_ablate_control": np.array([1., 2., 3., 4.]),
        }
        result = summarize_local_effects(arrays, ("dnase",), self.mask, "ablate")["dnase"]
        self.assertEqual(result["target_mean_on_element"], -0.5)
        self.assertAlmostEqual(result["target_mean_off_element"], 0.2)
        self.assertEqual(result["baseline_mae_on_element"], 0.0)
        self.assertEqual(result["baseline_mae_off_element"], 0.0)

    def test_local_plan_preserves_frozen_positive_selection(self):
        parent = {
            "format": "fullstream_pls_exploratory_cohort_plan_v1",
            "fold": "fold1", "sae_seed": 0, "tap": "resid_pre_b8",
            "concept": "cCRE_PLS", "target_feature": 199,
            "control_feature": 4247, "split": "test", "match_sign": 1,
            "primary_heads": ["dnase", "atac", "cage", "rna_seq"],
            "off_target_heads": ["procap", "chip_tf", "chip_histone"],
            "calibration_split": "dev", "calibration_levels": {"target": 5., "control": 3.},
            "source_paths": {}, "source_sha256": {},
            "windows": [
                {"chrom": "chr1", "start": i * 1_048_576,
                 "end": (i + 1) * 1_048_576, "positive_bins": 1,
                 "mode": "ablate"} for i in range(40)
            ] + [
                {"chrom": "chr2", "start": i * 1_048_576,
                 "end": (i + 1) * 1_048_576, "positive_bins": 0,
                 "mode": "inject"} for i in range(9)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_source, relocated_source = root / "old.bin", root / "new.bin"
            original_source.write_bytes(b"unchanged checkpoint bytes")
            relocated_source.write_bytes(original_source.read_bytes())
            parent["source_paths"] = {"sae": str(original_source)}
            parent["source_sha256"] = {"sae": sha256(original_source)}
            path = root / "v1.json"
            path.write_text(json.dumps(parent))
            local = build_local_plan(path)
            local_path = root / "v2.json"
            local_path.write_text(json.dumps(local))
            checked, first, resolved = load_frozen_plan(
                local_path, 0, source_overrides={"sae": relocated_source}
            )
            self.assertEqual(checked["format"], local["format"])
            self.assertEqual(first["start"], 0)
            self.assertEqual(resolved["sae"], relocated_source)
        self.assertEqual(len(local["windows"]), 40)
        self.assertTrue(all(window["mode"] == "paired_local" for window in local["windows"]))
        self.assertEqual(local["excluded_parent_absent_windows"], 9)
        self.assertIn("NOT absent-site sufficiency", local["interpretation"])


if __name__ == "__main__":
    unittest.main()

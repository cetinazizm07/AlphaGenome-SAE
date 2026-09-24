"""Strict receipt and readout checks for each frozen cohort window."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plan_fullstream_cohort import sha256
from run_frozen_pls_ablation_cohort import HEADS, validate_output


class FrozenCohortOutputTests(unittest.TestCase):
    def test_accepts_hash_valid_paired_local_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "window"
            out.mkdir()
            audit_path = root / "audit.json"
            audit_path.write_text("frozen audit")
            entry = {
                "window_index": 1,
                "window": {"chrom": "chr1", "start": 100,
                           "end": 1_048_676, "positive_bins": 3},
                "control_feature": 1349, "target_active_bins": 2,
                "control_audit_path": str(audit_path),
                "control_audit_sha256": sha256(audit_path),
            }
            cohort = {"plan_sha256": "plan-hash", "target_feature": 199}
            support = np.zeros(8192, dtype=bool)
            support[:2] = True
            concept = np.zeros(8192, dtype=bool)
            concept[:3] = True
            arrays = {"intervention_mask": support, "concept_label": concept}
            for head in HEADS:
                baseline = np.zeros(8192, dtype=np.float32)
                arrays[f"{head}_baseline"] = baseline
                arrays[f"{head}_original"] = baseline.copy()
                arrays[f"{head}_ablate_target"] = baseline.copy()
                arrays[f"{head}_ablate_control"] = baseline.copy()
            readouts_path = out / "readouts.npz"
            np.savez_compressed(readouts_path, **arrays)
            receipt = {
                "status": "measured",
                "format": "fullstream_pls_local_ablation_window_v6",
                "plan_sha256": "plan-hash", "window_index": 1,
                "window": entry["window"], "target_feature": 199,
                "control_feature": 1349, "n_intervened_bins": 2,
                "n_concept_positive_bins": 3,
                "window_control_audit_sha256": entry["control_audit_sha256"],
                "code_counts": {
                    "target_changed_on_element": 2,
                    "control_changed_on_element": 2,
                    "target_activation_changed_on_element": 2,
                    "control_activation_changed_on_element": 2,
                },
                "tap_intervention": {
                    "target_mean_l2_per_intervened_bin": 1.0,
                    "comparator_mean_l2_per_intervened_bin": 0.95,
                    "coactive_bins": 2,
                },
                "readouts_sha256": sha256(readouts_path),
            }
            (out / "receipt.json").write_text(json.dumps(receipt))
            result = validate_output(out, cohort, entry)
            self.assertEqual(result["control_feature"], 1349)
            self.assertEqual(result["target_active_bins"], 2)

    def test_rejects_intervention_outside_element_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "window"
            out.mkdir()
            audit_path = root / "audit.json"
            audit_path.write_text("frozen audit")
            entry = {
                "window_index": 1,
                "window": {"chrom": "chr1", "start": 100,
                           "end": 1_048_676, "positive_bins": 1},
                "control_feature": 1349, "target_active_bins": 1,
                "control_audit_path": str(audit_path),
                "control_audit_sha256": sha256(audit_path),
            }
            cohort = {"plan_sha256": "plan-hash", "target_feature": 199}
            support = np.zeros(8192, dtype=bool)
            support[1] = True
            concept = np.zeros(8192, dtype=bool)
            concept[0] = True
            arrays = {"intervention_mask": support, "concept_label": concept}
            for head in HEADS:
                baseline = np.zeros(8192, dtype=np.float32)
                arrays[f"{head}_baseline"] = baseline
                arrays[f"{head}_original"] = baseline.copy()
                arrays[f"{head}_ablate_target"] = baseline.copy()
                arrays[f"{head}_ablate_control"] = baseline.copy()
            readouts_path = out / "readouts.npz"
            np.savez_compressed(readouts_path, **arrays)
            receipt = {
                "status": "measured",
                "format": "fullstream_pls_local_ablation_window_v6",
                "plan_sha256": "plan-hash", "window_index": 1,
                "window": entry["window"], "target_feature": 199,
                "control_feature": 1349, "n_intervened_bins": 1,
                "n_concept_positive_bins": 1,
                "window_control_audit_sha256": entry["control_audit_sha256"],
                "code_counts": {
                    "target_changed_on_element": 1,
                    "control_changed_on_element": 1,
                    "target_activation_changed_on_element": 1,
                    "control_activation_changed_on_element": 1,
                },
                "tap_intervention": {
                    "target_mean_l2_per_intervened_bin": 1.0,
                    "comparator_mean_l2_per_intervened_bin": 1.0,
                    "coactive_bins": 1,
                },
                "readouts_sha256": sha256(readouts_path),
            }
            (out / "receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "subset of PLS bins"):
                validate_output(out, cohort, entry)


if __name__ == "__main__":
    unittest.main()

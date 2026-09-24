"""Frozen PLS-local ablation plan, control and dose bookkeeping tests."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from plan_fullstream_cohort import sha256
from plan_fullstream_pls_ablation_v3 import build_plan
from run_fullstream_local_window import HEADS
from run_fullstream_pls_ablation_v3 import (
    checked_plan,
    intervention_norms,
    validate_control_audit,
)


class PlsAblationV3Tests(unittest.TestCase):
    def test_plan_uses_dev_control_and_preserves_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_names = ("sae", "annotations", "firing", "auroc",
                            "activation_index", "weights", "manifest", "match", "calibration")
            paths = {}
            for name in source_names:
                path = root / name
                path.write_text("fixed-" + name)
                paths[name] = path
            sha = {name: sha256(path) for name, path in paths.items()}
            parent = {
                "format": "fullstream_pls_local_paired_plan_v2",
                "fold": "fold1", "tap": "resid_pre_b8", "sae_seed": 0,
                "concept": "cCRE_PLS", "target_feature": 199,
                "control_feature": 4247,
                "source_paths": {name: str(path) for name, path in paths.items()},
                "source_sha256": sha, "primary_heads": list(HEADS[:4]),
                "off_target_heads": list(HEADS[4:]),
                "selection": "fixed 40 positive windows",
                "windows": [{"chrom": "chr1", "start": i * 1_048_576,
                             "end": (i + 1) * 1_048_576,
                             "mode": "paired_local", "positive_bins": 3}
                            for i in range(40)],
            }
            control = {
                "format": "dev_pls_local_control_diagnostic_v2_recovered",
                "target_feature": 199, "previous_control_feature": 4247,
                "selected_control_feature": 5861,
                "coordinates": {"test_label_mismatches": 0},
                "local_rate_and_strength_within_25_percent_count": 7,
                "source_sha256": {
                    "sae": sha["sae"], "test_annotations": sha["annotations"],
                    "global_firing": sha["firing"],
                    "existing_auroc": sha["auroc"],
                    "activation_index": sha["activation_index"],
                },
                "closest_local_candidates": [{
                    "feature": 5861,
                    "relative_dev_pls_rate_difference": 0.003,
                    "relative_strength_difference": 0.039,
                    "max_existing_auroc": 0.544,
                }],
            }
            parent_path, control_path = root / "parent.json", root / "control.json"
            parent_path.write_text(json.dumps(parent))
            control_path.write_text(json.dumps(control))
            plan = build_plan(parent_path, control_path)
            self.assertEqual(plan["control_feature"], 5861)
            self.assertEqual(len(plan["windows"]), 40)
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            checked, window, used, control_used = checked_plan(
                plan_path, 0, None, None, {}
            )
            self.assertEqual(checked["control_feature"], 5861)
            self.assertEqual(window["positive_bins"], 3)
            self.assertEqual(used["sae"], paths["sae"])
            self.assertEqual(control_used.resolve(), control_path.resolve())
            paths["sae"].write_text("tampered")
            with self.assertRaises(ValueError):
                checked_plan(plan_path, 0, None, None, {})

    def test_per_window_dose_records_overlap(self):
        baseline = torch.zeros(1, 4, 2)
        target = baseline.clone()
        control = baseline.clone()
        target[0, 1, 0] = 2
        control[0, 1, 1] = 3
        control[0, 2, 1] = 1
        record = intervention_norms(
            baseline, target, control, np.array([False, True, True, False])
        )
        self.assertEqual(record["target_effective_bins"], 1)
        self.assertEqual(record["comparator_effective_bins"], 2)
        self.assertEqual(record["coactive_bins"], 1)
        self.assertEqual(record["target_mean_l2_per_intervened_bin"], 1.0)

    def test_control_audit_must_match_window_support_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, control_path = root / "plan.json", root / "control.json"
            plan_path.write_text("frozen plan")
            control_path.write_text("frozen selection")
            plan = {"target_feature": 199}
            window = {"chrom": "chr1", "start": 100, "end": 1_048_676}
            selection = {"closest_local_candidates": [{
                "feature": 1349,
                "relative_dev_pls_rate_difference": 0.1,
                "relative_strength_difference": 0.1,
                "max_existing_auroc": 0.5,
            }]}
            audit = {
                "format": "outcome_blind_pls_window_control_audit_v1",
                "interpretation": "Cached activations only; no model output inspected",
                "plan_sha256": sha256(plan_path),
                "selection_sha256": sha256(control_path),
                "window_index": 0, "window": window,
                "target_feature": 199, "candidate_features": [1349],
                "coordinates_sha256": "coords", "activation_sha256": "acts",
                "chosen_feature": 1349,
                "candidates": [{
                    "feature": 1349, "target_active_bins": 2,
                    "comparator_active_on_target_bins": 2,
                    "comparator_to_target_l2_on_target_bins": 0.95,
                }],
            }
            chosen = validate_control_audit(
                audit, plan, plan_path, control_path, selection,
                0, window, "coords", "acts"
            )
            self.assertEqual(chosen, 1349)
            audit["coordinates_sha256"] = "wrong-coordinates"
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_control_audit(
                    audit, plan, plan_path, control_path, selection,
                    0, window, "coords", "acts"
                )
            audit["coordinates_sha256"] = "coords"
            audit["candidates"][0]["comparator_active_on_target_bins"] = 1
            with self.assertRaisesRegex(ValueError, "same-support"):
                validate_control_audit(
                    audit, plan, plan_path, control_path, selection,
                    0, window, "coords", "acts"
                )


if __name__ == "__main__":
    unittest.main()

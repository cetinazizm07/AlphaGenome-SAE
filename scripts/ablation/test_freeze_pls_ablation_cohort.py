"""Freeze only hash-valid, same-support matched non-pilot windows."""

import json
from pathlib import Path
import tempfile
import unittest

from freeze_pls_ablation_cohort import freeze_cohort
from plan_fullstream_cohort import sha256


class FreezeCohortTests(unittest.TestCase):
    def test_freezes_matching_windows_and_excludes_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {
                "format": "fullstream_pls_local_ablation_plan_v3",
                "windows": [
                    {"chrom": "chr1", "start": 100, "end": 1_048_676},
                    {"chrom": "chr2", "start": 200, "end": 1_048_776},
                    {"chrom": "chr3", "start": 300, "end": 1_048_876},
                ],
                "primary_heads": ["dnase"], "off_target_heads": ["atac"],
                "source_sha256": {"sae": "pinned"},
            }
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            audits = []
            for index in range(3):
                directory_path = root / f"audit{index}"
                directory_path.mkdir()
                audit_path = directory_path / "audit.json"
                audit = {
                    "format": "outcome_blind_pls_window_control_audit_v1",
                    "plan_sha256": sha256(plan_path),
                    "window_index": index, "window": plan["windows"][index],
                    "chosen_feature": 100 + index,
                    "candidates": [{
                        "feature": 100 + index, "target_active_bins": 2,
                        "comparator_active_on_target_bins": 2,
                        "comparator_to_target_l2_on_target_bins": 0.95,
                    }],
                }
                audit_path.write_text(json.dumps(audit))
                audits.append({"window_index": index,
                               "path": str(audit_path),
                               "sha256": sha256(audit_path),
                               "chosen_feature": 100 + index})
            assignments = {
                "format": "outcome_blind_pls_control_assignment_cohort_v1",
                "plan_sha256": sha256(plan_path),
                "n_planned_windows": 3,
                "matched_window_indices": [0, 1, 2],
                "audits": audits,
            }
            assignment_path = root / "assignments.json"
            assignment_path.write_text(json.dumps(assignments))
            code_files = []
            for name in ("runner.py", "edit.py", "hook.py", "audit.py", "manager.py"):
                path = root / name
                path.write_text("pinned code")
                code_files.append(path)
            output = root / "frozen.json"
            result = freeze_cohort(
                plan_path, assignment_path, output, 0,
                *code_files, expected_windows=3,
            )
            frozen = json.loads(output.read_text())
            self.assertEqual(result["n_windows"], 2)
            self.assertEqual([row["window_index"] for row in frozen["windows"]], [1, 2])
            self.assertEqual(frozen["pilot_excluded"]["window_index"], 0)

    def test_rejects_modified_audit_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"format": "fullstream_pls_local_ablation_plan_v3",
                    "windows": [{"chrom": "chr1", "start": 100,
                                 "end": 1_048_676},
                                {"chrom": "chr2", "start": 200,
                                 "end": 1_048_776}],
                    "primary_heads": [], "off_target_heads": [],
                    "source_sha256": {}}
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            audit_path = root / "audit.json"
            audit_path.write_text("changed")
            assignments = {
                "format": "outcome_blind_pls_control_assignment_cohort_v1",
                "plan_sha256": sha256(plan_path), "n_planned_windows": 2,
                "matched_window_indices": [0, 1],
                "audits": [{"window_index": 0, "path": str(audit_path),
                            "sha256": "pilot", "chosen_feature": 5},
                           {"window_index": 1, "path": str(audit_path),
                            "sha256": "wrong", "chosen_feature": 5}],
            }
            assignment_path = root / "assignments.json"
            assignment_path.write_text(json.dumps(assignments))
            code_files = []
            for name in ("runner.py", "edit.py", "hook.py", "audit.py", "manager.py"):
                path = root / name
                path.write_text("pinned")
                code_files.append(path)
            with self.assertRaisesRegex(ValueError, "hash changed"):
                freeze_cohort(plan_path, assignment_path, root / "frozen.json",
                              0, *code_files, expected_windows=2)


if __name__ == "__main__":
    unittest.main()

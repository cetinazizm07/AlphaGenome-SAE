#!/usr/bin/env python3
"""Freeze the matched, non-pilot windows before model-output inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from plan_fullstream_cohort import sha256


def freeze_cohort(plan_path: Path, assignment_path: Path, out: Path,
                  pilot_index: int, runner: Path, local_edit: Path,
                  tower_hook: Path, audit_worker: Path, manager: Path,
                  expected_windows: int = 40) -> dict:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    plan = json.loads(plan_path.read_text())
    assignment = json.loads(assignment_path.read_text())
    plan_hash = sha256(plan_path)
    if (plan.get("format") != "fullstream_pls_local_ablation_plan_v3"
            or len(plan.get("windows", [])) != expected_windows
            or assignment.get("format") != "outcome_blind_pls_control_assignment_cohort_v1"
            or assignment.get("plan_sha256") != plan_hash
            or assignment.get("n_planned_windows") != expected_windows
            or not 0 <= pilot_index < expected_windows):
        raise ValueError("Assignment manifest does not match the frozen 40-window plan")

    selected = []
    for reference in assignment["audits"]:
        index = int(reference["window_index"])
        if index == pilot_index or index not in assignment["matched_window_indices"]:
            continue
        path = Path(reference["path"])
        if sha256(path) != reference["sha256"]:
            raise ValueError(f"Control audit hash changed for window {index}")
        audit = json.loads(path.read_text())
        if (audit.get("format") != "outcome_blind_pls_window_control_audit_v1"
                or audit.get("plan_sha256") != plan_hash
                or audit.get("window_index") != index
                or audit.get("window") != plan["windows"][index]
                or audit.get("chosen_feature") != reference["chosen_feature"]):
            raise ValueError(f"Control audit provenance mismatch at window {index}")
        chosen = int(audit["chosen_feature"])
        row = next((r for r in audit["candidates"] if r["feature"] == chosen), None)
        if (row is None or row["target_active_bins"] < 1
                or row["comparator_active_on_target_bins"] != row["target_active_bins"]
                or not 0.75 <= row["comparator_to_target_l2_on_target_bins"] <= 1.25):
            raise ValueError(f"Control fails frozen matching criteria at window {index}")
        selected.append({
            "window_index": index,
            "window": plan["windows"][index],
            "control_feature": chosen,
            "target_active_bins": row["target_active_bins"],
            "control_to_target_dose_ratio": row["comparator_to_target_l2_on_target_bins"],
            "control_audit_path": str(path),
            "control_audit_sha256": reference["sha256"],
        })
    selected.sort(key=lambda row: row["window_index"])
    if not selected:
        raise ValueError("No non-pilot matched windows are available")

    code_paths = {"runner": runner, "local_edit": local_edit,
                  "tower_hook": tower_hook, "audit_worker": audit_worker,
                  "freeze_script": Path(__file__), "manager": manager}
    result = {
        "format": "residual_preserving_fullstream_pls_ablation_cohort_v1",
        "status": "frozen_before_cohort_output_inference",
        "interpretation": (
            "Exploratory matched-window cohort. The window-0 pilot is excluded. "
            "Controls are selected from held-out activation support and dose only; "
            "no cohort intervention outputs are read during assignment."
        ),
        "candidate_pool_caveat": (
            "The dev activity/strength candidate pool was also filtered by an "
            "existing max-AUROC threshold computed on test labels; this is not a "
            "fully independent confirmatory control design."
        ),
        "fold": "fold1", "split": "test", "sae_seed": 0,
        "tap": "resid_pre_b8", "target_feature": 199,
        "concept": "cCRE_PLS", "intervention": "ablation",
        "plan_path": str(plan_path), "plan_sha256": plan_hash,
        "assignment_manifest_path": str(assignment_path),
        "assignment_manifest_sha256": sha256(assignment_path),
        "pilot_excluded": {
            "window_index": pilot_index,
            "reason": "used for method and control validation before cohort inference",
        },
        "n_frozen_plan_windows": expected_windows,
        "n_matched_nonpilot_windows": len(selected),
        "matching_rule": {
            "same_support": "candidate active in every target-active PLS bin",
            "dose_ratio_inclusive": [0.75, 1.25],
            "per_window_ranking": "closest log-dose ratio, with dev rank as tie-break",
            "candidate_pool": "independent dev PLS firing/tap-L2 pool in the frozen selection artifact",
        },
        "feature_edit": (
            "At the tower-block input, x' = x + decode(c_edited, params) "
            "- decode(c, params); preserve the raw SAE residual. Set only the "
            "selected feature code to zero at the same target-active PLS bins."
        ),
        "heads_primary": plan["primary_heads"],
        "heads_off_target": plan["off_target_heads"],
        "source_sha256": plan["source_sha256"],
        "code_sha256": {name: sha256(path) for name, path in code_paths.items()},
        "windows": selected,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"out": str(out), "n_windows": len(selected),
            "window_indices": [row["window_index"] for row in selected],
            "sha256": sha256(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--pilot-index", type=int, required=True)
    parser.add_argument("--expected-windows", type=int, default=40)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--local-edit", type=Path, required=True)
    parser.add_argument("--tower-hook", type=Path, required=True)
    parser.add_argument("--audit-worker", type=Path, required=True)
    parser.add_argument("--manager", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_cohort(
        args.plan, args.assignments, args.out, args.pilot_index,
        args.runner, args.local_edit, args.tower_hook, args.audit_worker,
        args.manager, args.expected_windows,
    )))


if __name__ == "__main__":
    main()

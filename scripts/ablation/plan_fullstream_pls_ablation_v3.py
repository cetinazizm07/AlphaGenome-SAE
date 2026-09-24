#!/usr/bin/env python3
"""Freeze PLS-only feature ablation with a dev-local activity control.

This inherits the same 40 held-out PLS-positive windows from v2. The old
globally matched feature 4247 is silent at PLS sites and is replaced for this
new exploratory protocol by feature 5861, chosen before any v3 model outputs.
Injection is a separate gene-TSS experiment and is not run in this plan.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from plan_fullstream_cohort import sha256


def build_plan(parent_path: Path, control_path: Path) -> dict:
    parent = json.loads(parent_path.read_text())
    control = json.loads(control_path.read_text())
    if (parent.get("format") != "fullstream_pls_local_paired_plan_v2"
            or parent.get("fold") != "fold1" or parent.get("tap") != "resid_pre_b8"
            or parent.get("sae_seed") != 0 or parent.get("concept") != "cCRE_PLS"
            or parent.get("target_feature") != 199
            or parent.get("control_feature") != 4247
            or len(parent.get("windows", [])) != 40):
        raise ValueError("Unexpected frozen PLS-positive parent windows")
    if (control.get("format") != "dev_pls_local_control_diagnostic_v2_recovered"
            or control.get("target_feature") != 199
            or control.get("previous_control_feature") != 4247
            or control.get("selected_control_feature") != 5861
            or control.get("coordinates", {}).get("test_label_mismatches") != 0
            or control.get("local_rate_and_strength_within_25_percent_count", 0) < 1):
        raise ValueError("Dev-local control is not validated")
    source = control["source_sha256"]
    for parent_name, control_name in (("sae", "sae"),
                                      ("annotations", "test_annotations"),
                                      ("firing", "global_firing"),
                                      ("auroc", "existing_auroc"),
                                      ("activation_index", "activation_index")):
        if parent["source_sha256"][parent_name] != source[control_name]:
            raise ValueError(f"Dev control and parent differ in {parent_name}")
    selected = next((row for row in control["closest_local_candidates"]
                     if row["feature"] == 5861), None)
    if (selected is None
            or selected["relative_dev_pls_rate_difference"] > 0.25
            or selected["relative_strength_difference"] > 0.25
            or selected["max_existing_auroc"] >= 0.55):
        raise ValueError("Selected control fails frozen matching criteria")
    return {
        "format": "fullstream_pls_local_ablation_plan_v3",
        "interpretation": (
            "Exploratory full-stream PLS-bin-only SAE code ablation. Control 5861 "
            "matches dev-PLS firing and mean raw-tap L2, not global firing. "
            "Existing AUROC filter was computed on test labels; not a fully "
            "independent negative control or confirmatory causal design."
        ),
        "parent_plan_path": str(parent_path.resolve()),
        "parent_plan_sha256": sha256(parent_path),
        "control_selection_path": str(control_path.resolve()),
        "control_selection_sha256": sha256(control_path),
        "fold": "fold1", "sae_seed": 0, "tap": "resid_pre_b8",
        "concept": "cCRE_PLS", "split": "test",
        "target_feature": 199, "control_feature": 5861,
        "control_match": selected,
        "control_source_note": "Dev PLS firing and L2; max existing test AUROC < 0.55",
        "primary_heads": parent["primary_heads"],
        "off_target_heads": parent["off_target_heads"],
        "source_paths": parent["source_paths"],
        "source_sha256": parent["source_sha256"],
        "intervention_scope": "only cCRE_PLS-positive 128-bp bins",
        "baseline_scope": "SAE reconstruction on PLS bins; original tower input elsewhere",
        "selection": parent["selection"],
        "windows": parent["windows"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--control-selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    plan = build_plan(args.parent_plan, args.control_selection)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".json.tmp")
    with temporary.open("x") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(args.out)
    print(json.dumps({"plan": str(args.out), "sha256": sha256(args.out),
                      "n_windows": len(plan["windows"]),
                      "control_feature": plan["control_feature"]}))


if __name__ == "__main__":
    main()

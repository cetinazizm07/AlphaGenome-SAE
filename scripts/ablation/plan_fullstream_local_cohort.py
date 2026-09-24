#!/usr/bin/env python3
"""Freeze a local, paired PLS experiment without rewriting the global v1 plan.

The 40 PLS-positive held-out windows are inherited from the already-frozen v1
selection. Its nine PLS-absent windows cannot support a PLS-bin-only edit, so
they are excluded from this *different*, exploratory gain-of-function protocol.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from plan_fullstream_cohort import sha256


def build_local_plan(parent_path: Path) -> dict:
    parent = json.loads(parent_path.read_text())
    if (parent.get("format") != "fullstream_pls_exploratory_cohort_plan_v1"
            or parent.get("fold") != "fold1" or parent.get("sae_seed") != 0
            or parent.get("tap") != "resid_pre_b8" or parent.get("concept") != "cCRE_PLS"
            or parent.get("target_feature") != 199
            or parent.get("control_feature") != 4247):
        raise ValueError("Unexpected frozen parent plan")
    present = [dict(window, parent_window_index=index, mode="paired_local")
               for index, window in enumerate(parent["windows"])
               if window["mode"] == "ablate"]
    absent = [window for window in parent["windows"] if window["mode"] == "inject"]
    if (len(present), len(absent)) != (40, 9) or any(
        window["positive_bins"] <= 0 for window in present
    ) or any(window["positive_bins"] != 0 for window in absent):
        raise ValueError("Frozen parent window counts or labels changed")
    if len({(window["chrom"], window["start"]) for window in present}) != 40:
        raise ValueError("Duplicate PLS-positive window")
    return {
        "format": "fullstream_pls_local_paired_plan_v2",
        "interpretation": (
            "Exploratory local ablation and local gain-of-function at existing PLS bins; "
            "injection here is NOT absent-site sufficiency. Do not merge with global v1."
        ),
        "parent_plan_path": str(parent_path.resolve()),
        "parent_plan_sha256": sha256(parent_path),
        "fold": parent["fold"], "sae_seed": parent["sae_seed"],
        "tap": parent["tap"], "concept": parent["concept"],
        "split": parent["split"],
        "target_feature": parent["target_feature"],
        "control_feature": parent["control_feature"],
        "match_sign": parent["match_sign"],
        "primary_heads": parent["primary_heads"],
        "off_target_heads": parent["off_target_heads"],
        "calibration_split": parent["calibration_split"],
        "calibration_levels": parent["calibration_levels"],
        "source_paths": parent["source_paths"],
        "source_sha256": parent["source_sha256"],
        "selection": "the same 40 preselected PLS-positive held-out windows in parent v1",
        "intervention_scope": "only cCRE_PLS-positive 128-bp bins",
        "baseline_scope": "SAE reconstruction on PLS bins; original tower input elsewhere",
        "injection_rule": "max(existing SAE code, dev positive-code p95) at PLS bins",
        "excluded_parent_absent_windows": len(absent),
        "windows": present,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    plan = build_local_plan(args.parent_plan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".json.tmp")
    with temporary.open("x") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(args.out)
    print(json.dumps({"plan": str(args.out), "sha256": sha256(args.out),
                      "n_windows": len(plan["windows"])}))


if __name__ == "__main__":
    main()

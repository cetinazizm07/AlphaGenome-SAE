#!/usr/bin/env python3
"""Freeze an exploratory fold1/b8/seed0 PLS window list before GPU inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def build_plan(paths: dict[str, Path]) -> dict:
    manifest = pd.read_parquet(paths["manifest"])
    annotations = pd.read_parquet(paths["annotations"])
    matches = pd.read_csv(paths["match"])
    selected = matches.loc[matches.concept == "cCRE_PLS"]
    if len(selected) != 1 or not bool(selected.iloc[0].recovered_v3):
        raise ValueError("No unique recovered_v3 PLS match")
    feature = int(str(selected.iloc[0].best_feature).removeprefix("sae_"))
    if feature != 199 or int(selected.iloc[0].match_sign) != 1:
        raise ValueError("The exploratory target feature/sign changed")
    firing = np.load(paths["firing"])
    auroc = np.load(paths["auroc"])
    if not 0 < float(firing[feature]) < 0.5:
        raise ValueError("The selected feature is no longer sparse")
    maximum = np.nanmax(auroc, axis=1)
    eligible = (np.isfinite(firing) & (firing > 0)
                & (np.abs(firing - firing[feature]) / firing[feature] <= 0.25)
                & np.isfinite(maximum) & (maximum < 0.55))
    eligible[feature] = False
    candidates = np.flatnonzero(eligible)
    control = int(np.random.default_rng(1234 + feature).choice(candidates))
    if control != 4247:
        raise ValueError("The deterministic control feature changed")
    calibration = json.loads(paths["calibration"].read_text())
    if (calibration["format"] != "sae_injection_dev_positive_p95_v1"
            or calibration["split"] != "dev" or calibration["tap"] != "resid_pre_b8"
            or calibration["target_feature"] != feature
            or calibration["control_feature"] != control
            or calibration["sha256"]["sae"] != sha256(paths["sae"])
            or calibration["sha256"]["activation_index"] != sha256(paths["activation_index"])):
        raise ValueError("Independent dev calibration does not match this frozen source")
    present, absent, excluded = [], [], []
    held = manifest.loc[manifest.split == "test"].sort_values(["chrom", "win_start"])
    for row in held.itertuples():
        chrom, start, end = str(row.chrom), int(row.win_start), int(row.win_end)
        bins = annotations.loc[
            (annotations.split == "test") & (annotations.chrom == chrom)
            & (annotations.bin_start >= start) & (annotations.bin_start < end)
        ].sort_values("bin_start")
        expected = start + np.arange(8192, dtype=np.int64) * 128
        if end - start != 1_048_576 or not np.array_equal(
            bins.bin_start.to_numpy(dtype=np.int64), expected
        ):
            excluded.append({"chrom": chrom, "start": start,
                             "observed_bins": int(len(bins)), "reason": "incomplete_128bp_grid"})
            continue
        valid = bins.n_mask.to_numpy(dtype=bool)
        if not valid.all():
            raise ValueError(f"Unexpected masked bin in complete-grid window {chrom}:{start}")
        positives = int((valid & bins.cCRE_PLS.to_numpy(dtype=bool)).sum())
        entry = {"chrom": chrom, "start": start, "end": end,
                 "valid_bins": int(valid.sum()), "positive_bins": positives,
                 "mode": "ablate" if positives else "inject"}
        (present if positives else absent).append(entry)
    if (len(held), len(present), len(absent), len(excluded)) != (200, 188, 9, 3):
        raise ValueError("Held-out eligibility counts changed")
    rng = np.random.default_rng(0)
    present_indices = sorted(rng.choice(len(present), 40, replace=False).tolist())
    windows = [present[index] for index in present_indices] + absent
    windows.sort(key=lambda item: (item["mode"], item["chrom"], item["start"]))
    if len({(w["chrom"], w["start"]) for w in windows}) != 49:
        raise ValueError("Duplicate selected window")
    hashes = {key: sha256(path) for key, path in paths.items()}
    return {
        "format": "fullstream_pls_exploratory_cohort_plan_v1",
        "interpretation": "Post-pilot exploratory b8 feature choice; not a preregistered confirmatory result.",
        "fold": "fold1", "sae_seed": 0, "tap": "resid_pre_b8", "concept": "cCRE_PLS",
        "split": "test", "target_feature": feature, "control_feature": control,
        "match_sign": 1, "calibration_split": "dev",
        "primary_heads": ["dnase", "atac", "cage", "rna_seq"],
        "off_target_heads": ["procap", "chip_tf", "chip_histone"],
        "selection": "all 9 full-grid PLS-absent windows; 40 of 188 full-grid PLS-present windows via numpy default_rng(0).choice without replacement after chrom/start sorting",
        "counts": {"manifest_test": len(held), "present_eligible": len(present),
                   "absent_eligible": len(absent), "excluded": len(excluded),
                   "present_selected": 40, "absent_selected": len(absent)},
        "excluded": excluded, "windows": windows,
        "source_paths": {key: str(path) for key, path in paths.items()},
        "source_sha256": hashes,
        "calibration_levels": {
            "target": float(calibration["features"][str(feature)]["positive_p95"]),
            "control": float(calibration["features"][str(control)]["positive_p95"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "annotations", "weights", "sae", "match", "firing",
                 "auroc", "calibration", "activation_index"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    paths = {name: getattr(args, name) for name in (
        "manifest", "annotations", "weights", "sae", "match", "firing",
        "auroc", "calibration", "activation_index")}
    plan = build_plan(paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(".json.tmp")
    with temporary.open("x") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(args.out)
    print(json.dumps({"plan": str(args.out), "sha256": sha256(args.out),
                      "counts": plan["counts"], "target": plan["target_feature"],
                      "control": plan["control_feature"]}))


if __name__ == "__main__":
    main()

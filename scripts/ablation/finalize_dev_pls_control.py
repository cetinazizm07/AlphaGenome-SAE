#!/usr/bin/env python3
"""Recover a complete dev-local control receipt from finished GPU counts.

The earlier measurement completed its activation reads and writes, then failed
only while JSON-serializing a NumPy scalar. This finalizer never recomputes SAE
codes and never overwrites the immutable count file or failed draft.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np

from measure_dev_pls_control import (
    choose_control,
    choose_local_control,
    dev_coordinates_and_labels,
    sha256,
)


def finalize(partial: Path, acts: Path, sae_path: Path, ccre: Path,
             test_annotations: Path, firing_path: Path,
             auroc_path: Path) -> dict:
    if (not partial.is_dir() or (partial / "selection.json").exists()
            or not (partial / "selection.json.tmp").exists()):
        raise ValueError("Expected incomplete measurement with no final selection")
    counts_path = partial / "dev_pls_firing_counts.npz"
    with np.load(counts_path, allow_pickle=False) as arrays:
        counts = arrays["fired"].copy()
        strengths = arrays["tap_l2_sum"].copy()
        n_dev_pls = int(arrays["n_dev_pls"])
    if (counts.shape != (8192,) or strengths.shape != counts.shape
            or n_dev_pls <= 0 or (counts < 0).any() or (counts > n_dev_pls).any()
            or not np.isfinite(strengths).all() or (strengths < 0).any()):
        raise ValueError("Stored GPU count or strength arrays are malformed")
    index_path = acts / "index.json"
    index = json.loads(index_path.read_text())
    labels, items, coordinate_stats = dev_coordinates_and_labels(
        acts, index, ccre, test_annotations
    )
    if int(labels.sum()) != n_dev_pls:
        raise ValueError("Recreated dev PLS labels differ from stored GPU count")
    verified_activations = {}
    for item, _left, _right in items:
        path = acts / item["activations"]
        observed = sha256(path)
        if observed != item["activations_sha256"]:
            raise ValueError(f"Activation bytes differ from frozen index: {path}")
        verified_activations[item["activations"]] = observed
    firing = np.load(firing_path, allow_pickle=False)
    auroc = np.load(auroc_path, allow_pickle=False)
    old_pool_choice, old_pool = choose_control(199, counts, n_dev_pls, firing, auroc)
    selected, local_pool = choose_local_control(199, counts, strengths,
                                                n_dev_pls, auroc)
    from ag_sae import concepts as concepts_module, sae as sae_module

    result = {
        "format": "dev_pls_local_control_diagnostic_v2_recovered",
        "interpretation": (
            "Exploratory dev-PLS control selection. Local firing and exact raw-tap "
            "perturbation strength are matched within 25%; max existing test "
            "concept AUROC < 0.55. Global firing is descriptive, not a local "
            "eligibility condition. No model prediction outcomes were used."
        ),
        "recovery": {
            "reason": "GPU measurement completed; JSON serialization of np.float64 failed",
            "old_worker_sha256": sha256(partial / "measure_dev_pls_control.py"),
            "failed_draft_sha256": sha256(partial / "selection.json.tmp"),
            "finalizer_sha256": sha256(Path(__file__)),
        },
        "target_feature": 199, "previous_control_feature": 4247,
        "selected_control_feature": selected,
        "old_global_pool_locally_matched_feature": old_pool_choice,
        "target_dev_pls_fired": int(counts[199]),
        "target_dev_pls_rate": float(counts[199] / n_dev_pls),
        "target_mean_tap_l2_per_pls_bin": float(strengths[199] / n_dev_pls),
        "previous_control_dev_pls_fired": int(counts[4247]),
        "previous_control_dev_pls_rate": float(counts[4247] / n_dev_pls),
        "previous_control_mean_tap_l2_per_pls_bin": float(strengths[4247] / n_dev_pls),
        "selected_control_dev_pls_fired": int(counts[selected]) if selected is not None else None,
        "selected_control_dev_pls_rate": float(counts[selected] / n_dev_pls)
        if selected is not None else None,
        "selected_control_mean_tap_l2_per_pls_bin": float(strengths[selected] / n_dev_pls)
        if selected is not None else None,
        "old_global_pool_count": len(old_pool),
        "local_eligible_count": len(local_pool),
        "local_rate_and_strength_within_25_percent_count": sum(
            row["relative_dev_pls_rate_difference"] <= 0.25 + 1e-12
            and row["relative_strength_difference"] <= 0.25 + 1e-12
            for row in local_pool
        ),
        "closest_local_candidates": local_pool[:20],
        "coordinates": coordinate_stats,
        "source_sha256": {
            "activation_index": sha256(index_path), "ccre_bed": sha256(ccre),
            "sae": sha256(sae_path), "test_annotations": sha256(test_annotations),
            "global_firing": sha256(firing_path), "existing_auroc": sha256(auroc_path),
            "labeling_code": sha256(Path(concepts_module.__file__)),
            "sae_code": sha256(Path(sae_module.__file__)),
        },
        "activation_sha256": verified_activations,
        "counts_sha256": sha256(counts_path),
    }
    shutil.copy2(__file__, partial / Path(__file__).name)
    final = partial / "selection.json"
    with final.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"out": str(partial), "selected": selected,
            "local_matches": result["local_rate_and_strength_within_25_percent_count"],
            "selection_sha256": sha256(final)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--acts", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--ccre", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--global-firing", type=Path, required=True)
    parser.add_argument("--existing-auroc", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.partial, args.acts, args.sae, args.ccre,
                              args.test_annotations, args.global_firing,
                              args.existing_auroc)))


if __name__ == "__main__":
    main()

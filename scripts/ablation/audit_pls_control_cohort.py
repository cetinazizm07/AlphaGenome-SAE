#!/usr/bin/env python3
"""Freeze per-window PLS controls using held-out activations only.

The plan, model outputs, and concept labels are never used to choose among
predictions. The only window-level inputs to control assignment are the cached
tower activations, the PLS-bin mask, and the dev-frozen candidate pool.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from audit_pls_window_controls import (
    activation_rows_for_window,
    eligible_dev_candidates,
    rank_window_candidates,
)
from plan_fullstream_cohort import sha256
from run_fullstream_pls_ablation_v3 import checked_plan


def collect_windows(plan: dict, paths: dict[str, Path],
                    annotations: pd.DataFrame,
                    manifest: pd.DataFrame) -> list[dict]:
    """Validate every frozen window against held-out manifest and label grid."""
    windows = []
    for index, window in enumerate(plan["windows"]):
        chrom, start, end = window["chrom"], int(window["start"]), int(window["end"])
        held = manifest.loc[(manifest.split == "test") & (manifest.chrom == chrom)
                            & (manifest.win_start == start)]
        if len(held) != 1 or int(held.iloc[0].win_end) != end:
            raise ValueError(f"Frozen window {index} is not uniquely held out")
        bins = annotations.loc[
            (annotations.split == "test") & (annotations.chrom == chrom)
            & (annotations.bin_start >= start) & (annotations.bin_start < end)
        ].sort_values("bin_start").reset_index(drop=True)
        expected = start + 128 * np.arange(8192, dtype=np.int64)
        if (not np.array_equal(bins.bin_start.to_numpy(dtype=np.int64), expected)
                or not bins.n_mask.to_numpy(dtype=bool).all()
                or bins.cCRE_PLS.isna().any()
                or not bins.cCRE_PLS.isin([False, True, 0, 1]).all()):
            raise ValueError(f"Frozen window {index} lacks a complete PLS grid")
        mask = bins.cCRE_PLS.to_numpy(dtype=bool)
        if int(mask.sum()) != int(window["positive_bins"]) or not mask.any():
            raise ValueError(f"Frozen PLS-bin count changed in window {index}")
        windows.append({"index": index, "window": window, "mask": mask,
                        "expected_bin_starts": expected})
    return windows


def index_window_coordinate_shards(windows: list[dict], activation_index: dict,
                                   acts: Path) -> dict[int, dict]:
    """Read each test coordinate shard once and resolve all plan windows."""
    wanted = {(entry["window"]["chrom"], int(entry["window"]["start"])): entry["index"]
              for entry in windows}
    resolved = {}
    for shard in activation_index["shards"]:
        if shard["split"] != "test" or "resid_pre_b8" not in shard["taps"]:
            continue
        item = shard["taps"]["resid_pre_b8"]
        coord_path = acts / item["coordinates"]
        if sha256(coord_path) != item["coordinates_sha256"]:
            raise ValueError(f"Coordinate shard hash changed: {coord_path}")
        full_coordinates = pd.read_parquet(coord_path)
        for key, group in full_coordinates.groupby(["chrom", "window_start"], sort=False):
            window_index = wanted.get((str(key[0]), int(key[1])))
            if window_index is None:
                continue
            if window_index in resolved:
                raise ValueError(f"Window {window_index} occurs in multiple activation shards")
            resolved[window_index] = {
                "tap_item": item,
                "coordinate_path": coord_path,
                "coordinate_sha256": item["coordinates_sha256"],
                "full_coordinates": full_coordinates,
                "window_coordinates": group.sort_values("bin_start").reset_index(drop=True),
            }
    missing = sorted(set(range(len(windows))) - set(resolved))
    if missing:
        raise ValueError(f"No held-out activation coordinates for windows {missing}")
    return resolved


@torch.inference_mode()
def audit_cohort(plan_path: Path, acts: Path, out: Path,
                 batch_size: int = 256) -> dict:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    plan, _, paths, selection_path = checked_plan(plan_path, 0, None, None, {})
    selection = json.loads(selection_path.read_text())
    candidates = eligible_dev_candidates(selection)
    manifest = pd.read_parquet(paths["manifest"])
    annotations = pd.read_parquet(paths["annotations"])
    windows = collect_windows(plan, paths, annotations, manifest)
    activation_index = json.loads(paths["activation_index"].read_text())
    shards = index_window_coordinate_shards(windows, activation_index, acts)

    activation_cache: dict[Path, np.ndarray] = {}
    activation_hashes: dict[Path, str] = {}
    raw_parts = []
    windows_with_slices = []
    cursor = 0
    for entry in windows:
        index = entry["index"]
        window = entry["window"]
        shard = shards[index]
        item = shard["tap_item"]
        coord_path = shard["coordinate_path"]
        expected = entry["expected_bin_starts"]
        coordinates = shard["window_coordinates"]
        rows = activation_rows_for_window(
            shard["full_coordinates"], coordinates,
            window["chrom"], int(window["start"]),
        )
        if (len(coordinates) != 8192
                or not (coordinates.split == "test").all()
                or not np.array_equal(coordinates.bin_start.to_numpy(dtype=np.int64), expected)):
            raise ValueError(f"Activation and annotation coordinates disagree at window {index}")
        activation_path = acts / item["activations"]
        if activation_path not in activation_cache:
            observed_sha = sha256(activation_path)
            if observed_sha != item["activations_sha256"]:
                raise ValueError(f"Activation shard hash changed: {activation_path}")
            values = np.load(activation_path, mmap_mode="r", allow_pickle=False)
            if (values.shape != (int(item["rows"]), int(item["channels"]))
                    or values.shape[1] != 1536):
                raise ValueError(f"Activation shard shape changed: {activation_path}")
            activation_cache[activation_path] = values
            activation_hashes[activation_path] = observed_sha
        selected_rows = rows[entry["mask"]]
        raw_parts.append(np.asarray(activation_cache[activation_path][selected_rows],
                                    dtype=np.float32))
        stop = cursor + len(selected_rows)
        windows_with_slices.append({
            **entry, "rows": rows, "coordinate_sha256": shard["coordinate_sha256"],
            "activation_shard_rows": len(shard["full_coordinates"]),
            "activation_path": activation_path,
            "activation_sha256": activation_hashes[activation_path],
            "slice": slice(cursor, stop),
        })
        cursor = stop

    from ag_sae.sae import BorzoiSAE

    sae = BorzoiSAE.from_checkpoint(paths["sae"], device="cuda")
    direction_norm = torch.linalg.vector_norm(
        sae.core.decoder.weight.detach().float()
        * sae.channel_scale.float()[:, None], dim=0
    )
    raw_all = np.concatenate(raw_parts, axis=0)
    feature_doses = []
    scale = sae.channel_scale.float()
    for offset in range(0, len(raw_all), batch_size):
        raw = torch.as_tensor(raw_all[offset:offset + batch_size], device="cuda")
        pre, params = sae.core.encode(raw / scale)
        codes = sae.core.get_sparse_activations(sae.core.activation(pre))
        doses = (codes.clamp_min(0).float() * params["std"].float()
                 * direction_norm[None, :])
        feature_doses.append(doses[:, [199, *candidates]].cpu().numpy())
    all_doses = np.concatenate(feature_doses, axis=0)

    worker_sha = sha256(Path(__file__))
    audits = []
    for entry in windows_with_slices:
        window = entry["window"]
        doses = all_doses[entry["slice"]]
        target_l2 = doses[:, 0]
        candidate_l2 = {feature: doses[:, i + 1]
                        for i, feature in enumerate(candidates)}
        chosen, rows = rank_window_candidates(target_l2, candidate_l2, candidates)
        audits.append({
            "format": "outcome_blind_pls_window_control_audit_v1",
            "interpretation": "Cached test activations only; no model output inspected or altered",
            "plan_sha256": sha256(plan_path),
            "selection_sha256": sha256(selection_path),
            "window_index": entry["index"], "window": window,
            "target_feature": 199, "candidate_features": candidates,
            "chosen_feature": chosen,
            "target_fired_on_pls": int(np.count_nonzero(target_l2 > 1e-8)),
            "target_l2_sum_on_pls": float(target_l2.sum()),
            "activation_shard_rows": entry["activation_shard_rows"],
            "activation_window_rows": len(entry["rows"]),
            "activation_window_row_min": int(entry["rows"].min()),
            "activation_window_row_max": int(entry["rows"].max()),
            "candidates": rows,
            "activation_sha256": entry["activation_sha256"],
            "coordinates_sha256": entry["coordinate_sha256"],
            "worker_sha256": worker_sha,
        })

    out.mkdir(parents=True, exist_ok=False)
    audit_paths = []
    for audit in audits:
        window = audit["window"]
        name = f"{audit['window_index']:02d}_{window['chrom']}_{window['start']}"
        directory = out / name
        directory.mkdir()
        path = directory / "audit.json"
        with path.open("x") as stream:
            json.dump(audit, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        audit_paths.append({"window_index": audit["window_index"],
                            "path": str(path), "sha256": sha256(path),
                            "chosen_feature": audit["chosen_feature"]})
    matched = [row for row in audits if row["chosen_feature"] is not None]
    manifest_out = {
        "format": "outcome_blind_pls_control_assignment_cohort_v1",
        "interpretation": "Feature controls chosen from held-out activations before reading any intervention outputs",
        "plan_sha256": sha256(plan_path),
        "selection_sha256": sha256(selection_path),
        "worker_sha256": worker_sha,
        "n_planned_windows": len(audits),
        "n_target_active_windows": sum(row["target_fired_on_pls"] > 0 for row in audits),
        "n_same_support_dose_matched_windows": len(matched),
        "n_unmatched_windows": len(audits) - len(matched),
        "matched_window_indices": [row["window_index"] for row in matched],
        "unmatched_window_indices": [row["window_index"] for row in audits
                                     if row["chosen_feature"] is None],
        "audits": audit_paths,
        "source_sha256": plan["source_sha256"],
    }
    manifest_path = out / "assignment_manifest.json"
    with manifest_path.open("x") as stream:
        json.dump(manifest_out, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    shutil.copy2(__file__, out / Path(__file__).name)
    return {"out": str(out), "planned": len(audits),
            "target_active": manifest_out["n_target_active_windows"],
            "matched": len(matched), "unmatched": len(audits) - len(matched),
            "manifest_sha256": sha256(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--acts-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(audit_cohort(args.plan, args.acts_dir, args.out, args.batch_size)))


if __name__ == "__main__":
    main()

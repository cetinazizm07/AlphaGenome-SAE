#!/usr/bin/env python3
"""Measure SAE firing on independently labeled dev PLS bins.

The original globally matched control need not fire where the PLS element is.
This diagnostic relabels the existing dev activation coordinates using the
same cCRE BED and overlap rule as the test annotation, then selects a neutral
control by dev-PLS firing similarity. No model prediction outcome is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_control(target: int, dev_pls_counts: np.ndarray, n_dev_pls: int,
                   global_firing: np.ndarray, auroc: np.ndarray,
                   tolerance: float = 0.25) -> tuple[int | None, list[dict]]:
    """Reuse the low-AUROC global pool, then match firing *inside dev PLS*."""
    if (dev_pls_counts.ndim != 1 or n_dev_pls < 1
            or global_firing.shape != dev_pls_counts.shape
            or auroc.ndim != 2 or auroc.shape[0] != len(dev_pls_counts)
            or not 0 <= target < len(dev_pls_counts)
            or dev_pls_counts[target] <= 0 or global_firing[target] <= 0
            or not 0 < tolerance < 1):
        raise ValueError("Invalid feature arrays or zero target dev-PLS activity")
    q = dev_pls_counts.astype(np.float64) / n_dev_pls
    max_auroc = np.nanmax(auroc, axis=1)
    global_pool = (
        np.isfinite(global_firing) & (global_firing > 0)
        & (np.abs(global_firing - global_firing[target]) / global_firing[target] <= 0.25)
        & np.isfinite(max_auroc) & (max_auroc < 0.55)
    )
    global_pool[target] = False
    pool = np.flatnonzero(global_pool)
    rows = [{
        "feature": int(feature), "dev_pls_fired": int(dev_pls_counts[feature]),
        "dev_pls_rate": float(q[feature]),
        "relative_dev_pls_rate_difference": float(abs(q[feature] - q[target]) / q[target]),
        "global_firing_rate": float(global_firing[feature]),
        "max_existing_auroc": float(max_auroc[feature]),
    } for feature in pool]
    rows.sort(key=lambda row: (row["relative_dev_pls_rate_difference"],
                               abs(row["global_firing_rate"] - global_firing[target]),
                               row["feature"]))
    within = [row for row in rows if row["relative_dev_pls_rate_difference"] <= tolerance + 1e-12
              and row["dev_pls_fired"] > 0]
    return (within[0]["feature"] if within else None), rows


def choose_local_control(target: int, counts: np.ndarray,
                         strength_sum: np.ndarray, n_dev_pls: int,
                         auroc: np.ndarray, tolerance: float = 0.25
                         ) -> tuple[int | None, list[dict]]:
    """Match both PLS-bin firing and per-PLS-bin tap L2, without global q.

    `strength_sum` accumulates each code times its exact raw decoder-direction
    norm, including token-specific SAE standard deviation and channel scaling.
    Only the previously computed low-AUROC filter remains from test analyses.
    """
    if (counts.ndim != 1 or strength_sum.shape != counts.shape
            or auroc.ndim != 2 or auroc.shape[0] != len(counts)
            or not 0 <= target < len(counts) or n_dev_pls < 1
            or counts[target] <= 0 or strength_sum[target] <= 0
            or not 0 < tolerance < 1):
        raise ValueError("Invalid PLS activity or target strength")
    max_auroc = np.nanmax(auroc, axis=1)
    eligible = np.flatnonzero((counts > 0) & np.isfinite(max_auroc) & (max_auroc < 0.55))
    rows = []
    for feature in eligible:
        if feature == target:
            continue
        rate_difference = float(
            abs(float(counts[feature] - counts[target])) / counts[target]
        )
        strength_difference = float(
            abs(float(strength_sum[feature] - strength_sum[target])) / strength_sum[target]
        )
        rows.append({
            "feature": int(feature), "dev_pls_fired": int(counts[feature]),
            "dev_pls_rate": float(counts[feature] / n_dev_pls),
            "mean_tap_l2_per_pls_bin": float(strength_sum[feature] / n_dev_pls),
            "relative_dev_pls_rate_difference": rate_difference,
            "relative_strength_difference": strength_difference,
            "max_existing_auroc": float(max_auroc[feature]),
        })
    rows.sort(key=lambda row: (max(row["relative_dev_pls_rate_difference"],
                                  row["relative_strength_difference"]),
                               row["relative_dev_pls_rate_difference"]
                               + row["relative_strength_difference"], row["feature"]))
    matched = [row for row in rows
               if row["relative_dev_pls_rate_difference"] <= tolerance + 1e-12
               and row["relative_strength_difference"] <= tolerance + 1e-12]
    return (matched[0]["feature"] if matched else None), rows


def dev_coordinates_and_labels(acts: Path, index: dict, ccre_path: Path,
                               test_annotations: Path):
    """Validate the old test labels and label only existing dev activation rows."""
    from ag_sae.concepts import label_bins, read_ccre_bed

    bed = read_ccre_bed(ccre_path, class_column=9)
    if "PLS" not in bed:
        raise ValueError("BED lacks the same PLS category as the test annotation")
    test = pd.read_parquet(test_annotations,
                           columns=["chrom", "bin_start", "bin_end", "cCRE_PLS"])
    reproduced = label_bins(test, bed["PLS"])
    mismatch = int(np.count_nonzero(reproduced != test.cCRE_PLS.to_numpy(dtype=bool)))
    if mismatch:
        raise ValueError(f"BED/overlap rule differs at {mismatch} existing test bins")
    frames = []
    items = []
    cursor = 0
    for row in index["shards"]:
        if row["split"] != "dev":
            continue
        item = row["taps"]["resid_pre_b8"]
        coord_path = acts / item["coordinates"]
        if sha256(coord_path) != item["coordinates_sha256"]:
            raise ValueError(f"Dev coordinate hash changed: {coord_path}")
        coords = pd.read_parquet(coord_path)
        if (len(coords) != item["rows"] or not (coords.split == "dev").all()
                or not ((coords.bin_end - coords.bin_start) == 128).all()):
            raise ValueError(f"Malformed dev 128-bp coordinate shard: {coord_path}")
        frames.append(coords[["chrom", "bin_start", "bin_end"]])
        items.append((item, cursor, cursor + len(coords)))
        cursor += len(coords)
    if not items:
        raise ValueError("No dev resid_pre_b8 activations")
    coordinates = pd.concat(frames, ignore_index=True)
    if coordinates.duplicated(["chrom", "bin_start"]).any():
        raise ValueError("Duplicate dev genomic bins")
    labels = label_bins(coordinates, bed["PLS"])
    return labels, items, {"test_rows_checked": len(test), "test_label_mismatches": 0,
                           "dev_rows": len(coordinates), "dev_pls_rows": int(labels.sum()),
                           "dev_shards": len(items)}


def measure(acts: Path, sae_path: Path, ccre: Path, test_annotations: Path,
            firing_path: Path, auroc_path: Path, out: Path,
            device: str, batch_size: int) -> dict:
    """Encode only dev PLS rows; keep all source hashes and full counts."""
    if out.exists() or batch_size < 1:
        raise ValueError("Output must be new and batch size positive")
    from ag_sae import concepts as concepts_module, sae as sae_module
    from ag_sae.sae import BorzoiSAE

    index_path = acts / "index.json"
    index = json.loads(index_path.read_text())
    labels, items, coordinate_stats = dev_coordinates_and_labels(
        acts, index, ccre, test_annotations
    )
    if not labels.any():
        raise ValueError("No dev PLS-positive bins")
    sae = BorzoiSAE.from_checkpoint(sae_path, device=device)
    sae.eval()
    scale = sae.channel_scale.float().to(device)
    counts = np.zeros(sae.hidden, dtype=np.int64)
    strength_sum = np.zeros(sae.hidden, dtype=np.float64)
    raw_direction_norm = torch.linalg.vector_norm(
        sae.core.decoder.weight.detach().float() * scale[:, None], dim=0
    )
    activation_hashes = {}
    for item, left, right in items:
        activation_path = acts / item["activations"]
        activation_hash = sha256(activation_path)
        if activation_hash != item["activations_sha256"]:
            raise ValueError(f"Dev activation hash changed: {activation_path}")
        activation_hashes[item["activations"]] = activation_hash
        selected = np.flatnonzero(labels[left:right])
        if not len(selected):
            continue
        values = np.load(activation_path, mmap_mode="r", allow_pickle=False)
        if values.shape != (item["rows"], sae.d_in):
            raise ValueError(f"Dev activation shape changed: {activation_path}")
        for offset in range(0, len(selected), batch_size):
            rows = selected[offset:offset + batch_size]
            raw = torch.as_tensor(np.array(values[rows], dtype=np.float32), device=device)
            with torch.inference_mode():
                pre, params = sae.core.encode(raw / scale)
                code = sae.core.get_sparse_activations(sae.core.activation(pre))
            counts += (code > 0).sum(dim=0).cpu().numpy().astype(np.int64)
            token_std = params["std"].float()
            strength_sum += (code.clamp_min(0).float() * token_std
                             * raw_direction_norm[None, :]).sum(dim=0).cpu().numpy()
    if (counts < 0).any() or (counts > coordinate_stats["dev_pls_rows"]).any():
        raise ValueError("Impossible feature firing count")
    firing = np.load(firing_path, allow_pickle=False)
    auroc = np.load(auroc_path, allow_pickle=False)
    chosen, ranked = choose_control(199, counts, coordinate_stats["dev_pls_rows"],
                                    firing, auroc)
    local_chosen, local_ranked = choose_local_control(
        199, counts, strength_sum, coordinate_stats["dev_pls_rows"], auroc
    )
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    count_path = out / "dev_pls_firing_counts.npz"
    np.savez_compressed(count_path, fired=counts, tap_l2_sum=strength_sum,
                        n_dev_pls=np.asarray(coordinate_stats["dev_pls_rows"]))
    result = {
        "format": "dev_pls_local_control_diagnostic_v1",
        "interpretation": "Exploratory control selection; existing global eligibility uses test AUROC/firing, local firing uses dev labels only; no model prediction outcomes used",
        "target_feature": 199, "previous_control_feature": 4247,
        "selected_control_feature": local_chosen,
        "old_global_pool_locally_matched_feature": chosen,
        "target_dev_pls_fired": int(counts[199]),
        "target_dev_pls_rate": float(counts[199] / coordinate_stats["dev_pls_rows"]),
        "target_mean_tap_l2_per_pls_bin": float(
            strength_sum[199] / coordinate_stats["dev_pls_rows"]),
        "previous_control_dev_pls_fired": int(counts[4247]),
        "previous_control_dev_pls_rate": float(counts[4247] / coordinate_stats["dev_pls_rows"]),
        "previous_control_mean_tap_l2_per_pls_bin": float(
            strength_sum[4247] / coordinate_stats["dev_pls_rows"]),
        "global_eligible_count": len(ranked),
        "local_within_25_percent_count": sum(
            row["relative_dev_pls_rate_difference"] <= 0.25 + 1e-12 for row in ranked
        ),
        "closest_candidates": ranked[:20],
        "local_eligible_count": len(local_ranked),
        "local_rate_and_strength_within_25_percent_count": sum(
            row["relative_dev_pls_rate_difference"] <= 0.25 + 1e-12
            and row["relative_strength_difference"] <= 0.25 + 1e-12
            for row in local_ranked
        ),
        "closest_local_candidates": local_ranked[:20],
        "coordinates": coordinate_stats,
        "source_sha256": {
            "activation_index": sha256(index_path), "ccre_bed": sha256(ccre),
            "sae": sha256(sae_path), "test_annotations": sha256(test_annotations),
            "global_firing": sha256(firing_path), "existing_auroc": sha256(auroc_path),
            "labeling_code": sha256(Path(concepts_module.__file__)),
            "sae_code": sha256(Path(sae_module.__file__)),
        },
        "worker_sha256": sha256(out / Path(__file__).name),
        "activation_sha256": activation_hashes,
        "counts_sha256": sha256(count_path),
    }
    temporary = out / "selection.json.tmp"
    with temporary.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(out / "selection.json")
    return {"out": str(out), "selected": local_chosen,
            "target_dev_pls_rate": result["target_dev_pls_rate"],
            "previous_control_dev_pls_rate": result["previous_control_dev_pls_rate"],
            "local_matches": result["local_rate_and_strength_within_25_percent_count"],
            "selection_sha256": sha256(out / "selection.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acts", type=Path, required=True)
    parser.add_argument("--sae", type=Path, required=True)
    parser.add_argument("--ccre", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--global-firing", type=Path, required=True)
    parser.add_argument("--existing-auroc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(measure(args.acts, args.sae, args.ccre,
                             args.test_annotations, args.global_firing,
                             args.existing_auroc, args.out, args.device,
                             args.batch_size)))


if __name__ == "__main__":
    main()

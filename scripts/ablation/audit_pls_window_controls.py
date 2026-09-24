#!/usr/bin/env python3
"""Outcome-blind PLS-window control audit from cached tower activations.

The seven candidates were selected on independent dev PLS bins. This audit
uses a held-out window's *input activation only*, never model readouts, to see
which candidates co-fire at the exact target-active PLS bins and have similar
decoder perturbation L2 there. No model forward is made.
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

from plan_fullstream_cohort import sha256
from run_cloud_ablation_pilot import load_sampled_coordinates
from run_fullstream_local_window import load_window_data
from run_fullstream_pls_ablation_v3 import checked_plan


def eligible_dev_candidates(selection: dict) -> list[int]:
    """Use the dev-frozen 25% firing/L2, low-AUROC pool; preserve its order."""
    result = [int(row["feature"]) for row in selection["closest_local_candidates"]
              if row["relative_dev_pls_rate_difference"] <= 0.25 + 1e-12
              and row["relative_strength_difference"] <= 0.25 + 1e-12
              and row["max_existing_auroc"] < 0.55]
    if len(result) != selection["local_rate_and_strength_within_25_percent_count"]:
        raise ValueError("Candidate list is truncated or differs from dev selection")
    return result


def rank_window_candidates(target_l2: np.ndarray, candidate_l2: dict[int, np.ndarray],
                           dev_order: list[int]) -> tuple[int | None, list[dict]]:
    """Demand identical active support and within-25% total tap L2 at that support."""
    target = np.asarray(target_l2, dtype=np.float64)
    if target.ndim != 1 or not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Invalid target per-PLS-bin dose")
    active = target > 1e-8
    if not active.any():
        return None, []
    rows = []
    for order, feature in enumerate(dev_order):
        values = np.asarray(candidate_l2[feature], dtype=np.float64)
        if values.shape != target.shape or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid comparator dose for feature {feature}")
        coactive = int(np.count_nonzero(values[active] > 1e-8))
        ratio = float(values[active].sum() / target[active].sum())
        rows.append({
            "feature": feature, "dev_rank": order,
            "target_active_bins": int(active.sum()),
            "comparator_active_on_target_bins": coactive,
            "comparator_active_on_all_pls_bins": int(np.count_nonzero(values > 1e-8)),
            "comparator_to_target_l2_on_target_bins": ratio,
        })
    valid = [row for row in rows
             if row["comparator_active_on_target_bins"] == row["target_active_bins"]
             and 0.75 <= row["comparator_to_target_l2_on_target_bins"] <= 1.25]
    valid.sort(key=lambda row: (abs(np.log(row["comparator_to_target_l2_on_target_bins"])),
                                row["dev_rank"], row["feature"]))
    return (valid[0]["feature"] if valid else None), rows


def activation_rows_for_window(shard_coordinates: pd.DataFrame,
                               window_coordinates: pd.DataFrame,
                               chrom: str, start: int,
                               expected_rows: int = 8192) -> np.ndarray:
    """Map one window's sorted bins to rows in a possibly multi-window shard."""
    required = {"chrom", "window_start", "bin_start", "split"}
    if not required.issubset(shard_coordinates) or not required.issubset(window_coordinates):
        raise ValueError("Coordinate tables lack required native fields")
    selector = ((shard_coordinates.chrom == chrom)
                & (shard_coordinates.window_start == start))
    positions = np.flatnonzero(selector.to_numpy())
    if len(positions) != expected_rows or len(window_coordinates) != expected_rows:
        raise ValueError("Coordinate shard does not contain exactly one full window")
    order = np.argsort(shard_coordinates.bin_start.to_numpy()[positions], kind="stable")
    positions = positions[order]
    selected = shard_coordinates.iloc[positions].reset_index(drop=True)
    expected = window_coordinates.sort_values("bin_start").reset_index(drop=True)
    for column in ("chrom", "window_start", "bin_start", "split"):
        if not np.array_equal(selected[column].to_numpy(), expected[column].to_numpy()):
            raise ValueError(f"Activation rows do not align with window coordinates: {column}")
    if (selected.split != "test").any():
        raise ValueError("Window coordinates are not all held out")
    return positions


def audit(plan_path: Path, index: int, acts: Path, fasta: Path,
          out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    plan, window, paths, selection_path = checked_plan(plan_path, index, None, None, {})
    _, _, mask, coord_sha, _ = load_window_data(window, paths, acts, fasta)
    selection = json.loads(selection_path.read_text())
    candidates = eligible_dev_candidates(selection)
    coords, coord_path = load_sampled_coordinates(
        acts, "resid_pre_b8", window["chrom"], window["start"]
    )
    legacy = json.loads((acts / "index.json").read_text())
    item = [row["taps"]["resid_pre_b8"] for row in legacy["shards"]
            if row["split"] == "test" and "resid_pre_b8" in row["taps"]
            and row["taps"]["resid_pre_b8"]["coordinates"] == coord_path.name]
    if len(item) != 1 or sha256(coord_path) != coord_sha:
        raise ValueError("Test activation coordinate shard is ambiguous")
    activation_path = acts / item[0]["activations"]
    if sha256(activation_path) != item[0]["activations_sha256"]:
        raise ValueError("Cached test activation bytes differ from index")
    values = np.load(activation_path, mmap_mode="r", allow_pickle=False)
    shard_coords = pd.read_parquet(coord_path)
    if (values.ndim != 2 or values.shape[1] != 1536
            or len(shard_coords) != len(values)
            or len(shard_coords) != int(item[0]["rows"])):
        raise ValueError("Cached activation shape differs from indexed coordinate shard")
    window_rows = activation_rows_for_window(
        shard_coords, coords, window["chrom"], int(window["start"])
    )
    from ag_sae.sae import BorzoiSAE

    sae = BorzoiSAE.from_checkpoint(paths["sae"], device="cuda")
    positions = window_rows[mask]
    raw = torch.as_tensor(np.array(values[positions], dtype=np.float32), device="cuda")
    with torch.inference_mode():
        pre, params = sae.core.encode(raw / sae.channel_scale)
        code = sae.core.get_sparse_activations(sae.core.activation(pre))
        direction_norm = torch.linalg.vector_norm(
            sae.core.decoder.weight.detach().float()
            * sae.channel_scale.float()[:, None], dim=0
        )
        doses = (code.clamp_min(0).float() * params["std"].float()
                 * direction_norm[None, :]).cpu().numpy()
    target_l2 = doses[:, 199]
    candidate_l2 = {feature: doses[:, feature] for feature in candidates}
    chosen, rows = rank_window_candidates(target_l2, candidate_l2, candidates)
    result = {
        "format": "outcome_blind_pls_window_control_audit_v1",
        "interpretation": "Cached test activations only; no model output inspected or altered",
        "plan_sha256": sha256(plan_path),
        "selection_sha256": sha256(selection_path),
        "window_index": index, "window": window,
        "target_feature": 199, "candidate_features": candidates,
        "chosen_feature": chosen, "target_fired_on_pls": int(np.count_nonzero(target_l2 > 1e-8)),
        "target_l2_sum_on_pls": float(target_l2.sum()),
        "activation_shard_rows": len(shard_coords),
        "activation_window_rows": len(window_rows),
        "activation_window_row_min": int(window_rows.min()),
        "activation_window_row_max": int(window_rows.max()),
        "candidates": rows,
        "activation_sha256": sha256(activation_path),
        "coordinates_sha256": coord_sha,
        "worker_sha256": sha256(Path(__file__)),
    }
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    with (out / "audit.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"out": str(out), "chosen": chosen,
            "target_fired_on_pls": result["target_fired_on_pls"],
            "audit_sha256": sha256(out / "audit.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--window-index", type=int, required=True)
    parser.add_argument("--acts-dir", type=Path, required=True)
    parser.add_argument("--fasta-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.plan, args.window_index, args.acts_dir,
                           args.fasta_dir, args.out)))


if __name__ == "__main__":
    main()

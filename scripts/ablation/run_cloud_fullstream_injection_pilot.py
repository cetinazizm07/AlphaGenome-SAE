#!/usr/bin/env python3
"""One PLS-absent held-out window: dev-calibrated full-stream b8 injection.

This exploratory technical pilot uses a sparse, positively matched feature.
It is not an inferential or causal conclusion and does not replace paired
ablation on concept-present windows. The earlier b0 pilot remains immutable.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch

from run_cloud_ablation_pilot import chrom_sequence, file_sha256, load_sampled_coordinates, read_head
from run_cloud_fullstream_control_pilot import FullStreamTower, run_replacement, select_control


def choose_absent_window(manifest, annotations, chrom: str, start: int):
    held = manifest.loc[
        (manifest.split == "test") & (manifest.chrom == chrom) & (manifest.win_start == start)
    ]
    if len(held) != 1:
        raise ValueError("Expected exactly one held-out window")
    row = held.iloc[0]
    bins = annotations.loc[
        (annotations.split == "test") & (annotations.chrom == chrom)
        & (annotations.bin_start >= row.win_start)
        & (annotations.bin_start < row.win_end)
    ].sort_values("bin_start").reset_index(drop=True)
    if len(bins) != 8192:
        raise ValueError("Expected the full 128-bp annotation grid")
    expected = start + np.arange(8192, dtype=np.int64) * 128
    if not np.array_equal(bins.bin_start.to_numpy(dtype=np.int64), expected):
        raise ValueError("Annotation coordinates differ from the genomic grid")
    valid = bins.n_mask.to_numpy(dtype=bool)
    if not valid.any() or (valid & bins.cCRE_PLS.to_numpy(dtype=bool)).any():
        raise ValueError("Injection window is not PLS-absent")
    if (valid & bins.cCRE_pELS.to_numpy(dtype=bool)).any():
        raise ValueError("Pilot window must also be pELS-absent")
    return row, bins


@torch.inference_mode()
def reconstructed_injection_triplet(sae, activation, target, control, positions,
                                     target_level, control_level, batch_size=256):
    if (activation.ndim != 3 or activation.shape[0] != 1
            or activation.shape[-1] != sae.d_in or batch_size < 1):
        raise ValueError("Invalid activation shape or batch size")
    indices = torch.as_tensor(positions, dtype=torch.long, device=activation.device)
    if (indices.ndim != 1 or not len(indices) or indices.unique().numel() != len(indices)
            or bool((indices < 0).any()) or bool((indices >= activation.shape[1]).any())):
        raise ValueError("Invalid native intervention positions")
    if (target == control or not 0 <= target < sae.hidden
            or not 0 <= control < sae.hidden):
        raise ValueError("Invalid target/control feature")
    if not np.isfinite(target_level) or target_level <= 0 or not np.isfinite(control_level) or control_level <= 0:
        raise ValueError("Injection levels must be positive and finite")
    baseline, target_edit, control_edit = (activation.clone() for _ in range(3))
    scale = sae.channel_scale.float()
    counts = {"target_was_zero": 0, "control_was_zero": 0,
              "target_changed": 0, "control_changed": 0}
    for offset in range(0, len(indices), batch_size):
        rows = indices[offset:offset + batch_size]
        raw = activation[0, rows].float()
        pre, params = sae.core.encode(raw / scale)
        codes = sae.core.get_sparse_activations(sae.core.activation(pre))
        if codes.shape != (len(rows), sae.hidden):
            raise ValueError("Unexpected SAE code shape")
        counts["target_was_zero"] += int((codes[:, target] == 0).sum().item())
        counts["control_was_zero"] += int((codes[:, control] == 0).sum().item())
        counts["target_changed"] += int((codes[:, target] != target_level).sum().item())
        counts["control_changed"] += int((codes[:, control] != control_level).sum().item())
        reconstructed = sae.core.decode(codes, params) * scale
        target_codes = codes.clone()
        target_codes[:, target] = target_level
        control_codes = codes.clone()
        control_codes[:, control] = control_level
        injected_target = sae.core.decode(target_codes, params) * scale
        injected_control = sae.core.decode(control_codes, params) * scale
        if not all(torch.isfinite(value).all() for value in
                   (reconstructed, injected_target, injected_control)):
            raise ValueError("Nonfinite SAE reconstruction or injection")
        baseline[0, rows] = reconstructed.to(activation.dtype)
        target_edit[0, rows] = injected_target.to(activation.dtype)
        control_edit[0, rows] = injected_control.to(activation.dtype)
    if not counts["target_changed"] or not counts["control_changed"]:
        raise ValueError("Injection did not change either feature")
    return baseline, target_edit, control_edit, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "sae", "match", "firing", "auroc", "calibration",
                 "manifest", "ann", "acts-dir", "fasta-dir", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--tap", default="resid_pre_b8")
    parser.add_argument("--concept", default="cCRE_PLS")
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.tap != "resid_pre_b8" or args.concept != "cCRE_PLS":
        raise ValueError("Exploratory pilot requires fold1 resid_pre_b8 / cCRE_PLS")

    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.utils import sequence_to_onehot
    from ag_sae.extract import TAPS
    from ag_sae.sae import BorzoiSAE

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    calibration_path = Path(args.calibration)
    calibration = json.loads(calibration_path.read_text())
    if (calibration["format"] != "sae_injection_dev_positive_p95_v1"
            or calibration["split"] != "dev" or calibration["tap"] != args.tap
            or calibration["n_rows"] != 819197):
        raise ValueError("Calibration does not match the frozen dev-split rule")
    matches = pd.read_csv(args.match)
    matched = matches.loc[matches.concept == args.concept]
    if (len(matched) != 1 or not bool(matched.iloc[0].recovered_v3)
            or int(matched.iloc[0].match_sign) != 1):
        raise ValueError("Missing unique, positive-sign recovered_v3 target match")
    target = int(str(matched.iloc[0].best_feature).removeprefix("sae_"))
    firing, auroc = np.load(args.firing), np.load(args.auroc)
    if not 0 < float(firing[target]) < 0.5:
        raise ValueError("Target is not sparse enough for the exploratory pilot")
    control, candidates = select_control(target, firing, auroc)
    if (target != calibration["target_feature"] or control != calibration["control_feature"]):
        raise ValueError("Calibration feature pair differs from precommitted match/control")
    if (file_sha256(args.sae) != calibration["sha256"]["sae"]
            or file_sha256(Path(args.acts_dir) / "index.json") != calibration["sha256"]["activation_index"]):
        raise ValueError("SAE or activation index changed since calibration")
    target_level = float(calibration["features"][str(target)]["positive_p95"])
    control_level = float(calibration["features"][str(control)]["positive_p95"])

    manifest = pd.read_parquet(args.manifest)
    ann = pd.read_parquet(args.ann)
    row, bins = choose_absent_window(manifest, ann, args.chrom, args.start)
    chrom, start = str(row.chrom), int(row.win_start)
    width = int(row.win_end - row.win_start)
    if width != 1_048_576:
        raise ValueError("Expected one deployed 1-Mb window")
    tap = TAPS[args.tap]
    if tap.kind != "tower" or tap.bin_bp != 128:
        raise ValueError("Unexpected injection tap")
    coords, coord_path = load_sampled_coordinates(Path(args.acts_dir), args.tap, chrom, start)
    relative_native = coords.bin_start.to_numpy(dtype=np.int64) - start
    if ((coords.chrom != chrom).any() or (coords.split != "test").any()
            or (relative_native < 0).any() or (relative_native >= width).any()
            or (relative_native % 128 != 0).any()
            or len(np.unique(relative_native)) != len(relative_native)):
        raise ValueError("Sampled native coordinates are not aligned")
    native_indices = relative_native // 128
    valid_by_bin = dict(zip(
        ((bins.bin_start.to_numpy(dtype=np.int64) - start) // 128).tolist(),
        bins.n_mask.to_numpy(dtype=bool).tolist(),
    ))
    use_native = np.array(
        [valid_by_bin.get(int(index), False) for index in native_indices], dtype=bool
    )
    native_indices = native_indices[use_native]
    coords_used = coords.loc[use_native].reset_index(drop=True)
    bins = bins.loc[bins.n_mask].reset_index(drop=True)
    bin_indices = (bins.bin_start.to_numpy(dtype=np.int64) - start) // 128
    sequence = chrom_sequence(args.fasta_dir, chrom)[start:start + width]
    if len(sequence) != width:
        raise ValueError("FASTA does not cover selected window")
    onehot = sequence_to_onehot(sequence).astype(np.float32)
    if onehot.shape != (width, 4):
        raise ValueError("Unexpected one-hot shape")
    native_valid = onehot.sum(-1).reshape(-1, tap.bin_bp).all(-1)
    if not native_valid[native_indices].all():
        raise ValueError("Intervention overlaps an ambiguous base")
    input_tensor = torch.from_numpy(onehot)[None].to("cuda")
    organism = torch.tensor([0], device="cuda")
    sae = BorzoiSAE.from_checkpoint(args.sae, device="cuda")
    if sae.d_in != tap.channels or target >= sae.hidden or control >= sae.hidden:
        raise ValueError("SAE dimension/feature mismatch")
    model = AlphaGenome.from_pretrained(
        args.weights, dtype_policy=DtypePolicy.mixed_precision(), device="cuda"
    )
    model.eval().requires_grad_(False)
    # PLS primary DNase and a predeclared off-target histone readout.
    heads = ("dnase", "chip_histone")
    forward_args = {"resolutions": (128,), "heads": heads, "channels_last": True}
    captured = []
    started = time.monotonic()

    def capture(value):
        captured.append(np.array(value, dtype=np.float32, copy=True))
        return value

    with torch.inference_mode(), FullStreamTower(model, tap, capture):
        original_output = model.predict(input_tensor, organism, **forward_args)
    if len(captured) != 1 or captured[0].shape != (width // 128, tap.channels):
        raise ValueError("Full-stream capture shape/call count differs")
    original = {head: read_head(original_output, head, width, 128, bin_indices) for head in heads}
    del original_output
    activation = torch.from_numpy(captured[0])[None].to("cuda")
    del captured
    baseline, target_edit, control_edit, counts = reconstructed_injection_triplet(
        sae, activation, target, control, native_indices,
        target_level, control_level, args.batch_size,
    )
    del activation
    torch.cuda.empty_cache()

    def predict_with(tensor):
        with torch.inference_mode(), FullStreamTower(model, tap, run_replacement(tensor)):
            result = model.predict(input_tensor, organism, **forward_args)
        values = {head: read_head(result, head, width, 128, bin_indices) for head in heads}
        del result
        torch.cuda.empty_cache()
        return values

    reconstruction = predict_with(baseline)
    target_output = predict_with(target_edit)
    control_output = predict_with(control_edit)
    del baseline, target_edit, control_edit
    elapsed = time.monotonic() - started
    readouts_path = out / "readouts.npz"
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "readouts.npz.tmp"
    arrays = {"bin_start": bins.bin_start.to_numpy(dtype=np.int64), "bin_index": bin_indices,
              "concept_label": bins.cCRE_PLS.to_numpy(dtype=bool)}
    effects = {}
    for head in heads:
        arrays[f"{head}_original"] = original[head]
        arrays[f"{head}_reconstruction_baseline"] = reconstruction[head]
        arrays[f"{head}_target_injected"] = target_output[head]
        arrays[f"{head}_control_injected"] = control_output[head]
        baseline_mean = reconstruction[head].mean(axis=1)
        scale = float(baseline_mean.std() + 1e-9)
        target_effect = float((target_output[head] - reconstruction[head]).mean(axis=1).mean())
        control_effect = float((control_output[head] - reconstruction[head]).mean(axis=1).mean())
        effects[head] = {
            "target_mean": target_effect, "control_mean": control_effect,
            "target_minus_control": target_effect - control_effect,
            "baseline_spatial_std": scale,
            "target_standardized": target_effect / scale,
            "control_standardized": control_effect / scale,
        }
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(readouts_path)
    bins.to_parquet(out / "window_annotations.parquet", index=False)
    coords_used.to_parquet(out / "sampled_native_coordinates.parquet", index=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    paths = {
        "weights": args.weights, "sae": args.sae, "match": args.match,
        "firing": args.firing, "auroc": args.auroc, "calibration": calibration_path,
        "manifest": args.manifest, "ann": args.ann,
        "activation_index": Path(args.acts_dir) / "index.json",
        "sampled_coordinates": coord_path, "runner": __file__,
    }
    receipt = {
        "format": "single_window_fullstream_dev_calibrated_injection_pilot_v2",
        "interpretation": "Exploratory forced-high pilot with a sparse positive-sign match on one PLS/pELS-absent window; no causal verdict.",
        "fold_window": {"fold": "fold1", "chrom": chrom, "start": start,
                        "end": start + width, "split": str(row.split)},
        "tap": args.tap, "tap_route": "tower_block_input_before_pair_update_mha_residual_mlp",
        "target_feature": target, "control_feature": control,
        "match_sign": int(matched.iloc[0].match_sign),
        "test_firing_rate_target": float(firing[target]),
        "test_firing_rate_control": float(firing[control]),
        "control_candidates": candidates,
        "calibration_split": "dev", "target_positive_p95": target_level,
        "control_positive_p95": control_level,
        "heads": {"primary": ["dnase"], "off_target": ["chip_histone"]},
        "n_native_positions": int(len(native_indices)),
        "n_valid_128bp_bins": int(len(bins)), "code_change_counts": counts,
        "effects": effects, "elapsed_seconds": elapsed,
        "sha256": {key: file_sha256(path) for key, path in paths.items()},
        "readouts_sha256": file_sha256(readouts_path),
    }
    receipt_tmp = out / "receipt.json.tmp"
    receipt_tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt_tmp.replace(out / "receipt.json")
    print(json.dumps({"out": str(out), "receipt": receipt}, indent=2))


if __name__ == "__main__":
    main()

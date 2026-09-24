#!/usr/bin/env python3
"""Single-window, paired full-stream SAE ablation with an activity-matched control.

This is a technical check of the tower intervention route, not the complete
necessity/sufficiency experiment. It deliberately uses the same held-out window,
checkpoint, feature match, coordinates and DNase readout as the MHA-only pilot.
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

from run_cloud_ablation_pilot import (
    chrom_sequence,
    choose_window,
    file_sha256,
    load_sampled_coordinates,
    read_head,
    reconstructed_ablation_pair,
)


class FullStreamTower:
    """Replace tower-block input before pair update, MHA, and residual paths."""

    def __init__(self, model, tap, replace):
        if tap.kind != "tower":
            raise ValueError("Full-stream pilot requires a tower tap")
        self.tower = model.tower
        self.block = self.tower.blocks[int(tap.key)]
        self.replace = replace
        self.calls = 0
        self.had_override = False
        self.previous = None

    def __enter__(self):
        self.had_override = "_forward_block" in vars(self.tower)
        self.previous = self.tower._forward_block

        def on_block(block, x, pair_x, compute_dtype):
            if block is self.block:
                current = x[0].detach().float().cpu().numpy()
                edited = np.asarray(self.replace(current), dtype=np.float32)
                if edited.shape != current.shape or not np.isfinite(edited).all():
                    raise ValueError("Full-stream editor changed shape or produced nonfinite values")
                replacement = x.clone()
                replacement[0] = torch.as_tensor(
                    np.ascontiguousarray(edited), dtype=x.dtype, device=x.device
                )
                x = replacement
                self.calls += 1
            return self.previous(block, x, pair_x, compute_dtype)

        self.tower._forward_block = on_block
        return self

    def __exit__(self, exc_type, _value, _traceback):
        if self.had_override:
            self.tower._forward_block = self.previous
        else:
            delattr(self.tower, "_forward_block")
        if exc_type is None and self.calls != 1:
            raise RuntimeError(f"Expected one full-stream tap call, got {self.calls}")


def select_control(feature: int, firing: np.ndarray, auroc: np.ndarray) -> tuple[int, list[int]]:
    """Frozen protocol: within 25% firing, max concept AUROC < 0.55, seeded draw."""
    if firing.ndim != 1 or auroc.ndim != 2 or auroc.shape[0] != len(firing):
        raise ValueError("Mismatched firing-rate and AUROC arrays")
    if not 0 <= feature < len(firing) or not np.isfinite(firing[feature]) or firing[feature] <= 0:
        raise ValueError("Invalid target firing rate")
    max_auroc = np.nanmax(auroc, axis=1)
    eligible = (np.isfinite(firing) & (firing > 0)
                & (np.abs(firing - firing[feature]) / firing[feature] <= 0.25)
                & np.isfinite(max_auroc) & (max_auroc < 0.55))
    eligible[feature] = False
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        raise ValueError("No activity-matched low-AUROC control feature")
    choice = int(np.random.default_rng(1234 + feature).choice(candidates))
    return choice, candidates.tolist()


def run_replacement(tensor: torch.Tensor):
    values = tensor[0].detach().float().cpu().numpy()

    def replace(current: np.ndarray) -> np.ndarray:
        if current.shape != values.shape:
            raise ValueError("Tap shape changed between paired forwards")
        return values

    return replace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "sae", "match", "firing", "auroc", "manifest", "ann",
                 "acts-dir", "fasta-dir", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--tap", default="resid_pre_b0")
    parser.add_argument("--concept", default="cCRE_PLS")
    parser.add_argument("--head", default="dnase")
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--legacy-readouts", default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.utils import sequence_to_onehot
    from ag_sae.ablate import Intervention
    from ag_sae.extract import TAPS
    from ag_sae.sae import BorzoiSAE

    tap = TAPS[args.tap]
    if tap.kind != "tower" or args.head != "dnase" or args.concept != "cCRE_PLS":
        raise ValueError("This controlled pilot is frozen to tower/PLS/DNase")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(args.manifest)
    ann = pd.read_parquet(args.ann)
    row, bins = choose_window(manifest, ann, args.concept, args.chrom, args.start)
    chrom, start = str(row.chrom), int(row.win_start)
    width = int(row.win_end - row.win_start)
    if width != 1_048_576:
        raise ValueError("Expected exactly one deployed 1-Mb window")
    match = pd.read_csv(args.match)
    selected = match.loc[match.concept == args.concept]
    if len(selected) != 1 or not bool(selected.iloc[0].recovered_v3):
        raise ValueError("Missing unique recovered_v3 match")
    feature = int(str(selected.iloc[0].best_feature).removeprefix("sae_"))
    firing = np.load(args.firing)
    auroc = np.load(args.auroc)
    control, candidates = select_control(feature, firing, auroc)

    coords, coord_path = load_sampled_coordinates(Path(args.acts_dir), args.tap, chrom, start)
    relative_native = coords.bin_start.to_numpy(dtype=np.int64) - start
    if ((coords.chrom != chrom).any() or (coords.split != "test").any()
            or (relative_native < 0).any() or (relative_native >= width).any()
            or (relative_native % tap.bin_bp != 0).any()
            or len(np.unique(relative_native)) != len(relative_native)):
        raise ValueError("Native sampled coordinates do not uniquely map to window")
    native_indices = relative_native // tap.bin_bp
    relative_bins = bins.bin_start.to_numpy(dtype=np.int64) - start
    if ((relative_bins < 0).any() or (relative_bins >= width).any()
            or (relative_bins % 128 != 0).any()):
        raise ValueError("Invalid 128-bp annotation coordinates")
    bin_indices = relative_bins // 128
    valid_by_bin = dict(zip(bin_indices.tolist(), bins.n_mask.to_numpy(dtype=bool).tolist()))
    use_native = np.array(
        [valid_by_bin.get(int(i), False) for i in relative_native // 128], dtype=bool
    )
    native_indices = native_indices[use_native]
    coords_used = coords.loc[use_native].reset_index(drop=True)
    readout_keep = bins.n_mask.to_numpy(dtype=bool)
    bins = bins.loc[readout_keep].reset_index(drop=True)
    bin_indices = bin_indices[readout_keep]
    labels = bins[args.concept].to_numpy(dtype=bool)
    if not labels.any() or not (~labels).any():
        raise ValueError("Window lacks one concept class")

    sequence = chrom_sequence(args.fasta_dir, chrom)[start:start + width]
    if len(sequence) != width:
        raise ValueError("FASTA does not cover selected window")
    onehot = sequence_to_onehot(sequence).astype(np.float32)
    if onehot.shape != (width, 4):
        raise ValueError("Unexpected onehot shape")
    native_valid = onehot.sum(-1).reshape(-1, tap.bin_bp).all(-1)
    if not native_valid[native_indices].all():
        raise ValueError("Native intervention positions overlap ambiguous bases")
    input_tensor = torch.from_numpy(onehot)[None].to("cuda")
    organism = torch.tensor([0], device="cuda")
    sae = BorzoiSAE.from_checkpoint(args.sae, device="cuda")
    if sae.d_in != tap.channels or not 0 <= feature < sae.hidden or not 0 <= control < sae.hidden:
        raise ValueError("SAE dimensions differ from tap or selected features")
    model = AlphaGenome.from_pretrained(
        args.weights, dtype_policy=DtypePolicy.mixed_precision(), device="cuda"
    )
    model.eval().requires_grad_(False)
    forward_args = {"resolutions": (128,), "heads": (args.head,), "channels_last": True}
    full_capture, mha_capture = [], []
    started = time.monotonic()

    def capture_full(value):
        full_capture.append(np.array(value, dtype=np.float32, copy=True))
        return value

    def capture_mha(value):
        mha_capture.append(np.array(value, dtype=np.float32, copy=True))
        return value

    with torch.inference_mode(), FullStreamTower(model, tap, capture_full), Intervention(
        model, tap, capture_mha
    ):
        original_output = model.predict(input_tensor, organism, **forward_args)
    expected_shape = (width // tap.bin_bp, tap.channels)
    if (len(full_capture) != 1 or len(mha_capture) != 1
            or full_capture[0].shape != expected_shape or mha_capture[0].shape != expected_shape
            or not np.array_equal(full_capture[0], mha_capture[0])):
        raise ValueError("Full-stream and old MHA tap captures differ")
    original = read_head(original_output, args.head, width, 128, bin_indices)
    del original_output, mha_capture
    activation = torch.from_numpy(full_capture[0])[None].to("cuda")
    del full_capture
    baseline, target_ablated, target_fired = reconstructed_ablation_pair(
        sae, activation, feature, native_indices, args.batch_size
    )
    control_baseline, control_ablated, control_fired = reconstructed_ablation_pair(
        sae, activation, control, native_indices, args.batch_size
    )
    if not torch.equal(baseline, control_baseline) or not target_fired or not control_fired:
        raise ValueError("Paired SAE baseline differs or target/control never fired")
    del activation, control_baseline
    torch.cuda.empty_cache()

    def predict_with(tensor):
        with torch.inference_mode(), FullStreamTower(model, tap, run_replacement(tensor)):
            result = model.predict(input_tensor, organism, **forward_args)
        readout = read_head(result, args.head, width, 128, bin_indices)
        del result
        torch.cuda.empty_cache()
        return readout

    reconstruction = predict_with(baseline)
    target = predict_with(target_ablated)
    control_out = predict_with(control_ablated)
    del baseline, target_ablated, control_ablated
    elapsed = time.monotonic() - started
    legacy = None
    if args.legacy_readouts:
        with np.load(args.legacy_readouts) as previous:
            if not np.array_equal(previous["bin_start"], bins.bin_start.to_numpy(dtype=np.int64)):
                raise ValueError("Legacy readout bin coordinates differ")
            legacy = {
                "original_max_abs_delta": float(np.max(np.abs(original - previous["original"]))),
                "reconstruction_max_abs_delta": float(np.max(np.abs(reconstruction - previous["reconstruction_baseline"]))),
                "target_ablated_max_abs_delta": float(np.max(np.abs(target - previous["ablated"]))),
            }

    readouts_path = out / "readouts.npz"
    temporary = out / "readouts.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, original=original, reconstruction_baseline=reconstruction,
            target_ablated=target, control_ablated=control_out,
            concept_label=labels, bin_index=bin_indices,
            bin_start=bins.bin_start.to_numpy(dtype=np.int64),
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(readouts_path)
    bins.to_parquet(out / "window_annotations.parquet", index=False)
    coords_used.to_parquet(out / "sampled_native_coordinates.parquet", index=False)
    shutil.copy2(__file__, out / Path(__file__).name)
    source_paths = {
        "weights": args.weights, "sae": args.sae, "match": args.match,
        "firing": args.firing, "auroc": args.auroc, "manifest": args.manifest,
        "ann": args.ann, "activation_index": Path(args.acts_dir) / "index.json",
        "sampled_coordinates": coord_path, "runner": __file__,
    }
    if args.legacy_readouts:
        source_paths["legacy_readouts"] = args.legacy_readouts
    effect_target = (target - reconstruction).mean(axis=1)
    effect_control = (control_out - reconstruction).mean(axis=1)
    receipt = {
        "format": "single_window_fullstream_matched_control_pilot_v1",
        "interpretation": "Technical paired ablation pilot, not causal support; injection and inference pending.",
        "fold_window": {"fold": "fold1", "chrom": chrom, "start": start,
                        "end": start + width, "split": str(row.split)},
        "tap": args.tap, "tap_route": "tower_block_input_before_pair_update_mha_residual_mlp",
        "head": args.head, "concept": args.concept, "target_feature": feature,
        "control_feature": control, "control_candidates": candidates,
        "control_rule": "firing within 25%, max AUROC < 0.55, rng seed 1234+target",
        "firing_target": float(firing[feature]), "firing_control": float(firing[control]),
        "max_auroc_control": float(np.nanmax(auroc[control])),
        "n_native_positions": int(len(native_indices)),
        "n_target_fired": target_fired, "n_control_fired": control_fired,
        "n_positive_bins": int(labels.sum()), "n_negative_bins": int((~labels).sum()),
        "target_mean_effect_positive": float(effect_target[labels].mean()),
        "target_mean_effect_negative": float(effect_target[~labels].mean()),
        "control_mean_effect_positive": float(effect_control[labels].mean()),
        "control_mean_effect_negative": float(effect_control[~labels].mean()),
        "legacy_mha_only_comparison": legacy,
        "elapsed_seconds": elapsed, "sha256": {k: file_sha256(v) for k, v in source_paths.items()},
        "readouts_sha256": file_sha256(readouts_path),
    }
    receipt_path = out / "receipt.json"
    receipt_tmp = out / "receipt.json.tmp"
    receipt_tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt_tmp.replace(receipt_path)
    print(json.dumps({"out": str(out), "receipt": receipt}, indent=2))


if __name__ == "__main__":
    main()

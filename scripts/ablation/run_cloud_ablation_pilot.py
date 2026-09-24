#!/usr/bin/env python3
"""Run one portable, held-out, reconstruction-baseline SAE ablation pilot.

The source activation pipeline samples native tap positions from 1 Mb windows.
This pilot reruns the *whole* 1 Mb sequence, then intervenes only at the exact
sampled positions recorded for the selected test window. It is a technical
pilot, not an inferential or causal conclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def chrom_sequence(fasta_root: str | Path, chrom: str) -> str:
    """Read one chromosome from a per-chromosome FASTA or the bundled genome."""
    root = Path(fasta_root)
    if root.is_file():
        candidates = [root]
    else:
        candidates = [
            root / f"Homo_sapiens.GRCh38.dna.chromosome.{chrom.removeprefix('chr')}.fa.gz",
            root / f"{chrom}.fa", root / f"{chrom}.fa.gz", root / f"{chrom}.fasta",
        ]
        if not any(p.is_file() for p in candidates):
            candidates.append(root / "GRCh38.primary_assembly.genome.fa")
    import gzip

    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No FASTA for {chrom} under {root}")
    opener = gzip.open if path.suffix == ".gz" else open
    chunks: list[str] = []
    active = False
    found = False
    with opener(path, "rt") as stream:
        for line in stream:
            if line.startswith(">"):
                if active:
                    break
                name = line[1:].split()[0]
                active = name == chrom or name == chrom.removeprefix("chr")
                found |= active
            elif active:
                chunks.append(line.strip())
    if not found:
        raise ValueError(f"Chromosome {chrom} was not present in {path}")
    return "".join(chunks).upper()


def load_sampled_coordinates(acts_dir: Path, tap_name: str, chrom: str,
                             start: int) -> tuple[pd.DataFrame, Path]:
    """Find the exact sampled native positions used to build the match matrix."""
    index_path = acts_dir / "index.json"
    index = json.loads(index_path.read_text())
    if not index.get("complete"):
        raise ValueError(f"Activation index is not complete: {index_path}")
    for shard in index["shards"]:
        if shard["split"] != "test" or tap_name not in shard["taps"]:
            continue
        coordinate_path = acts_dir / shard["taps"][tap_name]["coordinates"]
        coords = pd.read_parquet(coordinate_path)
        chosen = coords.loc[(coords.chrom == chrom) &
                            (coords.window_start == start)].copy()
        if not chosen.empty:
            return chosen.sort_values("bin_start").reset_index(drop=True), coordinate_path
    raise ValueError(f"No sampled {tap_name} coordinates for held-out {chrom}:{start}")


def choose_window(manifest: pd.DataFrame, ann: pd.DataFrame, concept: str,
                  chrom: str | None, start: int | None) -> tuple[pd.Series, pd.DataFrame]:
    held = manifest.loc[manifest.split == "test"]
    if chrom is not None:
        held = held.loc[(held.chrom == chrom) & (held.win_start == start)]
    for _, row in held.iterrows():
        width = int(row.win_end - row.win_start)
        bins = ann.loc[(ann.chrom == row.chrom) &
                       (ann.bin_start >= row.win_start) &
                       (ann.bin_start < row.win_end) &
                       (ann.split == "test")].sort_values("bin_start").reset_index(drop=True)
        if bins.empty or concept not in bins:
            continue
        labels = bins[concept].to_numpy(dtype=bool)
        valid = bins.n_mask.to_numpy(dtype=bool)
        relative = bins.bin_start.to_numpy(dtype=np.int64) - int(row.win_start)
        if ((relative < 0).any() or (relative >= width).any() or
                (relative % 128 != 0).any() or len(np.unique(relative)) != len(relative)):
            raise ValueError("Annotation bins do not map uniquely to the window's 128-bp grid")
        if (labels & valid).any() and ((~labels) & valid).any():
            return row, bins
    raise ValueError("No held-out window has both valid positive and negative concept bins")


@torch.inference_mode()
def reconstructed_ablation_pair(sae, activation: torch.Tensor, feature: int,
                                positions: np.ndarray, batch_size: int = 256):
    """Replace sampled positions with SAE reconstruction, then zero one feature."""
    if activation.ndim != 3 or activation.shape[0] != 1 or activation.shape[-1] != sae.d_in:
        raise ValueError("Expected one full-window NLC tap matching the SAE input width")
    if not 0 <= feature < sae.hidden:
        raise ValueError("Matched feature is outside this SAE dictionary")
    indices = torch.as_tensor(positions, dtype=torch.long, device=activation.device)
    if not len(indices) or indices.unique().numel() != indices.numel():
        raise ValueError("Expected nonempty, unique sampled positions")
    if bool((indices < 0).any()) or bool((indices >= activation.shape[1]).any()):
        raise ValueError("Sampled native position is outside the captured tap")

    baseline, ablated = activation.clone(), activation.clone()
    fired = 0
    scale = sae.channel_scale.float()
    for offset in range(0, len(indices), batch_size):
        rows = indices[offset:offset + batch_size]
        raw = activation[0, rows].float()
        pre, params = sae.core.encode(raw / scale)
        codes = sae.core.get_sparse_activations(sae.core.activation(pre))
        fired += int((codes[:, feature] > 0).sum().item())
        reconstructed = sae.core.decode(codes, params) * scale
        codes[:, feature] = 0
        changed = sae.core.decode(codes, params) * scale
        if not torch.isfinite(reconstructed).all() or not torch.isfinite(changed).all():
            raise ValueError("Nonfinite SAE reconstruction or intervention")
        baseline[0, rows] = reconstructed.to(activation.dtype)
        ablated[0, rows] = changed.to(activation.dtype)
    if fired == 0 or torch.equal(baseline, ablated):
        raise ValueError("The selected feature did not fire in this pilot window")
    return baseline, ablated, fired


def read_head(output, name: str, width: int, resolution_bp: int,
              bins_128: np.ndarray) -> np.ndarray:
    value = output[name]
    if not isinstance(value, dict) or resolution_bp not in value:
        raise ValueError(f"{name} did not return the required {resolution_bp}-bp resolution")
    value = value[resolution_bp]
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"Unexpected {name} prediction type or dimensions")
    expected_positions = width // resolution_bp
    if value.shape[1] != expected_positions:
        raise ValueError(f"Expected {expected_positions} positions at {resolution_bp} bp, got {value.shape[1]}")
    if resolution_bp == 1:
        full_bins = width // 128
        pooled = value[0].float().reshape(full_bins, 128, -1).mean(dim=1)
        selected = pooled[bins_128]
    elif resolution_bp == 128:
        selected = value[0].float()[bins_128]
    else:
        raise ValueError(f"Unsupported readout resolution: {resolution_bp}")
    selected = selected.detach().cpu().numpy()
    if not np.isfinite(selected).all():
        raise ValueError(f"Nonfinite {name} predictions")
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--match", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--acts-dir", required=True,
                    help="Legacy activation folder containing index.json and test coordinates")
    ap.add_argument("--fasta-dir", required=True)
    ap.add_argument("--tap", default="resid_pre_b0")
    ap.add_argument("--concept", default="cCRE_PLS")
    ap.add_argument("--chrom", default=None)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--head", default="dnase")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if (args.chrom is None) != (args.start is None):
        raise ValueError("Provide both --chrom and --start, or omit both for deterministic selection")

    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from alphagenome_pytorch.utils import sequence_to_onehot
    from ag_sae.ablate import Intervention
    from ag_sae.extract import TAPS
    from ag_sae.sae import BorzoiSAE

    if args.tap not in TAPS:
        raise ValueError(f"Unknown AlphaGenome tap: {args.tap}")
    tap = TAPS[args.tap]
    resolution_bp = 128 if tap.kind == "tower" else 1
    if args.head in ("splice_sites",) or args.head not in {
        "atac", "dnase", "procap", "cage", "rna_seq", "chip_tf", "chip_histone"
    }:
        raise ValueError(f"Unsupported 1-bp genome-track head: {args.head}")

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing pilot output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    manifest_path, annotation_path = Path(args.manifest), Path(args.ann)
    manifest = pd.read_parquet(manifest_path)
    ann = pd.read_parquet(annotation_path)
    if "n_mask" not in ann:
        raise ValueError("Annotation matrix is missing n_mask")
    row, bins = choose_window(manifest, ann, args.concept, args.chrom, args.start)
    chrom, start = str(row.chrom), int(row.win_start)
    width = int(row.win_end - row.win_start)
    if width != 1_048_576 or width % 128:
        raise ValueError(f"Expected the deployed 1-Mb window, got {width} bp")

    matches = pd.read_csv(args.match)
    selected = matches.loc[matches.concept == args.concept]
    if len(selected) != 1 or not bool(selected.iloc[0].recovered_v3):
        raise ValueError("The pilot concept must have exactly one recovered_v3 feature match")
    feature = int(str(selected.iloc[0].best_feature).removeprefix("sae_"))

    coords, coordinate_path = load_sampled_coordinates(
        Path(args.acts_dir), args.tap, chrom, start)
    relative_native = coords.bin_start.to_numpy(dtype=np.int64) - start
    if ((coords.chrom != chrom).any() or (coords.split != "test").any() or
            (relative_native < 0).any() or (relative_native >= width).any() or
            (relative_native % tap.bin_bp != 0).any() or
            len(np.unique(relative_native)) != len(relative_native)):
        raise ValueError("Sampled activation coordinates do not uniquely map to this window")
    native_indices = relative_native // tap.bin_bp
    relative_bins = bins.bin_start.to_numpy(dtype=np.int64) - start
    bin_indices = relative_bins // 128
    label_by_bin = dict(zip(bin_indices.tolist(), bins[args.concept].to_numpy(dtype=bool).tolist()))
    valid_by_bin = dict(zip(bin_indices.tolist(), bins.n_mask.to_numpy(dtype=bool).tolist()))
    sampled_bin = relative_native // 128
    use_native = np.array([valid_by_bin.get(int(index), False) for index in sampled_bin], dtype=bool)
    native_indices = native_indices[use_native]
    coords_used = coords.loc[use_native].reset_index(drop=True)
    readout_keep = bins.n_mask.to_numpy(dtype=bool)
    bins = bins.loc[readout_keep].reset_index(drop=True)
    bin_indices = bin_indices[readout_keep]
    labels = bins[args.concept].to_numpy(dtype=bool)
    if not (labels.any() and (~labels).any()):
        raise ValueError("Selected window lost one concept class after n_mask filtering")

    fasta = chrom_sequence(args.fasta_dir, chrom)
    sequence = fasta[start:start + width]
    if len(sequence) != width:
        raise ValueError("FASTA does not cover the selected 1-Mb window")
    onehot = sequence_to_onehot(sequence).astype(np.float32)
    if onehot.shape != (width, 4):
        raise ValueError(f"Unexpected one-hot sequence shape: {onehot.shape}")
    native_valid = onehot.sum(-1).reshape(-1, tap.bin_bp).all(-1)
    if not native_valid[native_indices].all():
        raise ValueError("A selected native intervention position overlaps an ambiguous base")
    input_tensor = torch.from_numpy(onehot)[None].to("cuda")
    organism = torch.tensor([0], device="cuda")

    sae = BorzoiSAE.from_checkpoint(args.sae, device="cuda")
    if sae.d_in != tap.channels or not 0 <= feature < sae.hidden:
        raise ValueError("SAE checkpoint dimensions or matched feature do not match this tap")
    # Match the L4/bfloat16 extraction that produced these SAE coordinates.
    policy = DtypePolicy.mixed_precision()
    model = AlphaGenome.from_pretrained(args.weights, dtype_policy=policy, device="cuda")
    model.eval().requires_grad_(False)
    forward_args = {"resolutions": (resolution_bp,), "heads": (args.head,),
                    "channels_last": True}

    captured: list[np.ndarray] = []
    started = time.monotonic()

    def capture(value: np.ndarray) -> np.ndarray:
        captured.append(np.array(value, dtype=np.float32, copy=True))
        return value

    with torch.inference_mode(), Intervention(model, tap, capture):
        raw_output = model.predict(input_tensor, organism, **forward_args)
    if len(captured) != 1 or captured[0].shape != (width // tap.bin_bp, tap.channels):
        raise ValueError(f"Unexpected native tap capture: {[x.shape for x in captured]}")
    original = read_head(raw_output, args.head, width, resolution_bp, bin_indices)
    del raw_output
    activation = torch.from_numpy(captured[0])[None].to("cuda")
    del captured
    reconstruction, ablation, fired = reconstructed_ablation_pair(
        sae, activation, feature, native_indices, args.batch_size)
    del activation
    torch.cuda.empty_cache()

    def run_replacement(tensor: torch.Tensor) -> np.ndarray:
        replacement = tensor[0].detach().float().cpu().numpy()

        def replace(current: np.ndarray) -> np.ndarray:
            if current.shape != replacement.shape:
                raise ValueError("Native tap shape changed across paired forwards")
            return replacement
        return replace

    with torch.inference_mode(), Intervention(model, tap, run_replacement(reconstruction)):
        baseline_output = model.predict(input_tensor, organism, **forward_args)
    baseline = read_head(baseline_output, args.head, width, resolution_bp, bin_indices)
    del baseline_output
    torch.cuda.empty_cache()

    with torch.inference_mode(), Intervention(model, tap, run_replacement(ablation)):
        ablated_output = model.predict(input_tensor, organism, **forward_args)
    edited = read_head(ablated_output, args.head, width, resolution_bp, bin_indices)
    del ablated_output
    elapsed = time.monotonic() - started

    effect = (edited - baseline).mean(axis=1)
    positive = labels
    negative = ~labels
    readouts_path = out / "readouts.npz"
    temporary = out / "readouts.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, original=original, reconstruction_baseline=baseline,
                            ablated=edited, concept_label=labels, valid_128bp=np.ones_like(labels),
                            bin_index=bin_indices, bin_start=bins.bin_start.to_numpy(dtype=np.int64))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(readouts_path)
    bins.to_parquet(out / "window_annotations.parquet", index=False)
    coords_used.to_parquet(out / "sampled_native_coordinates.parquet", index=False)
    shutil.copy2(__file__, out / Path(__file__).name)

    paths = {
        "weights": args.weights, "sae": args.sae, "match": args.match,
        "manifest": manifest_path, "ann": annotation_path,
        "activation_index": Path(args.acts_dir) / "index.json",
        "sampled_coordinates": coordinate_path, "runner": __file__,
    }
    receipt = {
        "format": "single_window_reconstruction_ablation_pilot_v2",
        "interpretation": "technical pilot; one held-out window, no inferential or causal verdict",
        "fold_window": {"fold": "fold1", "chrom": chrom, "start": start,
                        "end": start + width, "split": str(row.split)},
        "tap": args.tap, "tap_kind": tap.kind, "tap_key": tap.key,
        "native_bin_bp": tap.bin_bp, "readout_resolution_bp": resolution_bp,
        "readout_head": args.head, "concept": args.concept, "sae_feature": feature,
        "n_annotation_bins": int(len(bins)), "n_concept_positive_bins": int(labels.sum()),
        "n_concept_negative_bins": int((~labels).sum()),
        "n_sampled_native_positions_intervened": int(len(native_indices)),
        "n_feature_firing_sampled_positions": fired,
        "mean_effect_positive_bins": float(effect[positive].mean()),
        "mean_effect_negative_bins": float(effect[negative].mean()),
        "elapsed_seconds": elapsed, "torch": str(torch.__version__),
        "checkpoint_recipe": torch.load(args.sae, map_location="cpu", weights_only=True)["recipe"],
        "sha256": {name: file_sha256(path) for name, path in paths.items()},
        "readouts_sha256": file_sha256(readouts_path),
    }
    receipt_path = out / "receipt.json"
    receipt_tmp = receipt_path.with_suffix(".json.tmp")
    receipt_tmp.write_text(json.dumps(receipt, indent=2))
    receipt_tmp.replace(receipt_path)
    print(json.dumps({"out": str(out), "window": receipt["fold_window"],
                      "positive_bins": int(labels.sum()), "negative_bins": int((~labels).sum()),
                      "firing_positions": fired, "elapsed_seconds": elapsed,
                      "mean_effect_positive_bins": receipt["mean_effect_positive_bins"],
                      "mean_effect_negative_bins": receipt["mean_effect_negative_bins"]}, indent=2))


if __name__ == "__main__":
    main()

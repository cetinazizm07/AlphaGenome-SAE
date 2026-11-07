"""Causal intervention engine for AlphaGenome-SAE.

Implements two complementary causal interpretability protocols:
  1. Regulatory Triad Double-Dissociation Ablation:
     Tests the causal necessity and specificity of Promoter (PLS), Insulator (CTCF),
     and Enhancer (pELS) SAE latents compared to Shuffled Controls across held-out
     genomic test windows, evaluating chromatin accessibility, TF binding, and transcription.
  2. MANE Select Gene-TSS Causal Injection Cohort:
     Tests causal transcriptional activation (PRO-cap nascent initiation and RNA-seq)
     by clamping promoter features at the TSS with L2-dose matched negative controls.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None

from ag_sae.edit import (Condition, clamp_min_feature, run_conditions, sae_edit,
                         track_mean_profile, zero_feature)
from ag_sae.stats import StatisticalSummary, compute_paired_statistics, safe_wilcoxon

ALL_HEADS: tuple[str, ...] = (
    "dnase", "atac", "cage", "rna_seq", "procap", "chip_tf", "chip_histone"
)
TRANSCRIPTIONAL_HEADS: tuple[str, ...] = ("rna_seq", "cage", "procap")

# In-memory RAM cache for chromosome FASTA sequences (zero-copy per run)
CHROM_CACHE: dict[str, str] = {}

# In-memory cache for track metadata catalogs and queries
_CATALOG_CACHE: dict[str, Any] = {}


def cleanup_gpu() -> None:
    """Trigger Python garbage collection and empty PyTorch CUDA cache."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def sequence_to_onehot(sequence: str) -> np.ndarray:
    """Convert nucleotide string to (L, 4) float32 one-hot matrix with robust multi-path fallback."""
    try:
        from alphagenome_pytorch.utils.sequence import sequence_to_onehot as _up_ohe
        return _up_ohe(sequence)
    except ImportError:
        pass
    try:
        from alphagenome_pytorch.utils import sequence_to_onehot as _up_ohe
        return _up_ohe(sequence)
    except ImportError:
        pass
    # Pure NumPy fallback identical to ag_sae.extract.one_hot
    bases = "ACGT"
    codes = np.frombuffer(sequence.upper().encode(), dtype=np.uint8)
    out = np.zeros((len(codes), 4), dtype=np.float32)
    for index, base in enumerate(bases):
        out[codes == ord(base), index] = 1.0
    return out


def _atomic_write_json(path: Path | str, payload: Any) -> None:
    """Write JSON payload to disk atomically using a PID-tagged temporary file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(target)


def is_valid_json_file(path: Path | str | None) -> bool:
    """Return True if path points to an existing, non-empty, parseable JSON file."""
    if path is None:
        return False
    target = Path(path)
    if not target.is_file() or target.stat().st_size == 0:
        return False
    try:
        with open(target, "rb") as f:
            json.load(f)
        return True
    except Exception:
        return False


def map_annotation_column_to_grid(
    bins: pd.DataFrame, col: str, window_start: int, num_bins: int, bin_bp: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map an annotation column onto the dense grid, returning (pos_mask, observed_mask).
    
    - pos_mask: True where bin was observed and annotated as positive (True).
    - observed_mask: True where bin was observed/present in the annotations table.
    """
    pos_mask = np.zeros(num_bins, dtype=bool)
    observed_mask = np.zeros(num_bins, dtype=bool)
    if col not in bins.columns or len(bins) == 0:
        return pos_mask, observed_mask
    rel_indices = ((bins.bin_start - window_start) // bin_bp).to_numpy(dtype=np.int64)
    valid = (rel_indices >= 0) & (rel_indices < num_bins)
    observed_indices = rel_indices[valid]
    observed_mask[observed_indices] = True
    col_vals = bins[col].to_numpy(bool)
    active_indices = rel_indices[valid & col_vals]
    pos_mask[active_indices] = True
    return pos_mask, observed_mask




def chrom_sequence(fasta_root: str | Path, chrom: str) -> str:
    """Read one chromosome sequence with RAM caching."""
    if chrom in CHROM_CACHE:
        return CHROM_CACHE[chrom]

    root = Path(fasta_root)
    if root.is_file():
        candidates = [root]
    else:
        candidates = [
            root / f"Homo_sapiens.GRCh38.dna.chromosome.{chrom.removeprefix('chr')}.fa.gz",
            root / f"{chrom}.fa",
            root / f"{chrom}.fa.gz",
            root / f"{chrom}.fasta",
            root / "GRCh38.primary_assembly.genome.fa",
        ]

    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No FASTA file found for {chrom} under {root}")

    opener = gzip.open if path.suffix == ".gz" else open
    chunks: list[str] = []
    active = False
    with opener(path, "rt") as stream:
        for line in stream:
            if line.startswith(">"):
                header_chrom = line[1:].strip().split()[0]
                if header_chrom == chrom or header_chrom == chrom.removeprefix("chr"):
                    active = True
                elif active:
                    break
            elif active:
                chunks.append(line.strip())

    seq = "".join(chunks).upper()
    CHROM_CACHE[chrom] = seq
    return seq


_FASTA_INDEX: dict[str, Any] = {}


def window_sequence(fasta_path: str | Path, chrom: str, start: int, end: int) -> str:
    """Upper-case sequence of [start, end), read by seeking an indexed FASTA.

    An uncompressed FASTA is indexed once per process (extract.FastaIndex) and
    each window is a seek. A gzipped FASTA cannot be seeked, so it falls back to
    `chrom_sequence`, which scans and caches whole chromosomes.
    """
    path = Path(fasta_path)
    if not path.is_file() or path.suffix == ".gz":
        return chrom_sequence(path, chrom)[start:end]
    from ag_sae.extract import FastaIndex

    key = str(path.resolve())
    if key not in _FASTA_INDEX:
        _FASTA_INDEX[key] = FastaIndex(path)
    index = _FASTA_INDEX[key]
    name = chrom if chrom in index.records else chrom.removeprefix("chr")
    if name not in index.records:
        name = f"chr{chrom}"
    return index.fetch(name, int(start), int(end)).upper()


class FullStreamTower:
    """Intercept and edit AlphaGenome tower residual stream before downstream paths.

    Kept for the scripts under scripts/ablation. The engines now use
    `ag_sae.edit`, which edits whole batches on the GPU.
    """

    def __init__(self, model: Any, tap: Any, replace: Callable[[np.ndarray], np.ndarray]):
        if tap.kind != "tower":
            raise ValueError("Intervention requires a tower tap")
        self.tower = model.tower
        self.block = self.tower.blocks[int(tap.key)]
        self.replace = replace
        self.previous = None

    def __enter__(self) -> "FullStreamTower":
        self.previous = self.tower._forward_block

        def on_block(block, x, pair_x, compute_dtype):
            if block is self.block:
                current = x[0].detach().float().cpu().numpy()
                edited = np.asarray(self.replace(current), dtype=np.float32)
                if edited.shape != current.shape or not np.isfinite(edited).all():
                    raise ValueError("Editor produced invalid shape or non-finite values")
                replacement = x.clone()
                replacement[0] = torch.as_tensor(
                    np.ascontiguousarray(edited), dtype=x.dtype, device=x.device
                )
                return self.previous(block, replacement, pair_x, compute_dtype)
            return self.previous(block, x, pair_x, compute_dtype)

        self.tower._forward_block = on_block
        return self

    def __exit__(self, *_exc) -> None:
        if self.previous is not None:
            self.tower._forward_block = self.previous


def run_replacement(new_activation: torch.Tensor) -> Callable[[np.ndarray], np.ndarray]:
    """Factory creating an editor that replaces activation with an edited tensor."""
    replacement = new_activation[0].detach().float().cpu().numpy()

    def replace(_ignored: np.ndarray) -> np.ndarray:
        return replacement

    return replace


def checked_element_mask(labels: np.ndarray, n_positions: int = 8192) -> np.ndarray:
    """Validate a Boolean annotation mask of size n_positions."""
    mask = np.asarray(labels, dtype=bool)
    if mask.ndim != 1 or len(mask) != n_positions:
        raise ValueError(f"Mask must have length {n_positions}, got {len(mask)}")
    if not mask.any():
        raise ValueError("No element-positive bins in mask")
    return mask.copy()


@dataclass(frozen=True)
class DiscoveredFeatures:
    """Maps biological regulatory concepts to their highest-scoring SAE latent IDs."""

    promoter_feature_id: int
    insulator_feature_id: int
    enhancer_feature_id: int
    shuffled_control_feature_id: int

    def as_dict(self) -> dict[str, tuple[int, str]]:
        """Return mapping of {concept_name: (feature_id, annotation_column_name)}."""
        return {
            "PLS_Promoter": (self.promoter_feature_id, "cCRE_PLS"),
            "CTCF_Insulator": (self.insulator_feature_id, "cCRE_CTCF_only"),
            "Enhancer_pELS": (self.enhancer_feature_id, "cCRE_pELS"),
            "Shuffled_Control": (self.shuffled_control_feature_id, "shuffled_dELS"),
        }


def discover_concept_features(
    sae: Any,
    model: Any,
    tap: Any,
    calibration_windows: list[dict],
    annotations: pd.DataFrame,
    fasta_path: Path,
    organism: torch.Tensor,
    max_calibration_windows: int = 10,
) -> DiscoveredFeatures:
    """Score SAE latents on calibration windows to find top features per concept."""
    from ag_sae.edit import capture_residual

    concepts = ["cCRE_PLS", "cCRE_CTCF_only", "cCRE_pELS", "shuffled_dELS"]
    diffs: dict[str, list[np.ndarray]] = {c: [] for c in concepts}
    device = sae.channel_scale.device

    with torch.no_grad():
        for window in calibration_windows[:max_calibration_windows]:
            chrom, start, end = window["chrom"], int(window["start"]), int(window["end"])
            num_bins = (end - start) // tap.bin_bp
            seq = window_sequence(fasta_path, chrom, start, end)
            input_tensor = torch.from_numpy(sequence_to_onehot(seq).astype(np.float32))[None].to(device)

            bins = annotations.loc[
                (annotations.chrom == chrom)
                & (annotations.bin_start >= start) & (annotations.bin_start < end)
            ].sort_values("bin_start").reset_index(drop=True)

            # Only the part of the model before the tap runs.
            acts = capture_residual(model, int(tap.key), input_tensor)
            raw = acts[0].float()
            codes, _ = sae.encode_codes(raw)

            for c in concepts:
                col = c
                if c not in bins.columns and c == "cCRE_CTCF_only" and "cCRE_CTCF-only" in bins.columns:
                    col = "cCRE_CTCF-only"
                if col not in bins.columns:
                    continue

                pos_mask, observed_mask = map_annotation_column_to_grid(bins, col, start, num_bins, tap.bin_bp)
                neg_mask = observed_mask & ~pos_mask
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    pos = codes[pos_mask].mean(dim=0).detach().cpu().numpy()
                    neg = codes[neg_mask].mean(dim=0).detach().cpu().numpy()
                    diffs[c].append(pos - neg)

            del input_tensor, acts, raw, codes

    discovered: dict[str, int] = {}
    for c in concepts:
        if len(diffs[c]) > 0:
            avg_diff = np.mean(diffs[c], axis=0)
            discovered[c] = int(np.argmax(avg_diff))
        else:
            raise RuntimeError(
                f"[Feature Discovery] No contrastive bins found in calibration windows for concept '{c}'. "
                "Cannot reliably identify latent feature without contrast."
            )

    cleanup_gpu()
    return DiscoveredFeatures(
        promoter_feature_id=discovered["cCRE_PLS"],
        insulator_feature_id=discovered["cCRE_CTCF_only"],
        enhancer_feature_id=discovered["cCRE_pELS"],
        shuffled_control_feature_id=discovered["shuffled_dELS"],
    )


def read_head_summaries(output: dict, label: str, arrays: dict[str, np.ndarray]) -> None:
    """Store 128-bp resolution track predictions into the flat summary dictionary."""
    for head, val in output.items():
        if isinstance(val, dict) and 128 in val:
            tensor = val[128][0]
        elif isinstance(val, torch.Tensor):
            tensor = val[0]
        else:
            continue
        arrays[f"{head}_{label}"] = tensor.detach().float().cpu().numpy().mean(axis=-1)


def run_triad_ablation(
    sae: Any,
    model: Any,
    tap: Any,
    windows: list[dict],
    annotations: pd.DataFrame,
    features: DiscoveredFeatures,
    fasta_path: Path,
    organism: torch.Tensor,
    out_dir: Path,
    max_batch: int = 3,
) -> tuple[list[dict], dict[str, Any]]:
    """Execute Regulatory Triad Double-Dissociation ablation across test windows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = sae.channel_scale.device
    target_classes = features.as_dict()

    window_results: list[dict] = []
    t_start = time.time()

    for wi, window in enumerate(windows):
        t0 = time.time()
        chrom, start, end = window["chrom"], int(window["start"]), int(window["end"])
        num_bins = (end - start) // tap.bin_bp

        bins = annotations.loc[
            (annotations.split == "test") & (annotations.chrom == chrom)
            & (annotations.bin_start >= start) & (annotations.bin_start < end)
        ].sort_values("bin_start").reset_index(drop=True)

        intervention_mask = np.zeros(num_bins, dtype=bool)
        if "cCRE_any" in bins.columns:
            pos, _ = map_annotation_column_to_grid(bins, "cCRE_any", start, num_bins, tap.bin_bp)
            intervention_mask |= pos
        elif "cCRE_PLS" in bins.columns:
            pos, _ = map_annotation_column_to_grid(bins, "cCRE_PLS", start, num_bins, tap.bin_bp)
            intervention_mask |= pos

        if "shuffled_dELS" in bins.columns:
            pos, _ = map_annotation_column_to_grid(bins, "shuffled_dELS", start, num_bins, tap.bin_bp)
            intervention_mask |= pos

        selected_bins = np.flatnonzero(intervention_mask)
        if len(selected_bins) == 0:
            selected_bins = np.array([num_bins // 2])

        seq = window_sequence(fasta_path, chrom, start, end)
        input_tensor = torch.from_numpy(sequence_to_onehot(seq).astype(np.float32))[None].to(device)

        # Original, reconstruction and one ablation per class, batched into
        # ceil(6 / max_batch) forwards instead of six (plus no capture pass).
        conditions = [Condition("original"),
                      Condition("reconstruction", 0, selected_bins, sae_edit(sae, "reconstruction"))]
        conditions += [Condition(f"ablate_{cname}", 0, selected_bins,
                                 sae_edit(sae, f"ablate_{cname}", zero_feature(feat)))
                       for cname, (feat, _col) in target_classes.items()]
        memo: dict = {}
        profiles = run_conditions(model, int(tap.key), input_tensor, conditions,
                                  track_mean_profile(ALL_HEADS), heads=ALL_HEADS,
                                  max_batch=max_batch, memo=memo)
        arrays: dict[str, np.ndarray] = {f"{head}_{label}": profile[head]
                                         for label, profile in profiles.items()
                                         for head in ALL_HEADS}
        codes = memo["reconstruction"]

        win_data: dict[str, Any] = {
            "window_index": wi, "chrom": chrom, "start": start, "features": {}
        }

        # Effects of each class's ablation
        for cname, (feat, track_col) in target_classes.items():
            active_mask = (codes[:, feat] > 0).cpu().numpy()
            fired_count = int(active_mask.sum())

            actual_col = track_col
            if track_col not in bins.columns and track_col == "cCRE_CTCF_only" and "cCRE_CTCF-only" in bins.columns:
                actual_col = "cCRE_CTCF-only"

            if actual_col in bins.columns:
                elem_pos, _ = map_annotation_column_to_grid(bins, actual_col, start, num_bins, tap.bin_bp)
                elem_mask = elem_pos
            else:
                elem_mask = intervention_mask.copy()

            win_data["features"][cname] = {
                "feature_id": feat, "fired_bins": fired_count, "heads": {}
            }

            eval_mask = (elem_mask & intervention_mask) if (elem_mask & intervention_mask).any() else elem_mask
            has_concept_bins = bool(eval_mask.any())
            for head in ALL_HEADS:
                recon = arrays[f"{head}_reconstruction"]
                abl = arrays[f"{head}_ablate_{cname}"]
                delta = abl - recon
                win_data["features"][cname]["heads"][head] = {
                    "delta_on_concept": float(delta[eval_mask].mean()) if has_concept_bins else None,
                    "delta_on_all_regulatory": float(delta[intervention_mask].mean()) if intervention_mask.any() else 0.0,
                    "has_concept_bins": has_concept_bins,
                }

        window_results.append(win_data)
        elapsed = time.time() - t0
        p_atac = win_data["features"]["PLS_Promoter"]["heads"]["atac"]["delta_on_concept"]
        c_tf = win_data["features"]["CTCF_Insulator"]["heads"]["chip_tf"]["delta_on_concept"]
        p_str = f"{p_atac:+.4f}" if p_atac is not None else "N/A"
        c_str = f"{c_tf:+.4f}" if c_tf is not None else "N/A"
        if (wi + 1) % 5 == 0 or wi == 0:
            print(f"  [Ablation {wi+1:02d}/{len(windows)}] {chrom}:{start} | PLS ATAC: {p_str} | CTCF TF: {c_str} ({elapsed:.1f}s)", flush=True)

        del input_tensor, arrays, codes, memo, profiles

    _atomic_write_json(out_dir / "triad_40windows_results.json", window_results)

    summary_stats: dict[str, dict[str, Any]] = {}
    for cname, (feat, track_col) in target_classes.items():
        summary_stats[cname] = {"feature_id": feat, "heads": {}}
        for head in ALL_HEADS:
            deltas = np.array([
                w["features"][cname]["heads"][head]["delta_on_concept"]
                for w in window_results
                if w["features"][cname]["heads"][head]["delta_on_concept"] is not None
            ], dtype=np.float64)
            if len(deltas) >= 2:
                stats_summary = compute_paired_statistics(deltas)
                summary_stats[cname]["heads"][head] = stats_summary.to_dict()
                summary_stats[cname]["heads"][head]["n_measured_windows"] = len(deltas)
            else:
                summary_stats[cname]["heads"][head] = {
                    "mean_delta": float(np.mean(deltas)) if len(deltas) == 1 else 0.0,
                    "median_delta": float(np.median(deltas)) if len(deltas) == 1 else 0.0,
                    "t_stat": 0.0,
                    "t_p": 1.0,
                    "wilcoxon_w": 0.0,
                    "wilcoxon_p": 1.0,
                    "pct_negative": 0.0,
                    "n_measured_windows": len(deltas),
                }

    _atomic_write_json(out_dir / "regulatory_triad_summary.json", summary_stats)
    print(f"Ablation complete in {time.time()-t_start:.1f}s! Saved to {out_dir / 'regulatory_triad_summary.json'}")
    cleanup_gpu()
    return window_results, summary_stats


def match_control_norm(
    baseline: torch.Tensor, target: torch.Tensor, control: torch.Tensor, bin_index: int
) -> tuple[torch.Tensor, dict[str, float]]:
    """Match control's L2 tap-space change to target's L2 change at the specified bin."""
    if (baseline.shape != target.shape or baseline.shape != control.shape
            or baseline.ndim != 3 or baseline.shape[0] != 1
            or not 0 <= bin_index < baseline.shape[1]):
        raise ValueError("Mismatched tower activations or TSS bin index")
    target_delta = (target[0, bin_index] - baseline[0, bin_index]).float()
    control_delta = (control[0, bin_index] - baseline[0, bin_index]).float()
    target_norm = float(torch.linalg.vector_norm(target_delta))
    control_norm = float(torch.linalg.vector_norm(control_delta))
    if not np.isfinite(target_norm) or not np.isfinite(control_norm) or min(target_norm, control_norm) <= 1e-8:
        raise ValueError("Target or control has near-zero or non-finite norm")
    ratio = target_norm / control_norm
    matched = control.clone()
    matched[0, bin_index] = (baseline[0, bin_index].float() + control_delta * ratio).to(control.dtype)
    return matched, {"target_l2": target_norm, "control_before_l2": control_norm, "scale": ratio}


def matched_control_edit(sae: Any, target: int, control: int, target_level: float,
                         control_level: float):
    """Row edit for the injection control, dose-matched to the target in L2.

    Clamps `control` up to control_level, then rescales the change from the SAE
    reconstruction to the L2 size of the change that clamping `target` up to
    target_level makes at the same row (match_control_norm). If either change
    is ~0 the row is left at its reconstruction and the reason is stored as
    memo["unmeasurable"], so the caller can drop the gene.
    """
    def edit(rows: torch.Tensor, memo: dict) -> torch.Tensor:
        codes, params = sae.encode_codes(rows)
        target_codes, control_codes = codes.clone(), codes.clone()
        target_codes[:, target] = torch.clamp_min(codes[:, target], target_level)
        control_codes[:, control] = torch.clamp_min(codes[:, control], control_level)
        base = sae.decode_codes(codes, params)
        try:
            matched, _ = match_control_norm(base[None], sae.decode_codes(target_codes, params)[None],
                                            sae.decode_codes(control_codes, params)[None], 0)
        except ValueError as err:
            memo["unmeasurable"] = str(err)
            return base
        return matched[0]
    return edit


def local_edit_triplet(
    sae: Any,
    activation: torch.Tensor,
    element_mask: np.ndarray,
    target: int,
    control: int,
    mode: str,
    target_level: float,
    control_level: float,
    batch_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    """Construct SAE baseline, target edit, and control edit activations."""
    mask = checked_element_mask(element_mask, activation.shape[1])
    baseline, target_edit, control_edit = (activation.clone() for _ in range(3))
    selected = np.flatnonzero(mask)
    scale = sae.channel_scale.float().to(activation.device)

    for offset in range(0, len(selected), batch_size):
        positions = torch.as_tensor(selected[offset:offset + batch_size], dtype=torch.long, device=activation.device)
        raw = activation[0, positions].float()
        pre, params = sae.core.encode(raw / scale)
        codes = sae.core.get_sparse_activations(sae.core.activation(pre))

        target_codes, control_codes = codes.clone(), codes.clone()
        if mode == "ablate":
            target_codes[:, target] = 0
            control_codes[:, control] = 0
        else:
            target_codes[:, target] = torch.clamp_min(codes[:, target], target_level)
            control_codes[:, control] = torch.clamp_min(codes[:, control], control_level)

        rebuilt = (sae.core.decode(codes, params) * scale).to(activation.dtype)
        ch_target = (sae.core.decode(target_codes, params) * scale).to(activation.dtype)
        ch_control = (sae.core.decode(control_codes, params) * scale).to(activation.dtype)

        baseline[0, positions] = rebuilt
        target_edit[0, positions] = ch_target
        control_edit[0, positions] = ch_control

    return baseline, target_edit, control_edit, {"element_bins": int(len(selected))}


def select_tracks(strand: str) -> dict[str, list[dict]]:
    """Select strand-matched human RNA-seq, CAGE, and PRO-cap tracks."""
    cache_key = f"tracks_{strand}"
    if cache_key in _CATALOG_CACHE:
        return _CATALOG_CACHE[cache_key]

    if "human" not in _CATALOG_CACHE:
        from alphagenome_pytorch.named_outputs import TrackMetadataCatalog
        _CATALOG_CACHE["human"] = TrackMetadataCatalog.load_builtin("human")
    catalog = _CATALOG_CACHE["human"]

    selected = {}
    for output_name in ("rna_seq", "cage", "procap"):
        tracks = []
        for track in catalog.get_tracks(output_name, organism=0):
            extras = track.extras
            if extras.get("strand") != strand or extras.get("genetically_modified"):
                continue
            if output_name == "rna_seq" and extras.get("assay_title") != "total RNA-seq":
                continue
            tracks.append({
                "index": int(track.track_index), "name": track.track_name,
                "biosample": extras.get("biosample_name"), "assay": extras.get("assay_title"),
                "strand": strand,
            })
        if not tracks:
            raise ValueError(f"No unique tracks found for {output_name}")
        selected[output_name] = tracks
    _CATALOG_CACHE[cache_key] = selected
    return selected


def read_mane_transcripts(gtf: Path, chromosomes: set[str]) -> list[dict]:
    """Parse protein-coding MANE Select transcripts from GTF."""
    opener = gzip.open if gtf.suffix == ".gz" else open
    transcripts = []
    with opener(gtf, "rt") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t", 8)
            if len(fields) != 9 or fields[2] != "transcript" or fields[0] not in chromosomes:
                continue
            attrs = {p.strip().split(" ", 1)[0]: p.strip().split(" ", 1)[1].strip('"')
                     for p in fields[8].strip().split(";") if len(p.strip().split(" ", 1)) == 2}
            if ("MANE_Select" not in fields[8] or attrs.get("gene_type") != "protein_coding"
                    or attrs.get("transcript_type") != "protein_coding"):
                continue
            strand = fields[6]
            start0, end0 = int(fields[3]) - 1, int(fields[4])
            transcripts.append({
                "chrom": fields[0], "start0": start0, "end0": end0,
                "tss0": start0 if strand == "+" else end0 - 1,
                "strand": strand, "gene_id": attrs["gene_id"],
                "gene_name": attrs.get("gene_name", attrs["gene_id"]),
                "transcript_id": attrs["transcript_id"],
            })
    return transcripts


def read_selected_exons(gtf: Path, transcript_ids: set[str]) -> dict[str, list[tuple[int, int]]]:
    """Extract exon coordinates for candidate MANE transcripts."""
    opener = gzip.open if gtf.suffix == ".gz" else open
    exons: dict[str, list[tuple[int, int]]] = {t: [] for t in transcript_ids}
    with opener(gtf, "rt") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t", 8)
            if len(fields) != 9 or fields[2] != "exon":
                continue
            attrs = {p.strip().split(" ", 1)[0]: p.strip().split(" ", 1)[1].strip('"')
                     for p in fields[8].strip().split(";") if len(p.strip().split(" ", 1)) == 2}
            t_id = attrs.get("transcript_id")
            if t_id in exons:
                exons[t_id].append((int(fields[3]) - 1, int(fields[4])))

    merged = {}
    for t_id, intervals in exons.items():
        parts: list[list[int]] = []
        for s, e in sorted(intervals):
            if parts and s <= parts[-1][1]:
                parts[-1][1] = max(parts[-1][1], e)
            else:
                parts.append([s, e])
        merged[t_id] = [(s, e) for s, e in parts]
    return merged


def eligible_genes(
    gtf: Path,
    manifest: pd.DataFrame,
    annotations: pd.DataFrame,
    cache_dir: Path | None = None,
) -> list[dict]:
    """Select held-out MANE Select genes meeting coordinate and isolation criteria.

    Uses deterministic disk caching based on GTF metadata, manifest size, and
    annotation split distribution to avoid re-parsing the large GTF file.
    """
    gtf_path = Path(gtf)
    try:
        st = gtf_path.stat()
        gtf_info = f"{gtf_path.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        gtf_info = str(gtf_path)

    manifest_cols = [c for c in ["chrom", "win_start", "start", "end", "win_end", "split"] if c in manifest.columns]
    if manifest_cols:
        m_bytes = pd.util.hash_pandas_object(manifest[manifest_cols], index=False).to_numpy().tobytes()
        manifest_sig = hashlib.sha256(m_bytes).hexdigest()[:16]
    else:
        manifest_sig = str(len(manifest))

    ann_cols = [c for c in ["chrom", "bin_start", "cCRE_PLS", "split"] if c in annotations.columns]
    if ann_cols:
        a_bytes = pd.util.hash_pandas_object(annotations[ann_cols], index=False).to_numpy().tobytes()
        ann_sig = hashlib.sha256(a_bytes).hexdigest()[:16]
    else:
        ann_sig = str(len(annotations))

    key_str = f"gtf:{gtf_info}|m_sig:{manifest_sig}|a_sig:{ann_sig}|m_len:{len(manifest)}"
    cache_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

    cache_root = Path(cache_dir) if cache_dir is not None else Path("/tmp/ag_cache")
    cache_file = cache_root / f"eligible_genes_{cache_hash}.json"

    if cache_file.is_file():
        try:
            cached_data = json.loads(cache_file.read_text())
            for r in cached_data:
                if "exons_0based_halfopen" in r:
                    r["exons_0based_halfopen"] = [tuple(pair) for pair in r["exons_0based_halfopen"]]
            return cached_data
        except (json.JSONDecodeError, OSError):
            pass

    WINDOW_BP = 1_048_576
    BIN_BP = 128
    EDGE_MARGIN_BP = 131_072

    by_chrom = {chrom: part.sort_values("bin_start") for chrom, part in
                annotations.loc[annotations.split == "test"].groupby("chrom")}
    grids = {}
    for row in manifest.loc[manifest.split == "test"].itertuples():
        start, end = int(row.win_start), int(row.win_end)
        if end - start != WINDOW_BP or row.chrom not in by_chrom:
            continue
        chrom_rows = by_chrom[row.chrom]
        starts = chrom_rows.bin_start.to_numpy(dtype=np.int64)
        left, right = starts.searchsorted([start, end])
        bins = chrom_rows.iloc[left:right]
        if len(bins) != WINDOW_BP // BIN_BP:
            continue
        grids[(row.chrom, start)] = {"end": end, "pls": bins.cCRE_PLS.to_numpy(bool)}

    transcripts = read_mane_transcripts(gtf_path, {chrom for chrom, _ in grids})
    preliminary = []
    for t in transcripts:
        for (chrom, win_start), grid in grids.items():
            if chrom != t["chrom"]:
                continue
            if not (win_start + EDGE_MARGIN_BP <= t["tss0"] < grid["end"] - EDGE_MARGIN_BP
                    and win_start <= t["start0"] and t["end0"] <= grid["end"]):
                continue
            tss_bin = (t["tss0"] - win_start) // BIN_BP
            if grid["pls"][tss_bin]:
                continue
            preliminary.append(dict(t, window_start=win_start, window_end=grid["end"], tss_bin_index=int(tss_bin)))
            break

    exons = read_selected_exons(gtf_path, {r["transcript_id"] for r in preliminary})
    eligible = []
    for r in preliminary:
        t_exons = exons.get(r["transcript_id"], [])
        if not t_exons:
            continue
        weights = []
        for es, ee in t_exons:
            for bi in range((es - r["window_start"]) // BIN_BP, (ee - 1 - r["window_start"]) // BIN_BP + 1):
                bs = r["window_start"] + BIN_BP * bi
                weights.append({"bin_index": bi, "exon_bp": min(ee, bs + BIN_BP) - max(es, bs)})
        if sum(w["exon_bp"] for w in weights) < 2048 or len(weights) < 16:
            continue
        r["exons_0based_halfopen"] = t_exons
        r["exon_bin_weights"] = weights
        r["selection_hash"] = hashlib.sha256(f"{r['gene_id']}|{r['transcript_id']}".encode()).hexdigest()
        eligible.append(r)

    eligible.sort(key=lambda r: (r["selection_hash"], r["gene_id"]))

    try:
        _atomic_write_json(cache_file, eligible)
    except OSError:
        pass

    return eligible


def summarize_gene_effects(readouts: dict[str, np.ndarray], tracks: dict) -> dict:
    """Compute per-modality paired difference and log2 fold change across tracks."""
    summary = {}
    for head in ("rna_seq", "cage", "procap"):
        original = readouts[f"{head}_original"]
        baseline = readouts[f"{head}_reconstruction"]
        target = readouts[f"{head}_target"]
        control = readouts[f"{head}_control"]

        offset = 0.001 if head == "rna_seq" else 1.0
        target_lfc = np.log2((target + offset) / (baseline + offset))
        control_lfc = np.log2((control + offset) / (baseline + offset))
        diff = target - control

        summary[head] = {
            "n_tracks": len(tracks[head]),
            "target_median_log2_ratio": float(np.median(target_lfc)),
            "control_median_log2_ratio": float(np.median(control_lfc)),
            "paired_target_minus_control_median": float(np.median(target_lfc - control_lfc)),
            "paired_difference": float(np.median(diff)),
            "fraction_target_increase": float(np.mean(target_lfc > 0)),
        }
    return summary


def extract_transcriptional_readout(
    output_dict: dict, head: str, indices: list[int], gene: dict, batch_index: int = 0
) -> np.ndarray:
    """Extract modality-specific transcriptional readout per track.

    - PRO-cap & CAGE: 5-bin sum centered at the TSS: [tss_bin - 2, tss_bin + 2] inclusive
    - RNA-seq: Exon-weighted average over exonic bins
    """
    value = output_dict[head][128] if isinstance(output_dict[head], dict) else output_dict[head]
    tensor = value[batch_index]
    tss_bin = int(gene["tss_bin_index"])

    if head in ("procap", "cage"):
        start_bin = max(0, tss_bin - 2)
        end_bin = min(tensor.shape[0], tss_bin + 3)
        return tensor[start_bin:end_bin, indices].sum(dim=0).cpu().numpy().astype(np.float32)
    elif head == "rna_seq":
        weights = gene.get("exon_bin_weights", [])
        if weights and torch is not None and isinstance(tensor, torch.Tensor):
            bin_idx = torch.tensor([w["bin_index"] for w in weights], device=tensor.device, dtype=torch.long)
            bp_w = torch.tensor([w["exon_bp"] for w in weights], device=tensor.device, dtype=torch.float32).unsqueeze(1)
            exon_signals = tensor[bin_idx, :][:, indices]
            weighted_mean = (exon_signals * bp_w).sum(dim=0) / bp_w.sum()
            return weighted_mean.cpu().numpy().astype(np.float32)
        else:
            return tensor[:, indices].mean(dim=0).cpu().numpy().astype(np.float32)
    else:
        return tensor[:, indices].mean(dim=0).cpu().numpy().astype(np.float32)


def run_gene_injection_cohort(
    sae: Any,
    model: Any,
    tap: Any,
    manifest: pd.DataFrame,
    annotations: pd.DataFrame,
    gtf_path: Path,
    fasta_path: Path,
    promoter_feature_id: int,
    control_feature_id: int,
    organism: torch.Tensor,
    out_dir: Path,
    cache_dir: Path | None = None,
    max_batch: int = 4,
) -> tuple[list[dict], dict[str, Any]]:
    """Execute MANE Select Gene-TSS injection across eligible protein-coding genes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    device = sae.channel_scale.device
    candidates = eligible_genes(gtf_path, manifest, annotations, cache_dir=cache_dir)
    n_genes = len(candidates)

    target_level = float(sae.core.activation(torch.tensor([5.0])).item()) if hasattr(sae.core, "activation") else 5.0
    control_level = target_level * 0.7

    gene_results: list[dict] = []
    t_start = time.time()

    for rank, gene in enumerate(candidates):
        tg0 = time.time()
        chrom, start, end = gene["chrom"], gene["window_start"], gene["window_end"]
        gene_name, strand, tss0 = gene["gene_name"], gene["strand"], gene["tss0"]
        tss_bin = (tss0 - start) // 128
        tracks = select_tracks(strand)

        mask = np.zeros(8192, dtype=bool)
        mask[tss_bin] = True

        seq = window_sequence(fasta_path, chrom, start, end)
        input_tensor = torch.from_numpy(sequence_to_onehot(seq).astype(np.float32))[None].to(device)

        # Original, reconstruction, target injection and the L2-matched control,
        # in one batched forward (the first campaign used four, plus a capture).
        ctrl_feat = control_feature_id if control_feature_id != promoter_feature_id else (promoter_feature_id + 1)
        tss = [int(tss_bin)]
        conditions = [
            Condition("original"),
            Condition("reconstruction", 0, tss, sae_edit(sae, "reconstruction")),
            Condition("target", 0, tss, sae_edit(sae, "target",
                                                 clamp_min_feature(promoter_feature_id, target_level))),
            Condition("control", 0, tss, matched_control_edit(
                sae, promoter_feature_id, ctrl_feat, target_level, control_level)),
        ]
        track_indices = {head: [int(t["index"]) for t in tracks[head]] for head in TRANSCRIPTIONAL_HEADS}

        def gene_readout(outputs, b, gene=gene):
            return {head: extract_transcriptional_readout(outputs, head, track_indices[head], gene, b)
                    for head in TRANSCRIPTIONAL_HEADS}

        memo: dict = {}
        per_condition = run_conditions(model, int(tap.key), input_tensor, conditions, gene_readout,
                                       heads=TRANSCRIPTIONAL_HEADS, max_batch=max_batch, memo=memo)
        if "unmeasurable" in memo:
            gene_results.append({"rank": rank, "gene_name": gene_name, "status": "unmeasurable",
                                 "reason": memo["unmeasurable"]})
            continue
        readouts: dict[str, np.ndarray] = {f"{head}_{label}": values[head]
                                           for label, values in per_condition.items()
                                           for head in TRANSCRIPTIONAL_HEADS}

        effects = summarize_gene_effects(readouts, tracks)
        np.savez_compressed(out_dir / f"gene_rank_{rank:02d}_{gene_name}.npz", **readouts)
        gene_results.append({
            "rank": rank, "gene_name": gene_name, "chrom": chrom, "tss0": tss0,
            "status": "measured", "effects": effects,
        })

        if (rank + 1) % 5 == 0 or rank == 0:
            rna_d = effects["rna_seq"]["paired_difference"]
            pro_d = effects["procap"]["paired_difference"]
            print(f"  [Gene {rank+1:02d}/{n_genes}] {gene_name:<10} | RNA Diff: {rna_d:+.4f} | PRO-cap Diff: {pro_d:+.4f} ({time.time()-tg0:.1f}s)", flush=True)

        del input_tensor, readouts, per_condition, memo

    _atomic_write_json(out_dir / "gene_cohort_results.json", gene_results)

    measured_genes = [g for g in gene_results if g.get("status") == "measured"]
    gene_stats: dict[str, Any] = {
        "n_total_eligible": n_genes, "n_measured": len(measured_genes), "modalities": {}
    }

    if len(measured_genes) == 0:
        for m in TRANSCRIPTIONAL_HEADS:
            gene_stats["modalities"][m] = {
                "mean_paired_difference": 0.0,
                "median_paired_difference": 0.0,
                "t_stat": 0.0,
                "t_p_value": 1.0,
                "wilcoxon_w": 0.0,
                "wilcoxon_p_value": 1.0,
                "fraction_target_greater_than_control": 0.0,
            }
    else:
        for m in TRANSCRIPTIONAL_HEADS:
            diffs = [g["effects"][m]["paired_difference"] for g in measured_genes]
            stats_summary = compute_paired_statistics(np.array(diffs))
            gene_stats["modalities"][m] = {
                "mean_paired_difference": stats_summary.mean_delta,
                "median_paired_difference": stats_summary.median_delta,
                "t_stat": stats_summary.t_statistic,
                "t_p_value": stats_summary.t_p_value,
                "wilcoxon_w": stats_summary.wilcoxon_w,
                "wilcoxon_p_value": stats_summary.wilcoxon_p_value,
                "fraction_target_greater_than_control": float(np.mean(np.array(diffs) > 0)),
            }

    _atomic_write_json(out_dir / "gene_level_statistical_summary.json", gene_stats)
    print(f"Gene cohort complete in {time.time()-t_start:.1f}s! Saved to {out_dir / 'gene_level_statistical_summary.json'}")
    cleanup_gpu()
    return gene_results, gene_stats


def run_single_configuration(
    fold: str,
    layer: str,
    seed: str,
    weights_path: Path,
    sae_path: Path,
    manifest_path: Path,
    annotations_path: Path,
    fasta_path: Path,
    gtf_path: Path,
    out_dir: Path,
    plan_path: Path | None = None,
    cache_dir: Path | None = None,
    max_batch: int = 3,
) -> None:
    """Orchestrate evaluation for a single (fold, layer, seed) configuration."""
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from ag_sae.extract import TAPS
    from ag_sae.sae import BorzoiSAE

    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print(f"RUNNING CONFIGURATION: {fold} | {layer} | {seed}")
    print(f"Output directory: {out_dir}")
    print("=" * 80)

    # 1. Load metadata
    manifest = pd.read_parquet(manifest_path)
    annotations = pd.read_parquet(annotations_path)
    if plan_path and plan_path.is_file():
        print(f"Loading planned test windows from {plan_path}...")
        plan_data = json.loads(plan_path.read_text())
        plan_windows = plan_data.get("windows", [])
        # Validate planned windows against manifest test split for this fold
        test_manifest_windows = {
            (row.chrom, int(row.win_start), int(row.win_end))
            for row in manifest.loc[manifest.split == "test"].itertuples()
        }
        valid_windows = [
            w for w in plan_windows
            if (w["chrom"], int(w["start"]), int(w["end"])) in test_manifest_windows
        ]
        if len(valid_windows) < len(plan_windows):
            print(f"[Warning] Plan contained {len(plan_windows) - len(valid_windows)} windows not in the '{fold}' test split. Retaining {len(valid_windows)} matching test windows.")
        if not valid_windows:
            raise ValueError(f"None of the windows in plan {plan_path} belong to the '{fold}' test split.")
        windows = valid_windows
    else:
        windows = [
            {"chrom": row.chrom, "start": int(row.win_start), "end": int(row.win_end)}
            for row in manifest.loc[manifest.split == "test"].head(40).itertuples()
        ]

    # Use strictly disjoint calibration windows for feature discovery to guarantee zero data leakage
    # Calibration windows are drawn from manifest windows that are strictly disjoint from the evaluation cohort (windows)
    eval_keys = {(w["chrom"], int(w["start"]), int(w["end"])) for w in windows}
    # annotation_main.parquet provides annotations for the test partition; select disjoint test windows outside eval cohort
    test_slice = manifest.loc[manifest.split == "test"]
    calib_candidates = [
        {"chrom": row.chrom, "start": int(row.win_start), "end": int(row.win_end)}
        for row in test_slice.itertuples()
        if (row.chrom, int(row.win_start), int(row.win_end)) not in eval_keys
    ]
    if not calib_candidates:
        raise RuntimeError("Zero data leakage violated: no disjoint calibration windows available outside evaluation cohort.")
    calib_windows = calib_candidates[:10]

    # 2. Load models
    print(f"Loading SAE from {sae_path}...")
    tap = TAPS[layer]
    sae = BorzoiSAE.from_checkpoint(sae_path, device="cuda")

    print(f"Loading AlphaGenome from {weights_path}...")
    model = AlphaGenome.from_pretrained(weights_path, dtype_policy=DtypePolicy.mixed_precision(), device="cuda")
    model.eval().requires_grad_(False)
    organism = torch.tensor([0], device="cuda")

    # 3. Dynamic Feature Discovery
    features = discover_concept_features(sae, model, tap, calib_windows, annotations, fasta_path, organism)

    # 4. Regulatory Triad Ablation
    triad_summary_file = out_dir / "regulatory_triad_summary.json"
    if is_valid_json_file(triad_summary_file):
        print(f"[Ablation] Found valid {triad_summary_file}, skipping Part 1...")
    else:
        run_triad_ablation(sae, model, tap, windows, annotations, features, fasta_path, organism, out_dir,
                           max_batch=max_batch)

    # 5. Gene-TSS Injection Cohort
    gene_summary_file = out_dir / "gene_level_statistical_summary.json"
    if is_valid_json_file(gene_summary_file):
        print(f"[Injection] Found valid {gene_summary_file}, skipping Part 2...")
    else:
        run_gene_injection_cohort(
            sae=sae, model=model, tap=tap, manifest=manifest, annotations=annotations,
            gtf_path=gtf_path, fasta_path=fasta_path,
            promoter_feature_id=features.promoter_feature_id,
            control_feature_id=features.insulator_feature_id,
            organism=organism, out_dir=out_dir, cache_dir=cache_dir, max_batch=max_batch
        )

    print(f"\nConfiguration {fold} {layer} {seed} completed successfully in {time.time()-t_start:.1f}s!\n")


def cmd_intervene(argv: list[str] | None = None) -> int:
    """CLI dispatcher for ag_sae intervene."""
    parser = argparse.ArgumentParser(description="AlphaGenome-SAE Causal Interventions CLI")
    parser.add_argument("--fold", type=str, default="fold1")
    parser.add_argument("--layer", type=str, required=True, choices=["resid_pre_b0", "resid_pre_b4", "resid_pre_b8"])
    parser.add_argument("--seed", type=str, required=True, choices=["seed0", "seed1", "seed2"])
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--sae", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--fasta", type=Path, default=Path("/mnt/ag/data/GRCh38.primary_assembly.genome.fa"))
    parser.add_argument("--gtf", type=Path, default=Path("/mnt/ag/data/gencode.v50.basic.annotation.gtf.gz"))
    parser.add_argument("--cache-dir", type=Path, default=None, help="Directory to cache eligible genes JSON")
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sae-pool", type=str, default="sae",
                        help="SAE directory under /mnt/ag (sae, sae_k64, sae_v2, ...)")
    parser.add_argument("--max-batch", type=int, default=3,
                        help="conditions per forward; 3-4 on an 80 GB A100, 1-2 on 40 GB")

    args = parser.parse_args(argv)
    fold_num = args.fold.replace("fold", "")
    weights = args.weights or Path(f"/mnt/ag/weights/model_fold_{fold_num}.safetensors")
    sae = args.sae or Path(f"/mnt/ag/{args.sae_pool}/{args.fold}/{args.layer}_{args.seed}/best.pt")
    campaign = "campaign_matrix" if args.sae_pool == "sae" else f"campaign_matrix_{args.sae_pool}"
    out = args.out or Path(f"/mnt/ag/analysis_runs/{campaign}/{args.fold}_{args.layer}_{args.seed}")

    manifest_candidates = [
        Path(f"/mnt/ag/analysis-code-a1f31ce/manifests/window_manifest_fold{fold_num}.parquet"),
        Path(f"/home/azizcetin/ag/manifests/window_manifest_fold{fold_num}.parquet"),
        Path(f"/mnt/ag/manifests/window_manifest_fold{fold_num}.parquet"),
    ]
    manifest = args.manifest or next((p for p in manifest_candidates if p.is_file()), manifest_candidates[0])

    ann_candidates = [
        Path(f"/mnt/ag/analysis_runs/inputs/full128/{args.fold}/resid_pre_b8/annotation_main.parquet"),
        Path(f"/mnt/ag/analysis-code-a1f31ce/tmp/annotation_matrix_fold{fold_num}.parquet"),
    ]
    annotations = args.annotations or next((p for p in ann_candidates if p.is_file()), ann_candidates[0])

    fold_plan = Path(f"/mnt/ag/analysis-code-a1f31ce/tmp/local_plan_{args.fold}.json")
    default_plan = Path("/mnt/ag/analysis-code-a1f31ce/tmp/local_plan.json")
    plan = args.plan or (fold_plan if fold_plan.is_file() else (default_plan if default_plan.is_file() and args.fold == "fold1" else None))

    run_single_configuration(
        fold=args.fold, layer=args.layer, seed=args.seed,
        weights_path=weights, sae_path=sae,
        manifest_path=manifest, annotations_path=annotations,
        fasta_path=args.fasta, gtf_path=args.gtf,
        out_dir=out, plan_path=plan, cache_dir=args.cache_dir, max_batch=args.max_batch
    )
    return 0


if __name__ == "__main__":
    sys.exit(cmd_intervene())

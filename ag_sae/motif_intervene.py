"""Causal motif intervention engine and campaign runner for AlphaGenome-SAE.

Implements the 2x2 Factorial Causal Mediation Protocol:
  Condition 1 (WT): Natural sequence, baseline unperturbed residual stream (Y_W)
  Condition 2 (ISM): In Silico Mutagenesis on core motif anchor (Y_M)
  Condition 3 (SAE Ablation): Natural sequence with target TF latent ablated (Y_W_abl)
  Condition 4 (SAE Rescue): Mutated sequence with target TF latent restored (Y_M_rescue)

Evaluates:
  - Epigenomic heads: ChIP-TF, ATAC, DNase, CAGE, PRO-cap
  - Mediation ratio: Delta_abl / Delta_ISM
  - Rescue ratio: Delta_rescue / (Y_W - Y_M)
  - Paired hypothesis testing via Wilcoxon signed-rank and Student's t-test
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
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

from ag_sae.extract import TAPS
from ag_sae.motifs import JasparMotif, MotifHit, read_jaspar_records, scan_sequence_for_motifs
from ag_sae.stats import StatisticalSummary, compute_paired_statistics, safe_wilcoxon

DEFAULT_LAYERS: tuple[str, ...] = ("resid_pre_b8", "resid_pre_b4", "resid_pre_b0")
DEFAULT_SEEDS: tuple[str, ...] = ("seed0", "seed1", "seed2")
ALL_FOLDS: tuple[str, ...] = ("fold0", "fold1", "fold2", "fold3")

# Built-in canonical CTCF PFM (JASPAR CORE MA0139.1) as fail-safe fallback
CANONICAL_CTCF_COUNTS = np.array([
    [87., 291., 76., 760.],
    [167., 145., 414., 488.],
    [281., 49., 446., 438.],
    [56., 800., 92., 266.],
    [8., 1203., 8., 5.],
    [30., 1195., 35., 3.],
    [293., 39., 832., 50.],
    [155., 483., 29., 547.],
    [67., 782., 283., 82.],
    [131., 513., 43., 527.],
    [2., 1205., 6., 1.],
    [102., 19., 950., 143.],
    [343., 11., 799., 61.],
    [129., 308., 463., 314.],
    [147., 482., 286., 299.],
    [84., 339., 379., 412.],
    [16., 747., 125., 326.],
    [82., 480., 248., 404.],
    [202., 336., 296., 380.],
], dtype=np.float32)

MOTIF_HEADS: tuple[str, ...] = ("chip_tf", "atac", "dnase", "cage", "procap")

CHROM_CACHE: dict[str, str] = {}


def cleanup_gpu() -> None:
    """Trigger garbage collection and empty PyTorch CUDA cache."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def sequence_to_onehot(sequence: str) -> np.ndarray:
    """Convert nucleotide string to (L, 4) float32 one-hot matrix."""
    try:
        from alphagenome_pytorch.utils.sequence import sequence_to_onehot as _up_ohe
        return _up_ohe(sequence)
    except ImportError:
        codes = np.frombuffer(sequence.upper().encode(), dtype=np.uint8)
        out = np.zeros((len(codes), 4), dtype=np.float32)
        for i, b in enumerate("ACGT"):
            out[codes == ord(b), i] = 1.0
        return out


def chrom_sequence(path: Path, chrom: str) -> str:
    """Read full chromosome sequence into memory with RAM caching."""
    if chrom in CHROM_CACHE:
        return CHROM_CACHE[chrom]
    import gzip
    opener = gzip.open if Path(path).suffix == ".gz" else open
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


class FullStreamTower:
    """Intercept and edit AlphaGenome tower residual stream before downstream paths."""

    def __init__(self, model: Any, tap: Any, replace: Callable[[np.ndarray | torch.Tensor], np.ndarray | torch.Tensor]):
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
                edited = self.replace(current)
                if isinstance(edited, torch.Tensor):
                    replacement = x.clone()
                    replacement[0] = edited.to(device=x.device, dtype=x.dtype)
                    return self.previous(block, replacement, pair_x, compute_dtype)
                edited = np.asarray(edited, dtype=np.float32)
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



def is_valid_json_file(path: Path | str | None) -> bool:
    """Check if path points to an existing, non-empty, parseable JSON file."""
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


def is_valid_motif_summary(path: Path | str | None, min_windows: int = 5) -> bool:
    """Check if path points to a completed motif summary with sufficient measured windows."""
    if not is_valid_json_file(path):
        return False
    try:
        data = json.loads(Path(path).read_text())
        return int(data.get("n_measured_windows", 0)) >= min_windows
    except Exception:
        return False



def get_canonical_ctcf_motif(jaspar_file: Path | None = None) -> JasparMotif:
    """Load canonical CTCF JasparMotif object from file or built-in matrix."""
    if jaspar_file and jaspar_file.is_file():
        try:
            records = read_jaspar_records(jaspar_file)
            if "CTCF" in records:
                return records["CTCF"]
            # Look for MA0139 or any CTCF accession
            for k, m in records.items():
                if "CTCF" in m.name.upper() or "MA0139" in m.matrix_id:
                    return m
        except Exception:
            pass
    return JasparMotif(matrix_id="MA0139.1", name="CTCF", counts=CANONICAL_CTCF_COUNTS)


def mutate_motif_anchor(seq: str, hit: MotifHit, rng: np.random.Generator) -> str:
    """Generate in silico mutagenesis (ISM) sequence by shuffling the motif core."""
    motif_chars = list(hit.sequence)
    # Permute until different from wild-type sequence
    shuffled = "".join(rng.permutation(motif_chars))
    attempts = 0
    while shuffled == hit.sequence and attempts < 10:
        shuffled = "".join(rng.permutation(motif_chars))
        attempts += 1
    return seq[:hit.start] + shuffled + seq[hit.end:]


def extract_head_tensor(val: Any) -> torch.Tensor:
    """Extract tensor from model head prediction (dict[int, Tensor] or Tensor)."""
    if isinstance(val, dict) and 128 in val:
        return val[128]
    elif isinstance(val, torch.Tensor):
        return val
    raise ValueError(f"Unexpected prediction output format: {type(val)}")



def run_motif_causal_cohort(
    sae: Any,
    model: Any,
    tap: Any,
    windows: list[dict],
    fasta_path: Path,
    organism: torch.Tensor,
    target_feature_id: int,
    motif: JasparMotif,
    out_dir: Path,
    min_score_ratio: float = 0.5,
    max_measured_windows: int = 25,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute 2x2 factorial causal mediation protocol across genomic windows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scale = sae.channel_scale.float().to("cuda")
    forward_args = {"resolutions": (128,), "heads": MOTIF_HEADS, "channels_last": True}

    window_results: list[dict] = []
    measured_count = 0
    t_start = time.time()

    print(f"Scanning up to {len(windows)} windows for canonical {motif.name} ({motif.matrix_id}) anchors...")

    for wi, window in enumerate(windows):
        if measured_count >= max_measured_windows:
            break

        chrom, start, end = window["chrom"], int(window["start"]), int(window["end"])
        seq = chrom_sequence(fasta_path, chrom)[start:end]

        # 1. Scan sequence for motif hits
        hits = scan_sequence_for_motifs(seq, motif, min_score_ratio=min_score_ratio)
        if not hits:
            continue

        # Select highest-scoring motif hit
        hit = max(hits, key=lambda h: h.score)
        anchor_bin = hit.start // tap.bin_bp

        # Boundary check
        if anchor_bin >= 8192:
            continue

        # 2. Mutate DNA at motif anchor (In Silico Mutagenesis)
        mut_seq = mutate_motif_anchor(seq, hit, rng)

        wt_onehot = sequence_to_onehot(seq).astype(np.float32)
        mut_onehot = sequence_to_onehot(mut_seq).astype(np.float32)

        wt_tensor = torch.from_numpy(np.ascontiguousarray(wt_onehot[None])).to("cuda")
        mut_tensor = torch.from_numpy(np.ascontiguousarray(mut_onehot[None])).to("cuda")

        # ------------------------------------------------------------------
        # Condition 1: Wild-Type (WT) baseline reconstruction
        # ------------------------------------------------------------------
        captured_wt: list[np.ndarray] = []

        def cap_wt(v):
            captured_wt.append(np.array(v, dtype=np.float32, copy=True))
            return v

        with torch.inference_mode(), FullStreamTower(model, tap, cap_wt):
            _ = model.predict(wt_tensor, organism, **forward_args)

        raw_wt = torch.from_numpy(np.ascontiguousarray(captured_wt[0]))[None].to("cuda")
        pos_tensor = torch.tensor([anchor_bin], dtype=torch.long, device="cuda")
        raw_anchor = raw_wt[0, pos_tensor].float()
        pre_wt, params_wt = sae.core.encode(raw_anchor / scale)
        wt_codes = sae.core.get_sparse_activations(sae.core.activation(pre_wt))
        rebuilt_wt = (sae.core.decode(wt_codes, params_wt) * scale).to(raw_wt.dtype)

        wt_act = raw_wt.clone()
        wt_act[0, pos_tensor] = rebuilt_wt

        with torch.inference_mode(), FullStreamTower(model, tap, lambda _: wt_act[0]):
            out_wt = model.predict(wt_tensor, organism, **forward_args)

        # ------------------------------------------------------------------
        # Condition 2: In Silico Mutagenesis (ISM) on motif
        # ------------------------------------------------------------------
        captured_mut: list[np.ndarray] = []

        def cap_mut(v):
            captured_mut.append(np.array(v, dtype=np.float32, copy=True))
            return v

        with torch.inference_mode(), FullStreamTower(model, tap, cap_mut):
            _ = model.predict(mut_tensor, organism, **forward_args)

        raw_mut = torch.from_numpy(np.ascontiguousarray(captured_mut[0]))[None].to("cuda")
        raw_mut_anchor = raw_mut[0, pos_tensor].float()
        pre_mut, params_mut = sae.core.encode(raw_mut_anchor / scale)
        mut_codes = sae.core.get_sparse_activations(sae.core.activation(pre_mut))
        rebuilt_mut = (sae.core.decode(mut_codes, params_mut) * scale).to(raw_mut.dtype)

        mut_act = raw_mut.clone()
        mut_act[0, pos_tensor] = rebuilt_mut

        with torch.inference_mode(), FullStreamTower(model, tap, lambda _: mut_act[0]):
            out_ism = model.predict(mut_tensor, organism, **forward_args)

        # ------------------------------------------------------------------
        # Condition 3: SAE Latent Ablation on WT sequence
        # ------------------------------------------------------------------
        abl_codes = wt_codes.clone()
        abl_codes[:, target_feature_id] = 0.0
        rebuilt_abl = (sae.core.decode(abl_codes, params_wt) * scale).to(raw_wt.dtype)

        abl_act = raw_wt.clone()
        abl_act[0, pos_tensor] = rebuilt_abl

        with torch.inference_mode(), FullStreamTower(model, tap, lambda _: abl_act[0]):
            out_abl = model.predict(wt_tensor, organism, **forward_args)

        # ------------------------------------------------------------------
        # Condition 4: SAE Latent Rescue on Mutant sequence
        # ------------------------------------------------------------------
        # Clamp/restore the target feature in mutant to its WT firing level
        rescue_codes = mut_codes.clone()
        rescue_codes[:, target_feature_id] = wt_codes[:, target_feature_id]
        rebuilt_rescue = (sae.core.decode(rescue_codes, params_mut) * scale).to(raw_mut.dtype)

        rescue_act = raw_mut.clone()
        rescue_act[0, pos_tensor] = rebuilt_rescue

        with torch.inference_mode(), FullStreamTower(model, tap, lambda _: rescue_act[0]):
            out_rescue = model.predict(mut_tensor, organism, **forward_args)

        # ------------------------------------------------------------------
        # Extract Readouts across heads at the anchor bin neighborhood
        # ------------------------------------------------------------------
        eval_window_bins = slice(max(0, anchor_bin - 1), min(8192, anchor_bin + 2))
        win_metrics: dict[str, Any] = {
            "chrom": chrom,
            "start": start,
            "end": end,
            "anchor_start": hit.start,
            "anchor_end": hit.end,
            "anchor_strand": hit.strand,
            "anchor_score": hit.score,
            "anchor_seq": hit.sequence,
            "anchor_bin": anchor_bin,
            "wt_latent_val": float(wt_codes[0, target_feature_id].item()),
            "mut_latent_val": float(mut_codes[0, target_feature_id].item()),
            "heads": {},
        }

        for head in MOTIF_HEADS:
            t_wt = extract_head_tensor(out_wt[head])
            t_ism = extract_head_tensor(out_ism[head])
            t_abl = extract_head_tensor(out_abl[head])
            t_rescue = extract_head_tensor(out_rescue[head])

            y_wt = t_wt[0, eval_window_bins].mean().item()
            y_ism = t_ism[0, eval_window_bins].mean().item()
            y_abl = t_abl[0, eval_window_bins].mean().item()
            y_rescue = t_rescue[0, eval_window_bins].mean().item()

            delta_ism = y_ism - y_wt
            delta_abl = y_abl - y_wt
            delta_rescue = y_rescue - y_ism

            # Causal Mediation & Rescue Ratios
            mediation = (delta_abl / delta_ism) if abs(delta_ism) > 1e-4 else 0.0
            rescue_rate = (delta_rescue / (y_wt - y_ism)) if abs(y_wt - y_ism) > 1e-4 else 0.0

            win_metrics["heads"][head] = {
                "y_wt": y_wt,
                "y_ism": y_ism,
                "y_abl": y_abl,
                "y_rescue": y_rescue,
                "delta_ism": delta_ism,
                "delta_abl": delta_abl,
                "delta_rescue": delta_rescue,
                "mediation_ratio": float(np.clip(mediation, -2.0, 2.0)),
                "rescue_ratio": float(np.clip(rescue_rate, -2.0, 2.0)),
            }

        window_results.append(win_metrics)
        measured_count += 1

        tf_delta_ism = win_metrics["heads"]["chip_tf"]["delta_ism"]
        tf_delta_abl = win_metrics["heads"]["chip_tf"]["delta_abl"]
        tf_med = win_metrics["heads"]["chip_tf"]["mediation_ratio"]
        print(f"  [Motif {measured_count:02d}/{max_measured_windows}] {chrom}:{hit.start} ({hit.strand}) | "
              f"ChIP-TF ISM: {tf_delta_ism:+.3f} | Abl: {tf_delta_abl:+.3f} | Mediation: {tf_med*100:.1f}%", flush=True)

        del wt_tensor, mut_tensor, raw_wt, raw_mut, wt_act, mut_act, abl_act, rescue_act
        del out_wt, out_ism, out_abl, out_rescue
        cleanup_gpu()

    # Save detailed per-window records
    detail_file = out_dir / "motif_per_window_results.json"
    detail_file.write_text(json.dumps(window_results, indent=2))

    # ----------------------------------------------------------------------
    # Aggregate Statistical Summaries across Measured Windows
    # ----------------------------------------------------------------------
    summary_stats: dict[str, Any] = {
        "motif_id": motif.matrix_id,
        "motif_name": motif.name,
        "feature_id": target_feature_id,
        "n_measured_windows": len(window_results),
        "heads": {},
    }

    for head in MOTIF_HEADS:
        delta_ism_arr = np.array([w["heads"][head]["delta_ism"] for w in window_results], dtype=np.float64)
        delta_abl_arr = np.array([w["heads"][head]["delta_abl"] for w in window_results], dtype=np.float64)
        delta_rescue_arr = np.array([w["heads"][head]["delta_rescue"] for w in window_results], dtype=np.float64)
        mediation_arr = np.array([w["heads"][head]["mediation_ratio"] for w in window_results], dtype=np.float64)
        rescue_arr = np.array([w["heads"][head]["rescue_ratio"] for w in window_results], dtype=np.float64)

        summary_stats["heads"][head] = {
            "ism_stats": compute_paired_statistics(delta_ism_arr).to_dict() if len(delta_ism_arr) >= 2 else {},
            "abl_stats": compute_paired_statistics(delta_abl_arr).to_dict() if len(delta_abl_arr) >= 2 else {},
            "rescue_stats": compute_paired_statistics(delta_rescue_arr).to_dict() if len(delta_rescue_arr) >= 2 else {},
            "mean_mediation_ratio": float(np.mean(mediation_arr)) if len(mediation_arr) else 0.0,
            "median_mediation_ratio": float(np.median(mediation_arr)) if len(mediation_arr) else 0.0,
            "mean_rescue_ratio": float(np.mean(rescue_arr)) if len(rescue_arr) else 0.0,
            "median_rescue_ratio": float(np.median(rescue_arr)) if len(rescue_arr) else 0.0,
        }

    summary_file = out_dir / "motif_causal_summary.json"
    summary_file.write_text(json.dumps(summary_stats, indent=2))
    print(f"\nMotif causal cohort completed in {time.time()-t_start:.1f}s. Results written to {summary_file}.")
    return summary_stats


def run_motif_configuration(
    fold: str,
    layer: str,
    seed: str,
    base_dir: Path = Path("/mnt/ag"),
    out_dir: Path | None = None,
    triad_dir: Path | None = None,
) -> None:
    """Execute motif causal evaluation for a single (fold, layer, seed) configuration."""
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from ag_sae.sae import BorzoiSAE

    t0 = time.time()
    config_name = f"{fold}_{layer}_{seed}"
    effective_out = out_dir or (base_dir / "analysis_runs" / "motif_matrix" / config_name)
    effective_out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"STARTING MOTIF CAUSAL RUN: {config_name}")
    print(f"Output: {effective_out}")
    print("=" * 80)

    # 1. Determine target feature ID from previous campaign triad summary
    target_feature_id = 4586  # Safe default (CTCF in fold1 b8)
    triad_summary_file = (triad_dir or (base_dir / "analysis_runs" / "campaign_matrix" / config_name)) / "regulatory_triad_summary.json"
    if triad_summary_file.is_file():
        try:
            triad_data = json.loads(triad_summary_file.read_text())
            if "CTCF_Insulator" in triad_data and "feature_id" in triad_data["CTCF_Insulator"]:
                target_feature_id = int(triad_data["CTCF_Insulator"]["feature_id"])
                print(f"Loaded CTCF_Insulator feature ID from campaign summary: F{target_feature_id}")
        except Exception as e:
            print(f"Warning: Failed to load triad summary ({e}), using default F{target_feature_id}")

    # 2. Paths
    fold_num = fold.replace("fold", "")
    manifest_candidates = [
        base_dir / "analysis-code-a1f31ce" / "manifests" / f"window_manifest_fold{fold_num}.parquet",
        Path.home() / "ag" / "manifests" / f"window_manifest_fold{fold_num}.parquet",
        base_dir / "inputs" / f"manifest_{fold}.parquet",
    ]
    manifest_path = next((p for p in manifest_candidates if p.is_file()), manifest_candidates[0])

    fasta_candidates = [
        base_dir / "data" / "GRCh38.primary_assembly.genome.fa",
        base_dir / "inputs" / "genome.fa",
    ]
    fasta_path = next((p for p in fasta_candidates if p.is_file()), fasta_candidates[0])

    weights_candidates = [
        base_dir / "weights" / f"model_fold_{fold_num}.safetensors",
        base_dir / "weights" / fold,
    ]
    weights_path = next((p for p in weights_candidates if p.exists()), weights_candidates[0])

    sae_candidates = [
        base_dir / "sae" / fold / f"{layer}_{seed}" / "best.pt",
        base_dir / "sae" / fold / layer / seed / "checkpoints" / "best.pt",
    ]
    sae_path = next((p for p in sae_candidates if p.is_file()), sae_candidates[0])

    jaspar_candidates = [
        base_dir / "data" / "jaspar" / "JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
        Path("data/jaspar/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt"),
    ]
    jaspar_path = next((p for p in jaspar_candidates if p.is_file()), jaspar_candidates[0])

    motif = get_canonical_ctcf_motif(jaspar_path)
    tap = TAPS[layer]

    print(f"Loading SAE from {sae_path}...")
    sae = BorzoiSAE.from_checkpoint(sae_path, device="cuda")

    print(f"Loading AlphaGenome from {weights_path}...")
    model = AlphaGenome.from_pretrained(weights_path, dtype_policy=DtypePolicy.mixed_precision(), device="cuda")
    model.eval().requires_grad_(False)
    organism = torch.tensor([0], device="cuda")

    # 3. Load manifest test split windows
    manifest = pd.read_parquet(manifest_path)
    test_slice = manifest.loc[manifest.split == "test"]
    windows = [
        {"chrom": row.chrom, "start": int(row.win_start), "end": int(row.win_end)}
        for row in test_slice.itertuples()
    ]

    # 4. Execute 2x2 Factorial Causal Cohort
    run_motif_causal_cohort(
        sae=sae,
        model=model,
        tap=tap,
        windows=windows,
        fasta_path=fasta_path,
        organism=organism,
        target_feature_id=target_feature_id,
        motif=motif,
        out_dir=effective_out,
        min_score_ratio=0.5,
        max_measured_windows=25,
    )

    print(f"\nConfiguration {config_name} finished in {time.time()-t0:.1f}s!\n")


def run_motif_matrix_campaign(
    folds: tuple[str, ...] = ALL_FOLDS,
    layers: tuple[str, ...] = DEFAULT_LAYERS,
    seeds: tuple[str, ...] = DEFAULT_SEEDS,
    base_dir: Path = Path("/mnt/ag"),
    drive_remote: str = "gdrive:alphagenome_sae/motif_matrix",
    dry_run: bool = False,
) -> None:
    """Execute complete 36-configuration motif campaign matrix with process isolation."""
    base_out_dir = base_dir / "analysis_runs" / "motif_matrix"
    if not dry_run:
        try:
            base_out_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    configs = [(f, l, s) for f in folds for l in layers for s in seeds]
    total = len(configs)

    print("=" * 80)
    print(f"STARTING MOTIF CAMPAIGN MATRIX: {len(folds)} Folds x {len(layers)} Layers x {len(seeds)} Seeds ({total} Configs)")
    print(f"Output directory: {base_out_dir}")
    print(f"Drive destination: {drive_remote}")
    print("=" * 80)

    t_start = time.time()
    completed = 0

    for step, (fold, layer, seed) in enumerate(configs, 1):
        config_name = f"{fold}_{layer}_{seed}"
        out_dir = base_out_dir / config_name
        summary_file = out_dir / "motif_causal_summary.json"

        print(f"\n>>> [STEP {step}/{total}] Configuration: {config_name}")

        if is_valid_motif_summary(summary_file):
            if not dry_run and drive_remote:
                target_drive = f"{drive_remote.rstrip('/')}/{config_name}/"
                sync_cmd = ["rclone", "copy", f"{out_dir}/", target_drive, "--checksum", "-q"]
                subprocess.run(sync_cmd, capture_output=True, text=True)
            print(f"  [SKIPPED] {config_name} already completed and verified.")
            completed += 1
            continue

        cmd = [
            sys.executable, "-u", "-m", "ag_sae.motif_intervene",
            "--fold", fold,
            "--layer", layer,
            "--seed", seed,
            "--out", str(out_dir),
        ]

        if dry_run:
            print(f"  [DRY-RUN] Would execute: {' '.join(cmd)}")
            continue

        t_step = time.time()
        res = subprocess.run(cmd)

        if res.returncode != 0:
            print(f"\n[ERROR] Step {step}/{total} failed with returncode {res.returncode} for {config_name}!", file=sys.stderr)
            sys.exit(res.returncode)

        if not is_valid_motif_summary(summary_file):
            print(f"\n[ERROR] Output file {summary_file} missing or corrupt for {config_name}!", file=sys.stderr)
            sys.exit(1)

        print(f"  [SUCCESS] Finished {config_name} in {time.time()-t_step:.1f}s!")

        if drive_remote:
            target_drive = f"{drive_remote.rstrip('/')}/{config_name}/"
            print(f"  Syncing to Google Drive ({target_drive})...")
            sync_cmd = ["rclone", "copy", f"{out_dir}/", target_drive, "--checksum", "-q"]
            sync_res = subprocess.run(sync_cmd, capture_output=True, text=True)
            if sync_res.returncode != 0:
                print(f"  [Warning] Drive sync returned non-zero code {sync_res.returncode}")

        completed += 1

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"MOTIF CAMPAIGN MATRIX COMPLETED! {completed}/{total} configurations in {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    print("=" * 80)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AlphaGenome-SAE Motif Causal Intervention Engine")
    parser.add_argument("--campaign", action="store_true", help="Run full matrix campaign")
    parser.add_argument("--all-folds", action="store_true", help="Include all 4 genomic folds in campaign")
    parser.add_argument("--fold", type=str, default="fold1", help="Genomic fold (e.g. fold0, fold1)")
    parser.add_argument("--layer", type=str, default="resid_pre_b8", help="Tap layer name")
    parser.add_argument("--seed", type=str, default="seed0", help="Training seed (e.g. seed0)")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args(argv)

    if args.campaign:
        folds = ALL_FOLDS if args.all_folds else (args.fold,)
        run_motif_matrix_campaign(folds=folds, dry_run=args.dry_run)
    else:
        run_motif_configuration(fold=args.fold, layer=args.layer, seed=args.seed, out_dir=args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())

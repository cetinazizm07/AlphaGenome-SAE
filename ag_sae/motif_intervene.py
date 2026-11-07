"""Causal motif intervention engine and campaign runner for AlphaGenome-SAE.

Implements the 2x2 Factorial Causal Mediation Protocol:
  Condition 1 (WT): Natural sequence, baseline unperturbed residual stream (Y_W)
  Condition 2 (ISM): In Silico Mutagenesis on core motif anchor (Y_M)
  Condition 3 (SAE Ablation): Natural sequence with target TF latent ablated (Y_W_abl)
  Condition 4 (SAE Rescue): Mutated sequence with target TF latent restored (Y_M_rescue)

In every condition the anchor bin is replaced by its SAE reconstruction, so all
contrasts are between reconstructions.

Evaluates:
  - Epigenomic heads: ChIP-TF, ATAC, DNase, CAGE, PRO-cap
  - Mediation ratio: Delta_abl / Delta_ISM
  - Rescue ratio: Delta_rescue / (Y_W - Y_M)
  - Paired hypothesis testing via Wilcoxon signed-rank and Student's t-test

How it runs (A100 rewrite)
  * Motif sites are found with FIMO (memelite, tangermeme's scanner) at a
    p-value threshold, and the ISM sequence is made with tangermeme.ersatz.
  * Sites and ISM sequences are prepared once per fold and shared by every
    (layer, seed) configuration, so all configurations see the same windows
    and the same mutants.
  * The four conditions of a window run as one batched forward through
    `tm.FactorialMotifModel` and `tangermeme.predict` (the first campaign ran
    six batch-1 forwards per window, two of them only to capture the stream).
  * AlphaGenome is loaded once per fold, not once per configuration.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from ag_sae.extract import TAPS, one_hot
from ag_sae.motifs import JasparMotif, read_jaspar_records
from ag_sae.stats import compute_paired_statistics

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


def cleanup_gpu() -> None:
    """Trigger garbage collection and empty PyTorch CUDA cache."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


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
            for m in records.values():
                if "CTCF" in m.name.upper() or "MA0139" in m.matrix_id:
                    return m
        except Exception:
            pass
    return JasparMotif(matrix_id="MA0139.1", name="CTCF", counts=CANONICAL_CTCF_COUNTS)


def _atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


# --------------------------------------------------------------------------
# Sites and mutants, shared by every configuration of a fold
# --------------------------------------------------------------------------


@dataclass
class MotifWindow:
    """One measured window: its strongest motif site and the ISM sequence."""

    chrom: str
    start: int
    end: int
    site_start: int          # window-relative, like the first campaign's hit.start
    site_end: int
    strand: str
    score: float
    p_value: float
    site_sequence: str
    wt: Any                  # (1, 4, L) float32 one-hot, tangermeme layout
    mut: Any

    def anchor_bin(self, bin_bp: int) -> int:
        return self.site_start // bin_bp

    def record(self) -> dict:
        out = asdict(self)
        out.pop("wt"), out.pop("mut")
        return out


def prepare_motif_windows(
    windows: Sequence[dict],
    fasta_path: Path,
    motif: JasparMotif,
    *,
    n_windows: int = 25,
    p_threshold: float = 1e-5,
    seed: int = 42,
    shuffle_method: str = "shuffle",
    log=print,
) -> list[MotifWindow]:
    """Strongest FIMO site per window, in manifest order, until `n_windows`.

    The ISM sequence shuffles the site with tangermeme.ersatz; each window's
    shuffle has its own seed (seed + its rank) so the mutants do not depend on
    which configuration is being measured.
    """
    from ag_sae.intervene import window_sequence
    from ag_sae.tm import shuffle_site, strongest_sites, to_tangermeme

    out: list[MotifWindow] = []
    for window in windows:
        if len(out) >= n_windows:
            break
        chrom, start, end = window["chrom"], int(window["start"]), int(window["end"])
        sequence = window_sequence(fasta_path, chrom, start, end)
        x = to_tangermeme(one_hot(sequence))
        hits = strongest_sites(x, motif, name=motif.name, threshold=p_threshold)
        if hits.empty:
            continue
        hit = hits.iloc[0]
        s, e = int(hit["start"]), int(hit["end"])
        mut = shuffle_site(x, s, e, seed=seed + len(out), method=shuffle_method)
        out.append(MotifWindow(chrom, start, end, s, e, str(hit["strand"]),
                               float(hit["score"]), float(hit["p-value"]),
                               sequence[s:e], x, mut))
    log(f"prepared {len(out)} {motif.name} windows (FIMO p<{p_threshold:g}, "
        f"ISM={shuffle_method}) from {len(windows)} candidates")
    return out


# --------------------------------------------------------------------------
# The 2x2 measurement
# --------------------------------------------------------------------------


def _ratio(numerator: float, denominator: float) -> float:
    return float(np.clip(numerator / denominator, -2.0, 2.0)) if abs(denominator) > 1e-4 else 0.0


def measure_motif_windows(
    sae: Any,
    model: Any,
    tap: Any,
    motif_windows: Sequence[MotifWindow],
    target_feature_id: int,
    *,
    heads: Sequence[str] = MOTIF_HEADS,
    max_batch: int = 4,
    device: str = "cuda",
) -> list[dict]:
    """Per-window metrics for one SAE, in the first campaign's JSON schema."""
    from tangermeme.predict import predict
    from ag_sae.tm import FactorialMotifModel

    if not motif_windows:
        return []
    wrapper = FactorialMotifModel(model, sae, int(tap.key), target_feature_id, heads,
                                  max_batch=max_batch)
    x_wt = torch.cat([w.wt for w in motif_windows])
    x_mut = torch.cat([w.mut for w in motif_windows])
    anchors = torch.tensor([float(w.anchor_bin(tap.bin_bp)) for w in motif_windows])
    # dtype pinned: tangermeme casts every argument to it, and anchor bins
    # above 256 would round in bf16.
    readouts, latents = predict(wrapper, x_wt, args=(x_mut, anchors), batch_size=1,
                                dtype=torch.float32, device=device)
    readouts, latents = readouts.double().numpy(), latents.double().numpy()

    results = []
    for i, window in enumerate(motif_windows):
        record = {"chrom": window.chrom, "start": window.start, "end": window.end,
                  "anchor_start": window.site_start, "anchor_end": window.site_end,
                  "anchor_strand": window.strand, "anchor_score": window.score,
                  "anchor_pvalue": window.p_value, "anchor_seq": window.site_sequence,
                  "anchor_bin": window.anchor_bin(tap.bin_bp),
                  "wt_latent_val": float(latents[i, 0]),
                  "mut_latent_val": float(latents[i, 1]), "heads": {}}
        for h, head in enumerate(heads):
            y_wt, y_ism, y_abl, y_rescue = (float(v) for v in readouts[i, :, h])
            record["heads"][head] = {
                "y_wt": y_wt, "y_ism": y_ism, "y_abl": y_abl, "y_rescue": y_rescue,
                "delta_ism": y_ism - y_wt, "delta_abl": y_abl - y_wt,
                "delta_rescue": y_rescue - y_ism,
                "mediation_ratio": _ratio(y_abl - y_wt, y_ism - y_wt),
                "rescue_ratio": _ratio(y_rescue - y_ism, y_wt - y_ism),
            }
        results.append(record)
    return results


def summarize_motif_windows(window_results: list[dict], motif: JasparMotif,
                            feature_id: int, heads: Sequence[str] = MOTIF_HEADS) -> dict:
    """Aggregate statistics; same keys as the first campaign, plus ratio-of-means.

    The per-window ratios are clipped to [-2, 2] and set to 0 when the ISM
    effect is tiny, which biases their mean towards 0. The ratio of the mean
    deltas (`mediation_ratio_of_means`) has neither problem and is the one to
    report as "the share of the motif's effect that goes through the feature".
    """
    summary: dict[str, Any] = {"motif_id": motif.matrix_id, "motif_name": motif.name,
                               "feature_id": feature_id,
                               "n_measured_windows": len(window_results), "heads": {}}
    for head in heads:
        def column(key: str) -> np.ndarray:
            return np.array([w["heads"][head][key] for w in window_results], dtype=np.float64)

        ism, abl, rescue = column("delta_ism"), column("delta_abl"), column("delta_rescue")
        mediation, rescue_ratio = column("mediation_ratio"), column("rescue_ratio")
        usable = len(ism) > 0 and abs(ism.mean()) > 1e-12
        summary["heads"][head] = {
            "ism_stats": compute_paired_statistics(ism).to_dict() if len(ism) >= 2 else {},
            "abl_stats": compute_paired_statistics(abl).to_dict() if len(abl) >= 2 else {},
            "rescue_stats": compute_paired_statistics(rescue).to_dict() if len(rescue) >= 2 else {},
            "mean_mediation_ratio": float(mediation.mean()) if len(mediation) else 0.0,
            "median_mediation_ratio": float(np.median(mediation)) if len(mediation) else 0.0,
            "mean_rescue_ratio": float(rescue_ratio.mean()) if len(rescue_ratio) else 0.0,
            "median_rescue_ratio": float(np.median(rescue_ratio)) if len(rescue_ratio) else 0.0,
            "mediation_ratio_of_means": float(abl.mean() / ism.mean()) if usable else None,
            "rescue_ratio_of_means": float(rescue.mean() / -ism.mean()) if usable else None,
        }
    return summary


def run_motif_causal_cohort(sae, model, tap, motif_windows, motif, target_feature_id,
                            out_dir: Path, *, max_batch: int = 4, device: str = "cuda") -> dict:
    """Measure one configuration and write its two JSON files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    results = measure_motif_windows(sae, model, tap, motif_windows, target_feature_id,
                                    max_batch=max_batch, device=device)
    for n, record in enumerate(results, 1):
        tf = record["heads"]["chip_tf"]
        print(f"  [Motif {n:02d}/{len(results)}] {record['chrom']}:{record['start'] + record['anchor_start']} "
              f"({record['anchor_strand']}) | ChIP-TF ISM: {tf['delta_ism']:+.3f} | "
              f"Abl: {tf['delta_abl']:+.3f} | Mediation: {tf['mediation_ratio'] * 100:.1f}%")
    _atomic_json(out_dir / "motif_per_window_results.json", results)
    summary = summarize_motif_windows(results, motif, target_feature_id)
    _atomic_json(out_dir / "motif_causal_summary.json", summary)
    print(f"Motif cohort done in {time.time() - t_start:.1f}s -> {out_dir / 'motif_causal_summary.json'}")
    return summary


# --------------------------------------------------------------------------
# Paths, feature choice, and the fold/campaign drivers
# --------------------------------------------------------------------------


def _first(paths: Sequence[Path]) -> Path:
    return next((p for p in paths if p.exists()), paths[0])


def resolve_fold_paths(base_dir: Path, fold: str) -> dict[str, Path]:
    n = fold.replace("fold", "")
    return {
        "manifest": _first([base_dir / "analysis-code-a1f31ce" / "manifests" / f"window_manifest_fold{n}.parquet",
                            Path.home() / "ag" / "manifests" / f"window_manifest_fold{n}.parquet",
                            base_dir / "inputs" / f"manifest_{fold}.parquet"]),
        "fasta": _first([base_dir / "data" / "GRCh38.primary_assembly.genome.fa",
                         base_dir / "inputs" / "genome.fa"]),
        "weights": _first([base_dir / "weights" / f"model_fold_{n}.safetensors",
                           base_dir / "weights" / fold]),
        "jaspar": _first([base_dir / "data" / "jaspar" / "JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
                          Path("data/jaspar/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt")]),
    }


def resolve_sae_path(base_dir: Path, sae_pool: str, fold: str, layer: str, seed: str) -> Path:
    return _first([base_dir / sae_pool / fold / f"{layer}_{seed}" / "best.pt",
                   base_dir / sae_pool / fold / layer / seed / "checkpoints" / "best.pt",
                   base_dir / sae_pool / fold / layer / seed / "best.pt"])


def resolve_ctcf_feature(sae, model, tap, *, base_dir: Path, fold: str, layer: str,
                         config_name: str, sae_pool: str, manifest: pd.DataFrame,
                         fasta_path: Path, triad_dir: Path | None = None,
                         feature: int | None = None) -> tuple[int, str]:
    """The CTCF feature to test and where it came from.

    Order: an explicit --feature; the triad campaign's summary for this
    configuration; discovery on calibration windows. Unlike the first campaign
    there is no hard-coded fallback: a feature picked for another fold and
    layer would make the run meaningless, so failure stops the configuration.
    """
    if feature is not None:
        return int(feature), "explicit"
    if triad_dir is None:
        name = "campaign_matrix" if sae_pool == "sae" else f"campaign_matrix_{sae_pool}"
        triad_dir = base_dir / "analysis_runs" / name / config_name
    triad_file = triad_dir / "regulatory_triad_summary.json"
    if triad_file.is_file():
        data = json.loads(triad_file.read_text())
        return int(data["CTCF_Insulator"]["feature_id"]), str(triad_file)

    annotations_path = next((p for p in [
        base_dir / "analysis_runs" / "inputs" / "full128" / fold / layer / "annotation_main.parquet",
        base_dir / "inputs" / f"annotations_{fold}.parquet"] if p.is_file()), None)
    if annotations_path is None:
        raise FileNotFoundError(f"{config_name}: no triad summary and no annotations to discover "
                                "the CTCF feature; pass --feature")
    from ag_sae.intervene import discover_concept_features

    calibration = manifest.loc[manifest.split == "calibration"]
    if calibration.empty:
        calibration = manifest.loc[manifest.split == "test"].iloc[:10]
    windows = [{"chrom": r.chrom, "start": int(r.win_start), "end": int(r.win_end)}
               for r in calibration.itertuples()]
    found = discover_concept_features(sae=sae, model=model, tap=tap, calibration_windows=windows,
                                      annotations=pd.read_parquet(annotations_path),
                                      fasta_path=fasta_path, organism=torch.tensor([0], device="cuda"),
                                      max_calibration_windows=10)
    return int(found.insulator_feature_id), f"discovered:{annotations_path}"


def matrix_subdir(sae_pool: str) -> str:
    """Output folder of a pool: motif_matrix, motif_matrix_k64, motif_matrix_<pool>."""
    return "motif_matrix" if sae_pool == "sae" else f"motif_matrix_{sae_pool.removeprefix('sae_')}"


def _sync(out_dir: Path, remote: str | None) -> None:
    if remote:
        subprocess.run(["rclone", "copy", f"{out_dir}/", f"{remote.rstrip('/')}/{out_dir.name}/",
                        "--checksum", "-q"], capture_output=True, text=True)


def run_motif_fold(
    fold: str,
    layers: Sequence[str] = DEFAULT_LAYERS,
    seeds: Sequence[str] = DEFAULT_SEEDS,
    *,
    base_dir: Path = Path("/mnt/ag"),
    sae_pool: str = "sae",
    out_root: Path | None = None,
    out_dir: Path | None = None,
    triad_dir: Path | None = None,
    feature: int | None = None,
    drive_remote: str | None = None,
    n_windows: int = 25,
    p_threshold: float = 1e-5,
    shuffle_method: str = "shuffle",
    max_batch: int = 4,
    device: str = "cuda",
) -> int:
    """Every (layer, seed) of one fold with AlphaGenome loaded once. Returns #done."""
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    from ag_sae.sae import BorzoiSAE

    subdir = matrix_subdir(sae_pool)
    out_root = out_root or (base_dir / "analysis_runs" / subdir)
    configs = [(layer, seed) for layer in layers for seed in seeds]
    if out_dir is not None and len(configs) != 1:
        raise ValueError("out_dir is for a single configuration")
    targets = {c: out_dir or out_root / f"{fold}_{c[0]}_{c[1]}" for c in configs}
    todo = [c for c in configs if not is_valid_motif_summary(targets[c] / "motif_causal_summary.json")]
    for c in configs:
        if c not in todo:
            print(f"  [SKIPPED] {fold}_{c[0]}_{c[1]} already complete")
            _sync(targets[c], drive_remote)
    if not todo:
        return len(configs)

    paths = resolve_fold_paths(base_dir, fold)
    motif = get_canonical_ctcf_motif(paths["jaspar"])
    manifest = pd.read_parquet(paths["manifest"])
    test_windows = [{"chrom": r.chrom, "start": int(r.win_start), "end": int(r.win_end)}
                    for r in manifest.loc[manifest.split == "test"].itertuples()]
    motif_windows = prepare_motif_windows(test_windows, paths["fasta"], motif, n_windows=n_windows,
                                          p_threshold=p_threshold, shuffle_method=shuffle_method)

    print(f"Loading AlphaGenome {paths['weights']}...")
    model = AlphaGenome.from_pretrained(paths["weights"], dtype_policy=DtypePolicy.mixed_precision(),
                                        device=device)
    model.eval().requires_grad_(False)

    done = len(configs) - len(todo)
    for layer, seed in todo:
        t0 = time.time()
        config_name = f"{fold}_{layer}_{seed}"
        target = targets[(layer, seed)]
        target.mkdir(parents=True, exist_ok=True)
        tap = TAPS[layer]
        sae_path = resolve_sae_path(base_dir, sae_pool, fold, layer, seed)
        print("=" * 80 + f"\nMOTIF RUN {config_name} (pool {sae_pool}) <- {sae_path}\n" + "=" * 80)
        sae = BorzoiSAE.from_checkpoint(sae_path, device=device)
        feature_id, source = resolve_ctcf_feature(
            sae, model, tap, base_dir=base_dir, fold=fold, layer=layer, config_name=config_name,
            sae_pool=sae_pool, manifest=manifest, fasta_path=paths["fasta"],
            triad_dir=triad_dir, feature=feature)
        print(f"CTCF feature F{feature_id} ({source})")
        _atomic_json(target / "motif_windows.json", {
            "fold": fold, "feature_id": feature_id, "feature_source": source,
            "sae_checkpoint": str(sae_path), "p_threshold": p_threshold,
            "shuffle_method": shuffle_method, "windows": [w.record() for w in motif_windows]})
        run_motif_causal_cohort(sae, model, tap, motif_windows, motif, feature_id, target,
                                max_batch=max_batch, device=device)
        _sync(target, drive_remote)
        del sae
        cleanup_gpu()
        done += 1
        print(f"Configuration {config_name} finished in {time.time() - t0:.1f}s")
    del model
    cleanup_gpu()
    return done


def run_motif_matrix_campaign(
    folds: Sequence[str] = ALL_FOLDS,
    layers: Sequence[str] = DEFAULT_LAYERS,
    seeds: Sequence[str] = DEFAULT_SEEDS,
    base_dir: Path = Path("/mnt/ag"),
    drive_remote: str | None = None,
    sae_pool: str = "sae",
    dry_run: bool = False,
    **fold_kwargs,
) -> None:
    """Every fold in turn; each fold loads AlphaGenome once for all its configurations."""
    subdir = matrix_subdir(sae_pool)
    remote = drive_remote or f"gdrive:alphagenome_sae/{subdir}"
    total = len(folds) * len(layers) * len(seeds)
    print(f"MOTIF CAMPAIGN: {len(folds)} folds x {len(layers)} layers x {len(seeds)} seeds "
          f"= {total} configurations (pool {sae_pool}) -> {base_dir / 'analysis_runs' / subdir}")
    if dry_run:
        for fold in folds:
            for layer in layers:
                for seed in seeds:
                    print(f"  [DRY-RUN] {fold}_{layer}_{seed}")
        return
    t0, done = time.time(), 0
    for fold in folds:
        done += run_motif_fold(fold, layers, seeds, base_dir=base_dir, sae_pool=sae_pool,
                               drive_remote=remote, **fold_kwargs)
    print(f"MOTIF CAMPAIGN COMPLETE: {done}/{total} in {(time.time() - t0) / 60:.1f} min")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AlphaGenome-SAE Motif Causal Intervention Engine")
    parser.add_argument("--campaign", action="store_true", help="Run full matrix campaign")
    parser.add_argument("--all-folds", action="store_true", help="Include all 4 genomic folds in campaign")
    parser.add_argument("--fold", type=str, default="fold1", help="Genomic fold (e.g. fold0, fold1, fold3)")
    parser.add_argument("--layer", type=str, default="resid_pre_b8", help="Tap layer name")
    parser.add_argument("--seed", type=str, default="seed0", help="Training seed (e.g. seed0)")
    parser.add_argument("--sae-pool", type=str, default="sae",
                        help="SAE directory under --base-dir (sae, sae_k64, sae_v2, ...)")
    parser.add_argument("--base-dir", type=Path, default=Path("/mnt/ag"))
    parser.add_argument("--out", type=Path, default=None, help="Output directory (single config)")
    parser.add_argument("--feature", type=int, default=None, help="CTCF feature id to test (single config)")
    parser.add_argument("--n-windows", type=int, default=25)
    parser.add_argument("--p-threshold", type=float, default=1e-5, help="FIMO p-value for a site")
    parser.add_argument("--ism", choices=("shuffle", "dinucleotide"), default="shuffle")
    parser.add_argument("--max-batch", type=int, default=4,
                        help="conditions per forward; 4 on an 80 GB A100, 2 on 40 GB")
    parser.add_argument("--dry-run", action="store_true", help="Print configurations without executing")
    args = parser.parse_args(argv)

    fold_kwargs = dict(n_windows=args.n_windows, p_threshold=args.p_threshold,
                       shuffle_method=args.ism, max_batch=args.max_batch)
    if args.campaign:
        folds = (("fold3",) if args.sae_pool == "sae_k64" and not args.all_folds
                 else ALL_FOLDS if args.all_folds else (args.fold,))
        run_motif_matrix_campaign(folds=folds, base_dir=args.base_dir, sae_pool=args.sae_pool,
                                  dry_run=args.dry_run, **fold_kwargs)
    else:
        run_motif_fold(args.fold, (args.layer,), (args.seed,), base_dir=args.base_dir,
                       sae_pool=args.sae_pool, out_dir=args.out, feature=args.feature,
                       **fold_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

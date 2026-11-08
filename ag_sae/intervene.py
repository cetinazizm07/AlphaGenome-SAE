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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None

from ag_sae.edit import (Condition, add_to_feature, clamp_min_feature, run_conditions, sae_edit,
                         track_mean_profile, zero_feature, zero_features)
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


#: annotation column of each concept, in DiscoveredFeatures field order
CONCEPT_COLUMNS: tuple[str, ...] = ("cCRE_PLS", "cCRE_CTCF_only", "cCRE_pELS", "shuffled_dELS")
#: how a feature is ranked against a concept (see ConceptContrast)
SCORES: tuple[str, ...] = ("cohen_d", "mean_diff")


@dataclass(frozen=True)
class DiscoveredFeatures:
    """Maps biological regulatory concepts to their highest-scoring SAE latent IDs.

    `ranked` holds, per annotation column, the best few features in score
    order (the first is the field above) and `scores` their scores; both are
    empty when the IDs were given by hand.
    """

    promoter_feature_id: int
    insulator_feature_id: int
    enhancer_feature_id: int
    shuffled_control_feature_id: int
    ranked: dict[str, tuple[int, ...]] = field(default_factory=dict, compare=False)
    scores: dict[str, tuple[float, ...]] = field(default_factory=dict, compare=False)
    score: str = "given"
    #: share of the calibration windows' edited bins (edited_bins_mask) on
    #: which each feature fires, for matched nulls
    firing_rate: np.ndarray | None = field(default=None, compare=False, repr=False)
    #: the best NEAR_TOP features of every real concept, kept out of null pools
    near_top: tuple[int, ...] = field(default=(), compare=False, repr=False)

    def as_dict(self) -> dict[str, tuple[int, str]]:
        """Return mapping of {concept_name: (feature_id, annotation_column_name)}."""
        return {
            "PLS_Promoter": (self.promoter_feature_id, "cCRE_PLS"),
            "CTCF_Insulator": (self.insulator_feature_id, "cCRE_CTCF_only"),
            "Enhancer_pELS": (self.enhancer_feature_id, "cCRE_pELS"),
            "Shuffled_Control": (self.shuffled_control_feature_id, "shuffled_dELS"),
        }

    def top(self, column: str, n: int) -> tuple[int, ...]:
        """The `n` best features for an annotation column (at least the primary one)."""
        primary = {col: feat for feat, col in self.as_dict().values()}[column]
        return self.ranked.get(column, (primary,))[:max(1, n)]

    def to_json(self) -> dict:
        return {"score": self.score,
                "primary": {name: feat for name, (feat, _col) in self.as_dict().items()},
                "ranked": {col: list(ids) for col, ids in self.ranked.items()},
                "scores": {col: list(vals) for col, vals in self.scores.items()}}


class ConceptContrast:
    """Per-feature statistics of concept-positive against concept-negative bins.

    Both scores contrast bins within a window, so a feature that is simply
    higher in some windows (gene-dense, GC-rich) than in others does not win
    because the positive bins happen to sit in those windows:

    * ``mean_diff`` -- the first campaign's: positive-bin mean minus
      negative-bin mean, averaged over windows. It is in activation units, so
      features with large activations win regardless of how cleanly they
      separate the bins.
    * ``cohen_d`` -- the within-window mean difference (averaged over windows,
      weighted by their positive bins) over the pooled within-window standard
      deviation, which is scale-free. Features that fire on fewer than
      `min_fire_rate` of the positive bins are not eligible, so a rare, very
      large activation cannot win on its own.

    Each window's deviations are taken from its own means on the device
    (accurate in float32) and only then summed in float64.
    """

    def __init__(self) -> None:
        self.window_diffs: list[np.ndarray] = []
        self.weighted_diff: np.ndarray | None = None   # sum over windows of n_pos * diff
        self.within_ss: np.ndarray | None = None       # sum of squared within-window deviations
        self.fired: np.ndarray | None = None           # positive bins on which each feature fires
        self.n_pos = 0
        self.n_bins = 0
        self.n_windows = 0

    @staticmethod
    def _add(total, value):
        return value if total is None else total + value

    def add(self, codes: torch.Tensor, pos: np.ndarray, neg: np.ndarray) -> None:
        if not pos.any() or not neg.any():
            return
        p = codes[torch.as_tensor(pos, device=codes.device)].float()
        q = codes[torch.as_tensor(neg, device=codes.device)].float()
        mean_p, mean_q = p.mean(0), q.mean(0)
        diff = (mean_p - mean_q).cpu().numpy()
        ss = ((p - mean_p).square().sum(0) + (q - mean_q).square().sum(0)).cpu().double().numpy()
        self.window_diffs.append(diff)
        self.weighted_diff = self._add(self.weighted_diff, len(p) * diff.astype(np.float64))
        self.within_ss = self._add(self.within_ss, ss)
        self.fired = self._add(self.fired, (p > 0).sum(0).cpu().numpy().astype(np.int64))
        self.n_pos += len(p)
        self.n_bins += len(p) + len(q)
        self.n_windows += 1

    @property
    def empty(self) -> bool:
        return not self.window_diffs

    def score(self, kind: str, min_fire_rate: float = 0.05) -> np.ndarray:
        if self.empty:
            raise ValueError("no contrastive bins")
        if kind == "mean_diff":
            return np.mean(self.window_diffs, axis=0).astype(np.float64)
        if kind != "cohen_d":
            raise ValueError(f"unknown score {kind!r}; expected one of {SCORES}")
        # Two means estimated per window.
        pooled = np.sqrt(self.within_ss / max(self.n_bins - 2 * self.n_windows, 1))
        diff = self.weighted_diff / self.n_pos
        # A floor on the SD: a feature that separates the bins perfectly (no
        # spread within either group) ranks at the top instead of scoring 0.
        d = diff / np.maximum(pooled, 1e-6)
        return np.where(self.fired / self.n_pos >= min_fire_rate, d, -np.inf)


#: runners-up per concept kept out of null pools (DiscoveredFeatures.near_top)
NEAR_TOP = 20


def edited_bins_mask(bins: pd.DataFrame, start: int, num_bins: int, bin_bp: int) -> np.ndarray:
    """The bins the triad edits in a window: any cCRE (or PLS) plus shuffled_dELS."""
    mask = np.zeros(num_bins, dtype=bool)
    for column in (("cCRE_any",) if "cCRE_any" in bins.columns else ("cCRE_PLS",)) + ("shuffled_dELS",):
        if column in bins.columns:
            pos, _ = map_annotation_column_to_grid(bins, column, start, num_bins, bin_bp)
            mask |= pos
    return mask


def _concept_column(bins: pd.DataFrame, column: str) -> str | None:
    if column in bins.columns:
        return column
    if column == "cCRE_CTCF_only" and "cCRE_CTCF-only" in bins.columns:
        return "cCRE_CTCF-only"
    return None


def _window_codes(sae: Any, model: Any, tap: Any, fasta_path: Path, window: dict) -> torch.Tensor:
    """SAE codes of every bin of a window, from the stream entering the tap block."""
    from ag_sae.edit import capture_residual

    seq = window_sequence(fasta_path, window["chrom"], int(window["start"]), int(window["end"]))
    x = torch.from_numpy(sequence_to_onehot(seq).astype(np.float32))[None].to(sae.channel_scale.device)
    # Only the part of the model before the tap runs.
    acts = capture_residual(model, int(tap.key), x)
    codes, _ = sae.encode_codes(acts[0].float())
    return codes


def discover_concept_features(
    sae: Any,
    model: Any,
    tap: Any,
    calibration_windows: list[dict],
    annotations: pd.DataFrame,
    fasta_path: Path,
    organism: torch.Tensor,
    max_calibration_windows: int = 10,
    score: str = "cohen_d",
    top_n: int = 3,
    min_fire_rate: float = 0.05,
) -> DiscoveredFeatures:
    """Score SAE latents on calibration windows to find the top features per concept.

    `score="mean_diff"` reproduces the first campaign's choice exactly.
    """
    if score not in SCORES:
        raise ValueError(f"unknown score {score!r}; expected one of {SCORES}")
    contrasts = {c: ConceptContrast() for c in CONCEPT_COLUMNS}
    fired_bins, seen_bins = None, 0

    with torch.no_grad():
        for window in calibration_windows[:max_calibration_windows]:
            chrom, start, end = window["chrom"], int(window["start"]), int(window["end"])
            num_bins = (end - start) // tap.bin_bp
            bins = annotations.loc[
                (annotations.chrom == chrom)
                & (annotations.bin_start >= start) & (annotations.bin_start < end)
            ].sort_values("bin_start").reset_index(drop=True)
            codes = _window_codes(sae, model, tap, fasta_path, window)

            for c in CONCEPT_COLUMNS:
                col = _concept_column(bins, c)
                if col is None:
                    continue
                pos_mask, observed_mask = map_annotation_column_to_grid(bins, col, start, num_bins, tap.bin_bp)
                contrasts[c].add(codes, pos_mask, observed_mask & ~pos_mask)
            # Firing rate on the bins the triad edits, which is where a null
            # feature must fire as often as the concept feature to be a fair
            # comparison (a genome-wide rate says little about cCRE bins).
            edited = edited_bins_mask(bins, start, num_bins, tap.bin_bp)
            if edited.any():
                rows = codes[torch.as_tensor(edited, device=codes.device)]
                fired = (rows > 0).sum(0).cpu().numpy().astype(np.int64)
                fired_bins = fired if fired_bins is None else fired_bins + fired
                seen_bins += int(edited.sum())
            del codes

    ranked: dict[str, tuple[int, ...]] = {}
    scores: dict[str, tuple[float, ...]] = {}
    near_top: set[int] = set()
    for c in CONCEPT_COLUMNS:
        if contrasts[c].empty:
            raise RuntimeError(
                f"[Feature Discovery] No contrastive bins found in calibration windows for concept '{c}'. "
                "Cannot reliably identify latent feature without contrast.")
        values = contrasts[c].score(score, min_fire_rate)
        if not np.isfinite(values).any():
            raise RuntimeError(f"[Feature Discovery] no feature fires on {min_fire_rate:.0%} of the "
                               f"'{c}' bins; lower min_fire_rate")
        # Stable sort: ties keep the lowest ID, as np.argmax did.
        order = np.argsort(-values, kind="stable")[:max(1, top_n)]
        order = [int(i) for i in order if np.isfinite(values[i])]
        ranked[c] = tuple(order)
        if c != "shuffled_dELS":
            # Strong runners-up of any real concept are not null features.
            near_top |= {int(i) for i in np.argsort(-values, kind="stable")[:NEAR_TOP]
                         if np.isfinite(values[i])}
        scores[c] = tuple(float(values[i]) for i in order)

    cleanup_gpu()
    return DiscoveredFeatures(
        promoter_feature_id=ranked["cCRE_PLS"][0],
        insulator_feature_id=ranked["cCRE_CTCF_only"][0],
        enhancer_feature_id=ranked["cCRE_pELS"][0],
        shuffled_control_feature_id=ranked["shuffled_dELS"][0],
        ranked=ranked, scores=scores, score=score,
        firing_rate=None if fired_bins is None else fired_bins / max(seen_bins, 1),
        near_top=tuple(sorted(near_top)),
    )


def matched_null_pools(features: DiscoveredFeatures, top_n: int, pool_size: int = 50) -> dict[str, list[int]]:
    """Per concept, the `pool_size` features whose firing rate is closest to its feature's.

    A single "control" feature is one draw; the pilot showed it is not null
    (its ablation moved ChIP-TF). A pool of firing-rate-matched features gives
    a null distribution instead: a concept feature is specific only if it
    beats features that fire as often. Every concept's top-n features are
    excluded from every pool, and so are every real concept's NEAR_TOP best
    features and features that never fired. Rates are on the edited bins.
    """
    if features.firing_rate is None:
        raise ValueError("matched nulls need the firing rates from discover_concept_features")
    rate = np.asarray(features.firing_rate, dtype=np.float64)
    taken = {f for _name, (feat, col) in features.as_dict().items() for f in features.top(col, top_n)}
    taken |= set(features.near_top)
    eligible = np.array([i for i in range(len(rate)) if i not in taken and rate[i] > 0])
    if len(eligible) == 0:
        raise ValueError("no eligible null features")
    pools = {}
    for name, (feat, _col) in features.as_dict().items():
        if rate[feat] <= 0:
            raise ValueError(f"{name} feature {feat} never fires on the edited calibration bins; "
                             "no firing-rate match is possible")
        gap = np.abs(np.log(rate[eligible] + 1e-9) - np.log(rate[feat] + 1e-9))
        order = np.argsort(gap, kind="stable")[:pool_size]
        pools[name] = [int(i) for i in eligible[order]]
    return pools


def draw_null_features(pool: Sequence[int], k: int, seed: int, window_index: int,
                       concept_index: int) -> list[int]:
    """`k` distinct features from a pool, reproducible per (seed, window, concept)."""
    rng = np.random.default_rng([int(seed), int(window_index), int(concept_index)])
    return [int(f) for f in rng.choice(np.asarray(pool), size=min(k, len(pool)), replace=False)]


#: how evaluation windows are picked from a fold's test split
WINDOW_SAMPLINGS: tuple[str, ...] = ("random", "first")


def evaluation_windows(manifest: pd.DataFrame, n: int, sampling: str = "random",
                       seed: int = 2026) -> list[dict]:
    """The fold's evaluation windows: the same for every layer and seed of the fold.

    "first" is the first campaign's: the first `n` test windows in manifest
    order, which cluster on one or two chromosomes. "random" draws `n` test
    windows uniformly (fixed seed), so they spread over the held-out regions
    of every chromosome, and returns them in genome order.
    """
    test = manifest.loc[manifest.split == "test"]
    if sampling == "first":
        rows = test.head(n)
    elif sampling == "random":
        rows = test.sample(n=min(n, len(test)), random_state=seed).sort_values(["chrom", "win_start"])
    else:
        raise ValueError(f"unknown window sampling {sampling!r}; expected one of {WINDOW_SAMPLINGS}")
    return [{"chrom": r.chrom, "start": int(r.win_start), "end": int(r.win_end)} for r in rows.itertuples()]


def split_windows(manifest: pd.DataFrame, protocol: "Protocol",
                  windows: list[dict] | None = None, n_calibration: int = 10) -> tuple[list[dict], list[dict]]:
    """(evaluation windows, calibration windows) of a fold; the same for every layer and seed.

    Evaluation windows are `windows` when given (a plan), else
    evaluation_windows(). Calibration windows - which choose the features and
    the injection dose - are test windows that do not overlap any evaluation
    window: under v2 a fixed-seed random draw, under legacy the first ones in
    manifest order (compared by exact key, as the first campaign did).
    """
    if windows is None:
        windows = evaluation_windows(manifest, protocol.n_windows, protocol.window_sampling,
                                     protocol.window_seed)
    keys = {(w["chrom"], int(w["start"]), int(w["end"])) for w in windows}

    def overlaps(chrom: str, start: int, end: int) -> bool:
        if protocol.name == "legacy":
            return (chrom, start, end) in keys
        return any(w["chrom"] == chrom and start < int(w["end"]) and int(w["start"]) < end for w in windows)

    candidates = [{"chrom": r.chrom, "start": int(r.win_start), "end": int(r.win_end)}
                  for r in manifest.loc[manifest.split == "test"].itertuples()
                  if not overlaps(r.chrom, int(r.win_start), int(r.win_end))]
    if not candidates:
        raise RuntimeError("no test windows left for calibration outside the evaluation windows")
    if protocol.window_sampling == "random":
        pick = np.random.default_rng(protocol.window_seed + 1).permutation(len(candidates))[:n_calibration]
        return windows, [candidates[i] for i in sorted(pick)]
    return windows, candidates[:n_calibration]


def feature_dose(
    sae: Any,
    model: Any,
    tap: Any,
    windows: list[dict],
    fasta_path: Path,
    feature: int,
    quantile: float = 0.99,
    max_windows: int = 10,
) -> float:
    """A feature's natural 'strongly on' level: a quantile of its non-zero values.

    Taken over every bin of the (calibration) windows. Injecting a fixed
    level such as 5.0 means a different thing for every feature; this level
    is in the feature's own units, so 1x is a value the feature really
    reaches and multiples of it give a dose-response curve.
    """
    values = []
    with torch.no_grad():
        for window in windows[:max_windows]:
            column = _window_codes(sae, model, tap, fasta_path, window)[:, int(feature)]
            values.append(column[column > 0].float().cpu().numpy())
    fired = np.concatenate(values) if values else np.empty(0)
    if fired.size == 0:
        raise ValueError(f"feature {feature} never fires in {len(values)} windows; no natural dose")
    return float(np.quantile(fired, quantile))


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


#: per-window delta fields of the triad, under each feature or joint entry
TRIAD_METHODS: dict[str, tuple[str, str]] = {
    # field name: (condition-name suffix, condition it is compared with)
    "heads": ("", "reconstruction"),               # first campaign: x -> SAE(x) - a_i d_i
    "heads_error_preserving": ("_ep", "original"),  # x -> x - a_i d_i, error kept
}


def _head_deltas(arrays: dict[str, np.ndarray], label: str, baseline: str,
                 eval_mask: np.ndarray, intervention_mask: np.ndarray) -> dict[str, dict]:
    has_concept_bins = bool(eval_mask.any())
    out = {}
    for head in ALL_HEADS:
        delta = arrays[f"{head}_{label}"] - arrays[f"{head}_{baseline}"]

        def on_concept(values):
            return float(values[eval_mask].mean()) if has_concept_bins else None

        out[head] = {
            "delta_on_concept": on_concept(delta),
            "delta_on_all_regulatory": float(delta[intervention_mask].mean()) if intervention_mask.any() else 0.0,
            "has_concept_bins": has_concept_bins,
            # The signal the change is measured against, so effects can be
            # put on a common scale across heads (relative change, log ratio).
            "baseline_on_concept": on_concept(arrays[f"{head}_{baseline}"]),
            "original_on_concept": on_concept(arrays[f"{head}_original"]),
        }
    return out


def _minus_null(concept: dict[str, dict], nulls: list[dict[str, dict]]) -> dict[str, dict]:
    """Per head, a window's concept delta minus the mean of its null deltas."""
    out = {}
    for head in ALL_HEADS:
        c = concept[head]["delta_on_concept"]
        values = [n[head]["delta_on_concept"] for n in nulls if n[head]["delta_on_concept"] is not None]
        out[head] = {"delta_on_concept": None if c is None or not values else c - float(np.mean(values))}
    return out


def _summarize_heads(per_window: list[dict]) -> dict[str, dict]:
    """Paired statistics per head over windows; `per_window` holds `_head_deltas` results."""
    stats: dict[str, dict] = {}
    for head in ALL_HEADS:
        deltas = np.array([w[head]["delta_on_concept"] for w in per_window
                           if w[head]["delta_on_concept"] is not None], dtype=np.float64)
        if len(deltas) >= 2:
            stats[head] = compute_paired_statistics(deltas).to_dict()
        else:
            stats[head] = {
                "mean_delta": float(np.mean(deltas)) if len(deltas) == 1 else 0.0,
                "median_delta": float(np.median(deltas)) if len(deltas) == 1 else 0.0,
                "t_stat": 0.0, "t_p": 1.0, "wilcoxon_w": 0.0, "wilcoxon_p": 1.0,
                "pct_negative": 0.0,
            }
        stats[head]["n_measured_windows"] = len(deltas)
    return stats


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
    top_n: int = 3,
    null_per_window: int = 0,
    null_pool_size: int = 50,
    null_seed: int = 0,
) -> tuple[list[dict], dict[str, Any]]:
    """Execute Regulatory Triad Double-Dissociation ablation across test windows.

    With `null_per_window` > 0, each window also ablates (error-preserving)
    that many features drawn per concept from a firing-rate-matched pool
    (matched_null_pools), so every concept is compared with features that
    fire as often, on the same bins of the same window; across windows the
    draws build a null distribution over features. "concept_minus_null" is the
    per-window paired difference.

    Every concept's feature is ablated two ways (TRIAD_METHODS): through the
    SAE reconstruction, as in the first campaign, and error-preserving, which
    removes only the feature's own contribution from the real stream. When
    discovery ranked more than one feature for a concept, its top `top_n` are
    also ablated together (error-preserving), since a cleaner dictionary can
    split one concept over several features.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target_classes = features.as_dict()
    joint = {cname: features.top(col, top_n) for cname, (_feat, col) in target_classes.items()}
    joint = {cname: ids for cname, ids in joint.items() if len(ids) > 1}
    pools = matched_null_pools(features, top_n, null_pool_size) if null_per_window > 0 else {}

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

        intervention_mask = edited_bins_mask(bins, start, num_bins, tap.bin_bp)
        selected_bins = np.flatnonzero(intervention_mask)
        if len(selected_bins) == 0:
            selected_bins = np.array([num_bins // 2])

        seq = window_sequence(fasta_path, chrom, start, end)
        input_tensor = torch.from_numpy(sequence_to_onehot(seq).astype(np.float32))[None].to(sae.channel_scale.device)

        conditions = [Condition("original"),
                      Condition("reconstruction", 0, selected_bins, sae_edit(sae, "reconstruction"))]
        for cname, (feat, _col) in target_classes.items():
            conditions.append(Condition(f"ablate_{cname}", 0, selected_bins,
                                        sae_edit(sae, f"ablate_{cname}", zero_feature(feat))))
            conditions.append(Condition(f"ablate_{cname}_ep", 0, selected_bins,
                                        sae_edit(sae, f"ablate_{cname}_ep", zero_feature(feat),
                                                 preserve_error=True)))
        for cname, ids in joint.items():
            conditions.append(Condition(f"joint_{cname}_ep", 0, selected_bins,
                                        sae_edit(sae, f"joint_{cname}_ep", zero_features(ids),
                                                 preserve_error=True)))
        nulls = {cname: draw_null_features(pool, null_per_window, null_seed, wi, ci)
                 for ci, (cname, pool) in enumerate(pools.items())}
        for cname, drawn in nulls.items():
            for j, null_feat in enumerate(drawn):
                conditions.append(Condition(f"null_{cname}_{j}_ep", 0, selected_bins,
                                            sae_edit(sae, f"null_{cname}_{j}_ep", zero_feature(null_feat),
                                                     preserve_error=True)))
        memo: dict = {}
        profiles = run_conditions(model, int(tap.key), input_tensor, conditions,
                                  track_mean_profile(ALL_HEADS), heads=ALL_HEADS,
                                  max_batch=max_batch, memo=memo)
        arrays: dict[str, np.ndarray] = {f"{head}_{label}": profile[head]
                                         for label, profile in profiles.items()
                                         for head in ALL_HEADS}
        codes = memo["reconstruction"]

        win_data: dict[str, Any] = {
            "window_index": wi, "chrom": chrom, "start": start, "features": {}, "joint": {},
            "null": {}, "edit_norms": memo.get("edit_norms", {}),
        }

        for cname, (feat, track_col) in target_classes.items():
            col = _concept_column(bins, track_col)
            if col is not None:
                elem_mask, _ = map_annotation_column_to_grid(bins, col, start, num_bins, tap.bin_bp)
            else:
                elem_mask = intervention_mask.copy()
            eval_mask = (elem_mask & intervention_mask) if (elem_mask & intervention_mask).any() else elem_mask

            entry: dict[str, Any] = {"feature_id": feat,
                                     "fired_bins": int((codes[:, feat] > 0).sum())}
            for field_name, (suffix, baseline) in TRIAD_METHODS.items():
                entry[field_name] = _head_deltas(arrays, f"ablate_{cname}{suffix}", baseline,
                                                 eval_mask, intervention_mask)
            win_data["features"][cname] = entry
            if cname in joint:
                win_data["joint"][cname] = {
                    "feature_ids": list(joint[cname]),
                    "heads_error_preserving": _head_deltas(arrays, f"joint_{cname}_ep", "original",
                                                           eval_mask, intervention_mask),
                }
            if cname in nulls:
                win_data["null"][cname] = [
                    {"feature_id": null_feat,
                     "fired_bins": int((codes[:, null_feat] > 0).sum()),
                     "heads_error_preserving": _head_deltas(arrays, f"null_{cname}_{j}_ep", "original",
                                                            eval_mask, intervention_mask)}
                    for j, null_feat in enumerate(nulls[cname])]

        window_results.append(win_data)
        elapsed = time.time() - t0
        def shown(cname, head):
            values = [win_data["features"][cname][f][head]["delta_on_concept"] for f in TRIAD_METHODS]
            return "/".join("N/A" if v is None else f"{v:+.4f}" for v in values)
        if (wi + 1) % 5 == 0 or wi == 0:
            print(f"  [Ablation {wi+1:02d}/{len(windows)}] {chrom}:{start} | PLS ATAC: {shown('PLS_Promoter', 'atac')} "
                  f"| CTCF TF: {shown('CTCF_Insulator', 'chip_tf')} (reconstruction/error-preserving, "
                  f"{elapsed:.1f}s)", flush=True)

        del input_tensor, arrays, codes, memo, profiles

    _atomic_write_json(out_dir / "triad_40windows_results.json", window_results)

    summary_stats: dict[str, dict[str, Any]] = {}
    for cname, (feat, _col) in target_classes.items():
        summary_stats[cname] = {"feature_id": feat}
        for field_name in TRIAD_METHODS:
            summary_stats[cname][field_name] = _summarize_heads(
                [w["features"][cname][field_name] for w in window_results])
        if cname in joint:
            shared = {other: sorted(set(joint[cname]) & set(ids)) for other, ids in joint.items()
                      if other != cname and set(joint[cname]) & set(ids)}
            summary_stats[cname]["joint"] = {
                "feature_ids": list(joint[cname]),
                # Features also in another concept's set blur the joint dissociation.
                "shared_with": shared,
                "heads_error_preserving": _summarize_heads(
                    [w["joint"][cname]["heads_error_preserving"] for w in window_results]),
            }
        if cname in pools:
            null_entries = [n["heads_error_preserving"] for w in window_results for n in w["null"][cname]]
            summary_stats[cname]["null"] = {
                "pool": pools[cname], "per_window": null_per_window,
                "heads_error_preserving": _summarize_heads(null_entries),
                "concept_minus_null": _summarize_heads([
                    _minus_null(w["features"][cname]["heads_error_preserving"],
                                [n["heads_error_preserving"] for n in w["null"][cname]])
                    for w in window_results]),
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


#: how an injection raises a feature: "clamp" up to a level (first campaign),
#: or "add" a fixed amount (the same edit for every gene)
INJECTION_MODES: tuple[str, ...] = ("add", "clamp")


def matched_control_edit(sae: Any, target: int, control: int, target_level: float,
                         control_level: float, *, preserve_error: bool = False,
                         flag: str = "unmeasurable", mode: str = "clamp"):
    """Row edit for the injection control, dose-matched to the target in L2.

    Raises `control` (mode "clamp": up to control_level; "add": by
    control_level), then rescales the change from the SAE reconstruction to
    the L2 size of the change the same edit of `target` by target_level makes
    at the same row (match_control_norm). If either change is ~0 the row is
    left unchanged and the reason is stored as memo[flag], so the caller can
    drop the gene. With `preserve_error` the matched change is added to the
    real rows instead of to their reconstruction.
    """
    if mode not in INJECTION_MODES:
        raise ValueError(f"unknown injection mode {mode!r}; expected one of {INJECTION_MODES}")

    def raised(codes, feature, level):
        out = codes.clone()
        out[:, feature] = (torch.clamp_min(codes[:, feature], level) if mode == "clamp"
                           else codes[:, feature] + level)
        return out

    def edit(rows: torch.Tensor, memo: dict) -> torch.Tensor:
        codes, params = sae.encode_codes(rows)
        target_codes, control_codes = raised(codes, target, target_level), raised(codes, control, control_level)
        base = sae.decode_codes(codes, params)
        try:
            matched, _ = match_control_norm(base[None], sae.decode_codes(target_codes, params)[None],
                                            sae.decode_codes(control_codes, params)[None], 0)
        except ValueError as err:
            memo[flag] = str(err)
            return rows if preserve_error else base
        return rows + (matched[0] - base) if preserve_error else matched[0]
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


def summarize_gene_effects(readouts: dict[str, np.ndarray], tracks: dict, *,
                           target: str = "target", control: str = "control",
                           baseline: str = "reconstruction") -> dict:
    """Compute per-modality paired difference and log2 fold change across tracks."""
    summary = {}
    for head in ("rna_seq", "cage", "procap"):
        base = readouts[f"{head}_{baseline}"]
        t = readouts[f"{head}_{target}"]
        c = readouts[f"{head}_{control}"]

        offset = 0.001 if head == "rna_seq" else 1.0
        target_lfc = np.log2((t + offset) / (base + offset))
        control_lfc = np.log2((c + offset) / (base + offset))
        diff = t - c

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


def dose_tag(multiplier: float) -> str:
    """Condition suffix of a dose multiplier: 1.0 -> 'x1', 0.5 -> 'x0.5'."""
    return f"x{float(multiplier):g}"


def _inside_any(chrom: str, position: int, windows: Sequence[dict]) -> bool:
    return any(w["chrom"] == chrom and int(w["start"]) <= position < int(w["end"]) for w in windows)


def _modality_stats(measured: list[dict]) -> dict[str, dict]:
    stats = {}
    for m in TRANSCRIPTIONAL_HEADS:
        if not measured:
            stats[m] = {"mean_paired_difference": 0.0, "median_paired_difference": 0.0,
                        "t_stat": 0.0, "t_p_value": 1.0, "wilcoxon_w": 0.0, "wilcoxon_p_value": 1.0,
                        "fraction_target_greater_than_control": 0.0}
            continue
        diffs = np.array([g[m]["paired_difference"] for g in measured])
        s = compute_paired_statistics(diffs)
        stats[m] = {
            "mean_paired_difference": s.mean_delta,
            "median_paired_difference": s.median_delta,
            "t_stat": s.t_statistic,
            "t_p_value": s.t_p_value,
            "wilcoxon_w": s.wilcoxon_w,
            "wilcoxon_p_value": s.wilcoxon_p_value,
            "fraction_target_greater_than_control": float(np.mean(diffs > 0)),
        }
    return stats


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
    dose: float | None = None,
    dose_multipliers: Sequence[float] = (1.0,),
    preserve_error: bool = False,
    dose_source: str = "fixed",
    injection: str = "clamp",
    exclude_windows: Sequence[dict] = (),
) -> tuple[list[dict], dict[str, Any]]:
    """Execute MANE Select Gene-TSS injection across eligible protein-coding genes.

    For every multiplier the promoter feature is raised at the TSS bin by
    ``multiplier * dose`` (injection "add") or up to it ("clamp", the first
    campaign), each with its own L2-matched control. Clamping does nothing
    where the feature already exceeds the level, so its genes and its real
    code change differ per dose; adding is the same edit for every gene.
    `dose` is the feature's natural level (feature_dose); None keeps the first
    campaign's fixed 5.0. With `preserve_error` the edits keep the SAE's
    reconstruction error and are compared with the unedited model instead of
    the reconstruction.

    The 1x dose is the primary result ("status", "effects", "modalities").
    Every dose is under "doses" / "dose_response", and "dose_response_common"
    repeats the curve on the genes measured at every dose. Genes whose TSS
    lies in one of `exclude_windows` (the calibration windows that chose the
    feature and its dose) are left out.
    """
    multipliers = tuple(float(m) for m in dose_multipliers)
    tags = {dose_tag(m): m for m in multipliers}
    if 1.0 not in multipliers or any(m <= 0 for m in multipliers) or len(tags) != len(multipliers):
        raise ValueError(f"dose multipliers must be positive, distinct and include 1.0, got {dose_multipliers}")
    if injection not in INJECTION_MODES:
        raise ValueError(f"unknown injection mode {injection!r}; expected one of {INJECTION_MODES}")
    out_dir.mkdir(parents=True, exist_ok=True)
    eligible = eligible_genes(gtf_path, manifest, annotations, cache_dir=cache_dir)
    candidates = [g for g in eligible if not _inside_any(g["chrom"], int(g["tss0"]), exclude_windows)]
    n_genes = len(candidates)

    if dose is None:
        dose = float(sae.core.activation(torch.tensor([5.0])).item()) if hasattr(sae.core, "activation") else 5.0
    if not np.isfinite(dose) or dose <= 0:
        raise ValueError(f"injection dose must be positive, got {dose}")
    levels = {tag: m * dose for tag, m in tags.items()}
    primary = dose_tag(1.0)
    baseline = "original" if preserve_error else "reconstruction"
    ctrl_feat = control_feature_id if control_feature_id != promoter_feature_id else (promoter_feature_id + 1)
    raise_target = add_to_feature if injection == "add" else clamp_min_feature

    gene_results: list[dict] = []
    t_start = time.time()

    for rank, gene in enumerate(candidates):
        tg0 = time.time()
        chrom, start, end = gene["chrom"], gene["window_start"], gene["window_end"]
        gene_name, strand, tss0 = gene["gene_name"], gene["strand"], gene["tss0"]
        tss_bin = (tss0 - start) // 128
        tracks = select_tracks(strand)

        seq = window_sequence(fasta_path, chrom, start, end)
        input_tensor = torch.from_numpy(sequence_to_onehot(seq).astype(np.float32))[None].to(sae.channel_scale.device)

        # Original, reconstruction, and a target injection with its L2-matched
        # control per dose, batched max_batch conditions per forward.
        tss = [int(tss_bin)]
        conditions = [Condition("original"),
                      Condition("reconstruction", 0, tss, sae_edit(sae, "reconstruction"))]
        for tag, level in levels.items():
            conditions.append(Condition(f"target_{tag}", 0, tss, sae_edit(
                sae, f"target_{tag}", raise_target(promoter_feature_id, level),
                preserve_error=preserve_error)))
            conditions.append(Condition(f"control_{tag}", 0, tss, matched_control_edit(
                sae, promoter_feature_id, ctrl_feat, level, 0.7 * level,
                preserve_error=preserve_error, flag=f"unmeasurable_{tag}", mode=injection)))
        track_indices = {head: [int(t["index"]) for t in tracks[head]] for head in TRANSCRIPTIONAL_HEADS}

        def gene_readout(outputs, b, gene=gene):
            return {head: extract_transcriptional_readout(outputs, head, track_indices[head], gene, b)
                    for head in TRANSCRIPTIONAL_HEADS}

        memo: dict = {}
        per_condition = run_conditions(model, int(tap.key), input_tensor, conditions, gene_readout,
                                       heads=TRANSCRIPTIONAL_HEADS, max_batch=max_batch, memo=memo)
        readouts: dict[str, np.ndarray] = {f"{head}_{label}": values[head]
                                           for label, values in per_condition.items()
                                           for head in TRANSCRIPTIONAL_HEADS}
        a0 = float(memo["reconstruction"][0, promoter_feature_id])
        doses: dict[str, dict] = {}
        for tag, level in levels.items():
            code_change = level if injection == "add" else max(level - a0, 0.0)
            entry = {"level": level, "code_change": code_change,
                     "edit_norms": {k: memo.get("edit_norms", {}).get(f"{k}_{tag}")
                                    for k in ("target", "control")}}
            reason = memo.get(f"unmeasurable_{tag}")
            if reason is not None:
                entry.update(status="unmeasurable", reason=reason)
            else:
                entry.update(status="measured", effects=summarize_gene_effects(
                    readouts, tracks, target=f"target_{tag}", control=f"control_{tag}", baseline=baseline))
            doses[tag] = entry

        np.savez_compressed(out_dir / f"gene_rank_{rank:02d}_{gene_name}.npz", **readouts)
        record = {"rank": rank, "gene_name": gene_name, "chrom": chrom, "tss0": tss0,
                  "feature_at_tss": a0,
                  "control_at_tss": float(memo["reconstruction"][0, ctrl_feat]),
                  "status": doses[primary]["status"], "doses": doses}
        if doses[primary]["status"] == "measured":
            record["effects"] = doses[primary]["effects"]
        else:
            record["reason"] = doses[primary]["reason"]
        gene_results.append(record)

        if ((rank + 1) % 5 == 0 or rank == 0) and "effects" in record:
            rna_d = record["effects"]["rna_seq"]["paired_difference"]
            pro_d = record["effects"]["procap"]["paired_difference"]
            print(f"  [Gene {rank+1:02d}/{n_genes}] {gene_name:<10} | RNA Diff: {rna_d:+.4f} | PRO-cap Diff: {pro_d:+.4f} ({time.time()-tg0:.1f}s)", flush=True)

        del input_tensor, readouts, per_condition, memo

    _atomic_write_json(out_dir / "gene_cohort_results.json", gene_results)

    def curve(genes: list[dict]) -> dict:
        out = {}
        for tag, level in levels.items():
            measured = [g["doses"][tag]["effects"] for g in genes if g["doses"][tag]["status"] == "measured"]
            out[tag] = {"multiplier": tags[tag], "level": level,
                        "n_measured": len(measured), "modalities": _modality_stats(measured)}
        return out

    dose_response = curve(gene_results)
    common = [g for g in gene_results if all(d["status"] == "measured" for d in g["doses"].values())]
    gene_stats: dict[str, Any] = {
        "n_total_eligible": n_genes,
        "n_excluded_calibration": len(eligible) - n_genes,
        "n_measured": dose_response[primary]["n_measured"],
        "dose": {"level": dose, "source": dose_source, "multipliers": list(multipliers),
                 "injection": injection},
        "baseline": baseline,
        "promoter_feature_id": int(promoter_feature_id),
        "control_feature_id": int(ctrl_feat),
        "modalities": dose_response[primary]["modalities"],
        "dose_response": dose_response,
        "n_measured_every_dose": len(common),
        "dose_response_common": curve(common),
    }

    _atomic_write_json(out_dir / "gene_level_statistical_summary.json", gene_stats)
    print(f"Gene cohort complete in {time.time()-t_start:.1f}s! Saved to {out_dir / 'gene_level_statistical_summary.json'}")
    cleanup_gpu()
    return gene_results, gene_stats


@dataclass(frozen=True)
class Protocol:
    """How a configuration's features are chosen and edited.

    The defaults are the v2 protocol; `Protocol.legacy()` is the first
    campaign's (largest window-averaged mean difference, one feature, fixed
    dose 5.0 clamped in, edits through the reconstruction), so old SAEs can
    be re-run as they were. The triad always reports both edit methods
    (TRIAD_METHODS); under v2 the error-preserving one is primary.
    """

    name: str = "v2"
    score: str = "cohen_d"
    top_n: int = 3
    min_fire_rate: float = 0.05
    fixed_dose: float | None = None          # None: the feature's natural level
    dose_quantile: float = 0.99
    dose_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    injection: str = "add"
    preserve_error: bool = True               # injection edits
    n_windows: int = 40
    window_sampling: str = "random"           # "first": the first campaign's
    window_seed: int = 2026
    null_per_window: int = 1                  # matched null features per concept and window
    null_pool_size: int = 50

    def __post_init__(self):
        if self.name not in PROTOCOLS:
            raise ValueError(f"unknown protocol {self.name!r}; expected one of {PROTOCOLS}")
        if self.score not in SCORES:
            raise ValueError(f"unknown score {self.score!r}; expected one of {SCORES}")
        if self.injection not in INJECTION_MODES:
            raise ValueError(f"unknown injection {self.injection!r}; expected one of {INJECTION_MODES}")
        if self.window_sampling not in WINDOW_SAMPLINGS:
            raise ValueError(f"unknown window sampling {self.window_sampling!r}")
        if self.n_windows < 1 or self.null_per_window < 0 or self.null_pool_size < 1 or self.top_n < 1:
            raise ValueError("n_windows, null_pool_size and top_n must be >= 1, null_per_window >= 0")

    @classmethod
    def legacy(cls, **overrides) -> "Protocol":
        base = dict(name="legacy", score="mean_diff", top_n=1, min_fire_rate=0.0, fixed_dose=5.0,
                    dose_multipliers=(1.0,), injection="clamp", preserve_error=False,
                    window_sampling="first", null_per_window=0)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def named(cls, name: str, **overrides) -> "Protocol":
        return cls.legacy(**overrides) if name == "legacy" else cls(name=name, **overrides)

    @property
    def triad_primary(self) -> str:
        return "heads" if self.name == "legacy" else "heads_error_preserving"

    def to_json(self) -> dict:
        from dataclasses import asdict
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


#: protocol names; results of each go to their own folders (protocol_suffix)
PROTOCOLS: tuple[str, ...] = ("v2", "legacy")


def protocol_suffix(protocol: str) -> str:
    """Folder suffix of a protocol: none for legacy (the first campaign's folders)."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    return "" if protocol == "legacy" else f"_protocol_{protocol}"


def campaign_name(sae_pool: str, protocol: str = "legacy") -> str:
    """analysis_runs folder (and Drive folder) of a pool's triad/injection results.

    campaign_matrix, campaign_matrix_<pool>, or with _protocol_v2 appended,
    so a v2 run never finds, skips over or mixes with the first campaign's
    files.
    """
    base = "campaign_matrix" if sae_pool == "sae" else f"campaign_matrix_{sae_pool}"
    return base + protocol_suffix(protocol)


def check_resumable(out_dir: Path, protocol: Protocol) -> None:
    """Refuse to continue a configuration folder written under another protocol."""
    record = out_dir / "feature_discovery.json"
    done = [f.name for f in (out_dir / "regulatory_triad_summary.json",
                             out_dir / "gene_level_statistical_summary.json") if is_valid_json_file(f)]
    if record.is_file():
        stored = json.loads(record.read_text()).get("protocol")
        if stored != protocol.to_json():
            raise RuntimeError(f"{out_dir} was run with protocol {stored}, not {protocol.to_json()}; "
                               "use another --out")
    elif done and protocol != Protocol.legacy():
        raise RuntimeError(f"{out_dir} holds {done} from an unrecorded (first-campaign) protocol; "
                           "use another --out")


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
    protocol: Protocol | None = None,
) -> None:
    """Orchestrate evaluation for a single (fold, layer, seed) configuration."""
    protocol = protocol or Protocol()
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
        windows = None
    windows, calib_windows = split_windows(manifest, protocol, windows)
    # Fail before the models are loaded if this folder belongs to another protocol.
    check_resumable(out_dir, protocol)

    # 2. Load models
    print(f"Loading SAE from {sae_path}...")
    tap = TAPS[layer]
    sae = BorzoiSAE.from_checkpoint(sae_path, device="cuda")

    print(f"Loading AlphaGenome from {weights_path}...")
    model = AlphaGenome.from_pretrained(weights_path, dtype_policy=DtypePolicy.mixed_precision(), device="cuda")
    model.eval().requires_grad_(False)
    organism = torch.tensor([0], device="cuda")

    # 3. Dynamic Feature Discovery
    features = discover_concept_features(sae, model, tap, calib_windows, annotations, fasta_path, organism,
                                         score=protocol.score, top_n=protocol.top_n,
                                         min_fire_rate=protocol.min_fire_rate)
    if protocol.fixed_dose is not None:
        dose, dose_source = float(protocol.fixed_dose), "fixed"
    else:
        dose = feature_dose(sae, model, tap, calib_windows, fasta_path, features.promoter_feature_id,
                            quantile=protocol.dose_quantile)
        dose_source = f"q{protocol.dose_quantile:g} of non-zero values in {len(calib_windows)} calibration windows"
    _atomic_write_json(out_dir / "feature_discovery.json", {
        **features.to_json(), "protocol": protocol.to_json(),
        "triad_primary_method": protocol.triad_primary,
        "evaluation_windows": windows,
        "injection_dose": {"feature_id": features.promoter_feature_id, "level": dose, "source": dose_source},
        "calibration_windows": calib_windows,
    })
    print(f"[Discovery] {features.to_json()['primary']} (score {protocol.score}); "
          f"injection dose {dose:.3f} ({dose_source})")

    # 4. Regulatory Triad Ablation
    triad_summary_file = out_dir / "regulatory_triad_summary.json"
    if is_valid_json_file(triad_summary_file):
        print(f"[Ablation] Found valid {triad_summary_file}, skipping Part 1...")
    else:
        run_triad_ablation(sae, model, tap, windows, annotations, features, fasta_path, organism, out_dir,
                           max_batch=max_batch, top_n=protocol.top_n,
                           null_per_window=protocol.null_per_window,
                           null_pool_size=protocol.null_pool_size, null_seed=protocol.window_seed)

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
            organism=organism, out_dir=out_dir, cache_dir=cache_dir, max_batch=max_batch,
            dose=dose, dose_multipliers=protocol.dose_multipliers,
            preserve_error=protocol.preserve_error, dose_source=dose_source,
            injection=protocol.injection,
            # Genes in the windows that chose the feature and its dose would leak.
            exclude_windows=() if protocol.name == "legacy" else calib_windows,
        )

    print(f"\nConfiguration {fold} {layer} {seed} completed successfully in {time.time()-t_start:.1f}s!\n")


def add_protocol_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", choices=list(PROTOCOLS), default="v2",
                        help="v2: within-window Cohen's d top-3, natural additive dose-response, "
                             "error-preserving injection; legacy: the first campaign's choices")
    parser.add_argument("--score", choices=list(SCORES), default=None, help="override the protocol's score")
    parser.add_argument("--top-n", type=int, default=None, help="features ranked (and jointly ablated) per concept")
    parser.add_argument("--min-fire-rate", type=float, default=None,
                        help="cohen_d: minimum share of positive bins a feature must fire on")
    dose = parser.add_mutually_exclusive_group()
    dose.add_argument("--fixed-dose", type=float, default=None,
                      help="inject this level instead of the feature's natural one")
    dose.add_argument("--natural-dose", action="store_true",
                      help="use the feature's natural level (the v2 default) under any protocol")
    parser.add_argument("--dose-quantile", type=float, default=None)
    parser.add_argument("--dose-multipliers", type=float, nargs="+", default=None,
                        help="multiples of the dose to inject; must include 1")
    parser.add_argument("--injection", choices=list(INJECTION_MODES), default=None)
    parser.add_argument("--preserve-error", action=argparse.BooleanOptionalAction, default=None,
                        help="injection edits keep the SAE reconstruction error")
    parser.add_argument("--n-windows", type=int, default=None, help="evaluation windows per fold")
    parser.add_argument("--null-per-window", type=int, default=None,
                        help="matched null features per concept and window (0: none)")


def protocol_from_args(args: argparse.Namespace) -> Protocol:
    overrides = {"score": args.score, "top_n": args.top_n, "min_fire_rate": args.min_fire_rate,
                 "fixed_dose": args.fixed_dose, "dose_quantile": args.dose_quantile,
                 "dose_multipliers": tuple(args.dose_multipliers) if args.dose_multipliers else None,
                 "injection": args.injection, "preserve_error": args.preserve_error,
                 "n_windows": args.n_windows, "null_per_window": args.null_per_window}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if args.natural_dose:
        overrides["fixed_dose"] = None
    if args.protocol == "legacy" and overrides.get("score") == "cohen_d" and args.min_fire_rate is None:
        overrides["min_fire_rate"] = Protocol().min_fire_rate   # keep the guard cohen_d needs
    return Protocol.named(args.protocol, **overrides)


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
    add_protocol_arguments(parser)

    args = parser.parse_args(argv)
    protocol = protocol_from_args(args)
    fold_num = args.fold.replace("fold", "")
    weights = args.weights or Path(f"/mnt/ag/weights/model_fold_{fold_num}.safetensors")
    sae = args.sae or Path(f"/mnt/ag/{args.sae_pool}/{args.fold}/{args.layer}_{args.seed}/best.pt")
    campaign = campaign_name(args.sae_pool, protocol.name)
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
    # The first campaign's hand-made plans apply to it only; v2 picks its own windows.
    auto_plan = (fold_plan if fold_plan.is_file() else
                 (default_plan if default_plan.is_file() and args.fold == "fold1" else None))
    plan = args.plan or (auto_plan if protocol.name == "legacy" else None)

    run_single_configuration(
        fold=args.fold, layer=args.layer, seed=args.seed,
        weights_path=weights, sae_path=sae,
        manifest_path=manifest, annotations_path=annotations,
        fasta_path=args.fasta, gtf_path=args.gtf,
        out_dir=out, plan_path=plan, cache_dir=args.cache_dir, max_batch=args.max_batch,
        protocol=protocol,
    )
    return 0


if __name__ == "__main__":
    sys.exit(cmd_intervene())

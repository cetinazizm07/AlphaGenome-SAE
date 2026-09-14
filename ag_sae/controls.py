"""Controls that ask whether a concept match means what it looks like.

Three questions the main matcher cannot answer on its own.

* Would the feature still separate the concept if the obvious confound were
  held fixed? `stratified_auroc`.
* Does the separation survive on territory the backbone never trained on?
  `memorisation_gap`.
* Do independently seeded dictionaries agree about which feature carries the
  concept? `concept_seed_agreement`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

#: Fewest bins of each class a stratum needs before it contributes.
MIN_PER_STRATUM = 10


# --------------------------------------------------------------------------
# Conditioning on a confound
# --------------------------------------------------------------------------


def quantile_strata(values: np.ndarray, n_strata: int = 5) -> np.ndarray:
    """Split a covariate into roughly equal-sized strata.

    Returns -1 where the covariate is missing, so those bins drop out rather
    than joining an arbitrary stratum.
    """
    values = np.asarray(values, dtype=float)
    if n_strata < 2:
        raise ValueError("Need at least two strata")
    ok = np.isfinite(values)
    out = np.full(values.size, -1, dtype=np.int64)
    if not ok.any():
        return out
    edges = np.quantile(values[ok], np.linspace(0, 1, n_strata + 1)[1:-1])
    out[ok] = np.searchsorted(edges, values[ok], side="right")
    return out


def _wins_and_pairs(values: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney concordant-pair count and the number of pairs."""
    from scipy.stats import rankdata

    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos < 1 or n_neg < 1:
        return 0.0, 0.0
    ranks = rankdata(np.asarray(values, dtype=np.float32), method="average")
    wins = float(ranks[labels].sum(dtype=np.float64) - n_pos * (n_pos + 1) / 2)
    return wins, float(n_pos) * n_neg


def stratified_auroc(
    values: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    min_per_stratum: int = MIN_PER_STRATUM,
) -> dict:
    """AUROC computed inside strata, then pooled over comparable pairs.

    A feature can look like it marks promoters when what it really tracks is GC
    content, because promoters are GC rich. Inside a stratum the covariate is
    nearly constant, so whatever separation survives is separation the covariate
    does not explain.

    Pooling weights each stratum by its pair count, n_pos * n_neg, which is how
    many comparisons it contributes. The pooled value is then the fraction of
    concordant pairs among pairs drawn from the same stratum, which is the
    quantity a reader means by "controlling for GC".
    """
    values = np.asarray(values, dtype=float).ravel()
    labels = np.asarray(labels, dtype=bool).ravel()
    strata = np.asarray(strata, dtype=np.int64).ravel()
    if not (values.size == labels.size == strata.size):
        raise ValueError("values, labels and strata must describe the same bins")

    total_wins = total_pairs = 0.0
    used, skipped = [], []
    for stratum in np.unique(strata):
        if stratum < 0:
            continue
        mask = strata == stratum
        in_labels = labels[mask]
        n_pos = int(in_labels.sum())
        n_neg = int(in_labels.size - n_pos)
        if n_pos < min_per_stratum or n_neg < min_per_stratum:
            skipped.append(int(stratum))
            continue
        wins, pairs = _wins_and_pairs(values[mask], in_labels)
        total_wins += wins
        total_pairs += pairs
        used.append({"stratum": int(stratum), "n_positive": n_pos,
                     "n_negative": n_neg, "auroc": wins / pairs})
    if total_pairs == 0:
        raise ValueError("No stratum had enough of both classes")
    unadjusted_wins, unadjusted_pairs = _wins_and_pairs(values, labels)
    return {
        "pooled_auroc": total_wins / total_pairs,
        "unadjusted_auroc": unadjusted_wins / unadjusted_pairs,
        "strata_used": len(used),
        "strata_skipped": skipped,
        "per_stratum": pd.DataFrame(used),
    }


def gc_fraction(bins: pd.DataFrame, fasta: str | Path) -> np.ndarray:
    """GC content of every bin, straight from the genome.

    NaN where the bin holds an ambiguous base, so `quantile_strata` drops it
    rather than placing it by a guess.
    """
    from ag_sae.extract import FastaIndex

    index = FastaIndex(fasta)
    try:
        out = np.empty(len(bins), dtype=float)
        for position, row in enumerate(bins.itertuples(index=False)):
            sequence = index.fetch(row.chrom, int(row.bin_start), int(row.bin_end)).upper()
            counts = {base: sequence.count(base) for base in "ACGT"}
            known = sum(counts.values())
            out[position] = (np.nan if known < len(sequence)
                             else (counts["G"] + counts["C"]) / known)
        return out
    finally:
        index.close()


# --------------------------------------------------------------------------
# Did the backbone simply remember it
# --------------------------------------------------------------------------


def memorisation_gap(
    unseen: pd.DataFrame, trained: pd.DataFrame, column: str = "sae_excess"
) -> pd.DataFrame:
    """Per-concept difference between held-out and trained-on territory.

    `test` and `test_trained` are the same size by construction, so the two runs
    differ only in whether the backbone saw the sequence. A concept that scores
    much higher on trained territory is telling you about memorisation in the
    backbone, not about a feature that generalises.
    """
    for frame, label in ((unseen, "unseen"), (trained, "trained")):
        for needed in ("concept", column):
            if needed not in frame.columns:
                raise ValueError(f"{label} frame has no {needed!r} column")
    merged = unseen[["concept", column]].merge(
        trained[["concept", column]], on="concept", how="inner",
        suffixes=("_unseen", "_trained"))
    if merged.empty:
        raise ValueError("The two runs share no concepts")
    merged["gap"] = merged[f"{column}_unseen"] - merged[f"{column}_trained"]
    return merged.sort_values("gap")


# --------------------------------------------------------------------------
# Do the seeds agree
# --------------------------------------------------------------------------


def concept_seed_agreement(
    decoders: Mapping[int, np.ndarray],
    best_feature: Mapping[int, Mapping[str, int]],
    threshold: float = 0.7,
) -> pd.DataFrame:
    """For each concept and seed pair, did both seeds land on the same feature.

    Dictionary-level stability says how much of the dictionary recurs. This asks
    the sharper question: the feature that carried the concept in one seed, is
    it the same direction as the one that carried it in another. A concept that
    fails here was found by one initialisation and is not a property of the
    model.

    `nearest_is_best` is the strict reading: the closest match to seed A's
    winner is exactly seed B's winner. `cosine` is the readable number.
    """
    seeds = sorted(decoders)
    if len(seeds) < 2:
        raise ValueError("Need at least two seeds to compare")
    shapes = {np.asarray(decoders[s]).shape for s in seeds}
    if len(shapes) != 1:
        raise ValueError(f"Decoders disagree in shape: {sorted(shapes)}")

    unit = {}
    for seed in seeds:
        matrix = np.asarray(decoders[seed], dtype=float)
        unit[seed] = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)

    rows = []
    concepts = sorted(set().union(*(set(best_feature[s]) for s in seeds)))
    for concept in concepts:
        for i, left in enumerate(seeds):
            for right in seeds[i + 1:]:
                if concept not in best_feature[left] or concept not in best_feature[right]:
                    continue
                a = int(best_feature[left][concept])
                b = int(best_feature[right][concept])
                similarity = unit[right] @ unit[left][a]
                rows.append({
                    "concept": concept,
                    "pair": f"{left}-{right}",
                    "feature_a": a,
                    "feature_b": b,
                    "cosine": float(similarity[b]),
                    "nearest_is_best": bool(int(similarity.argmax()) == b),
                    "best_available_cosine": float(similarity.max()),
                    "agrees": bool(similarity[b] >= threshold),
                })
    if not rows:
        raise ValueError("No concept was scored in two or more seeds")
    return pd.DataFrame(rows)

"""Match a feature's sequence preference against a motif database.

This is the arm that does not depend on the AUROC pipeline. A feature whose
top-activating sequences build a position weight matrix that matches a known
transcription factor motif is evidence of a different kind: it survives even if
something in the concept matching turns out to be wrong.

The comparison is written here rather than shelled out to TOMTOM, so the study
carries no MEME suite dependency. It is not TOMTOM's statistic. The score is the
best Pearson correlation over all ungapped alignments of the two matrices, and
the p-value is empirical, from column-shuffled copies of the query. Report it as
that, never as a TOMTOM q-value.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

BASES = "ACGT"
#: Columns narrower than this contribute little and add noise to an alignment.
MIN_OVERLAP = 5


def _open(path: str | Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "rt")


def read_jaspar(path: str | Path) -> dict[str, np.ndarray]:
    """Read a JASPAR PFM file into (length, 4) probability matrices.

    Accepts the stacked format, where each motif is a `>ID name` line followed
    by four rows of counts in A, C, G, T order.
    """
    motifs: dict[str, np.ndarray] = {}
    name, rows = None, []
    with _open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    motifs[name] = _counts_to_probabilities(rows)
                parts = line[1:].split(None, 1)
                name = parts[1].strip() if len(parts) > 1 else parts[0]
                rows = []
            else:
                numbers = re.findall(r"-?\d+\.?\d*", line)
                if numbers:
                    rows.append([float(v) for v in numbers])
    if name is not None:
        motifs[name] = _counts_to_probabilities(rows)
    if not motifs:
        raise ValueError(f"{path}: no motifs found")
    return motifs


def _counts_to_probabilities(rows: Sequence[Sequence[float]]) -> np.ndarray:
    if len(rows) != 4:
        raise ValueError(f"A PFM needs four rows, got {len(rows)}")
    counts = np.asarray(rows, dtype=float).T          # (length, 4)
    total = counts.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise ValueError("A PFM column has no counts")
    return counts / total


def pwm_from_sequences(
    sequences: Sequence[str], weights: np.ndarray | None = None,
    pseudocount: float = 0.25,
) -> np.ndarray:
    """Position weight matrix of equal-length sequences, as probabilities.

    Weighting by activation lets strongly firing sites count for more. The
    pseudocount keeps a column that happens to be pure from producing a zero
    that no database motif can ever match.
    """
    if not sequences:
        raise ValueError("No sequences")
    length = len(sequences[0])
    if any(len(s) != length for s in sequences):
        raise ValueError("Sequences must all be the same length")
    weights = (np.ones(len(sequences)) if weights is None
               else np.clip(np.asarray(weights, dtype=float), 0, None))
    if weights.shape != (len(sequences),):
        raise ValueError("One weight per sequence")
    if weights.sum() <= 0:
        weights = np.ones_like(weights)

    counts = np.full((length, 4), pseudocount, dtype=float)
    index = {base: i for i, base in enumerate(BASES)}
    for sequence, weight in zip(sequences, weights):
        for position, base in enumerate(sequence.upper()):
            if base in index:
                counts[position, index[base]] += weight
    return counts / counts.sum(axis=1, keepdims=True)


def reverse_complement(pwm: np.ndarray) -> np.ndarray:
    """The matrix a motif would have on the other strand."""
    return pwm[::-1, ::-1]


def align_score(query: np.ndarray, target: np.ndarray,
                min_overlap: int = MIN_OVERLAP) -> tuple[float, int]:
    """Best Pearson correlation over ungapped offsets, and the offset used.

    Correlation rather than a distance, because the two matrices come from
    different sources and only their shape should matter, not their scale.
    """
    query = np.asarray(query, dtype=float)
    target = np.asarray(target, dtype=float)
    if query.ndim != 2 or query.shape[1] != 4 or target.shape[1] != 4:
        raise ValueError("Both matrices must be (length, 4)")
    best, best_offset = -1.0, 0
    for offset in range(-(len(target) - min_overlap), len(query) - min_overlap + 1):
        q_lo, t_lo = max(offset, 0), max(-offset, 0)
        span = min(len(query) - q_lo, len(target) - t_lo)
        if span < min_overlap:
            continue
        a = query[q_lo:q_lo + span].ravel()
        b = target[t_lo:t_lo + span].ravel()
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        score = float(np.corrcoef(a, b)[0, 1])
        if score > best:
            best, best_offset = score, offset
    return best, best_offset


def compare(query: np.ndarray, target: np.ndarray,
            min_overlap: int = MIN_OVERLAP) -> tuple[float, str]:
    """Best score over both strands of the target."""
    forward, _ = align_score(query, target, min_overlap)
    reverse, _ = align_score(query, reverse_complement(target), min_overlap)
    return (forward, "+") if forward >= reverse else (reverse, "-")


def match_motifs(
    query: np.ndarray,
    database: Mapping[str, np.ndarray],
    *,
    n_null: int = 200,
    seed: int = 0,
    top_n: int = 10,
    min_overlap: int = MIN_OVERLAP,
) -> pd.DataFrame:
    """Rank database motifs against one query matrix, with an empirical p.

    The null shuffles the query's columns. That keeps each column's base
    composition and destroys only their order, so the p-value asks whether the
    match is about the motif's arrangement rather than its overall letter mix.
    An unshuffled query against a large database will always match something.
    """
    if not database:
        raise ValueError("Empty motif database")
    rng = np.random.default_rng(seed)
    query = np.asarray(query, dtype=float)

    scores = {name: compare(query, target, min_overlap)
              for name, target in database.items()}
    null_best = np.empty(n_null, dtype=float)
    for draw in range(n_null):
        shuffled = query[rng.permutation(len(query))]
        null_best[draw] = max(
            compare(shuffled, target, min_overlap)[0] for target in database.values())

    rows = []
    for name, (score, strand) in scores.items():
        # (1 + count) / (1 + n) so a p-value is never reported as exactly zero.
        p = float((1 + int((null_best >= score).sum())) / (1 + n_null))
        rows.append({"motif": name, "score": score, "strand": strand,
                     "p_value": p, "null_p95": float(np.quantile(null_best, 0.95))})
    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    return out.head(top_n).reset_index(drop=True)

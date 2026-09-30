"""Match a feature's sequence preference against a motif database using standard bioinformatics APIs.

This module refactors AlphaGenome-SAE motif handling on top of industry-standard
computational genomics libraries:
  - Biopython (Bio.motifs): standard representation for PFMs, PWMs, PSSMs, and fast
    sequence scanning on both strands.
  - Logomaker: publication-ready vector sequence logo visualizations.

Main Capabilities:
  1. Parsing: Read JASPAR PFM collections into standard Bio.motifs and JasparMotif objects.
  2. Construction: Build weighted PWMs from top-activating sequences.
  3. Scanning: Scan genomic sequences on forward and reverse strands for exact motif anchors.
  4. Alignment & Empirical Null: Compute ungapped Pearson correlation over alignments
     and rank motifs against a column-shuffled null distribution.
  5. Visualization: Generate publication-ready sequence logos via Logomaker.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import io
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from Bio import motifs as bio_motifs
from Bio.Seq import Seq
import numpy as np
import pandas as pd

BASES = "ACGT"
#: Minimum overlap in bases for matrix alignment
MIN_OVERLAP = 5


def _open_handle(path: str | Path):
    """Open a text file or gzipped file for reading."""
    p = Path(path)
    return gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" else open(p, "rt", encoding="utf-8")


@dataclass(frozen=True)
class MotifHit:
    """Exact transcription factor binding site match on a genomic sequence."""

    motif_id: str
    motif_name: str
    start: int
    end: int
    strand: str  # '+' or '-'
    score: float
    sequence: str

    @property
    def length(self) -> int:
        return self.end - self.start


class JasparMotif:
    """High-level, object-oriented representation of a JASPAR motif wrapping Bio.motifs.

    Provides standardized access to counts, PWM, PSSM, sequence scanning,
    reverse complementation, and logo visualization.
    """

    def __init__(
        self,
        matrix_id: str,
        name: str,
        counts: np.ndarray,
        bio_motif: bio_motifs.Motif | None = None,
        pseudocount: float = 0.25,
    ):
        self.matrix_id = matrix_id
        self.name = name or matrix_id
        counts = np.asarray(counts, dtype=float)
        if counts.ndim != 2 or counts.shape[1] != 4:
            raise ValueError(f"Counts must be (length, 4), got {counts.shape}")
        self.counts = counts
        self._pseudocount = pseudocount

        total = counts.sum(axis=1, keepdims=True)
        if (total <= 0).any():
            raise ValueError("A PFM column has no counts")
        self.pwm = counts / total

        if bio_motif is not None:
            self._bio_motif = bio_motif
        else:
            # Build underlying Bio.motifs object from counts dict
            counts_dict = {
                b: list(self.counts[:, i]) for i, b in enumerate(BASES)
            }
            self._bio_motif = bio_motifs.Motif(alphabet="ACGT", counts=counts_dict)
            self._bio_motif.matrix_id = self.matrix_id
            self._bio_motif.name = self.name

        eff_pseudocount = pseudocount if pseudocount > 0 else (0.0 if (self.counts > 0).all() else 0.25)
        self._bio_pwm = self._bio_motif.counts.normalize(pseudocounts=eff_pseudocount)
        self._bio_pssm = self._bio_pwm.log_odds()

    @property
    def length(self) -> int:
        return len(self.counts)

    @property
    def consensus(self) -> str:
        return str(self._bio_motif.consensus)

    @property
    def degenerate_consensus(self) -> str:
        return str(self._bio_motif.degenerate_consensus)

    @property
    def bio_motif(self) -> bio_motifs.Motif:
        return self._bio_motif

    @property
    def bio_pwm(self):
        return self._bio_pwm

    @property
    def bio_pssm(self):
        return self._bio_pssm

    def reverse_complement(self) -> JasparMotif:
        """Return the reverse complement motif as a new JasparMotif object."""
        rc_bio = self._bio_motif.reverse_complement()
        rc_counts = self.counts[::-1, ::-1].copy()
        return JasparMotif(
            matrix_id=f"{self.matrix_id}_rc",
            name=f"{self.name}_rc",
            counts=rc_counts,
            bio_motif=rc_bio,
            pseudocount=self._pseudocount,
        )

    def scan(
        self,
        sequence: str | Seq,
        threshold: float | None = None,
        min_score_ratio: float = 0.8,
    ) -> list[MotifHit]:
        """Scan a DNA sequence for motif instances on both forward and reverse strands."""
        return scan_sequence_for_motifs(
            sequence=sequence,
            motif=self,
            threshold=threshold,
            min_score_ratio=min_score_ratio,
        )

    def to_dataframe(self, use_probabilities: bool = True) -> pd.DataFrame:
        """Convert motif matrix to a pandas DataFrame suitable for Logomaker."""
        matrix = self.pwm if use_probabilities else self.counts
        return pd.DataFrame(matrix, columns=list(BASES))

    def plot_logo(
        self,
        ax: Any | None = None,
        title: str | None = None,
        color_scheme: str = "classic",
    ) -> Any:
        """Plot a sequence logo for this motif using Logomaker."""
        return plot_motif_logo(
            matrix_or_motif=self,
            ax=ax,
            title=title or f"{self.name} ({self.matrix_id})",
            color_scheme=color_scheme,
        )

    def __repr__(self) -> str:
        return f"<JasparMotif {self.matrix_id} ({self.name}), length={self.length}, consensus={self.consensus}>"


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _read_jaspar_raw_blocks(path: str | Path) -> list[tuple[str, str, list[list[float]]]]:
    """Parse raw lines of JASPAR stacked PFM file, validating structure."""
    blocks: list[tuple[str, str, list[list[float]]]] = []
    current_id, current_name = None, None
    current_rows: list[list[float]] = []

    with _open_handle(path) as handle:
        lines = [line.strip() for line in handle]

    non_empty = [l for l in lines if l and not l.startswith(("#", "track", "browser"))]
    if not non_empty:
        raise ValueError(f"{path}: no motifs found")

    for line in non_empty:
        if line.startswith(">"):
            if current_id is not None:
                if len(current_rows) != 4:
                    raise ValueError(f"A PFM needs four rows, got {len(current_rows)}")
                blocks.append((current_id, current_name, current_rows))
            header = line[1:].strip()
            parts = header.split(None, 1)
            current_id = parts[0]
            current_name = parts[1].strip() if len(parts) > 1 else parts[0]
            current_rows = []
        else:
            numbers = re.findall(r"-?\d+\.?\d*", line)
            if numbers:
                current_rows.append([float(v) for v in numbers])

    if current_id is not None:
        if len(current_rows) != 4:
            raise ValueError(f"A PFM needs four rows, got {len(current_rows)}")
        blocks.append((current_id, current_name, current_rows))

    if not blocks:
        raise ValueError(f"{path}: no motifs found")
    return blocks


def read_jaspar_records(path: str | Path) -> dict[str, JasparMotif]:
    """Read a JASPAR PFM file into a dictionary of high-level JasparMotif objects."""
    blocks = _read_jaspar_raw_blocks(path)
    records: dict[str, JasparMotif] = {}
    for matrix_id, name, rows in blocks:
        counts = np.asarray(rows, dtype=float).T
        record = JasparMotif(matrix_id=matrix_id, name=name, counts=counts)
        records[name] = record
    return records


def read_jaspar(path: str | Path) -> dict[str, np.ndarray]:
    """Read a JASPAR PFM file into (length, 4) probability matrices.

    Maintains 100% backward compatibility with legacy pipelines, returning
    {motif_name: (length, 4) numpy array}.
    """
    records = read_jaspar_records(path)
    return {name: motif.pwm for name, motif in records.items()}


# --------------------------------------------------------------------------
# Construction from Sequences
# --------------------------------------------------------------------------


def pwm_from_sequences(
    sequences: Sequence[str],
    weights: np.ndarray | None = None,
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


def motif_from_sequences(
    sequences: Sequence[str],
    weights: np.ndarray | None = None,
    name: str = "learned_latent_motif",
    matrix_id: str = "SAE0001",
    pseudocount: float = 0.25,
) -> JasparMotif:
    """Construct a full JasparMotif object from top-activating sequences with weighted counts."""
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

    raw_counts = (
        np.full((length, 4), pseudocount, dtype=float)
        if pseudocount > 0
        else np.zeros((length, 4), dtype=float)
    )
    index = {base: i for i, base in enumerate(BASES)}
    for sequence, weight in zip(sequences, weights):
        for position, base in enumerate(sequence.upper()):
            if base in index:
                raw_counts[position, index[base]] += weight

    # If any position has no counts (e.g. all ambiguous bases 'N' with pseudocount=0),
    # assign uniform baseline counts so the motif is always well-formed.
    zero_mask = raw_counts.sum(axis=1) <= 0
    if zero_mask.any():
        raw_counts[zero_mask, :] = 0.25

    return JasparMotif(
        matrix_id=matrix_id,
        name=name,
        counts=raw_counts,
        pseudocount=0.0 if pseudocount > 0 else 0.25,
    )


# --------------------------------------------------------------------------
# Sequence Scanning (Bio.motifs PSSM Engine)
# --------------------------------------------------------------------------


def scan_sequence_for_motifs(
    sequence: str | Seq,
    motif: JasparMotif | bio_motifs.Motif | np.ndarray,
    threshold: float | None = None,
    min_score_ratio: float = 0.8,
) -> list[MotifHit]:
    """Scan a genomic DNA sequence on both strands using standard Bio.motifs PSSM search.

    Args:
        sequence: The DNA sequence string or Bio.Seq.Seq to scan.
        motif: A JasparMotif, Bio.motifs.Motif, or (length, 4) probability matrix.
        threshold: Absolute log-odds score threshold. If None, set to min_score_ratio * pssm.max.
        min_score_ratio: Ratio of maximum theoretical score to accept (default: 0.8).

    Returns:
        List of MotifHit instances sorted by genomic start coordinate.
    """
    if isinstance(motif, JasparMotif):
        pssm = motif.bio_pssm
        motif_id = motif.matrix_id
        motif_name = motif.name
        m_len = motif.length
    elif isinstance(motif, bio_motifs.Motif):
        pssm = motif.counts.normalize(0.25).log_odds()
        motif_id = getattr(motif, "matrix_id", "MOTIF")
        motif_name = getattr(motif, "name", "MOTIF")
        m_len = motif.length
    elif isinstance(motif, np.ndarray):
        jm = JasparMotif(matrix_id="QUERY", name="QUERY", counts=motif * 100)
        pssm = jm.bio_pssm
        motif_id = jm.matrix_id
        motif_name = jm.name
        m_len = jm.length
    else:
        raise TypeError(f"Unsupported motif type: {type(motif)}")

    seq_obj = Seq(str(sequence).upper())
    effective_thresh = pssm.max * min_score_ratio if threshold is None else float(threshold)

    hits: list[MotifHit] = []
    # Biopython pssm.search scans both strands:
    seq_len = len(seq_obj)
    for pos, score in pssm.search(seq_obj, threshold=effective_thresh):
        if pos >= 0:
            strand = "+"
            start = int(pos)
        else:
            strand = "-"
            start = int(pos + seq_len)
        end = start + m_len
        matched_seq = str(seq_obj[start:end])
        hits.append(MotifHit(
            motif_id=motif_id,
            motif_name=motif_name,
            start=start,
            end=end,
            strand=strand,
            score=float(score),
            sequence=matched_seq,
        ))

    hits.sort(key=lambda h: (h.start, -h.score))
    return hits


# --------------------------------------------------------------------------
# Alignment & Matrix Correlation
# --------------------------------------------------------------------------


def reverse_complement(pwm: np.ndarray) -> np.ndarray:
    """The matrix a motif would have on the other strand."""
    return np.asarray(pwm)[::-1, ::-1]


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
    database: Mapping[str, np.ndarray | JasparMotif],
    *,
    n_null: int = 200,
    seed: int = 0,
    top_n: int = 10,
    min_overlap: int = MIN_OVERLAP,
) -> pd.DataFrame:
    """Rank database motifs against one query matrix, with an empirical p-value.

    The null distribution shuffles the query's columns (preserving base composition
    while disrupting sequence arrangement).
    """
    if not database:
        raise ValueError("Empty motif database")
    rng = np.random.default_rng(seed)
    query = np.asarray(query, dtype=float)

    # Standardize database to numpy arrays
    db_arrays: dict[str, np.ndarray] = {
        name: (item.pwm if isinstance(item, JasparMotif) else np.asarray(item, dtype=float))
        for name, item in database.items()
    }

    scores = {name: compare(query, target, min_overlap)
              for name, target in db_arrays.items()}
    null_best = np.empty(n_null, dtype=float)
    for draw in range(n_null):
        shuffled = query[rng.permutation(len(query))]
        null_best[draw] = max(
            compare(shuffled, target, min_overlap)[0] for target in db_arrays.values())

    rows = []
    for name, (score, strand) in scores.items():
        # (1 + count) / (1 + n) so empirical p-value is never reported as zero.
        p = float((1 + int((null_best >= score).sum())) / (1 + n_null))
        rows.append({
            "motif": name,
            "score": score,
            "strand": strand,
            "p_value": p,
            "null_p95": float(np.quantile(null_best, 0.95)),
        })
    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    return out.head(top_n).reset_index(drop=True)


# --------------------------------------------------------------------------
# Publication-Quality Visualization (Logomaker)
# --------------------------------------------------------------------------


def plot_motif_logo(
    matrix_or_motif: np.ndarray | JasparMotif | pd.DataFrame,
    ax: Any | None = None,
    title: str | None = None,
    color_scheme: str = "classic",
    out_path: str | Path | None = None,
) -> Any:
    """Plot a publication-grade sequence logo using Logomaker.

    Args:
        matrix_or_motif: A JasparMotif, (length, 4) numpy array, or pandas DataFrame.
        ax: Existing matplotlib Axes, or None to create a new figure.
        title: Optional title string for the logo.
        color_scheme: Logomaker color scheme (default: 'classic').
        out_path: Optional path to save the logo figure (PNG, PDF, or SVG).

    Returns:
        The matplotlib Figure object.
    """
    try:
        import logomaker
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "logomaker and matplotlib are required for sequence logo visualization. "
            "Install them via `pip install logomaker matplotlib`."
        ) from exc

    if isinstance(matrix_or_motif, JasparMotif):
        df = matrix_or_motif.to_dataframe(use_probabilities=True)
        default_title = f"{matrix_or_motif.name} ({matrix_or_motif.matrix_id})"
    elif isinstance(matrix_or_motif, pd.DataFrame):
        df = matrix_or_motif
        default_title = "Motif Logo"
    elif isinstance(matrix_or_motif, np.ndarray):
        df = pd.DataFrame(matrix_or_motif, columns=list(BASES))
        default_title = "Motif Logo"
    else:
        raise TypeError(f"Unsupported matrix type: {type(matrix_or_motif)}")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4.0, len(df) * 0.4), 2.2))
    else:
        fig = ax.figure

    logomaker.Logo(df, ax=ax, color_scheme=color_scheme)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_xlabel("Motif Position", fontsize=10)
    ax.set_title(title or default_title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight", dpi=300)

    return fig

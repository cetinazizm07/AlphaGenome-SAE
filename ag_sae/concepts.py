"""Match SAE features to ENCODE cCRE classes over a genomic interval.

Runs on a frozen SAE checkpoint, after training. Kept out of the training loop
so concept recovery cannot influence checkpoint selection.

Outputs:
  per-concept  best feature for each cCRE class, with a null and a raw baseline
  per-feature  where each feature fires and how those bins are annotated

Every AUROC is reported next to the same number computed on the tap's raw
channels. If the SAE does not beat the raw channels, it has added nothing.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

#: Ranks below this many valid bins make an AUROC meaningless to report.
MIN_BINS = 100
#: A concept with too few positives cannot be distinguished from noise.
MIN_POSITIVES = 20


# --------------------------------------------------------------------------
# cCRE intervals
# --------------------------------------------------------------------------


def read_ccre_bed(path: str | Path, class_column: int = 5) -> dict[str, dict[str, np.ndarray]]:
    """Read an ENCODE SCREEN cCRE BED into {class: {chrom: (starts, ends)}}.

    BED is 0-based half-open: `chr19 100 200` covers bases 100..199. The class
    column may hold several comma-separated labels; each becomes its own concept.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    per_class: dict[str, dict[str, list[tuple[int, int]]]] = {}
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= class_column:
                raise ValueError(f"{path}: expected >{class_column} columns, got {len(fields)}")
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            if end <= start:
                raise ValueError(f"{path}: invalid interval {chrom}:{start}-{end}")
            for label in fields[class_column].split(","):
                label = label.strip()
                if label and label != ".":
                    per_class.setdefault(label, {}).setdefault(chrom, []).append((start, end))

    out: dict[str, dict[str, np.ndarray]] = {}
    for label, by_chrom in per_class.items():
        out[label] = {}
        for chrom, spans in by_chrom.items():
            array = np.asarray(sorted(spans), dtype=np.int64)
            out[label][chrom] = array
    return out


def label_bins(bins: pd.DataFrame, intervals: dict[str, np.ndarray]) -> np.ndarray:
    """Mark bins that overlap any interval of one concept.

    Overlap means `interval.start < bin.end and interval.end > bin.start`, so
    touching at a boundary does not count. A positive bin intersects an element;
    it does not mean the whole 128 bp is one.

    Uses searchsorted plus a running max of interval ends. A per-bin scan would
    be ~10^6 intervals x ~10^7 bins and never finish.
    """
    positive = np.zeros(len(bins), dtype=bool)
    for chrom, group in bins.groupby("chrom", sort=False):
        spans = intervals.get(str(chrom))
        if spans is None or len(spans) == 0:
            continue
        starts, ends = spans[:, 0], spans[:, 1]
        # cummax makes "is there an interval starting at or before this bin
        # whose end reaches past the bin start" answerable in one lookup.
        reach = np.maximum.accumulate(ends)
        bin_start = group.bin_start.to_numpy()
        bin_end = group.bin_end.to_numpy()
        # Index of the last interval whose start is strictly before bin_end.
        last = np.searchsorted(starts, bin_end, side="left") - 1
        hit = np.zeros(len(group), dtype=bool)
        valid = last >= 0
        hit[valid] = reach[last[valid]] > bin_start[valid]
        positive[group.index.to_numpy()] = hit
    return positive


def bin_grid(chrom: str, start: int, end: int, bin_bp: int) -> pd.DataFrame:
    """Bins tiling [start, end) on one chromosome, aligned to `start`."""
    if end <= start or bin_bp < 1 or (end - start) % bin_bp:
        raise ValueError("Interval must be a positive whole number of bins")
    edges = np.arange(start, end, bin_bp, dtype=np.int64)
    return pd.DataFrame({"chrom": chrom, "bin_start": edges, "bin_end": edges + bin_bp})


# --------------------------------------------------------------------------
# AUROC for sparse, non-negative scores
# --------------------------------------------------------------------------


def auroc_from_nonzero(
    values: np.ndarray, is_positive: np.ndarray, n_pos: int, n_neg: int
) -> float:
    """Exact AUROC for a score vector that is zero outside `values`.

    `values` holds one feature's positive activations, `is_positive` the labels
    of those same bins. Every other bin is tied at zero.

    Ties count as half a win (Mann-Whitney). This matters: with TopK a feature
    is zero on most bins, so ignoring ties or dropping zeros shifts the AUROC a
    lot. Pairs split into four cases:

      positive nonzero vs negative zero   -> win
      positive zero vs negative nonzero   -> loss
      positive zero vs negative zero      -> tie, half credit
      both nonzero                        -> rank sum among the nonzeros

    A feature that never fires scores exactly 0.5.
    """
    if n_pos < 1 or n_neg < 1:
        raise ValueError("AUROC needs at least one positive and one negative bin")
    if values.shape != is_positive.shape:
        raise ValueError("Values and labels must describe the same bins")
    if values.size and (values <= 0).any():
        raise ValueError("Only strictly positive activations belong here")

    a = int(is_positive.sum())          # positives among the nonzeros
    b = int(values.size - a)            # negatives among the nonzeros
    pos_zero = n_pos - a
    neg_zero = n_neg - b
    if pos_zero < 0 or neg_zero < 0:
        raise ValueError("More nonzero bins than bins of that class")

    wins = a * neg_zero + 0.5 * pos_zero * neg_zero
    if a and b:
        from scipy.stats import rankdata

        ranks = rankdata(np.asarray(values, dtype=np.float32), method="average")
        wins += ranks[is_positive].sum(dtype=np.float64) - a * (a + 1) / 2
    return float(wins / (n_pos * n_neg))


def auroc_all_features(
    codes: "sparse.csc_matrix", labels: np.ndarray
) -> np.ndarray:
    """AUROC of every feature against one binary concept.

    Loops over features because each needs its own rank ordering; the loop body
    touches only that feature's nonzeros, so the cost is the number of stored
    activations, not features x bins.
    """
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    out = np.full(codes.shape[1], 0.5, dtype=np.float64)
    indptr, indices, data = codes.indptr, codes.indices, codes.data
    for feature in range(codes.shape[1]):
        lo, hi = indptr[feature], indptr[feature + 1]
        if lo == hi:
            continue  # never fired: stays at chance by definition
        rows = indices[lo:hi]
        out[feature] = auroc_from_nonzero(data[lo:hi], labels[rows], n_pos, n_neg)
    return out


class RankedCodes:
    """Sparse codes with each feature's value ranks precomputed.

    Permuting labels does not move the activations, only which bins are
    positive. Ranking is the expensive part of an AUROC, so do it once here;
    each later evaluation is then two segment sums. This takes a best-of-8192
    null from minutes per concept to milliseconds.

    Gives the same numbers as `auroc_all_features`, which stays as the plain
    reference implementation. A test checks that they agree.
    """

    def __init__(self, codes: "sparse.csc_matrix") -> None:
        from scipy.stats import rankdata

        codes = codes.tocsc()
        self.n_bins, self.n_features = codes.shape
        self.indptr = np.asarray(codes.indptr, dtype=np.int64)
        self.rows = np.asarray(codes.indices, dtype=np.int64)
        self.sizes = np.diff(self.indptr)
        self.ranks = np.empty(codes.nnz, dtype=np.float64)
        for feature in range(self.n_features):
            lo, hi = self.indptr[feature], self.indptr[feature + 1]
            if lo < hi:
                self.ranks[lo:hi] = rankdata(codes.data[lo:hi], method="average")
        self.fired = self.sizes > 0

    def _segment_sum(self, values: np.ndarray) -> np.ndarray:
        """Sum `values` within each feature's slice of the nonzero array.

        reduceat returns the element itself for a zero-length segment, so empty
        features are left out of the index list and filled in as zero.
        """
        out = np.zeros(self.n_features, dtype=np.float64)
        if values.size and self.fired.any():
            out[self.fired] = np.add.reduceat(values, self.indptr[:-1][self.fired])
        return out

    def auroc(self, labels: np.ndarray) -> np.ndarray:
        """AUROC of every feature against one binary concept."""
        n_pos = float(labels.sum())
        n_neg = float(labels.size - n_pos)
        if n_pos < 1 or n_neg < 1:
            raise ValueError("AUROC needs both classes present")
        at_nonzero = labels[self.rows]
        a = self._segment_sum(at_nonzero.astype(np.float64))   # positives that fire
        rank_sum = self._segment_sum(self.ranks * at_nonzero)
        b = self.sizes - a                                      # negatives that fire
        # Same four cases as auroc_from_nonzero, all features at once. A
        # never-firing feature has a = b = 0 and lands on 0.5.
        wins = a * (n_neg - b) + 0.5 * (n_pos - a) * (n_neg - b) + rank_sum - a * (a + 1) / 2
        return wins / (n_pos * n_neg)


# --------------------------------------------------------------------------
# Null calibration
# --------------------------------------------------------------------------


def concept_domains(bins: pd.DataFrame) -> list[tuple[str, int, int, np.ndarray]]:
    """Circular domains for the null: (chrom, start, end, row indices).

    One domain per extraction window when the coordinates carry `window_start`,
    otherwise one per chromosome. The domain is the span a concept is allowed to
    slide inside, so keeping it to a window holds local composition fixed.
    """
    has_window = "window_start" in bins.columns
    keys = ["chrom", "window_start"] if has_window else ["chrom"]
    domains = []
    for key, group in bins.groupby(keys, sort=False):
        chrom = str(key[0] if isinstance(key, tuple) else key)
        start = int(group.window_start.iloc[0]) if has_window else int(group.bin_start.min())
        end = int(group.bin_end.max())
        if end > start:
            domains.append((chrom, start, end, group.index.to_numpy()))
    return domains


def shift_intervals(spans: np.ndarray, start: int, end: int, offset: int) -> np.ndarray:
    """Slide intervals by `offset` inside [start, end), wrapping at the edge.

    Element count, element lengths and the gaps between elements all travel
    together, so the concept keeps its spatial arrangement and only changes
    where it sits. An element crossing the far edge comes back at the near one
    as two pieces, which is what makes the shift circular rather than a
    truncation.
    """
    length = end - start
    if length <= 0:
        raise ValueError("Domain must be a positive span")
    spans = np.asarray(spans, dtype=np.int64)
    if spans.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    width = np.minimum(spans[:, 1] - spans[:, 0], length)
    lo = (spans[:, 0] - start + offset) % length
    hi = lo + width
    inside = np.stack([lo, np.minimum(hi, length)], axis=1)
    over = hi > length
    pieces = [inside]
    if over.any():
        pieces.append(np.stack([np.zeros(int(over.sum()), dtype=np.int64),
                                hi[over] - length], axis=1))
    out = np.concatenate(pieces) + start
    return out[np.argsort(out[:, 0], kind="stable")]


class ShiftedConcept:
    """Draws of one concept's labels under a circular shift of its elements.

    The earlier null moved labels between the rows that happened to be sampled.
    That only works when neighbouring rows are adjacent on the genome. It is at
    128 bp, where every bin of a window is kept, and it is not at the conv taps,
    where 8192 of 262,144 bins are sampled: neighbours sit tens of bins apart,
    runs collapse to one or two bins, and the shift leaves 99.9% of labels where
    they were. Moving the concept instead is independent of which bins were
    sampled and works the same at every tap.
    """

    def __init__(self, bins: pd.DataFrame, intervals: Mapping[str, np.ndarray]) -> None:
        self.n_bins = len(bins)
        self.parts = []
        for chrom, start, end, rows in concept_domains(bins):
            spans = np.asarray(intervals.get(chrom, np.empty((0, 2))), dtype=np.int64)
            if spans.size:
                keep = (spans[:, 1] > start) & (spans[:, 0] < end)
                spans = spans[keep]
            frame = bins.loc[rows, ["chrom", "bin_start", "bin_end"]]
            self.parts.append((chrom, start, end, rows, spans, frame))

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        labels = np.zeros(self.n_bins, dtype=bool)
        for chrom, start, end, rows, spans, frame in self.parts:
            if spans.size == 0:
                continue
            offset = int(rng.integers(end - start))
            moved = shift_intervals(spans, start, end, offset)
            labels[rows] = label_bins(frame.reset_index(drop=True), {chrom: moved})
        return labels


def shifted_labels(
    bins: pd.DataFrame,
    intervals: Mapping[str, np.ndarray],
    n_permutations: int,
    seed: int,
) -> list[np.ndarray]:
    """One shifted label vector per permutation.

    Returned as a list so the SAE null and the raw null score the identical
    draws. That pairing takes the permutation noise out of the comparison
    between them.
    """
    shifter = ShiftedConcept(bins, intervals)
    rng = np.random.default_rng(seed)
    return [shifter.draw(rng) for _ in range(n_permutations)]


def circular_null(
    codes: RankedCodes,
    draws: Sequence[np.ndarray],
    fired: np.ndarray | None = None,
) -> np.ndarray:
    """Best feature AUROC across the draws, the statistic that is reported.

    The null takes the max over features because the reported statistic is also
    a max over features. Against a single-feature null, a large dictionary would
    make almost anything look significant.

    Two details keep null and observed identical. AUROC is folded onto [0.5, 1]
    first, because the observed best may mark a concept by going either way. And
    dead features are excluded, because the observed best is picked only from
    features that fire.
    """
    best = np.empty(len(draws), dtype=np.float64)
    for index, shifted in enumerate(draws):
        # A draw can land with no positives if the concept has few elements and
        # the shift moves them off the sampled bins. AUROC is undefined there,
        # so it scores chance. `degenerate_draws` counts these; a large count
        # means the null is being built from too little, not that it is strict.
        if not _usable(shifted):
            best[index] = 0.5
            continue
        folded, _ = directed(codes.auroc(shifted))
        if fired is not None:
            folded = np.where(fired, folded, 0.0)
        best[index] = folded.max()
    return best


def _usable(labels: np.ndarray) -> bool:
    """True when a draw has both classes, so an AUROC exists."""
    n_pos = int(labels.sum())
    return 0 < n_pos < labels.size


def circular_null_dense(
    activations: np.ndarray,
    draws: Sequence[np.ndarray],
    block: int = 256,
) -> np.ndarray:
    """The same null for the raw channels, so the baseline is calibrated too.

    Without this the two numbers are not comparable. The SAE reports the best of
    8192 features; the raw baseline the best of two directions on each of d_in
    channels. Two things pull the ceilings apart: more candidates raise a
    maximum even with no signal, while TopK sparsity ties most bins at zero and
    narrows the spread the maximum is drawn from. They act in opposite
    directions, so the sign of the gap cannot be reasoned out in advance and has
    to be measured for each tap.

    Ranking does not depend on the labels, so each column block is ranked once
    and every draw reuses it. Blocks keep the rank matrix off the heap: ranking
    all d_in columns of a test split at once would need tens of GB.
    """
    from scipy.stats import rankdata

    if not draws:
        raise ValueError("Need at least one permutation")
    best = np.full(len(draws), -np.inf, dtype=np.float64)
    for start in range(0, activations.shape[1], block):
        chunk = np.asarray(activations[:, start:start + block], dtype=np.float32)
        ranks = rankdata(chunk, method="average", axis=0)
        for index, shifted in enumerate(draws):
            if not _usable(shifted):
                best[index] = max(best[index], 0.5)
                continue
            n_pos = int(shifted.sum())
            n_neg = int(shifted.size - n_pos)
            r_pos = ranks[shifted].sum(axis=0, dtype=np.float64)
            auroc = (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
            folded, _ = directed(auroc)
            best[index] = max(best[index], float(folded.max()))
    return best


# --------------------------------------------------------------------------
# Frozen SAE
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenSAE:
    """Everything needed to encode, and nothing that can be trained.

    The checkpoint records its own preprocessing. Getting it wrong fails
    silently: the codes still look reasonable but are not the ones the decoder
    was trained with, and every AUROC downstream is then meaningless.
    """

    W_enc: np.ndarray          # (d, H)
    b_enc: np.ndarray          # (H,)
    b_pre: np.ndarray          # (d,)
    k: int
    channel_scale: np.ndarray | None = None   # (d,), divide raw input by this
    token_layernorm: bool = False             # per-bin mean/std over channels

    def __post_init__(self) -> None:
        d, hidden = self.W_enc.shape
        if self.b_enc.shape != (hidden,) or self.b_pre.shape != (d,):
            raise ValueError("Encoder bias shapes disagree with the weight matrix")
        if not 1 <= self.k <= hidden:
            raise ValueError(f"k={self.k} outside 1..{hidden}")
        if self.channel_scale is not None:
            if self.channel_scale.shape != (d,):
                raise ValueError("channel_scale must have one entry per input channel")
            if not np.isfinite(self.channel_scale).all() or (self.channel_scale <= 0).any():
                raise ValueError("channel_scale must be positive and finite")
        for name in ("W_enc", "b_enc", "b_pre"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} contains nonfinite values")

    @property
    def d_in(self) -> int:
        return self.W_enc.shape[0]

    @property
    def n_features(self) -> int:
        return self.W_enc.shape[1]

    @classmethod
    def from_torch_checkpoint(cls, path: str | Path) -> "FrozenSAE":
        """Load a checkpoint written by the training stage.

        `weights_only=True` because a checkpoint is data, not code to run.
        """
        import torch

        blob = torch.load(Path(path), map_location="cpu", weights_only=True)
        state = blob["state"]
        recipe = blob.get("recipe", {})

        def array(key: str) -> np.ndarray:
            if key not in state:
                raise KeyError(f"checkpoint lacks {key!r}; keys: {sorted(state)}")
            return state[key].detach().cpu().float().numpy()

        weight = array("encoder.weight")          # torch Linear stores (H, d)
        scale = array("channel_scale") if "channel_scale" in state else None
        return cls(
            W_enc=np.ascontiguousarray(weight.T),
            b_enc=array("latent_bias"),
            b_pre=array("pre_bias"),
            k=int(recipe["k"]),
            channel_scale=scale,
            token_layernorm=bool(recipe.get("token_layernorm", True)),
        )

    def encode(self, x: np.ndarray, chunk: int = 8192) -> "sparse.csc_matrix":
        """Sparse TopK codes for raw activations, shaped (n_bins, n_features).

        Chunked so the dense (n_bins, n_features) array never exists. At
        H=8192 that would be ~32 GB for one chromosome; the sparse result is
        about 5% of it.
        """
        from scipy import sparse

        if x.ndim != 2 or x.shape[1] != self.d_in:
            raise ValueError(f"Expected (n_bins, {self.d_in}), got {x.shape}")
        blocks = []
        for start in range(0, len(x), chunk):
            block = np.asarray(x[start:start + chunk], dtype=np.float32)
            if self.channel_scale is not None:
                block = block / self.channel_scale
            if self.token_layernorm:
                # Per-bin normalisation across channels. Matches the training
                # recipe, including the eps used upstream.
                mean = block.mean(axis=1, keepdims=True)
                block = block - mean
                # ddof=1 to match torch.std, which uses correction=1 by
                # default. numpy defaults to 0, which silently rescales every
                # code and every AUROC downstream.
                block = block / (block.std(axis=1, keepdims=True, ddof=1) + 1e-5)
            hidden = (block - self.b_pre) @ self.W_enc + self.b_enc
            # TopK picks by pre-activation value, then ReLU clips negatives,
            # so the actual L0 can be below k. argpartition only needs the k-th
            # boundary, not a full sort.
            cut = hidden.shape[1] - self.k
            keep = np.argpartition(hidden, cut, axis=1)[:, cut:]
            values = np.take_along_axis(hidden, keep, axis=1)
            values = np.maximum(values, 0.0)
            rows = np.repeat(np.arange(len(block)), self.k)
            block_sparse = sparse.coo_matrix(
                (values.ravel(), (rows, keep.ravel())),
                shape=(len(block), self.n_features),
            )
            block_sparse.eliminate_zeros()
            blocks.append(block_sparse)
        return sparse.vstack(blocks).tocsc()


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------


def auroc_dense(x: np.ndarray, labels: np.ndarray, block: int = 256) -> np.ndarray:
    """AUROC of every column of a dense, possibly signed score matrix.

    Used for the raw-channel baseline. Raw AlphaGenome channels are signed, so
    unlike SAE codes they have no preferred direction.
    """
    from scipy.stats import rankdata

    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos < 1 or n_neg < 1:
        raise ValueError("AUROC needs both classes present")
    out = np.empty(x.shape[1], dtype=np.float64)
    # Ranked in column blocks. A test split is millions of rows, and ranking
    # every channel at once would hold tens of GB of float ranks at peak.
    for start in range(0, x.shape[1], block):
        # rankdata keeps the input float dtype. Shards are stored float16,
        # whose integers stop being exact above 2048 and which overflows to inf
        # when the ranks are summed, so cast first and accumulate in float64.
        chunk = np.asarray(x[:, start:start + block], dtype=np.float32)
        ranks = rankdata(chunk, method="average", axis=0)
        r_pos = ranks[labels].sum(axis=0, dtype=np.float64)
        out[start:start + chunk.shape[1]] = (
            (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return out


def directed(auroc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold AUROC onto [0.5, 1] and report which direction was used.

    A channel that marks a concept by going down is as informative as one that
    goes up. Allowing both directions doubles the search space and so makes the
    raw baseline harder to beat, which is the conservative choice here.
    """
    inverse = auroc < 0.5
    return np.where(inverse, 1.0 - auroc, auroc), inverse


def top_bins_report(
    codes: "sparse.csc_matrix",
    bins: pd.DataFrame,
    concepts: dict[str, np.ndarray],
    top_n: int,
    features: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Where each feature fires hardest, and how those bins are annotated."""
    indptr, indices, data = codes.indptr, codes.indices, codes.data
    chosen = range(codes.shape[1]) if features is None else features
    rows = []
    for feature in chosen:
        lo, hi = indptr[feature], indptr[feature + 1]
        if lo == hi:
            continue
        values, bin_rows = data[lo:hi], indices[lo:hi]
        order = np.argsort(values)[::-1][:top_n]
        frame = bins.iloc[bin_rows[order]].reset_index(drop=True)
        frame.insert(0, "feature", feature)
        frame["activation"] = values[order]
        frame["rank"] = np.arange(1, len(order) + 1)
        for name, labels in concepts.items():
            frame[name] = labels[bin_rows[order]]
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["feature", "chrom", "bin_start", "bin_end", "activation", "rank"])
    return pd.concat(rows, ignore_index=True)


def summarise_feature_concepts(top: pd.DataFrame, concept_names: Sequence[str]) -> pd.DataFrame:
    """Per feature: the share of its top bins carrying each concept.

    A high share describes where the feature fires. It is not evidence that the
    feature encodes the concept; that needs the AUROC against a null, and
    causality needs ablation.
    """
    if top.empty:
        return pd.DataFrame(columns=["feature", "n_top_bins", *concept_names, "dominant", "dominant_share"])
    grouped = top.groupby("feature")
    if not len(concept_names):
        # Every concept was too rare to score. The firing locations are still
        # worth reporting; there is simply nothing to be dominant over.
        summary = grouped.size().rename("n_top_bins").reset_index()
        summary["dominant"] = "none"
        summary["dominant_share"] = 0.0
        return summary
    shares = grouped[list(concept_names)].mean()
    shares.insert(0, "n_top_bins", grouped.size())
    values = shares[list(concept_names)].to_numpy()
    best = values.argmax(axis=1)
    shares["dominant"] = [concept_names[i] for i in best]
    shares["dominant_share"] = values[np.arange(len(best)), best]
    # A feature whose top bins carry no annotation at all has no dominant class.
    shares.loc[shares.dominant_share == 0, "dominant"] = "none"
    return shares.reset_index()


@dataclass
class MatchResult:
    per_concept: pd.DataFrame
    per_feature: pd.DataFrame
    top_bins: pd.DataFrame
    summary: dict


def match_concepts(
    sae: FrozenSAE,
    activations: np.ndarray,
    bins: pd.DataFrame,
    ccre: dict[str, dict[str, np.ndarray]],
    *,
    bin_bp: int = 128,
    n_permutations: int = 200,
    seed: int = 0,
    top_n: int = 20,
    raw_baseline: bool = True,
) -> MatchResult:
    """Full comparison of one frozen SAE against the cCRE classes.

    `bins` must be row-aligned with `activations` and must already drop bins
    with ambiguous bases. The same mask has to be used at every stage, or the
    stages describe different sets of bins.
    """
    if len(bins) != len(activations):
        raise ValueError("Bins and activations must describe the same rows")
    if len(bins) < MIN_BINS:
        raise ValueError(f"Need at least {MIN_BINS} bins, got {len(bins)}")
    bins = bins.reset_index(drop=True)
    # The null shifts labels inside contiguous runs, and runs are found by
    # comparing the gap between bins to bin_bp. A wrong bin_bp makes every bin
    # its own run, the shift becomes the identity, and nothing is ever called
    # recovered. Fail loudly instead.
    widths = np.unique(bins.bin_end.to_numpy() - bins.bin_start.to_numpy())
    if widths.size != 1:
        raise ValueError(f"bins have mixed widths {widths.tolist()}")
    if int(widths[0]) != int(bin_bp):
        raise ValueError(
            f"bin_bp={bin_bp} but these bins are {int(widths[0])} bp wide")

    concepts = {name: label_bins(bins, spans) for name, spans in ccre.items()}
    # The null shifts each concept's own intervals, so a concept with no
    # intervals on these chromosomes cannot be calibrated and is dropped above.
    usable = {n: v for n, v in concepts.items()
              if MIN_POSITIVES <= int(v.sum()) <= len(bins) - MIN_POSITIVES}
    skipped = {n: int(v.sum()) for n, v in concepts.items() if n not in usable}

    codes = sae.encode(activations)
    ranked = RankedCodes(codes)
    fired = ranked.fired

    records = []
    for name, labels in usable.items():
        sae_dir, sae_inv = directed(ranked.auroc(labels))
        # Only firing features can be picked; a dead one sits at 0.5.
        best = int(np.argmax(np.where(fired, sae_dir, 0.0)))
        # One set of shifted draws, scored by both nulls, so the SAE ceiling
        # and the raw ceiling are paired. The null is folded and fired-masked
        # exactly like the observed value, so the two are the same statistic.
        draws = shifted_labels(bins, ccre[name], n_permutations, seed)
        null = circular_null(ranked, draws, fired)
        null_p95 = float(np.quantile(null, 0.95))
        row = {
            "concept": name,
            "n_positive_bins": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "best_feature": best,
            "best_auroc": float(sae_dir[best]),
            "best_inverse": bool(sae_inv[best]),
            "null_p95": null_p95,
            "sae_excess": float(sae_dir[best] - null_p95),
            "degenerate_draws": int(sum(not _usable(d) for d in draws)),
            "recovered": bool(sae_dir[best] > null_p95),
        }
        if raw_baseline:
            raw_dir, _ = directed(auroc_dense(activations, labels))
            # Same shifts as the SAE null, so the two ceilings are paired.
            raw_null = circular_null_dense(activations, draws)
            raw_null_p95 = float(np.quantile(raw_null, 0.95))
            row["raw_best_channel"] = int(raw_dir.argmax())
            row["raw_best_auroc"] = float(raw_dir.max())
            row["raw_null_p95"] = raw_null_p95
            row["raw_excess"] = float(raw_dir.max() - raw_null_p95)
            # Uncalibrated, kept only for reference. It compares a max over the
            # dictionary with a max over twice the channel count, and more
            # candidates give a higher max even with no signal.
            row["sae_minus_raw"] = float(row["best_auroc"] - row["raw_best_auroc"])
            # The number to report: each side measured against its own ceiling.
            row["calibrated_advantage"] = float(row["sae_excess"] - row["raw_excess"])
            row["sae_beats_raw"] = bool(row["calibrated_advantage"] > 1e-6)
        records.append(row)

    top = top_bins_report(codes, bins, usable, top_n)
    per_feature = summarise_feature_concepts(top, list(usable))
    summary = {
        "n_bins": int(len(bins)),
        "n_features": int(sae.n_features),
        "n_features_fired": int(fired.sum()),
        "effective_dictionary_fraction": float(fired.mean()),
        "mean_l0": float(codes.nnz / len(bins)),
        "k": int(sae.k),
        "bin_bp": int(bin_bp),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "concepts_used": sorted(usable),
        "concepts_skipped_too_rare": skipped,
        "region": {
            "chroms": sorted(bins.chrom.unique().tolist()),
            "span_bp": int((bins.bin_end - bins.bin_start).sum()),
        },
    }
    return MatchResult(pd.DataFrame(records), per_feature, top, summary)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_region(text: str) -> tuple[str, int, int]:
    """'chr19:1,000,000-2,000,000' -> ('chr19', 1000000, 2000000)."""
    chrom, _, span = text.partition(":")
    start, _, end = span.replace(",", "").partition("-")
    if not chrom or not start or not end:
        raise ValueError(f"Malformed region {text!r}; expected chrom:start-end")
    return chrom, int(start), int(end)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sae", required=True, help="frozen SAE checkpoint (.pt)")
    parser.add_argument("--activations", required=True, help=".npy, (n_bins, d), row-aligned with --bins")
    parser.add_argument("--bins", required=True, help="parquet with chrom/bin_start/bin_end")
    parser.add_argument("--ccre", required=True, help="ENCODE SCREEN cCRE BED (.bed or .bed.gz)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--region", help="restrict to chrom:start-end")
    parser.add_argument("--class-column", type=int, default=5, help="0-based cCRE class column (default 5)")
    parser.add_argument("--bin-bp", type=int, default=0,
                        help="bin width; 0 reads it from the coordinates file")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=20, help="top firing bins recorded per feature")
    parser.add_argument("--no-raw-baseline", action="store_true")
    args = parser.parse_args(argv)

    bins = pd.read_parquet(args.bins)
    missing = {"chrom", "bin_start", "bin_end"} - set(bins.columns)
    if missing:
        raise ValueError(f"--bins lacks columns: {sorted(missing)}")
    activations = np.load(args.activations, mmap_mode="r")
    if len(activations) != len(bins):
        raise ValueError(f"{len(activations)} activation rows vs {len(bins)} bins")

    if args.region:
        chrom, start, end = parse_region(args.region)
        keep = ((bins.chrom == chrom) & (bins.bin_start >= start) & (bins.bin_end <= end)).to_numpy()
        if not keep.any():
            raise ValueError(f"No bins inside {args.region}")
        bins, activations = bins[keep], np.asarray(activations[keep])
        print(f"region {args.region}: {len(bins)} bins")
    else:
        activations = np.asarray(activations)

    result = match_concepts(
        FrozenSAE.from_torch_checkpoint(args.sae),
        activations,
        bins,
        read_ccre_bed(args.ccre, args.class_column),
        bin_bp=args.bin_bp or int(bins.bin_end.iloc[0] - bins.bin_start.iloc[0]),
        n_permutations=args.permutations,
        seed=args.seed,
        top_n=args.top_n,
        raw_baseline=not args.no_raw_baseline,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.per_concept.to_parquet(out / "per_concept.parquet", index=False)
    result.per_feature.to_parquet(out / "per_feature.parquet", index=False)
    result.top_bins.to_parquet(out / "top_bins.parquet", index=False)
    (out / "summary.json").write_text(json.dumps(result.summary, indent=2))

    columns = [c for c in ("concept", "n_positive_bins", "best_feature", "best_auroc",
                           "null_p95", "sae_excess", "raw_best_auroc",
                           "raw_null_p95", "raw_excess", "calibrated_advantage",
                           "recovered")
               if c in result.per_concept.columns]
    print(result.per_concept[columns].to_string(index=False))
    print(f"\neffective dictionary: {result.summary['n_features_fired']}/{result.summary['n_features']}"
          f"  mean L0: {result.summary['mean_l0']:.1f}")
    if result.summary["concepts_skipped_too_rare"]:
        print(f"skipped (too few positive bins): {result.summary['concepts_skipped_too_rare']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

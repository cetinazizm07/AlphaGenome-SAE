"""Build the per-fold window manifest from the published fold definition.

The split is not ours to invent. Borzoi assigns every genomic region to one of
eight folds (`sequences_human.bed.gz`), and AlphaGenome's own
`fold_intervals.py` (Apache-2.0, google-deepmind/alphagenome_research) says
which of those eight each model version held out:

    model fold 0 -> valid fold0, test fold1
    model fold 1 -> valid fold3, test fold4
    model fold 2 -> valid fold2, test fold5
    model fold 3 -> valid fold6, test fold7

Everything else was trained on. The pairs are disjoint across model folds, so
no region is held out by more than one model: each fold is evaluated on its own
territory and there is no shared test set. That comes from the published
definition, not from us.

One detail changes the numbers: a training example predicts 196,608 bp but
reads 1,048,576 bp. A window can miss every training target and still have been
read. Activations depend on the input sequence, so read means seen. Training
regions are widened by the context margin before subtraction. That costs about
30 Mb per fold and is what keeps the holdout clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

#: From google-deepmind/alphagenome_research fold_intervals.py.
VALID_FOLD: Mapping[int, str] = {0: "fold0", 1: "fold3", 2: "fold2", 3: "fold6"}
TEST_FOLD: Mapping[int, str] = {0: "fold1", 1: "fold4", 2: "fold5", 3: "fold7"}
ALL_FOLDS = tuple(f"fold{i}" for i in range(8))

#: Sequence AlphaGenome reads per example, and the part it predicts.
CONTEXT_BP = 1_048_576
TARGET_BP = 196_608

Intervals = dict[str, np.ndarray]


# --------------------------------------------------------------------------
# Interval algebra
# --------------------------------------------------------------------------


def merge(frame: pd.DataFrame) -> Intervals:
    """Sort and union overlapping/abutting intervals, per chromosome."""
    out: Intervals = {}
    for chrom, group in frame.groupby("chrom", sort=False):
        raw = group[["start", "end"]].to_numpy(dtype=np.int64)
        if (raw[:, 1] <= raw[:, 0]).any():
            raise ValueError(f"{chrom}: interval with end <= start")
        raw = raw[np.argsort(raw[:, 0], kind="stable")]
        merged = [raw[0].copy()]
        for start, end in raw[1:]:
            if start <= merged[-1][1]:           # touching counts as joined
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append(np.array([start, end], dtype=np.int64))
        out[str(chrom)] = np.asarray(merged, dtype=np.int64)
    return out


def span(intervals: Intervals) -> int:
    return int(sum(int((v[:, 1] - v[:, 0]).sum()) for v in intervals.values()))


def subtract(left: Intervals, right: Intervals) -> Intervals:
    """left minus right, both assumed merged and sorted."""
    out: Intervals = {}
    for chrom, keep in left.items():
        cuts = right.get(chrom)
        if cuts is None or not len(cuts):
            out[chrom] = keep.copy()
            continue
        starts = cuts[:, 0]
        pieces: list[tuple[int, int]] = []
        for start, end in keep:
            # Step back one: the cut before the first that starts at or after
            # `start` may still reach into this interval.
            first = max(int(np.searchsorted(starts, start, side="right")) - 1, 0)
            cursor = int(start)
            for cut_start, cut_end in cuts[first:]:
                if cut_start >= end:
                    break
                if cut_end <= cursor:
                    continue
                if cut_start > cursor:
                    pieces.append((cursor, int(min(cut_start, end))))
                cursor = int(max(cursor, cut_end))
                if cursor >= end:
                    break
            if cursor < end:
                pieces.append((cursor, int(end)))
        if pieces:
            out[chrom] = np.asarray(pieces, dtype=np.int64)
    return out


def pad(intervals: Intervals, margin: int) -> Intervals:
    """Widen every interval by `margin` on both sides, then re-merge."""
    if margin < 0:
        raise ValueError("margin must be non-negative")
    rows = [pd.DataFrame({"chrom": chrom,
                          "start": np.maximum(iv[:, 0] - margin, 0),
                          "end": iv[:, 1] + margin})
            for chrom, iv in intervals.items() if len(iv)]
    return merge(pd.concat(rows, ignore_index=True)) if rows else {}


def tile(intervals: Intervals, window_bp: int) -> pd.DataFrame:
    """Whole, non-overlapping windows that fit inside the intervals.

    A window never crosses an interval edge, or it would contain sequence the
    label does not cover. A tail shorter than one window is dropped rather than
    shortened, since a short window has fewer bins and would reweight the pool.
    """
    if window_bp < 1:
        raise ValueError("window_bp must be positive")
    rows = []
    for chrom in sorted(intervals):
        for start, end in intervals[chrom]:
            count = (int(end) - int(start)) // window_bp
            if count < 1:
                continue
            offsets = int(start) + np.arange(count, dtype=np.int64) * window_bp
            rows.append(pd.DataFrame({"chrom": chrom, "win_start": offsets,
                                      "win_end": offsets + window_bp}))
    if not rows:
        return pd.DataFrame(columns=["chrom", "win_start", "win_end"])
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------
# Territory per model fold
# --------------------------------------------------------------------------


def read_fold_bed(path: str | Path) -> pd.DataFrame:
    """Borzoi's sequences_human.bed.gz: chrom, start, end, fold name."""
    frame = pd.read_csv(path, sep="\t", names=["chrom", "start", "end", "fold"],
                        dtype={"chrom": str, "fold": str})
    unknown = set(frame.fold) - set(ALL_FOLDS)
    if unknown:
        raise ValueError(f"Unexpected fold names {sorted(unknown)}; expected {list(ALL_FOLDS)}")
    if frame.empty:
        raise ValueError(f"{path}: no regions")
    return frame


@dataclass(frozen=True)
class Territory:
    """The two disjoint region sets a model fold offers."""

    model_fold: int
    valid_fold: str
    test_fold: str
    heldout: Intervals          # never trained on, not even read as context
    trained: Intervals          # trained on

    def summary(self) -> dict:
        return {"model_fold": self.model_fold, "valid_fold": self.valid_fold,
                "test_fold": self.test_fold,
                "heldout_mb": round(span(self.heldout) / 1e6, 1),
                "trained_mb": round(span(self.trained) / 1e6, 1)}


def territory(
    bed: pd.DataFrame,
    model_fold: int,
    *,
    context_bp: int = CONTEXT_BP,
    target_bp: int = TARGET_BP,
    exclude: Intervals | None = None,
) -> Territory:
    """Split the genome into unseen and seen regions for one model fold."""
    if model_fold not in VALID_FOLD:
        raise ValueError(f"model_fold must be one of {sorted(VALID_FOLD)}")
    if context_bp < target_bp:
        raise ValueError("context cannot be shorter than the prediction target")

    valid_name, test_name = VALID_FOLD[model_fold], TEST_FOLD[model_fold]
    train_names = [f for f in ALL_FOLDS if f not in (valid_name, test_name)]

    heldout_targets = merge(bed[bed.fold.isin([valid_name, test_name])])
    train_targets = merge(bed[bed.fold.isin(train_names)])
    # Widen to everything a training example could read, not just predict.
    margin = (context_bp - target_bp) // 2
    heldout = subtract(heldout_targets, pad(train_targets, margin))
    # Keep the two sets disjoint so no window lands in both.
    trained = subtract(train_targets, heldout)
    if exclude:
        heldout, trained = subtract(heldout, exclude), subtract(trained, exclude)
    return Territory(model_fold, valid_name, test_name, heldout, trained)


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

#: Splits per territory. `test` is locked and used once; `dev` is the only one
#: tuning may look at; `test_trained` is a size-matched control on territory the
#: backbone did train on, so the test vs test_trained gap measures memorisation.
HELDOUT_SPLITS = ("test", "dev")
TRAINED_SPLITS = ("train", "val", "test_trained")


def _draw(pool: pd.DataFrame, counts: Mapping[str, int], rng: np.random.Generator,
          label: str) -> pd.DataFrame:
    """Assign windows to splits without replacement, in a fixed random order."""
    order = rng.permutation(len(pool))
    taken, cursor = [], 0
    for split, want in counts.items():
        take = len(pool) - cursor if want == 0 else int(want)
        if take < 0:
            raise ValueError(f"{split}: negative window count")
        if cursor + take > len(pool):
            raise ValueError(
                f"{label}: asked for {take} '{split}' windows, only "
                f"{len(pool) - cursor} left of {len(pool)}")
        chunk = pool.iloc[order[cursor:cursor + take]].copy()
        chunk["split"] = split
        taken.append(chunk)
        cursor += take
    frame = pd.concat(taken, ignore_index=True) if taken else pool.iloc[:0].copy()
    frame["ag_label"] = label
    return frame


def build_manifest(
    bed: pd.DataFrame,
    model_fold: int,
    *,
    window_bp: int = CONTEXT_BP,
    test_windows: int = 200,
    dev_windows: int = 100,
    train_windows: int = 1000,
    val_windows: int = 100,
    test_trained_windows: int | None = None,
    seed: int = 0,
    context_bp: int = CONTEXT_BP,
    target_bp: int = TARGET_BP,
    exclude: Intervals | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Windows and splits for one model fold, plus a provenance record.

    `test_trained_windows` defaults to the size of `test` so the memorisation
    comparison uses equally sized sets. A count of 0 means "everything left",
    which only makes sense for the last split of a territory.
    """
    place = territory(bed, model_fold, context_bp=context_bp,
                      target_bp=target_bp, exclude=exclude)
    rng = np.random.default_rng(seed)
    matched = test_windows if test_trained_windows is None else test_trained_windows

    heldout_pool = tile(place.heldout, window_bp)
    trained_pool = tile(place.trained, window_bp)
    frame = pd.concat([
        _draw(heldout_pool, {"test": test_windows, "dev": dev_windows}, rng, "heldout"),
        _draw(trained_pool, {"train": train_windows, "val": val_windows,
                             "test_trained": matched}, rng, "trained"),
    ], ignore_index=True)

    frame["model_fold"] = model_fold
    frame = frame.sort_values(["chrom", "win_start"]).reset_index(drop=True)
    frame.insert(0, "window_idx", np.arange(len(frame), dtype=np.int64))

    # Cheap checks, run every time.
    if frame.duplicated(["chrom", "win_start"]).any():
        raise ValueError("duplicate windows in the manifest")
    for _, group in frame.groupby("chrom"):
        edges = group.sort_values("win_start")
        if (edges.win_start.to_numpy()[1:] < edges.win_end.to_numpy()[:-1]).any():
            raise ValueError("overlapping windows in the manifest")
    if not (frame.win_end - frame.win_start == window_bp).all():
        raise ValueError("windows are not all the requested length")

    record = {**place.summary(), "window_bp": window_bp, "seed": seed,
              "context_bp": context_bp, "target_bp": target_bp,
              "context_margin_bp": (context_bp - target_bp) // 2,
              "available_windows": {"heldout": int(len(heldout_pool)),
                                    "trained": int(len(trained_pool))},
              "selected_windows": frame.split.value_counts().to_dict(),
              "chroms": sorted(frame.chrom.unique().tolist())}
    return frame, record


# --------------------------------------------------------------------------
# Assembly gaps
# --------------------------------------------------------------------------


def n_runs(sequence: np.ndarray, min_length: int) -> np.ndarray:
    """Start/end of every run of N at least `min_length` long."""
    is_n = (sequence == ord("N")) | (sequence == ord("n"))
    if not is_n.any():
        return np.empty((0, 2), dtype=np.int64)
    # Pad with False so a run touching either end still shows an edge.
    edges = np.diff(np.r_[False, is_n, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1).astype(np.int64)
    ends = np.flatnonzero(edges == -1).astype(np.int64)
    keep = (ends - starts) >= min_length
    return np.stack([starts[keep], ends[keep]], axis=1)


def read_fasta(path: str | Path):
    """Yield (chrom, sequence as uint8) for each record. Handles .gz."""
    import gzip

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    chrom, chunks = None, []
    with opener(path, "rb") as handle:
        for line in handle:
            if line.startswith(b">"):
                if chrom is not None:
                    yield chrom, np.frombuffer(b"".join(chunks), dtype=np.uint8)
                chrom = line[1:].split()[0].decode()
                chunks = []
            else:
                chunks.append(line.strip())
    if chrom is not None:
        yield chrom, np.frombuffer(b"".join(chunks), dtype=np.uint8)


def gaps_from_fasta(paths: Sequence[str | Path], min_length: int = 1000) -> Intervals:
    """Assembly gaps, found by scanning the FASTA rather than trusting a track.

    Windows over long N runs produce no usable bins, so they waste a forward
    pass. Scanning is exact and needs no extra download, since the FASTAs are
    already required for extraction.
    """
    rows = []
    for path in paths:
        for chrom, sequence in read_fasta(path):
            runs = n_runs(sequence, min_length)
            if len(runs):
                rows.append(pd.DataFrame({"chrom": chrom, "start": runs[:, 0],
                                          "end": runs[:, 1]}))
    return merge(pd.concat(rows, ignore_index=True)) if rows else {}


def read_exclude_bed(path: str | Path) -> Intervals:
    """Regions to keep out of every split: assembly gaps, blacklist, anything."""
    frame = pd.read_csv(path, sep="\t", header=None, comment="#",
                        usecols=[0, 1, 2], names=["chrom", "start", "end"],
                        dtype={"chrom": str})
    return merge(frame) if len(frame) else {}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import hashlib
    import json

    top = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = top.add_subparsers(dest="command", required=True)

    gaps = sub.add_parser("gaps", help="scan FASTAs for assembly gaps, write a BED")
    gaps.add_argument("--fasta", nargs="+", required=True, help="chromosome FASTA files (.fa or .fa.gz)")
    gaps.add_argument("--out", required=True, help="output BED path")
    gaps.add_argument("--min-length", type=int, default=1000, help="shortest N run to record")

    parser = sub.add_parser("build", help="build per-fold window manifests")
    parser.add_argument("--bed", required=True,
                        help="Borzoi sequences_human.bed.gz (chrom, start, end, fold)")
    parser.add_argument("--out", required=True, help="directory for the manifests")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2],
                        choices=sorted(VALID_FOLD))
    parser.add_argument("--window-bp", type=int, default=CONTEXT_BP)
    parser.add_argument("--test-windows", type=int, default=200)
    parser.add_argument("--dev-windows", type=int, default=100)
    parser.add_argument("--train-windows", type=int, default=1000)
    parser.add_argument("--val-windows", type=int, default=100)
    parser.add_argument("--test-trained-windows", type=int,
                        help="default: same as --test-windows")
    parser.add_argument("--exclude", help="BED of regions to drop (gaps, blacklist)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-bp", type=int, default=CONTEXT_BP)
    parser.add_argument("--target-bp", type=int, default=TARGET_BP)
    args = top.parse_args(argv)

    if args.command == "gaps":
        intervals = gaps_from_fasta(args.fasta, args.min_length)
        rows = [pd.DataFrame({"chrom": c, "start": iv[:, 0], "end": iv[:, 1]})
                for c, iv in sorted(intervals.items())]
        frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["chrom", "start", "end"])
        frame.to_csv(args.out, sep="\t", header=False, index=False)
        print(f"{len(frame)} gaps, {span(intervals) / 1e6:.1f} Mb -> {args.out}")
        return 0

    bed = read_fold_bed(args.bed)
    exclude = read_exclude_bed(args.exclude) if args.exclude else None
    digest = hashlib.sha256(Path(args.bed).read_bytes()).hexdigest()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    heldout_sets = {}
    for fold in args.folds:
        frame, record = build_manifest(
            bed, fold, window_bp=args.window_bp, test_windows=args.test_windows,
            dev_windows=args.dev_windows, train_windows=args.train_windows,
            val_windows=args.val_windows,
            test_trained_windows=args.test_trained_windows, seed=args.seed,
            context_bp=args.context_bp, target_bp=args.target_bp, exclude=exclude)
        record["source_bed_sha256"] = digest
        record["exclude_bed"] = args.exclude
        frame.to_parquet(out / f"window_manifest_fold{fold}.parquet", index=False)
        (out / f"window_manifest_fold{fold}.json").write_text(json.dumps(record, indent=2))
        heldout = frame[frame.ag_label == "heldout"]
        heldout_sets[fold] = set(map(tuple, heldout[["chrom", "win_start"]].to_numpy()))
        print(f"fold {fold}: {record['heldout_mb']:.0f} Mb unseen "
              f"({record['available_windows']['heldout']} windows available), "
              f"selected {record['selected_windows']}")

    # The published mapping makes these disjoint. Check it rather than assume
    # it: a silent overlap would invalidate every fold-to-fold comparison.
    folds = sorted(heldout_sets)
    for i, a in enumerate(folds):
        for b in folds[i + 1:]:
            shared = heldout_sets[a] & heldout_sets[b]
            if shared:
                raise ValueError(f"folds {a} and {b} share {len(shared)} held-out windows")
    if len(folds) > 1:
        print(f"held-out windows are disjoint across folds {folds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

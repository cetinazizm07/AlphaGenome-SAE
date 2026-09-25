"""Validated genomic rows, memory-mapped activations and atomic local output."""
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from .config import BIN_BP, WIN

COORDINATES = ["chrom", "bin_start", "bin_end", "split"]


def log(*args):
    print(f"[{time.strftime('%H:%M:%S')}]", *args, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@contextmanager
def atomic_file(path):
    """Publish only a closed, flushed file on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_json(path, value):
    with atomic_file(path) as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False).encode())


def atomic_numpy(path, array):
    with atomic_file(path) as stream:
        np.save(stream, array, allow_pickle=False)


def boolean_mask(values):
    """Accept boolean or numeric 0/1 masks, reject missing/ambiguous values."""
    values = np.asarray(values)
    if values.dtype.kind not in "biuf" or not np.isin(values, [0, 1]).all():
        raise ValueError("n_mask must contain only nonmissing boolean or numeric 0/1 values")
    return values.astype(bool)


def validate_coordinates(frame):
    if not set(COORDINATES) <= set(frame.columns) or frame.empty:
        raise ValueError("Missing or empty genomic coordinates/splits")
    if frame[COORDINATES].isna().any().any():
        raise ValueError("Missing genomic coordinates/splits")
    for c in ("bin_start", "bin_end"):
        if frame[c].dtype.kind not in "iu":
            raise ValueError(f"{c} must contain integer coordinates")
    if (frame.bin_start < 0).any() or (frame.bin_end - frame.bin_start != BIN_BP).any():
        raise ValueError("Expected nonnegative, half-open 128-bp coordinates")
    if frame.duplicated(["chrom", "bin_start"]).any():
        raise ValueError("Duplicate genomic coordinates")


def read_annotation(path):
    ann = pd.read_parquet(path).reset_index(drop=True)
    validate_coordinates(ann)
    if "n_mask" not in ann:
        raise ValueError("Missing n_mask")
    ann["n_mask"] = boolean_mask(ann.n_mask)
    return ann


def coordinates_from_manifest(manifest, window_bp=WIN):
    required = {"chrom", "win_start", "win_end", "split"}
    if not required <= set(manifest) or manifest.empty:
        raise ValueError("Missing or empty window manifest")
    if manifest[list(required)].isna().any().any():
        raise ValueError("Missing manifest values")
    if any(manifest[c].dtype.kind not in "iu" for c in ("win_start", "win_end")):
        raise ValueError("Window coordinates must be integers")
    if window_bp % BIN_BP or (manifest.win_end - manifest.win_start != window_bp).any():
        raise ValueError(f"Manifest windows must be {window_bp} bp wide")
    bins = window_bp // BIN_BP
    start = np.repeat(manifest.win_start.to_numpy(), bins) + np.tile(np.arange(bins) * BIN_BP, len(manifest))
    coords = pd.DataFrame({"chrom": np.repeat(manifest.chrom.to_numpy(), bins),
                           "bin_start": start, "bin_end": start + BIN_BP,
                           "split": np.repeat(manifest.split.to_numpy(), bins)})
    validate_coordinates(coords)
    return coords


def validate_alignment(ann, coords):
    validate_coordinates(coords)
    if len(ann) != len(coords) or any(
        not np.array_equal(ann[c].to_numpy(), coords[c].to_numpy()) for c in COORDINATES
    ):
        raise ValueError("Activation/annotation coordinates and splits are not row-aligned")


def load_chrom_fasta(fasta_dir, chrom):
    root = Path(fasta_dir)
    names = [f"Homo_sapiens.GRCh38.dna.chromosome.{chrom.removeprefix('chr')}.fa.gz",
             f"{chrom}.fa", f"{chrom}.fa.gz", f"{chrom}.fasta"]
    path = next((root / name for name in names if (root / name).is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing FASTA for {chrom} in {root}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return "".join(line.strip() for line in stream if not line.startswith(">")).upper()


class ActivationStore:
    """Masked split rows mapped to immutable, coordinate-checked .npy shards.

    Logical indices retain the annotation order. Random batches may span shards;
    no full activation concatenation or change to the sampling order is needed.
    """
    def __init__(self, directory, split, ann):
        self.directory = Path(directory)
        index_path = self.directory / "shard_index.parquet"
        coord_path = self.directory / "row_coordinates.parquet"
        meta_path = self.directory / "extraction_meta.json"
        if not all(p.is_file() for p in (index_path, coord_path, meta_path)):
            raise ValueError("Incomplete activation cache or missing row coordinates; rerun extract")
        meta = json.loads(meta_path.read_text())
        if meta.get("format_version") != 2 or meta.get("coordinates_sha256") != sha256(coord_path):
            raise ValueError("Invalid activation coordinate metadata; rerun extract")
        if meta.get("shard_index_sha256") != sha256(index_path):
            raise ValueError("Activation shard index checksum mismatch")
        coords = pd.read_parquet(coord_path)
        validate_alignment(ann, coords)
        mask = boolean_mask(ann.n_mask)
        index = pd.read_parquet(index_path).sort_values("row_start")
        required = {"shard", "split", "row_start", "row_end", "sha256"}
        if not required <= set(index) or index.empty:
            raise ValueError("Invalid activation shard index")
        if index.shard.duplicated().any():
            raise ValueError("Duplicate activation shard")
        cursor = 0
        self.arrays, sizes = [], []
        for row in index.itertuples(index=False):
            if row.row_start != cursor or row.row_end <= cursor or row.row_end > len(coords):
                raise ValueError("Activation shard ranges must cover rows without gaps or overlaps")
            if not (coords.split.iloc[cursor:row.row_end] == row.split).all():
                raise ValueError("Activation shard split does not match its coordinates")
            cursor = row.row_end
            if Path(row.shard).name != row.shard:
                raise ValueError("Shard paths must be local filenames")
            if row.split != split:
                continue
            path = self.directory / row.shard
            if not path.is_file() or sha256(path) != row.sha256:
                raise ValueError(f"Missing or corrupt activation shard: {row.shard}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 2 or array.shape != (row.row_end - row.row_start, meta["dim"]):
                raise ValueError(f"Activation shard shape mismatch: {row.shard}")
            if array.dtype not in (np.dtype("float16"), np.dtype("float32")):
                raise ValueError("Activations must be float16 or float32")
            self.arrays.append(array)
            sizes.append(len(array))
        if cursor != len(coords) or cursor != meta["total_rows"]:
            raise ValueError("Activation cache does not cover all annotation rows")
        self.rows = np.flatnonzero(mask[ann.split.to_numpy() == split])
        if not self.arrays or not len(self.rows):
            raise ValueError(f"Split {split} has no valid activation rows")
        self.ends = np.cumsum(sizes)
        self.starts = np.r_[0, self.ends[:-1]]
        self.dim = int(meta["dim"])
        self.identity = {"index": sha256(index_path), "coordinates": sha256(coord_path),
                         "mask": hashlib.sha256(mask.tobytes()).hexdigest(), "split": split}

    def __len__(self):
        return len(self.rows)

    def take(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if (indices.ndim != 1 or (indices < 0).any()
                or (indices >= len(self)).any()):
            raise IndexError("Activation row index out of bounds")
        rows = self.rows[indices]
        shards = np.searchsorted(self.ends, rows, side="right")
        result = np.empty((len(rows), self.dim), dtype=np.float32)
        for shard in np.unique(shards):
            positions = np.flatnonzero(shards == shard)
            result[positions] = self.arrays[shard][rows[positions] - self.starts[shard]]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite activation values in selected bins")
        return result

    def batches(self, batch_size=8192):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self), batch_size):
            yield self.take(np.arange(start, min(start + batch_size, len(self))))

    def moments(self):
        """Stable float64 parallel variance; scans and validates the split once."""
        count, mean, m2 = 0, np.zeros(self.dim), np.zeros(self.dim)
        for batch in self.batches():
            batch = batch.astype(np.float64)
            n = len(batch)
            local_mean = batch.mean(0)
            delta = local_mean - mean
            m2 += ((batch - local_mean) ** 2).sum(0) + delta ** 2 * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
        return mean.astype(np.float32), float(m2.sum())


class MatchShardStore:
    """Coordinate-align a native ``extract.py`` tower cache for matching.

    The trainer's ``ShardStore`` layout is multi-tap and uses ``index.json``;
    the historical ``ActivationStore`` layout is a single-tap v2 cache. This
    adapter lets matching consume the native layout without rewriting data.
    Concept annotations are defined on the 128-bp grid, so encoder taps at
    4/16/64 bp are deliberately rejected rather than silently misaligned.
    """

    def __init__(self, directory, tap, split, ann):
        from .extract import ShardStore, TAPS

        ann = ann.reset_index(drop=True)
        validate_coordinates(ann)
        if "n_mask" not in ann:
            raise ValueError("Annotation table is missing n_mask")
        if tap not in TAPS:
            raise ValueError(f"Unknown activation tap: {tap!r}")
        if TAPS[tap].bin_bp != BIN_BP:
            raise ValueError(
                f"Matching requires a 128-bp tower tap; {tap!r} is {TAPS[tap].bin_bp} bp"
            )
        self.store = ShardStore(directory, tap, split)
        self.dim = self.store.dim
        coords = self.store.coordinates().reset_index(drop=True)
        validate_coordinates(coords)
        if not (coords.split == split).all():
            raise ValueError("Activation shard contains rows from another split")

        # Index annotations by exact half-open coordinates. The activation
        # cache may sample a subset of bins, so equality of whole tables is
        # neither expected nor required; every cached row must map uniquely.
        lookup = {}
        for idx, row in ann.iterrows():
            if str(row.split) != str(split):
                continue
            key = (str(row.chrom), int(row.bin_start), int(row.bin_end), str(row.split))
            if key in lookup:
                raise ValueError(f"Duplicate annotation coordinate: {key}")
            lookup[key] = int(idx)

        annotation_rows = []
        for row in coords.itertuples(index=False):
            key = (str(row.chrom), int(row.bin_start), int(row.bin_end), str(row.split))
            if key not in lookup:
                raise ValueError(f"Activation coordinate is absent from annotations: {key}")
            annotation_rows.append(lookup[key])
        mapped = np.asarray(annotation_rows, dtype=np.int64)
        valid = boolean_mask(ann.n_mask.to_numpy())[mapped]
        self.row_indices = np.flatnonzero(valid)
        self.annotation_indices = mapped[self.row_indices]
        self._coordinates = coords.iloc[self.row_indices].reset_index(drop=True)
        if not len(self.row_indices):
            raise ValueError(f"Split {split} has no valid matched activation rows")

        # Verify both arrays and coordinate sidecars against the extraction
        # index before trusting their row correspondence.
        for record in self.store.records:
            for field, checksum in (("activations", "activations_sha256"),
                                    ("coordinates", "coordinates_sha256")):
                path = Path(directory) / record[field]
                if not path.is_file() or sha256(path) != record[checksum]:
                    raise ValueError(f"Missing or corrupt activation shard: {path.name}")
        self.identity = {**self.store.identity,
                         "annotation_indices_sha256": hashlib.sha256(
                             self.annotation_indices.tobytes()).hexdigest()}

    def __len__(self):
        return len(self.row_indices)

    def take(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if (indices < 0).any() or (indices >= len(self)).any():
            raise IndexError("Activation row index out of bounds")
        return self.store.take(self.row_indices[indices])

    def batches(self, batch_size=8192):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(self), batch_size):
            yield self.take(np.arange(start, min(start + batch_size, len(self))))

    def coordinates(self):
        return self._coordinates.copy()

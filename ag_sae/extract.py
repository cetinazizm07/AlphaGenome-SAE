"""Capture AlphaGenome activations at several depths in one forward pass.

Six taps, all on the same pass so extra taps cost disk and not GPU time:

  bin_size_4/16/64   encoder feature maps, from the encoder's return value
  resid_pre_b0/4/8   residual stream entering tower blocks 0, 4 and 8

The tower taps need a pre-hook, not a forward hook. `tower.blocks[i]` is a
ModuleDict with no forward of its own, and the residual adds happen in
`TransformerTower._forward_block`, which is a plain method. So the residual
stream is nobody's output. It is `mha`'s input, and a forward hook on the block
silently never fires.

Each tap keeps a fixed number of positions per window rather than all of them.
At 4 bp a 1 Mb window holds 262,144 positions and at 128 bp only 8,192, so
storing every position would spend almost all the disk on the finest tap. Bins
overlapping an ambiguous base are dropped first.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Tap:
    """Where to read, and at what resolution."""

    name: str
    kind: str        # "encoder" or "tower"
    key: str | int   # intermediates key, or tower block index
    bin_bp: int
    channels: int


TAPS: Mapping[str, Tap] = {t.name: t for t in (
    Tap("bin_size_4", "encoder", "bin_size_4", 4, 1024),
    Tap("bin_size_16", "encoder", "bin_size_16", 16, 1280),
    Tap("bin_size_64", "encoder", "bin_size_64", 64, 1536),
    Tap("resid_pre_b0", "tower", 0, 128, 1536),
    Tap("resid_pre_b4", "tower", 4, 128, 1536),
    Tap("resid_pre_b8", "tower", 8, 128, 1536),
)}

BASES = "ACGT"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def one_hot(sequence: str) -> np.ndarray:
    """(L, 4) float32. Anything that is not ACGT becomes an all-zero row."""
    codes = np.frombuffer(sequence.upper().encode(), dtype=np.uint8)
    out = np.zeros((len(codes), 4), dtype=np.float32)
    for index, base in enumerate(BASES):
        out[codes == ord(base), index] = 1.0
    return out


def valid_bins(onehot: np.ndarray, bin_bp: int) -> np.ndarray:
    """True for bins whose every base is a known nucleotide."""
    if onehot.shape[0] % bin_bp:
        raise ValueError(f"sequence length {onehot.shape[0]} is not a multiple of {bin_bp}")
    known = onehot.sum(-1) > 0
    return known.reshape(-1, bin_bp).all(-1)


def choose_positions(valid: np.ndarray, keep: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted indices of up to `keep` valid bins, drawn without replacement.

    Random rather than strided: a stride can line up with periodic features and
    the SAE sees rows independently anyway, so contiguity buys nothing.
    """
    available = np.flatnonzero(valid)
    if keep <= 0 or keep >= available.size:
        return available
    return np.sort(rng.choice(available, keep, replace=False))


class Capture:
    """Installs hooks on a loaded model and holds the last forward's taps."""

    def __init__(self, model, taps: Sequence[Tap]) -> None:
        self.model = model
        self.taps = list(taps)
        self._store: dict[str, np.ndarray] = {}
        self._handles: list = []

    def __enter__(self) -> "Capture":
        encoder_taps = [t for t in self.taps if t.kind == "encoder"]
        if encoder_taps:
            def on_encoder(_module, _args, output):
                # SequenceEncoder returns (trunk, intermediates); the
                # intermediates are NCL, so transpose to (positions, channels).
                intermediates = output[1]
                for tap in encoder_taps:
                    if tap.key not in intermediates:
                        raise KeyError(f"encoder has no {tap.key!r}; got {sorted(intermediates)}")
                    self._store[tap.name] = self._to_numpy(intermediates[tap.key][0].T)
            self._handles.append(self.model.encoder.register_forward_hook(on_encoder))

        for tap in (t for t in self.taps if t.kind == "tower"):
            block = self.model.tower.blocks[int(tap.key)]

            def on_mha(_module, args, _kwargs, name=tap.name):
                # args[0] is the residual stream entering the block, NLC.
                self._store[name] = self._to_numpy(args[0][0])
                return None                      # do not modify the input

            self._handles.append(
                block["mha"].register_forward_pre_hook(on_mha, with_kwargs=True))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._store.clear()

    @staticmethod
    def _to_numpy(tensor) -> np.ndarray:
        return tensor.detach().float().cpu().numpy()

    def outputs(self) -> dict[str, np.ndarray]:
        missing = [t.name for t in self.taps if t.name not in self._store]
        if missing:
            raise RuntimeError(f"hooks did not fire for {missing}")
        return dict(self._store)

    def clear(self) -> None:
        self._store.clear()


# --------------------------------------------------------------------------
# Shard writing
# --------------------------------------------------------------------------


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Write through a handle: given a path whose name does not end in .npy,
    # np.save appends the extension itself and the rename would then miss.
    with open(tmp, "wb") as handle:
        np.save(handle, array, allow_pickle=False)
    tmp.replace(path)


def _atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def window_rows(
    row, sequence: str, forward, taps: Sequence[Tap], keep: int, seed: int
) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    """Activations and coordinates for one window, per tap."""
    onehot = one_hot(sequence)
    if onehot.shape[0] != row.win_end - row.win_start:
        raise ValueError(f"{row.chrom}:{row.win_start} sequence length mismatch")
    captured = forward(onehot)
    # One RNG per window so a re-run picks the same positions. zlib.crc32 and
    # not hash(): Python randomises str hashing per process, so hash() would
    # give different positions on every run.
    rng = np.random.default_rng([seed, int(row.win_start),
                                 zlib.crc32(str(row.chrom).encode())])
    out = {}
    for tap in taps:
        array = captured[tap.name]
        if array.shape != (onehot.shape[0] // tap.bin_bp, tap.channels):
            raise ValueError(f"{tap.name}: got {array.shape}, expected "
                             f"({onehot.shape[0] // tap.bin_bp}, {tap.channels})")
        chosen = choose_positions(valid_bins(onehot, tap.bin_bp), keep, rng)
        if not chosen.size:
            continue
        starts = int(row.win_start) + chosen * tap.bin_bp
        coords = pd.DataFrame({"chrom": row.chrom, "bin_start": starts,
                               "bin_end": starts + tap.bin_bp,
                               "split": row.split, "window_start": int(row.win_start)})
        out[tap.name] = (array[chosen].astype(np.float16), coords)
    return out


def run_extraction(
    manifest: pd.DataFrame,
    sequence_for,
    forward,
    taps: Sequence[Tap],
    out_dir: str | Path,
    *,
    positions_per_window: int = 2048,
    windows_per_shard: int = 8,
    seed: int = 0,
    provenance: dict | None = None,
    log=print,
) -> dict:
    """Write one shard set per tap. Resumable: finished shards are skipped."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "version": 1,
        "taps": [t.name for t in taps],
        "positions_per_window": positions_per_window,
        "windows_per_shard": windows_per_shard,
        "seed": seed,
        "storage": "float16",
        "manifest_sha256": hashlib.sha256(
            pd.util.hash_pandas_object(manifest, index=False).values.tobytes()).hexdigest(),
        "provenance": provenance or {},
    }
    index_path = out_dir / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["identity"] != identity:
            raise ValueError("this directory holds a different extraction; use a new one")
        if index.get("complete"):
            log("activation cache already complete")
            return index
    else:
        index = {"identity": identity, "shards": [], "complete": False}
        _atomic_json(index_path, index)

    done = len(index["shards"])
    groups = [g for _, g in manifest.groupby(
        [manifest.split, manifest.chrom,
         np.arange(len(manifest)) // windows_per_shard], sort=False)]

    for shard_id, group in enumerate(groups):
        if shard_id < done:
            continue
        buffers: dict[str, list[np.ndarray]] = {t.name: [] for t in taps}
        coords: dict[str, list[pd.DataFrame]] = {t.name: [] for t in taps}
        for row in group.itertuples(index=False):
            sequence = sequence_for(row.chrom, int(row.win_start), int(row.win_end))
            for name, (array, frame) in window_rows(
                    row, sequence, forward, taps, positions_per_window, seed).items():
                buffers[name].append(array)
                coords[name].append(frame)

        record = {"shard": shard_id, "split": str(group.split.iloc[0]),
                  "n_windows": int(len(group)), "taps": {}}
        for tap in taps:
            if not buffers[tap.name]:
                continue
            stem = f"{shard_id:05d}_{group.split.iloc[0]}_{tap.name}"
            array = np.concatenate(buffers[tap.name])
            if not np.isfinite(array).all():
                raise ValueError(f"{stem}: non-finite activations or float16 overflow")
            act_path, coord_path = out_dir / f"{stem}.npy", out_dir / f"{stem}.parquet"
            _atomic_npy(act_path, array)
            pd.concat(coords[tap.name], ignore_index=True).to_parquet(coord_path, index=False)
            record["taps"][tap.name] = {
                "activations": act_path.name, "coordinates": coord_path.name,
                "rows": int(len(array)), "channels": int(array.shape[1]),
                "activations_sha256": sha256(act_path),
                "coordinates_sha256": sha256(coord_path)}
        index["shards"].append(record)
        _atomic_json(index_path, index)
        log(f"shard {shard_id}/{len(groups)} split={record['split']} "
            f"rows={ {k: v['rows'] for k, v in record['taps'].items()} }")

    index["complete"] = True
    _atomic_json(index_path, index)
    return index


class ShardStore:
    """Read-only view of one tap and split from an extraction directory."""

    def __init__(self, directory: str | Path, tap: str, split: str) -> None:
        self.directory = Path(directory)
        index = json.loads((self.directory / "index.json").read_text())
        if not index.get("complete"):
            raise ValueError("extraction is not complete")
        self.records = [r["taps"][tap] for r in index["shards"]
                        if r["split"] == split and tap in r["taps"] and r["taps"][tap]["rows"]]
        if not self.records:
            raise ValueError(f"no rows for tap {tap!r} split {split!r}")
        self.arrays = [np.load(self.directory / r["activations"], mmap_mode="r",
                               allow_pickle=False) for r in self.records]
        self.dim = int(self.records[0]["channels"])
        if any(a.shape != (r["rows"], self.dim) for a, r in zip(self.arrays, self.records)):
            raise ValueError("shard shape does not match the index")
        self.ends = np.cumsum([len(a) for a in self.arrays])
        self.tap, self.split = tap, split
        self.identity = {"index": sha256(self.directory / "index.json"),
                         "tap": tap, "split": split}

    def __len__(self) -> int:
        return int(self.ends[-1])

    def take(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= len(self)).any():
            raise ValueError("row index out of range")
        out = np.empty((len(indices), self.dim), dtype=np.float32)
        shard_of = np.searchsorted(self.ends, indices, side="right")
        starts = np.r_[0, self.ends[:-1]]
        for shard in np.unique(shard_of):
            mask = shard_of == shard
            out[mask] = self.arrays[shard][indices[mask] - starts[shard]]
        return out

    def order(self, seed: int, epoch: int) -> np.ndarray:
        """Row order for one pass: shards shuffled, rows shuffled within each.

        Keeps reads local to one memory-mapped file at a time.
        """
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        starts = np.r_[0, self.ends[:-1]]
        return np.concatenate([rng.permutation(len(self.arrays[i])) + starts[i]
                               for i in rng.permutation(len(self.arrays))])

    def batches(self, size: int) -> Iterator[np.ndarray]:
        for offset in range(0, len(self), size):
            yield self.take(np.arange(offset, min(offset + size, len(self))))

    def coordinates(self) -> pd.DataFrame:
        return pd.concat([pd.read_parquet(self.directory / r["coordinates"])
                          for r in self.records], ignore_index=True)


# --------------------------------------------------------------------------
# Reading sequence
# --------------------------------------------------------------------------


class FastaIndex:
    """Random access into an uncompressed FASTA, samtools faidx style.

    Built by one scan of the file. A plain .gz cannot be seeked, so the genome
    has to be uncompressed first; at 1 Mb per window, reading the whole
    chromosome for every window would otherwise dominate the run.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.suffix == ".gz":
            raise ValueError(f"{self.path.name} is gzipped; uncompress it first")
        self.records: dict[str, tuple[int, int, int, int]] = {}
        name, offset, length, line_bases, line_width = None, 0, 0, 0, 0
        with open(self.path, "rb") as handle:
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.startswith(b">"):
                    if name is not None:
                        self.records[name] = (offset, length, line_bases, line_width)
                    name = line[1:].split()[0].decode()
                    offset, length, line_bases, line_width = handle.tell(), 0, 0, 0
                else:
                    stripped = len(line.rstrip())
                    if line_bases == 0:
                        line_bases, line_width = stripped, len(line)
                    length += stripped
            if name is not None:
                self.records[name] = (offset, length, line_bases, line_width)
        if not self.records:
            raise ValueError(f"{self.path}: no FASTA records")
        self._handle = open(self.path, "rb")

    def length(self, chrom: str) -> int:
        return self.records[chrom][1]

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Sequence for [start, end). Past the chromosome end is padded with N."""
        if chrom not in self.records:
            raise KeyError(f"{chrom} not in {self.path.name}; have {len(self.records)} records")
        offset, length, line_bases, line_width = self.records[chrom]
        if start < 0 or end <= start:
            raise ValueError(f"bad range {chrom}:{start}-{end}")
        stop = min(end, length)
        if stop <= start:
            return "N" * (end - start)
        first = offset + (start // line_bases) * line_width + (start % line_bases)
        last = offset + (stop // line_bases) * line_width + (stop % line_bases)
        self._handle.seek(first)
        raw = self._handle.read(last - first).replace(b"\n", b"").replace(b"\r", b"")
        return raw.decode() + "N" * (end - stop)

    def close(self) -> None:
        self._handle.close()


def chrom_sizes(index: FastaIndex) -> dict[str, int]:
    return {name: index.length(name) for name in index.records}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", required=True, help="window manifest parquet")
    parser.add_argument("--weights", required=True, help="AlphaGenome fold weights")
    parser.add_argument("--fasta", required=True, help="uncompressed genome FASTA")
    parser.add_argument("--out", required=True, help="output directory for the shards")
    parser.add_argument("--taps", nargs="+", default=list(TAPS),
                        choices=list(TAPS), help="which taps to capture")
    parser.add_argument("--positions-per-window", type=int, default=8192)
    parser.add_argument("--windows-per-shard", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--float32", action="store_true",
                        help="full float32 compute; default is bfloat16 where supported")
    parser.add_argument("--limit-windows", type=int, default=0, help="stop after N windows, for a smoke test")
    parser.add_argument("--splits", nargs="+", help="only these splits")
    args = parser.parse_args(argv)

    import torch
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy

    manifest = pd.read_parquet(args.manifest)
    if args.splits:
        manifest = manifest[manifest.split.isin(args.splits)]
    if args.limit_windows:
        manifest = manifest.groupby("split", group_keys=False).head(args.limit_windows)
    if manifest.empty:
        raise ValueError("no windows selected")
    taps = [TAPS[name] for name in args.taps]

    index = FastaIndex(args.fasta)
    bf16 = (not args.float32 and args.device.startswith("cuda")
            and torch.cuda.is_bf16_supported())
    policy = DtypePolicy.mixed_precision() if bf16 else DtypePolicy.full_float32()
    print(f"loading weights on {args.device}, bf16={bf16}")
    model = AlphaGenome.from_pretrained(str(args.weights), dtype_policy=policy,
                                        device=args.device)
    model.eval().requires_grad_(False)

    # Hooks are installed once and the buffer cleared per window; re-registering
    # them for every forward would be pure overhead.
    with Capture(model, taps) as capture:
        def forward(onehot: np.ndarray) -> dict[str, np.ndarray]:
            capture.clear()
            x = torch.from_numpy(onehot).unsqueeze(0).to(args.device)
            with torch.inference_mode():
                model.encode(x, organism_index=0, resolutions=(128,))
            del x
            return capture.outputs()

        provenance = {"weights": str(args.weights), "fasta": Path(args.fasta).name,
                      "manifest": str(args.manifest),
                      "compute": "bfloat16" if bf16 else "float32",
                      "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device}
        index_json = run_extraction(
            manifest, index.fetch, forward, taps, args.out,
            positions_per_window=args.positions_per_window,
            windows_per_shard=args.windows_per_shard, seed=args.seed,
            provenance=provenance)

    index.close()
    rows = {tap.name: sum(r["taps"].get(tap.name, {}).get("rows", 0) for r in index_json["shards"])
            for tap in taps}
    print(f"done: {len(index_json['shards'])} shards, rows per tap {rows}")
    if args.device.startswith("cuda"):
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

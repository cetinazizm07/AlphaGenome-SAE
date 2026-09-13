"""Build one BED per concept from the frozen panel description.

Reads concept_panel.json, derives every concept from the pinned ENCODE and
GENCODE files, and writes them in a single format so `concepts.py` can treat
them alike. Source checksums are verified first: a panel that silently used a
different GENCODE release would change what the numbers mean.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ag_sae.windows import Intervals, merge, span, subtract


def md5(path: str | Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: str | Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "rt")


def frame_from(chrom: Iterable[str], start: Iterable[int], end: Iterable[int]) -> pd.DataFrame:
    return pd.DataFrame({"chrom": list(chrom), "start": np.asarray(start, dtype=np.int64),
                         "end": np.asarray(end, dtype=np.int64)})


# --------------------------------------------------------------------------
# ENCODE cCREs
# --------------------------------------------------------------------------


def ccre_intervals(path: str | Path, label: str | None, class_column: int) -> Intervals:
    """Intervals of one cCRE label, or of every cCRE when label is None.

    The class field holds comma-separated labels such as 'pELS,CTCF-bound', so
    membership is tested per label rather than by string equality.
    """
    rows = []
    with open_text(path) as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            if label is not None:
                if len(fields) <= class_column:
                    raise ValueError(f"{path}: fewer than {class_column + 1} columns")
                if label not in [p.strip() for p in fields[class_column].split(",")]:
                    continue
            rows.append((fields[0], int(fields[1]), int(fields[2])))
    if not rows:
        raise ValueError(f"{path}: no intervals for label {label!r}")
    chrom, start, end = zip(*rows)
    return merge(frame_from(chrom, start, end))


# --------------------------------------------------------------------------
# GENCODE
# --------------------------------------------------------------------------


def gencode_records(path: str | Path, feature: str) -> pd.DataFrame:
    """GTF rows of one feature type, converted to 0-based half-open."""
    wanted, rows = f"\t{feature}\t", []
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or wanted not in line:
                continue
            fields = line.split("\t")
            if fields[2] != feature:
                continue
            transcript = ""
            for part in fields[8].split(";"):
                part = part.strip()
                if part.startswith("transcript_id"):
                    transcript = part.split('"')[1]
                    break
            rows.append((fields[0], int(fields[3]) - 1, int(fields[4]), fields[6], transcript))
    if not rows:
        raise ValueError(f"{path}: no {feature} records")
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "strand", "transcript"])


def splice_sites(gtf: str | Path, site: str, flank_bp: int) -> Intervals:
    """Donor or acceptor sites, from the gaps between consecutive exons.

    An intron runs from one exon's end to the next exon's start. On the plus
    strand the donor is the intron start and the acceptor its end; on the minus
    strand they swap.
    """
    if site not in ("donor", "acceptor"):
        raise ValueError("site must be 'donor' or 'acceptor'")
    exons = gencode_records(gtf, "exon").sort_values(["transcript", "start"])
    chrom = exons.chrom.to_numpy()
    strand = exons.strand.to_numpy()
    transcript = exons.transcript.to_numpy()
    intron_start = exons.end.to_numpy()[:-1]
    intron_end = exons.start.to_numpy()[1:]
    # Only gaps inside one transcript are introns.
    same = (transcript[:-1] == transcript[1:]) & (intron_end > intron_start)
    if not same.any():
        raise ValueError("no introns found; is this a single-exon annotation?")
    plus = strand[:-1][same] == "+"
    starts, ends = intron_start[same], intron_end[same]
    wanted_start = site == "donor"
    site_pos = np.where(plus == wanted_start, starts, ends)
    return merge(frame_from(chrom[:-1][same], np.maximum(site_pos - flank_bp, 0),
                            site_pos + flank_bp))


def tss_windows(gtf: str | Path, window_bp: int, feature: str = "transcript") -> Intervals:
    """Transcript start sites, extended by window_bp on both sides."""
    records = gencode_records(gtf, feature)
    start = np.where(records.strand.to_numpy() == "+",
                     records.start.to_numpy(), records.end.to_numpy() - 1)
    return merge(frame_from(records.chrom, np.maximum(start - window_bp, 0),
                            start + window_bp + 1))


# --------------------------------------------------------------------------
# Derived from sequence
# --------------------------------------------------------------------------


def gc_quantile_bins(fasta: str | Path, bin_bp: int, quantile: float,
                     min_bins: int = 10) -> Intervals:
    """Bins in the top GC fraction of their own chromosome.

    Per chromosome, not genome wide: GC varies enough between chromosomes that
    a single threshold would mostly select whole chromosomes.
    """
    from ag_sae.windows import read_fasta

    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1")
    rows = []
    for chrom, sequence in read_fasta(fasta):
        usable = (len(sequence) // bin_bp) * bin_bp
        if usable < bin_bp:
            continue
        binned = sequence[:usable].reshape(-1, bin_bp)
        gc = np.isin(binned, [ord(c) for c in "GCgc"]).sum(1)
        known = np.isin(binned, [ord(c) for c in "ACGTacgt"]).all(1)
        if known.sum() < min_bins:
            continue        # a quantile over a handful of bins means nothing
        fraction = gc / bin_bp
        threshold = np.quantile(fraction[known], quantile)
        chosen = np.flatnonzero(known & (fraction >= threshold))
        if chosen.size:
            rows.append(frame_from([chrom] * chosen.size, chosen * bin_bp,
                                   (chosen + 1) * bin_bp))
    if not rows:
        raise ValueError(f"{fasta}: no usable bins")
    return merge(pd.concat(rows, ignore_index=True))


def shuffle_intervals(source: Intervals, sizes: Mapping[str, int], seed: int,
                      exclude: Intervals | None = None) -> Intervals:
    """Relocate intervals at random within their own chromosome.

    Keeps the count and length distribution and avoids excluded regions, so the
    only thing removed is the genomic position. Anything this scores above the
    null ceiling is a bug in the mask or the metric.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for chrom, intervals in source.items():
        size = sizes.get(chrom)
        if size is None:
            raise KeyError(f"no length known for {chrom}")
        blocked = (exclude or {}).get(chrom, np.empty((0, 2), dtype=np.int64))
        lengths = intervals[:, 1] - intervals[:, 0]
        placed, attempts = [], 0
        while len(placed) < len(lengths) and attempts < 50 * len(lengths) + 1000:
            attempts += 1
            length = int(lengths[len(placed)])
            start = int(rng.integers(0, max(size - length, 1)))
            if len(blocked) and ((blocked[:, 0] < start + length) & (blocked[:, 1] > start)).any():
                continue
            placed.append((start, start + length))
        if len(placed) < len(lengths):
            raise RuntimeError(f"{chrom}: could not place {len(lengths)} intervals")
        arr = np.asarray(placed, dtype=np.int64)
        rows.append(frame_from([chrom] * len(arr), arr[:, 0], arr[:, 1]))
    return merge(pd.concat(rows, ignore_index=True))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def verify_sources(panel: dict, paths: Mapping[str, str | Path], strict: bool = True) -> dict:
    """Check each supplied file against the md5 recorded in the panel."""
    report = {}
    for name, spec in panel["sources"].items():
        path = paths.get(name)
        if path is None:
            report[name] = "not supplied"
            continue
        got = md5(path)
        ok = got == spec["md5"]
        report[name] = "ok" if ok else f"md5 mismatch: expected {spec['md5']}, got {got}"
        if strict and not ok:
            raise ValueError(f"{name}: {report[name]}")
    return report


def build_concept(spec: dict, panel: dict, paths: Mapping[str, str | Path],
                  built: Mapping[str, Intervals],
                  chrom_sizes: Mapping[str, int] | None,
                  exclude: Intervals | None) -> Intervals:
    """Derive one concept's intervals from its recipe."""
    how = spec["derivation"]
    kind = how["kind"]
    if kind in ("ccre_class", "ccre_union"):
        source = panel["sources"][how["source"]]
        return ccre_intervals(paths[how["source"]],
                              how.get("label") if kind == "ccre_class" else None,
                              source["class_column_0based"])
    if kind == "gencode_splice":
        return splice_sites(paths[how["source"]], how["site"], int(how["flank_bp"]))
    if kind == "gencode_tss_window":
        return tss_windows(paths[how["source"]], int(how["window_bp"]),
                           how.get("feature", "transcript"))
    if kind == "gc_quantile":
        return gc_quantile_bins(paths[how["source"]], int(panel["bin_bp"]),
                                float(how["quantile"]))
    if kind == "shuffle_of":
        if chrom_sizes is None:
            raise ValueError("shuffle_of needs chromosome sizes")
        return shuffle_intervals(built[how["concept"]], chrom_sizes,
                                 int(how["seed"]), exclude)
    raise ValueError(f"unknown derivation {kind!r}")


def write_bed(path: Path, name: str, intervals: Intervals) -> int:
    rows = [frame_from([chrom] * len(iv), iv[:, 0], iv[:, 1])
            for chrom, iv in sorted(intervals.items()) if len(iv)]
    frame = pd.concat(rows, ignore_index=True) if rows else frame_from([], [], [])
    frame["name"] = name
    frame.to_csv(path, sep="\t", header=False, index=False)
    return len(frame)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", default="concept_panel.json")
    parser.add_argument("--encode-ccre", help="ENCFF234XEZ.bed.gz")
    parser.add_argument("--gencode-gtf", help="gencode.v50.basic.annotation.gtf.gz")
    parser.add_argument("--genome-fasta", help="GRCh38.primary_assembly.genome.fa.gz")
    parser.add_argument("--chrom-sizes", help="two-column TSV, needed for shuffled controls")
    parser.add_argument("--exclude", help="BED of assembly gaps, kept clear of shuffled intervals")
    parser.add_argument("--out", required=True, help="directory for the concept BEDs")
    parser.add_argument("--only", nargs="+", help="build just these concepts")
    parser.add_argument("--skip-md5", action="store_true", help="skip source verification")
    args = parser.parse_args(argv)

    panel = json.loads(Path(args.panel).read_text())
    paths = {k: v for k, v in (("encode_ccre", args.encode_ccre),
                               ("gencode_gtf", args.gencode_gtf),
                               ("genome_fasta", args.genome_fasta)) if v}
    print(json.dumps(verify_sources(panel, paths, strict=not args.skip_md5), indent=2))

    sizes = None
    if args.chrom_sizes:
        table = pd.read_csv(args.chrom_sizes, sep="\t", header=None, names=["chrom", "size"])
        sizes = dict(zip(table.chrom.astype(str), table["size"].astype(int)))
    exclude = None
    if args.exclude:
        from ag_sae.windows import read_exclude_bed
        exclude = read_exclude_bed(args.exclude)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    built: dict[str, Intervals] = {}
    summary = []
    for spec in panel["concepts"]:
        name = spec["name"]
        if args.only and name not in args.only:
            continue
        how = spec["derivation"]
        needed = how.get("source")
        if needed and needed not in paths:
            print(f"skip {name}: needs --{needed.replace('_', '-')}")
            continue
        if how["kind"] == "shuffle_of":
            if sizes is None:
                print(f"skip {name}: needs --chrom-sizes")
                continue
            if how["concept"] not in built:
                print(f"skip {name}: {how['concept']} was not built")
                continue
        intervals = build_concept(spec, panel, paths, built, sizes, exclude)
        built[name] = intervals
        count = write_bed(out / f"{name}.bed", name, intervals)
        row = {"concept": name, "role": spec["role"], "intervals": count,
               "covered_mb": round(span(intervals) / 1e6, 2)}
        expected = spec.get("expected_elements")
        if expected:
            # Merging joins overlapping elements, so the count can fall below
            # the number of records in the source file.
            row["expected_elements"] = expected
            row["merged_ratio"] = round(count / expected, 3)
        summary.append(row)
        print(f"  {name:<22} {row['role']:<18} {count:>9} intervals  {row['covered_mb']:>8.2f} Mb")

    (out / "panel_summary.json").write_text(json.dumps(
        {"panel": panel["schema"], "frozen_at": panel["frozen_at"], "concepts": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

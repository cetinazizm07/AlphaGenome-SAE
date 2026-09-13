"""Write SAE feature activations as genome browser tracks.

The figures in `figures.py` are for a paper: static, vector, laid out by hand.
This module is for looking around. It writes the same numbers as bedGraph and
BED, which IGV, igv.js, igv-notebook and the UCSC browser all read, so nothing
here depends on a particular viewer.

bedGraph rather than BigWig because BigWig needs a compiled writer, and at one
value per 128 bp bin a window is small enough that plain text is fine.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

#: Track colours as IGV wants them, "r,g,b".
def _rgb(colour: str) -> str:
    colour = colour.lstrip("#")
    return ",".join(str(int(colour[i:i + 2], 16)) for i in (0, 2, 4))


def _open(path: Path):
    return gzip.open(path, "wt") if path.suffix == ".gz" else open(path, "w")


def write_bedgraph(
    bins: pd.DataFrame,
    values: np.ndarray,
    path: str | Path,
    *,
    name: str,
    description: str = "",
    colour: str = "#2a78d6",
    drop_zeros: bool = True,
) -> Path:
    """One bedGraph of a feature's activation, row-aligned with `bins`.

    Zero rows are dropped by default. Under TopK most bins are exactly zero, so
    keeping them multiplies the file size for no information; a browser draws a
    gap and a zero the same way.
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.size != len(bins):
        raise ValueError(f"{values.size} values for {len(bins)} bins")
    for column in ("chrom", "bin_start", "bin_end"):
        if column not in bins.columns:
            raise ValueError(f"bins needs a {column!r} column")
    if not np.isfinite(values).all():
        raise ValueError("activation contains non-finite values")

    frame = pd.DataFrame({
        "chrom": bins.chrom.to_numpy(),
        "start": bins.bin_start.to_numpy(dtype=np.int64),
        "end": bins.bin_end.to_numpy(dtype=np.int64),
        "value": values,
    })
    if (frame.end <= frame.start).any():
        raise ValueError("every bin must have end > start")
    if drop_zeros:
        frame = frame[frame.value != 0.0]
    frame = frame.sort_values(["chrom", "start"], kind="stable")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open(path) as handle:
        handle.write(
            f'track type=bedGraph name="{name}" description="{description or name}" '
            f'color={_rgb(colour)} visibility=full autoScale=on\n')
        for row in frame.itertuples(index=False):
            handle.write(f"{row.chrom}\t{row.start}\t{row.end}\t{row.value:.6g}\n")
    return path


def write_interval_bed(
    intervals: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    name: str,
    colour: str = "#8a8983",
) -> Path:
    """A BED of one concept's intervals, keyed by chromosome."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for chrom, spans in intervals.items():
        spans = np.asarray(spans, dtype=np.int64)
        if spans.size == 0:
            continue
        if spans.ndim != 2 or spans.shape[1] != 2:
            raise ValueError(f"{chrom}: intervals must be (n, 2)")
        for start, end in spans:
            rows.append((chrom, int(start), int(end)))
    rows.sort()
    with _open(path) as handle:
        handle.write(f'track name="{name}" itemRgb="On" color={_rgb(colour)}\n')
        for chrom, start, end in rows:
            handle.write(f"{chrom}\t{start}\t{end}\t{name}\t0\t.\n")
    return path


def igv_config(
    tracks: Sequence[Mapping[str, str]],
    *,
    locus: str,
    genome: str = "hg38",
) -> dict:
    """The dictionary igv.js and igv-notebook both take.

    Kept as plain data so it can be written to JSON, handed to igv-notebook, or
    pasted into an igv.js page, without importing a viewer here.
    """
    if not locus:
        raise ValueError("locus is required, for example 'chr7:30,000,000-30,153,600'")
    return {"genome": genome, "locus": locus, "tracks": [dict(t) for t in tracks]}


def bedgraph_track(path: str | Path, *, name: str, colour: str = "#2a78d6",
                   height: int = 50) -> dict:
    """One track entry for `igv_config`, pointing at a local bedGraph."""
    return {"name": name, "url": str(path), "format": "bedgraph",
            "type": "wig", "color": colour, "height": height,
            "autoscale": True}


def annotation_track(path: str | Path, *, name: str, colour: str = "#8a8983",
                     height: int = 30) -> dict:
    """One track entry for `igv_config`, pointing at a local BED."""
    return {"name": name, "url": str(path), "format": "bed",
            "type": "annotation", "color": colour, "height": height}


def show(config: Mapping) -> "object":
    """Open the config in igv-notebook, inside a Jupyter session.

    Imported here and not at module scope: igv-notebook is only needed for the
    interactive view, and the export functions above must work without it.
    """
    try:
        import igv_notebook
    except ImportError as error:                      # pragma: no cover
        raise ImportError(
            "igv-notebook is not installed. Either `pip install igv-notebook`, "
            "or load the files this module writes in IGV desktop or UCSC."
        ) from error
    igv_notebook.init()
    browser = igv_notebook.Browser(
        {"genome": config["genome"], "locus": config["locus"]})
    for track in config["tracks"]:
        browser.load_track(dict(track))
    return browser

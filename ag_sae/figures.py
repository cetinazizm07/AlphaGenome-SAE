"""Publication figures for SAE concept matching.

Each figure is written as PDF, SVG and 300 dpi PNG, plus a CSV of the numbers
behind it. The CSV is required, not a convenience: three palette slots fall
below 3:1 contrast on white, and the rule is that colour always comes with a
table view.

Form choices:

* Recovery is a dumbbell. The reader needs the gap between the SAE and the raw
  channels; grouped bars would make them subtract it themselves.
* Depth is small multiples, one panel per concept. The claim is the shape of
  each curve (flat = built by convolution, rising = built by attention).
  Overlaying eight concepts hides that and exceeds the series cap.
* The locus view gives each cCRE class its own lane, so identity comes from
  position and label rather than hue alone.
* No dual axes. Colour is either one hue of magnitude or an identity that a
  label repeats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Palette and style
# --------------------------------------------------------------------------

#: Validated categorical slots (light surface). The order is what makes them
#: colour-vision safe; re-ordering invalidates the checks.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
#: Grey for the context series in an emphasis pair.
CONTEXT = "#a8a7a0"
GRID = "#e6e5e1"
#: Single-hue ramp for magnitude. The mid step clears 3:1 on light.
SEQ_MID = "#2a78d6"
SEQ_LIGHT = "#cde2fb"

#: Depth order. A rule is drawn at the conv/tower boundary, which is where
#: "did attention add anything" is read.
TAP_ORDER = ("bin_size_4", "bin_size_16", "bin_size_64",
             "resid_pre_b0", "resid_pre_b4", "resid_pre_b8")
TAP_LABEL = {"bin_size_4": "4 bp", "bin_size_16": "16 bp", "bin_size_64": "64 bp",
             "resid_pre_b0": "blk 0", "resid_pre_b4": "blk 4", "resid_pre_b8": "blk 8"}
#: Most marks a profile panel can show before they stop being separable.
MAX_PROFILE_POINTS = 900
N_CONV_TAPS = 3


def use_style() -> None:
    """Light axes, hairline grid, text in ink colours. Safe to call twice."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        # DejaVu ships with matplotlib, so figures render the same on the
        # cluster and on a laptop. Helvetica first if a journal asks for it.
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.6,
        "axes.labelcolor": INK_SECONDARY, "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": False,
        "legend.frameon": False, "legend.handlelength": 1.4,
        "lines.linewidth": 1.8, "lines.markersize": 6,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # keep text editable
        "svg.fonttype": "none",
    })


def save(fig, out: str | Path, table: pd.DataFrame | None = None) -> list[Path]:
    """Write PDF + SVG + 300 dpi PNG, and the panel's source numbers as CSV."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, kwargs in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 300})):
        path = out.with_suffix(suffix)
        fig.savefig(path, **kwargs)
        written.append(path)
    if table is not None:
        path = out.with_suffix(".csv")
        table.to_csv(path, index=False)
        written.append(path)
    return written


# --------------------------------------------------------------------------
# Figure 1 - concept recovery
# --------------------------------------------------------------------------


def figure_recovery(per_concept: pd.DataFrame, out: str | Path, *, title: str = "") -> list[Path]:
    """Per concept: the SAE's best feature against the raw channels and the null.

    A dumbbell, because the gap between the two dots is the point. Filled
    marker = the SAE cleared its permutation ceiling, open = it did not.
    Recovery is shown by fill rather than colour so it survives greyscale
    printing and colour-vision deficiency.
    """
    import matplotlib.pyplot as plt

    use_style()
    frame = per_concept.sort_values("best_auroc").reset_index(drop=True)
    y = np.arange(len(frame))
    has_raw = "raw_best_auroc" in frame.columns

    # Height leaves a fixed strip at the bottom for the legend; placing it
    # inside the axes overlapped the lowest concept.
    legend_inches = 0.82          # legend row (~0.25 in) + note line + margins
    height = 0.34 * len(frame) + 1.0 + legend_inches
    fig, ax = plt.subplots(figsize=(3.6, height))
    if has_raw:
        # The connector is context: thin, and under both dots.
        ax.hlines(y, frame.raw_best_auroc, frame.best_auroc,
                  color=GRID, linewidth=2.2, zorder=1)
        ax.scatter(frame.raw_best_auroc, y, s=42, color=CONTEXT, zorder=2,
                   label="Raw channel", linewidths=0)
    ax.scatter(frame.best_auroc, y, s=52, zorder=3, label="SAE feature",
               facecolors=np.where(frame.recovered, SERIES[0], SURFACE),
               edgecolors=SERIES[0], linewidths=1.5)
    # A tick, not a dot: the ceiling is a threshold, not a measurement.
    ax.scatter(frame.null_p95, y, marker="|", s=110, color=INK_MUTED,
               linewidths=1.3, zorder=2, label="Null ceiling")

    for row, value in zip(y, frame.best_auroc):
        ax.annotate(f"{value:.2f}", (value, row), xytext=(7, 0),
                    textcoords="offset points", va="center",
                    fontsize=7, color=INK_SECONDARY)

    ax.set_yticks(y, frame.concept)
    ax.set_xlabel("AUROC against the annotation")
    ax.set_xlim(0.48, min(1.035, max(1.0, frame.best_auroc.max() + 0.09)))
    ax.axvline(0.5, color=GRID, linewidth=0.8, zorder=0)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if title:
        ax.set_title(title, loc="left", color=INK_PRIMARY, pad=8)
    fig.tight_layout(rect=(0, legend_inches / height, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0), labelcolor=INK_SECONDARY, columnspacing=1.2)
    fig.text(0.5, 0.34 / height, "filled marker = clears the null ceiling",
             ha="center", va="bottom", fontsize=7, color=INK_MUTED)

    columns = [c for c in ("concept", "n_positive_bins", "prevalence", "best_feature",
                           "best_auroc", "raw_best_auroc", "sae_minus_raw",
                           "null_p95", "recovered") if c in frame.columns]
    return save(fig, out, frame[columns])


# --------------------------------------------------------------------------
# Figure 2 - depth
# --------------------------------------------------------------------------


def figure_depth(
    by_tap: Mapping[str, pd.DataFrame], out: str | Path, *, n_columns: int = 3
) -> list[Path]:
    """One panel per concept: AUROC across taps, conv left of the rule.

    Small multiples rather than one multi-line chart, because the claim is the
    shape of each curve: flat across the conv taps and rising after the rule
    would mean attention built the concept.

    The grey band is everything at or below that tap's permutation ceiling. A
    point inside it is not distinguishable from chance, so the band does the
    job of a significance marker.
    """
    import matplotlib.pyplot as plt

    use_style()
    taps = [t for t in TAP_ORDER if t in by_tap]
    unknown = set(by_tap) - set(TAP_ORDER)
    if unknown:
        raise ValueError(f"Unknown taps {sorted(unknown)}; expected {list(TAP_ORDER)}")
    if len(taps) < 2:
        raise ValueError("The depth figure needs at least two taps")

    tidy = pd.concat(
        [frame.assign(tap=tap, tap_index=index) for index, (tap, frame)
         in enumerate((t, by_tap[t]) for t in taps)],
        ignore_index=True,
    )
    concepts = (tidy.groupby("concept").best_auroc.max()
                .sort_values(ascending=False).index.tolist())
    n_rows = int(np.ceil(len(concepts) / n_columns))
    legend_inches = 0.34
    height = 1.7 * n_rows + legend_inches
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(2.25 * n_columns, height),
                             sharex=True, sharey=True, squeeze=False)
    x = np.arange(len(taps))
    # Sits between the last conv tap and the first tower tap.
    boundary = sum(1 for t in taps if t.startswith("bin_size")) - 0.5

    for axis, concept in zip(axes.ravel(), concepts):
        rows = tidy[tidy.concept == concept].set_index("tap").reindex(taps)
        axis.fill_between(x, 0.0, rows.null_p95.to_numpy(), step="mid",
                          color=GRID, linewidth=0, zorder=0)
        if 0 < boundary < len(taps) - 1:
            axis.axvline(boundary, color=INK_MUTED, linewidth=0.7,
                         linestyle=(0, (3, 3)), zorder=1)
        if "raw_best_auroc" in rows:
            axis.plot(x, rows.raw_best_auroc, marker="o", color=CONTEXT,
                      markeredgewidth=0, zorder=2, label="Raw channels")
        axis.plot(x, rows.best_auroc, marker="o", color=SERIES[0],
                  markeredgewidth=0, zorder=3, label="SAE features")
        axis.set_title(concept, loc="left", color=INK_PRIMARY, fontsize=8.5, pad=4)
        axis.yaxis.grid(True, zorder=0)
        axis.set_axisbelow(True)

    for axis in axes.ravel()[len(concepts):]:
        axis.set_visible(False)
    for axis in axes.ravel():
        axis.set_xticks(x, [TAP_LABEL.get(t, t) for t in taps])
        axis.set_ylim(0.45, 1.02)
    for axis in axes[:, 0]:
        axis.set_ylabel("AUROC")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=GRID))
    labels.append("At or below null")
    if 0 < boundary < len(taps) - 1:
        handles.append(plt.Line2D([0], [0], color=INK_MUTED, linewidth=0.7,
                                  linestyle=(0, (3, 3))))
        labels.append("conv | tower boundary")
    fig.tight_layout(rect=(0, legend_inches / height, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               labelcolor=INK_SECONDARY, columnspacing=1.4)

    columns = [c for c in ("tap", "concept", "best_feature", "best_auroc",
                           "raw_best_auroc", "null_p95", "recovered") if c in tidy.columns]
    return save(fig, out, tidy[columns].sort_values(["concept", "tap"]))


# --------------------------------------------------------------------------
# Figure 3 - locus view
# --------------------------------------------------------------------------


def figure_locus(
    bins: pd.DataFrame,
    profile: np.ndarray,
    ccre: Mapping[str, Mapping[str, np.ndarray]],
    out: str | Path,
    *,
    feature: int | None = None,
    classes: Sequence[str] | None = None,
    top_n: int = 10,
    max_points: int = MAX_PROFILE_POINTS,
) -> list[Path]:
    """One feature's activation along the genome, over annotation lanes.

    Shows whether a feature's peaks sit on annotated elements, and on which
    class. Each cCRE class gets its own lane, so identity comes from position
    and the left-hand label; hue is only a scanning aid.

    A 1 Mb window is 8192 bins. Past `max_points` the profile is max-pooled and
    the y-label says so; treat that as an overview. For a figure where single
    bins must be separable, pass a narrower region (50-150 kb works well).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    use_style()
    if len(bins) != len(profile):
        raise ValueError("Profile must have one value per bin")
    if bins.chrom.nunique() != 1:
        raise ValueError("A locus view covers a single chromosome")
    chrom = str(bins.chrom.iloc[0])
    names = list(classes) if classes is not None else sorted(ccre)
    missing = [n for n in names if n not in ccre]
    if missing:
        raise ValueError(f"No intervals for {missing}")

    # 8192 bins is more marks than a 7-inch panel can separate; they merge
    # into a solid block. Max-pool to the panel's budget. Max rather than mean
    # so peak heights stay honest.
    pooled = len(bins) > max_points
    if pooled:
        groups = np.minimum(np.arange(len(bins)) * max_points // len(bins), max_points - 1)
        order = np.argsort(groups, kind="stable")
        edges = np.searchsorted(groups[order], np.arange(max_points))
        profile = np.maximum.reduceat(profile[order], edges)
        starts = bins.bin_start.to_numpy()[order][edges]
        ends = np.r_[starts[1:], bins.bin_end.to_numpy().max()]
        bin_bp = int(np.median(np.diff(starts))) if max_points > 1 else 0
        plotted = pd.DataFrame({"bin_start": starts, "bin_end": ends})
    else:
        plotted = bins[["bin_start", "bin_end"]].reset_index(drop=True)
        bin_bp = 0
    start_mb = plotted.bin_start.to_numpy() / 1e6
    end_mb = plotted.bin_end.to_numpy() / 1e6
    lane_height = 0.26
    fig, (track, lanes) = plt.subplots(
        2, 1, figsize=(7.2, 2.1 + lane_height * len(names)),
        gridspec_kw={"height_ratios": [2.2, lane_height * len(names)], "hspace": 0.12},
        sharex=True)

    # Step, not line: a bin is an interval, so the value is flat across it.
    track.fill_between(start_mb, 0, profile, step="post", color=SEQ_LIGHT, linewidth=0)
    track.step(start_mb, profile, where="post", color=SEQ_MID, linewidth=1.2)
    if top_n and (profile > 0).any():
        order = np.argsort(profile)[::-1][:top_n]
        order = order[profile[order] > 0]
        track.scatter((start_mb[order] + end_mb[order]) / 2, profile[order],
                      s=14, facecolors=SURFACE, edgecolors=SERIES[0],
                      linewidths=1.1, zorder=4)
    label = f"Feature {feature}" if feature is not None else "Feature"
    unit = f"\nmax per {bin_bp:,} bp" if pooled else ""
    track.set_ylabel(f"{label}\nactivation{unit}")
    track.set_ylim(bottom=0)
    track.yaxis.grid(True)
    track.set_axisbelow(True)
    track.spines["bottom"].set_visible(False)
    track.tick_params(axis="x", length=0)
    if top_n:
        track.annotate(f"open circles: {top_n} strongest", (1.0, 1.012),
                       xycoords="axes fraction", ha="right", va="bottom",
                       fontsize=7, color=INK_MUTED)

    window = (float(bins.bin_start.min()), float(bins.bin_end.max()))
    for row, name in enumerate(names):
        spans = np.asarray(ccre[name].get(chrom, np.empty((0, 2))), dtype=np.int64)
        colour = SERIES[row % len(SERIES)]
        for lo, hi in spans:
            if hi <= window[0] or lo >= window[1]:
                continue  # element lies outside the plotted region
            lanes.add_patch(Rectangle((lo / 1e6, row + 0.14), (hi - lo) / 1e6, 0.72,
                                      facecolor=colour, edgecolor="none"))
        drawn = int(((spans[:, 1] > window[0]) & (spans[:, 0] < window[1])).sum()) if len(spans) else 0
        # Always label directly: three palette slots are below 3:1 contrast
        # here, so the text is what makes the lane readable.
        lanes.annotate(f"{name}  ({drawn})", (-0.012, row + 0.5), xycoords=("axes fraction", "data"),
                       ha="right", va="center", fontsize=7.5, color=INK_SECONDARY)

    lanes.set_ylim(len(names), 0)
    lanes.set_yticks([])
    lanes.set_xlim(window[0] / 1e6, window[1] / 1e6)
    lanes.set_xlabel(f"{chrom} position (Mb)")
    for side in ("left", "top", "right"):
        lanes.spines[side].set_visible(False)

    table = pd.DataFrame({"chrom": chrom, "bin_start": plotted.bin_start.to_numpy(),
                          "bin_end": plotted.bin_end.to_numpy(), "activation": profile})
    return save(fig, out, table)


# --------------------------------------------------------------------------
# Figure 4 - how concept-specific are the features
# --------------------------------------------------------------------------


def figure_specificity(
    per_feature: pd.DataFrame, out: str | Path, *, chance: float | None = None
) -> list[Path]:
    """Distribution of how concentrated each feature's top bins are.

    An ECDF rather than a histogram: the question is cumulative ("what fraction
    of features are at least this specific") and an ECDF has no bin width to
    argue about.

    `chance` is what a randomly firing feature would reach, i.e. the prevalence
    of the commonest concept. Without it the curve looks impressive for reasons
    unrelated to the SAE.
    """
    import matplotlib.pyplot as plt

    use_style()
    shares = np.sort(per_feature.dominant_share.to_numpy(dtype=float))
    if not shares.size:
        raise ValueError("No features to summarise")
    fraction = 1.0 - np.arange(shares.size) / shares.size

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.step(shares, fraction, where="post", color=SERIES[0], linewidth=1.8, zorder=3)
    if chance is not None:
        ax.axvline(chance, color=INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)), zorder=2)
        ax.annotate("chance", (chance, 1.0), xytext=(4, -2), textcoords="offset points",
                    fontsize=7, color=INK_MUTED, va="top")
    ax.set_xlabel("Share of a feature's top bins in its dominant concept")
    ax.set_ylabel("Fraction of features at least this specific")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    median = float(np.median(shares))
    ax.annotate(f"median {median:.2f}   n = {shares.size:,}", (0.99, 0.99),
                xycoords="axes fraction", ha="right", va="top",
                fontsize=7.5, color=INK_SECONDARY)
    fig.tight_layout()
    return save(fig, out, per_feature)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def feature_profile(sae, activations: np.ndarray, feature: int) -> np.ndarray:
    """Dense activation of one feature across every bin, from the sparse codes."""
    codes = sae.encode(activations).tocsc()
    if not 0 <= feature < codes.shape[1]:
        raise ValueError(f"Feature {feature} outside 0..{codes.shape[1] - 1}")
    return np.asarray(codes[:, [feature]].todense()).ravel()


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("recovery", help="Figure 1: SAE vs raw vs null, per concept")
    p.add_argument("--per-concept", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")

    p = sub.add_parser("depth", help="Figure 2: AUROC across taps, one panel per concept")
    p.add_argument("--tap", action="append", required=True, metavar="NAME=PATH",
                   help="repeat once per tap, e.g. --tap resid_pre_b4=out_b4/per_concept.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--columns", type=int, default=4)

    p = sub.add_parser("locus", help="Figure 3: one feature along the genome, over cCRE lanes")
    p.add_argument("--sae", required=True)
    p.add_argument("--activations", required=True)
    p.add_argument("--bins", required=True)
    p.add_argument("--ccre", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--feature", type=int, help="feature index; default: best feature for --concept")
    p.add_argument("--concept", help="pick the winning feature for this concept")
    p.add_argument("--per-concept", help="per_concept.parquet, needed with --concept")
    p.add_argument("--region", help="chrom:start-end")
    p.add_argument("--class-column", type=int, default=5)

    p = sub.add_parser("specificity", help="Figure 4: how concept-specific the features are")
    p.add_argument("--per-feature", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chance", type=float, help="prevalence of the commonest concept")

    args = parser.parse_args(argv)

    if args.command == "recovery":
        written = figure_recovery(pd.read_parquet(args.per_concept), args.out, title=args.title)
    elif args.command == "depth":
        by_tap = {}
        for item in args.tap:
            name, _, path = item.partition("=")
            if not path:
                raise ValueError(f"--tap wants NAME=PATH, got {item!r}")
            by_tap[name] = pd.read_parquet(path)
        written = figure_depth(by_tap, args.out, n_columns=args.columns)
    elif args.command == "specificity":
        written = figure_specificity(pd.read_parquet(args.per_feature), args.out, chance=args.chance)
    else:
        from ag_sae.concepts import FrozenSAE, parse_region, read_ccre_bed

        bins = pd.read_parquet(args.bins)
        activations = np.load(args.activations, mmap_mode="r")
        if len(bins) != len(activations):
            raise ValueError(f"{len(activations)} activation rows vs {len(bins)} bins")
        if args.region:
            chrom, lo, hi = parse_region(args.region)
            keep = ((bins.chrom == chrom) & (bins.bin_start >= lo) & (bins.bin_end <= hi)).to_numpy()
            if not keep.any():
                raise ValueError(f"No bins inside {args.region}")
            bins, activations = bins[keep].reset_index(drop=True), np.asarray(activations[keep])
        else:
            activations = np.asarray(activations)

        feature = args.feature
        if feature is None:
            if not (args.concept and args.per_concept):
                raise ValueError("Pass --feature, or --concept together with --per-concept")
            table = pd.read_parquet(args.per_concept).set_index("concept")
            if args.concept not in table.index:
                raise ValueError(f"{args.concept!r} not in {sorted(table.index)}")
            feature = int(table.loc[args.concept, "best_feature"])

        sae = FrozenSAE.from_torch_checkpoint(args.sae)
        written = figure_locus(bins, feature_profile(sae, activations, feature),
                               read_ccre_bed(args.ccre, args.class_column), args.out,
                               feature=feature)

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

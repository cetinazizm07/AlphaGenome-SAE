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
# Figure 5 - reconstruction quality across runs
# --------------------------------------------------------------------------


def _spread_labels(anchors, gap: float):
    """Push label anchors apart vertically so none of them overprint.

    Takes [x, y, payload] rows, returns them with y adjusted. Runs that land on
    the same point would otherwise stack their labels into an unreadable blob.
    """
    ordered = sorted(anchors, key=lambda row: row[1])
    for lower, upper in zip(ordered, ordered[1:]):
        if upper[1] - lower[1] < gap:
            upper[1] = lower[1] + gap
    return ordered


def figure_pareto(runs: pd.DataFrame, out: str | Path, *, title: str = "") -> list[Path]:
    """Reconstruction against sparsity, one mark per training run.

    Two layouts, chosen from the data. If `mean_l0` varies the runs form a
    sparsity sweep and the figure is a true trade-off plot with the
    non-dominated frontier drawn. If `mean_l0` is the same everywhere, TopK
    pinned it to k and a trade-off plot would be a vertical stripe, so the x
    axis becomes depth instead and the fixed sparsity is stated on the panel.

    Colour separates conv from tower only. Six taps exceed the categorical
    slots, so tap identity comes from the axis and its label.
    """
    import matplotlib.pyplot as plt

    required = {"tap", "seed", "mean_l0", "fvu"}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"runs is missing {sorted(missing)}")
    if runs.empty:
        raise ValueError("No runs to plot")

    use_style()
    table = runs.copy()
    table["tap"] = pd.Categorical(table.tap, categories=[t for t in TAP_ORDER
                                                        if t in set(table.tap)],
                                  ordered=True)
    table = table.sort_values(["tap", "seed"])
    taps = list(table.tap.cat.categories)
    l0 = table.mean_l0.to_numpy(dtype=float)
    swept = bool(l0.size and (l0.max() - l0.min()) > 1e-6 * max(1.0, abs(l0.max())))

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    # Classify by which tap it is, not where it sits in this subset. Passing
    # only the tower taps must not colour them as convolutional.
    conv = frozenset(TAP_ORDER[:N_CONV_TAPS])
    colour = {t: (SERIES[0] if t in conv else SERIES[1]) for t in taps}
    boundary = sum(1 for t in taps if t in conv)

    if swept:
        anchors = []
        for tap in taps:
            sub = table[table.tap == tap].sort_values("mean_l0")
            ax.plot(sub.mean_l0, sub.fvu, "-o", color=colour[tap], markersize=4,
                    linewidth=1.2, markeredgewidth=0, zorder=3, alpha=0.9)
            end = sub.iloc[-1]
            anchors.append([float(end.mean_l0), float(end.fvu), tap])
        for x, y, tap in _spread_labels(anchors, gap=0.045):
            ax.annotate(TAP_LABEL.get(tap, tap), (x, y), xytext=(6, 0),
                        textcoords="offset points", fontsize=7,
                        color=colour[tap], va="center")
        # Frontier: the runs that nothing else beats on both axes at once.
        order = table.sort_values("mean_l0")
        keep, floor = [], np.inf
        for row in order.itertuples(index=False):
            if row.fvu < floor:
                keep.append((row.mean_l0, row.fvu))
                floor = row.fvu
        if len(keep) > 1:
            ax.step(*zip(*keep), where="post", color=INK_MUTED, linewidth=1.0,
                    linestyle=(0, (3, 3)), zorder=2)
        ax.set_xlabel("Mean L0 (active features per bin)")
        note = "dashed line: non-dominated runs"
    else:
        positions = {t: i for i, t in enumerate(taps)}
        # Seeds usually land within a hair of each other, so nudge them apart.
        # Without this the three runs print as one blob and the spread, which
        # is the reason for training three seeds, becomes invisible.
        seed_values = sorted(table.seed.unique())
        nudge = {s: (i - (len(seed_values) - 1) / 2) * 0.13
                 for i, s in enumerate(seed_values)}
        for tap in taps:
            sub = table[table.tap == tap]
            x = np.array([positions[tap] + nudge[s] for s in sub.seed])
            ax.plot(x, sub.fvu, "o", color=colour[tap], markersize=5,
                    markeredgewidth=0, zorder=3)
            ax.plot([positions[tap]], [sub.fvu.mean()], "_", color=INK_PRIMARY,
                    markersize=16, markeredgewidth=1.4, zorder=4)
        ax.set_xticks(range(len(taps)), [TAP_LABEL.get(t, t) for t in taps])
        ax.set_xlabel("Tap, shallow to deep")
        ax.set_xlim(-0.6, len(taps) - 0.4)
        if 0 < boundary < len(taps):
            ax.axvline(boundary - 0.5, color=INK_MUTED, linewidth=0.7,
                       linestyle=(0, (3, 3)), zorder=1)
        handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=SERIES[0]),
                   plt.Line2D([0], [0], marker="o", linestyle="", color=SERIES[1]),
                   plt.Line2D([0], [0], marker="_", linestyle="", color=INK_PRIMARY,
                              markeredgewidth=1.4)]
        ax.legend(handles, ["Convolutional", "Transformer", "Seed mean"],
                  loc="upper right", labelcolor=INK_SECONDARY, ncol=1,
                  handletextpad=0.5)
        note = f"sparsity fixed, L0 = {l0[0]:.0f}"

    ax.set_ylabel("Fraction of variance unexplained")
    ax.set_ylim(0, max(1.0, float(table.fvu.max()) * 1.1))
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.annotate(note, (0.01, 0.02) if not swept else (0.99, 0.99),
                xycoords="axes fraction",
                ha="left" if not swept else "right",
                va="bottom" if not swept else "top",
                fontsize=7, color=INK_SECONDARY)
    if title:
        ax.set_title(title, loc="left", color=INK_PRIMARY, fontsize=9, pad=6)
    fig.tight_layout()
    return save(fig, out, table)


# --------------------------------------------------------------------------
# Figure 6 - one feature in detail
# --------------------------------------------------------------------------

#: Base colours, taken from the validated slots rather than the usual ad hoc
#: DNA scheme, so the letters stay separable under colour-vision deficiency.
BASE_COLOUR = {"A": SERIES[2], "C": SERIES[0], "G": SERIES[3], "T": SERIES[1]}


def information_content(sequences: Sequence[str], weights: np.ndarray | None = None,
                        alphabet: str = "ACGT") -> np.ndarray:
    """Per-position letter heights in bits, shaped (length, 4).

    Standard sequence-logo arithmetic: a position's total height is 2 bits
    minus its entropy, split between letters by frequency. Weighting by
    activation lets strongly firing sites count for more.
    """
    if not sequences:
        raise ValueError("No sequences")
    length = len(sequences[0])
    if any(len(s) != length for s in sequences):
        raise ValueError("Sequences must all be the same length")
    weights = (np.ones(len(sequences)) if weights is None
               else np.asarray(weights, dtype=float))
    if weights.shape != (len(sequences),):
        raise ValueError("One weight per sequence")
    weights = np.clip(weights, 0, None)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)

    counts = np.zeros((length, len(alphabet)))
    index = {b: i for i, b in enumerate(alphabet)}
    for sequence, weight in zip(sequences, weights):
        for position, base in enumerate(sequence.upper()):
            if base in index:
                counts[position, index[base]] += weight
    total = counts.sum(axis=1, keepdims=True)
    frequency = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(frequency > 0, frequency * np.log2(frequency), 0.0),
                             axis=1)
    return frequency * np.clip(np.log2(len(alphabet)) - entropy, 0, None)[:, None]


def _draw_logo(ax, heights: np.ndarray, alphabet: str = "ACGT") -> None:
    """Draw a sequence logo with logomaker, in the module's base colours.

    logomaker is the field-standard renderer, so the letters look like every
    other logo a reader has seen. It wants a (position, letter) frame, which is
    what `information_content` already returns once it is labelled.
    """
    import logomaker

    frame = pd.DataFrame(heights, columns=list(alphabet))
    frame.index.name = "pos"
    # big_on_top is the convention: readers take the top letter of each stack
    # as the consensus base, so inverting the order inverts the reading.
    logomaker.Logo(frame, ax=ax, color_scheme=dict(BASE_COLOUR),
                   show_spines=False, vpad=0.02, stack_order="big_on_top")
    ax.set_xlim(-0.5, heights.shape[0] - 0.5)
    ax.set_ylim(0, max(float(heights.sum(axis=1).max()) * 1.05, 0.1))
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(INK_MUTED)
    ax.spines["left"].set_linewidth(0.6)


def figure_feature_card(
    feature: int,
    activation: np.ndarray,
    sequences: Sequence[str],
    out: str | Path,
    *,
    concept_auroc: Mapping[str, float] | None = None,
    top_n: int = 12,
    sample_n: int = 12,
    seed: int = 0,
) -> list[Path]:
    """Everything about one feature on one page.

    The lower strip is the point of the figure. Showing only the strongest
    sites makes any feature look clean, so the same number of sites is drawn
    from across the firing range. If the two strips disagree, the feature is
    not monosemantic no matter how good its top examples look.
    """
    import matplotlib.pyplot as plt

    activation = np.asarray(activation, dtype=float).ravel()
    if activation.size != len(sequences):
        raise ValueError("One sequence per activation value")
    firing = np.flatnonzero(activation > 0)
    if firing.size < 2:
        raise ValueError("Feature fires on fewer than two sites")

    use_style()
    rng = np.random.default_rng(seed)
    strongest = firing[np.argsort(activation[firing])[::-1][:top_n]]
    # Sample the firing range in equal slices so the middle is represented.
    ranked = firing[np.argsort(activation[firing])[::-1]]
    edges = np.linspace(0, ranked.size, min(sample_n, ranked.size) + 1).astype(int)
    spread = np.array([rng.integers(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])
                       if hi > lo])
    spread = ranked[spread]

    height = 6.0 if concept_auroc else 4.2
    fig = plt.figure(figsize=(5.6, height))
    rows = 4 if concept_auroc else 3
    grid = fig.add_gridspec(rows, 1,
                            height_ratios=([1.0, 0.85, 1.0, 1.0] if concept_auroc
                                           else [1.0, 0.85, 1.0]),
                            hspace=0.62, top=0.94, bottom=0.08,
                            left=0.16, right=0.97)

    top_logo = fig.add_subplot(grid[0])
    _draw_logo(top_logo, information_content([sequences[i] for i in strongest],
                                             activation[strongest]))
    top_logo.set_title(f"Feature {feature}: top {len(strongest)} sites",
                       loc="left", color=INK_PRIMARY, fontsize=8.5, pad=4)
    top_logo.set_ylabel("bits")

    histogram = fig.add_subplot(grid[1])
    histogram.hist(activation[firing], bins=40, color=SEQ_MID, edgecolor="none")
    histogram.set_yscale("log")
    histogram.set_xlabel("Activation where the feature fires")
    histogram.set_ylabel("Sites")
    histogram.annotate(
        f"fires on {firing.size / activation.size:.2%} of bins",
        (0.99, 0.92), xycoords="axes fraction", ha="right", va="top",
        fontsize=7, color=INK_SECONDARY)

    spread_logo = fig.add_subplot(grid[2])
    _draw_logo(spread_logo, information_content([sequences[i] for i in spread],
                                                activation[spread]))
    spread_logo.set_title("Sites drawn evenly across the firing range",
                          loc="left", color=INK_SECONDARY, fontsize=8, pad=4)
    spread_logo.set_ylabel("bits")

    for axis in (top_logo, spread_logo):
        axis.set_xticks([])
        axis.spines["left"].set_visible(True)

    if concept_auroc:
        bars = fig.add_subplot(grid[3])
        names = list(concept_auroc)
        values = [float(concept_auroc[n]) for n in names]
        order = np.argsort(values)
        # Bars grow from 0.5, not from the axis edge. An AUROC bar drawn from
        # an arbitrary left limit exaggerates small differences, and chance is
        # the only meaningful zero here.
        bars.barh([names[i] for i in order], [values[i] - 0.5 for i in order],
                  left=0.5, color=SEQ_MID, height=0.62)
        bars.axvline(0.5, color=INK_MUTED, linewidth=0.9, linestyle=(0, (3, 3)))
        bars.set_xlim(0.45, 1.0)
        bars.set_title("Concept AUROC, measured from chance", loc="left",
                       color=INK_SECONDARY, fontsize=8, pad=4)
        bars.set_xlabel("AUROC for this feature")
        bars.tick_params(axis="y", length=0)

    table = pd.DataFrame({
        "rank": np.arange(len(strongest)),
        "site": strongest,
        "activation": activation[strongest],
        "sequence": [sequences[i] for i in strongest],
    })
    return save(fig, out, table)


# --------------------------------------------------------------------------
# Figure 7 - do seeds find the same features
# --------------------------------------------------------------------------


def match_across_seeds(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For every feature in `a`, its best cosine similarity to any in `b`.

    Both are (n_features, d) decoder directions. Rows are unit-normalised
    first, so this is cosine and not a dot product that rewards long vectors.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Both matrices must be (n_features, d) with the same d")
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a @ b.T).max(axis=1)


def figure_seed_stability(
    decoders: Mapping[int, np.ndarray],
    out: str | Path,
    *,
    threshold: float = 0.7,
    seed: int = 0,
) -> list[Path]:
    """How much of the dictionary survives a change of initialisation.

    A null is drawn alongside, because the best match out of thousands of
    directions is high by chance alone. Without it the curve says nothing; the
    gap between the curve and the null is the whole result.
    """
    import matplotlib.pyplot as plt

    seeds = sorted(decoders)
    if len(seeds) < 2:
        raise ValueError("Need at least two seeds to compare")
    shapes = {np.asarray(decoders[s]).shape for s in seeds}
    if len(shapes) != 1:
        raise ValueError(f"Decoders disagree in shape: {sorted(shapes)}")

    use_style()
    rng = np.random.default_rng(seed)
    n_features, d = next(iter(shapes))

    pairs, records = [], []
    for i, left in enumerate(seeds):
        for right in seeds[i + 1:]:
            best = match_across_seeds(decoders[left], decoders[right])
            pairs.append((f"seed {left} vs {right}", best))
            records.append({"pair": f"{left}-{right}",
                            "median_cosine": float(np.median(best)),
                            "matched_fraction": float((best >= threshold).mean())})

    random_a = rng.normal(size=(n_features, d))
    random_b = rng.normal(size=(n_features, d))
    null = match_across_seeds(random_a, random_b)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))

    def ecdf(values, **kwargs):
        ordered = np.sort(values)
        ax.step(ordered, 1.0 - np.arange(ordered.size) / ordered.size,
                where="post", **kwargs)

    for index, (label, best) in enumerate(pairs):
        ecdf(best, color=SERIES[index % len(SERIES)], linewidth=1.8, zorder=3,
             label=label)
    ecdf(null, color=CONTEXT, linewidth=1.4, linestyle=(0, (3, 3)), zorder=2,
         label="random directions")

    ax.axvline(threshold, color=INK_MUTED, linewidth=0.9, linestyle=(0, (1, 2)),
               zorder=1)
    ax.set_xlabel("Best cosine similarity to a feature in the other seed")
    ax.set_ylabel("Fraction of features at least this well matched")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    matched = np.mean([r["matched_fraction"] for r in records])
    ax.annotate(f"matched at {threshold:g}: {matched:.0%} of features",
                (0.02, 0.03), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=7.5, color=INK_SECONDARY)
    ax.legend(loc="upper right", labelcolor=INK_SECONDARY)
    fig.tight_layout()
    return save(fig, out, pd.DataFrame(records))


# --------------------------------------------------------------------------
# Figure 9 - genome browser view
# --------------------------------------------------------------------------


def centromere_from_gaps(gaps: pd.DataFrame, chrom: str) -> tuple[int, int] | None:
    """Largest assembly gap on a chromosome.

    In GRCh38 the centromere is modelled as a long run of N, so the biggest gap
    on a chromosome is it. This avoids pulling in a cytoband file for what is a
    single landmark on the ideogram.
    """
    on_chrom = gaps[gaps.chrom == chrom]
    if on_chrom.empty:
        return None
    widest = (on_chrom.end - on_chrom.start).idxmax()
    return int(on_chrom.loc[widest, "start"]), int(on_chrom.loc[widest, "end"])


def _coordinate_label(value: float, _position: int = 0) -> str:
    """Axis ticks as Mb or kb, whichever keeps the number short."""
    if abs(value) >= 1e6:
        return f"{value / 1e6:g} Mb"
    if abs(value) >= 1e3:
        return f"{value / 1e3:g} kb"
    return f"{value:g}"


def _draw_ideogram(ax, chrom: str, length: int, window: tuple[int, int],
                   centromere: tuple[int, int] | None) -> None:
    """The whole chromosome as one bar, with the viewed window marked."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    ax.add_patch(FancyBboxPatch(
        (0, 0.32), length, 0.36,
        boxstyle="round,pad=0,rounding_size=" + str(length * 0.004),
        linewidth=0.8, edgecolor=INK_MUTED, facecolor=GRID, zorder=2))
    if centromere is not None:
        start, end = centromere
        ax.add_patch(Rectangle((start, 0.32), max(end - start, length * 0.003),
                               0.36, linewidth=0, facecolor=INK_MUTED, zorder=3))
    # The window is usually far too narrow to see, so it gets a minimum width
    # and a label rather than a faithful rectangle.
    left, right = window
    width = max(right - left, length * 0.004)
    ax.add_patch(Rectangle((left, 0.22), width, 0.56, linewidth=1.2,
                           edgecolor=SERIES[1], facecolor="none", zorder=4))
    span = (right - left) / 1e3
    ax.annotate(f"{chrom}:{left:,}-{right:,}  ({span:,.0f} kb)",
                (left + width / 2, 0.86), ha="center", va="bottom",
                fontsize=7, color=INK_SECONDARY, zorder=5)
    ax.set_xlim(-length * 0.01, length * 1.01)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([])
    ax.set_xticks([0, length])
    ax.set_xticklabels(["0", _coordinate_label(length)], fontsize=7)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, pad=1)


def figure_browser(
    bins: pd.DataFrame,
    tracks: Mapping[str, np.ndarray],
    out: str | Path,
    *,
    ccre: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    genes: pd.DataFrame | None = None,
    chrom_length: int | None = None,
    centromere: tuple[int, int] | None = None,
    max_points: int = MAX_PROFILE_POINTS,
    title: str = "",
) -> list[Path]:
    """A browser view: ideogram, feature tracks, annotation lanes, one ruler.

    Everything below the ideogram shares the coordinate axis, so a peak in a
    track sits directly above the element it overlaps. That vertical alignment
    is the whole reason to draw this rather than four separate panels.

    Tracks are max-pooled above `max_points`, because a 1 Mb window at 128 bp
    is 8192 bins and a line with 8192 vertices on a 6 inch axis is a block of
    ink. Pooling keeps the peaks; it is stated on the axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FuncFormatter

    if not tracks:
        raise ValueError("Need at least one track")
    if len(tracks) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} tracks; got {len(tracks)}")
    chroms = bins.chrom.unique()
    if len(chroms) != 1:
        raise ValueError(f"A browser view covers one chromosome, got {list(chroms)}")
    for name, values in tracks.items():
        if len(values) != len(bins):
            raise ValueError(f"Track {name!r} has {len(values)} values for {len(bins)} bins")

    use_style()
    chrom = str(chroms[0])
    starts = bins.bin_start.to_numpy(dtype=np.int64)
    ends = bins.bin_end.to_numpy(dtype=np.int64)
    window = (int(starts.min()), int(ends.max()))

    classes = list(ccre) if ccre else []
    n_lanes = len(classes) + (1 if genes is not None and not genes.empty else 0)
    rows = 1 + len(tracks) + (1 if n_lanes else 0)
    heights = [0.55] + [1.0] * len(tracks) + ([0.26 * max(n_lanes, 1)] if n_lanes else [])
    figure_height = 1.1 + 0.95 * len(tracks) + 0.34 * n_lanes
    fig = plt.figure(figsize=(6.4, figure_height))
    grid = fig.add_gridspec(rows, 1, height_ratios=heights, hspace=0.18,
                            left=0.13, right=0.98, top=0.90, bottom=0.13)

    ideogram = fig.add_subplot(grid[0])
    _draw_ideogram(ideogram, chrom, int(chrom_length or window[1]), window, centromere)
    if title:
        ideogram.set_title(title, loc="left", color=INK_PRIMARY, fontsize=9, pad=10)

    pooled_note = ""
    axes = []
    for index, (name, values) in enumerate(tracks.items()):
        axis = fig.add_subplot(grid[1 + index], sharex=axes[0] if axes else None)
        axes.append(axis)
        x, y = starts.astype(float), np.asarray(values, dtype=float)
        if x.size > max_points:
            fold = int(np.ceil(x.size / max_points))
            usable = (x.size // fold) * fold
            x = x[:usable].reshape(-1, fold)[:, 0]
            y = y[:usable].reshape(-1, fold).max(axis=1)
            pooled_note = f"peak of every {fold} bins"
        axis.fill_between(x, 0, y, color=SERIES[index], linewidth=0, alpha=0.9,
                          step="post")
        axis.set_ylabel(name, rotation=0, ha="right", va="center",
                        fontsize=7.5, color=INK_SECONDARY, labelpad=6)
        axis.set_ylim(0, max(float(y.max()) * 1.08, 1e-9))
        axis.set_yticks([0, float(y.max())])
        axis.set_yticklabels(["0", f"{y.max():.3g}"], fontsize=6.5)
        axis.spines["bottom"].set_visible(False)
        axis.tick_params(axis="x", length=0, labelbottom=False)

    if n_lanes:
        lanes = fig.add_subplot(grid[-1], sharex=axes[0])
        axes.append(lanes)
        labels = []
        for lane, name in enumerate(classes):
            spans = ccre[name].get(chrom, np.empty((0, 2), dtype=np.int64))
            for span_start, span_end in np.asarray(spans, dtype=np.int64):
                if span_end <= window[0] or span_start >= window[1]:
                    continue
                lanes.add_patch(Rectangle(
                    (span_start, lane + 0.15), max(span_end - span_start, 1), 0.7,
                    linewidth=0, facecolor=SERIES[lane % len(SERIES)]))
            labels.append(name)
        if genes is not None and not genes.empty:
            lane = len(classes)
            for gene in genes.itertuples(index=False):
                lanes.add_patch(Rectangle((gene.start, lane + 0.38),
                                          max(gene.end - gene.start, 1), 0.24,
                                          linewidth=0, facecolor=INK_SECONDARY))
                # The lane axis is inverted, so larger y is lower on screen.
                # va="top" is what puts the name under the gene body instead
                # of printing it across the bar.
                lanes.annotate(gene.name, ((gene.start + gene.end) / 2, lane + 0.66),
                               ha="center", va="top", fontsize=6,
                               color=INK_SECONDARY, style="italic")
            labels.append("genes")
        lanes.set_ylim(len(labels), 0)
        lanes.set_yticks(np.arange(len(labels)) + 0.5, labels, fontsize=7)
        lanes.tick_params(axis="y", length=0)
        for side in ("left", "right", "top"):
            lanes.spines[side].set_visible(False)

    ruler = axes[-1]
    ruler.tick_params(axis="x", length=3, labelbottom=True)
    ruler.spines["bottom"].set_visible(True)
    ruler.xaxis.set_major_formatter(FuncFormatter(_coordinate_label))
    ruler.set_xlim(window)
    ruler.set_xlabel(f"{chrom}" + (f"   ({pooled_note})" if pooled_note else ""))

    table = pd.DataFrame({"chrom": chrom, "bin_start": starts, "bin_end": ends,
                          **{name: np.asarray(v, dtype=float) for name, v in tracks.items()}})
    return save(fig, out, table)


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

"""Hierarchical statistics and figures for a triad/injection/motif campaign.

The unit of evidence follows how the data were made:

* **fold** - the four AlphaGenome models, each with its own held-out test
  regions, SAEs and (here) windows/genes. Folds are the independent
  replications; the headline number is the mean over folds, reported with
  how many folds agree in sign.
* **window / gene** - the sampling unit inside a fold. Every layer and seed of
  a fold measures the *same* windows and genes, so they are not independent
  of each other.
* **seed** - dictionary noise. Seeds are averaged per window/gene first and
  never counted as separate evidence.
* **layer** - a factor of interest, compared, never pooled.

Per fold, uncertainty comes from a genomic block bootstrap over windows or
genes (10 Mb blocks, so neighbouring units are resampled together); the
across-fold interval resamples blocks within every fold and averages the
fold means. Within-fold Wilcoxon p-values are BH-adjusted per table, and an
across-fold one-sample t-test on the fold means (n = number of folds) is the
replication test. RNA-seq "no effect" claims use TOST equivalence, not a
non-significant p-value.

Usage::

    python -m ag_sae analyze \\
        --campaign sae_v2=/mnt/ag/analysis_runs/campaign_matrix_sae_v2_protocol_v2 \\
        --motif sae_v2=/mnt/ag/analysis_runs/motif_matrix_v2_protocol_v2 \\
        --out /mnt/ag/analysis_runs/analysis_v2 [--sae-root /mnt/ag]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from ag_sae.results import collect, read_tables, write_tables
from ag_sae.statistics import genomic_block_ids
from ag_sae.stats import safe_wilcoxon

BLOCK_BP = 10_000_000
LAYER_ORDER = ("resid_pre_b0", "resid_pre_b4", "resid_pre_b8")
HEAD_ORDER = ("atac", "dnase", "chip_tf", "chip_histone", "cage", "procap", "rna_seq")
CONCEPT_ORDER = ("PLS_Promoter", "CTCF_Insulator", "Enhancer_pELS", "Shuffled_Control")


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _blocks(frame: pd.DataFrame, position: str) -> np.ndarray:
    return genomic_block_ids(frame["chrom"].astype(str).to_numpy(),
                             frame[position].astype(np.int64).to_numpy(), block_bp=BLOCK_BP)


def _block_sums(values: np.ndarray, blocks: np.ndarray, weights: np.ndarray | None = None):
    weights = np.ones_like(values) if weights is None else weights
    n = int(blocks.max()) + 1
    return (np.bincount(blocks, weights=values * weights, minlength=n),
            np.bincount(blocks, weights=weights, minlength=n))


def block_bootstrap(numerator: np.ndarray, blocks: np.ndarray, denominator: np.ndarray | None = None,
                    n_boot: int = 2000, seed: int = 0, rng: np.random.Generator | None = None
                    ) -> np.ndarray:
    """Bootstrap replicates of sum(numerator) / sum(denominator) over resampled blocks.

    With no denominator this is the mean. Returns (n_boot,) replicates.
    """
    rng = rng or np.random.default_rng(seed)
    num, count = _block_sums(numerator, blocks)
    den = count if denominator is None else _block_sums(denominator, blocks)[0]
    draws = rng.integers(0, len(num), size=(n_boot, len(num)))
    bottom = den[draws].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(bottom) > 1e-12, num[draws].sum(1) / bottom, np.nan)


def _interval(replicates: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0:
        return np.nan, np.nan
    alpha = (1 - confidence) / 2
    low, high = np.quantile(finite, [alpha, 1 - alpha])
    return float(low), float(high)


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaN p-values stay NaN and are not counted."""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = sps.false_discovery_control(np.clip(p[ok], 0, 1), method="bh")
    return q


def tost_p(values: np.ndarray, margin: float) -> float:
    """Two one-sided t-tests: p < alpha means the mean lies inside (-margin, margin)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or margin <= 0:
        return np.nan
    se = values.std(ddof=1) / np.sqrt(len(values))
    if se == 0:
        return 0.0 if abs(values.mean()) < margin else 1.0
    df = len(values) - 1
    p_low = sps.t.sf((values.mean() + margin) / se, df)      # H0: mean <= -margin
    p_high = sps.t.cdf((values.mean() - margin) / se, df)    # H0: mean >= +margin
    return float(max(p_low, p_high))


def seed_average(frame: pd.DataFrame, value: str, keys: Sequence[str]) -> pd.DataFrame:
    """Mean over seeds per unit; `keys` must identify the unit without the seed."""
    out = frame.groupby(list(keys), dropna=False, sort=False)[value].mean().reset_index()
    return out[np.isfinite(out[value])]


def hierarchical(units: pd.DataFrame, value: str, groups: Sequence[str], position: str,
                 n_boot: int = 2000, seed: int = 0, ratio_of: tuple[str, str] | None = None,
                 family: Sequence[str] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold and across-fold summaries of seed-averaged units.

    `units` has one row per (groups, fold, unit) with chrom and `position`.
    With `ratio_of=(num, den)` the statistic is sum(num) / sum(den) - a ratio
    of means, which a few units with a near-zero denominator cannot dominate
    the way a mean of per-unit ratios can - and the Wilcoxon test is on num.

    Across folds two intervals are given. `ci_low/ci_high` resample blocks
    within every fold and average the fold estimates: they are conditional on
    these folds and stay narrow even when folds disagree. `t_ci_low/t_ci_high`
    is a t-interval over the fold estimates (df = folds - 1): the replication
    interval, which widens when folds disagree. BH q-values are computed
    within each `family` (columns of `groups`), not over the whole table.
    """
    rng = np.random.default_rng(seed)
    per_fold, across = [], []
    for key, group in units.groupby(list(groups), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        label = dict(zip(groups, key))
        fold_reps, fold_rows = [], []
        for fold, part in group.groupby("fold", sort=True):
            blocks = _blocks(part, position)
            if ratio_of is None:
                x = part[value].to_numpy(float)
                estimate = float(x.mean())
                reps = block_bootstrap(x, blocks, rng=rng, n_boot=n_boot)
            else:
                x, den = (part[c].to_numpy(float) for c in ratio_of)
                estimate = float(x.sum() / den.sum()) if abs(den.sum()) > 1e-12 else np.nan
                reps = block_bootstrap(x, blocks, denominator=den, rng=rng, n_boot=n_boot)
            _w, p = safe_wilcoxon(x) if len(x) >= 2 else (np.nan, np.nan)
            low, high = _interval(reps)
            fold_reps.append(reps)
            fold_rows.append({**label, "fold": fold, "estimate": estimate, "ci_low": low, "ci_high": high,
                              "p_wilcoxon": float(p) if p is not None else np.nan,
                              "fraction_positive": float(np.mean(x > 0)),
                              "n_units": len(part), "n_blocks": int(blocks.max()) + 1})
        per_fold += fold_rows
        estimates = np.array([r["estimate"] for r in fold_rows], dtype=float)
        finite = estimates[np.isfinite(estimates)]
        pooled = np.nanmean(np.vstack(fold_reps), axis=0)
        low, high = _interval(pooled)
        mean = float(np.mean(finite)) if finite.size else np.nan
        if finite.size >= 2 and np.std(finite) > 0:
            t_p = float(sps.ttest_1samp(finite, 0.0).pvalue)
            half = sps.t.ppf(0.975, finite.size - 1) * finite.std(ddof=1) / np.sqrt(finite.size)
            t_low, t_high = mean - half, mean + half
        else:
            t_p, t_low, t_high = np.nan, np.nan, np.nan
        across.append({**label, "estimate": mean, "ci_low": low, "ci_high": high,
                       "t_ci_low": t_low, "t_ci_high": t_high,
                       "n_folds": int(finite.size),
                       "folds_same_sign": int(np.sum(np.sign(finite) == np.sign(mean))),
                       "fold_min": float(np.min(finite)) if finite.size else np.nan,
                       "fold_max": float(np.max(finite)) if finite.size else np.nan,
                       "p_folds_ttest": t_p})
    per_fold, across = pd.DataFrame(per_fold), pd.DataFrame(across)
    if not per_fold.empty:
        per_fold["q_wilcoxon"] = _bh_within(per_fold, "p_wilcoxon", list(family) + ["fold"])
    if not across.empty:
        across["q_folds_ttest"] = _bh_within(across, "p_folds_ttest", list(family))
        across["replicated"] = ((across.n_folds >= 2) & (across.folds_same_sign == across.n_folds)
                                & ((across.t_ci_low > 0) | (across.t_ci_high < 0)))
    return per_fold, across


def _bh_within(frame: pd.DataFrame, column: str, family: Sequence[str]) -> np.ndarray:
    if not family:
        return bh_adjust(frame[column])
    q = np.full(len(frame), np.nan)
    for _key, index in frame.groupby(list(family), sort=False).indices.items():
        q[index] = bh_adjust(frame[column].to_numpy()[index])
    return q


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

TRIAD_UNIT = ["pool", "layer", "concept", "head", "fold", "window", "chrom", "start"]
TRIAD_KEYS = ["pool", "fold", "layer", "seed", "window", "chrom", "start", "concept", "head"]


def triad_measures(triad: pd.DataFrame) -> pd.DataFrame:
    """Seed-level (delta, baseline) rows per measure: feature (both methods), joint, null, specificity.

    Specificity is the feature's delta minus the mean delta of the window's
    matched nulls; both are error-preserving edits measured on the same bins
    against the same unedited baseline, so they share the denominator.
    """
    t = triad[np.isfinite(triad["delta"])]
    parts = []
    for (kind, method), name in {("feature", "error_preserving"): "feature_ep",
                                 ("feature", "reconstruction"): "feature_recon",
                                 ("joint", "error_preserving"): "joint_ep"}.items():
        sel = t[(t.kind == kind) & (t.method == method)]
        parts.append(sel[TRIAD_KEYS + ["delta", "baseline"]].assign(measure=name))
    null = t[t.kind == "null"].groupby(TRIAD_KEYS, as_index=False)[["delta", "baseline"]].mean()
    if not null.empty:
        parts.append(null.assign(measure="null_ep"))
        feature = t[(t.kind == "feature") & (t.method == "error_preserving")][TRIAD_KEYS + ["delta", "baseline"]]
        spec = feature.merge(null[TRIAD_KEYS + ["delta"]], on=TRIAD_KEYS, suffixes=("", "_null"))
        spec["delta"] = spec["delta"] - spec["delta_null"]
        parts.append(spec[TRIAD_KEYS + ["delta", "baseline"]].assign(measure="specificity_ep"))
    return pd.concat(parts, ignore_index=True)


def analyze_triad(triad: pd.DataFrame, n_boot: int, seed: int,
                  value: str = "relative_change") -> dict[str, pd.DataFrame]:
    """`relative_change`: sum(delta) / sum(baseline) per fold, comparable across heads.

    `delta` keeps raw units. Runs made before the baseline signal was stored
    have no baseline; they fall back to delta with a note in the output.
    """
    if value == "relative_change" and not np.isfinite(triad["baseline"]).any():
        print("[analyze] no baseline signal in these runs; reporting raw delta instead of relative change")
        value = "delta"
    measures = triad_measures(triad)
    units = (measures.groupby(["measure"] + TRIAD_UNIT, sort=False)[["delta", "baseline"]].mean()
             .reset_index())
    groups = ["measure", "pool", "layer", "concept", "head"]
    # Separate BH families: each measure, and the shuffled control apart from real concepts.
    units["family"] = units.measure + np.where(units.concept == "Shuffled_Control", ":control", "")
    if value == "relative_change":
        units = units[np.isfinite(units.delta) & np.isfinite(units.baseline)]
        per_fold, across = hierarchical(units, "delta", groups + ["family"], "start", n_boot=n_boot,
                                        seed=seed, ratio_of=("delta", "baseline"), family=["family", "pool"])
    else:
        units = units[np.isfinite(units.delta)]
        per_fold, across = hierarchical(units, "delta", groups + ["family"], "start", n_boot=n_boot,
                                        seed=seed, family=["family", "pool"])
    per_fold["value"], across["value"] = value, value
    return {"triad_per_fold": per_fold, "triad_across_folds": across,
            "triad_null_diagnostics": null_diagnostics(triad)}


def null_diagnostics(triad: pd.DataFrame) -> pd.DataFrame:
    """Do the nulls edit as much as the concept features? (firing on edited bins, edit size)."""
    rows = triad[(triad.method == "error_preserving") & triad.kind.isin(["feature", "null"])]
    if rows.empty or "fired_bins" not in rows:
        return pd.DataFrame()
    per_edit = rows.drop_duplicates(TRIAD_KEYS[:-1] + ["kind", "null_index"])   # one row per edit, not per head
    out = per_edit.groupby(["pool", "fold", "layer", "concept", "kind"]).agg(
        fired_bins_mean=("fired_bins", "mean"), edit_norm_mean=("edit_norm", "mean"),
        zero_fire_share=("fired_bins", lambda s: float(np.mean(s == 0))), n_edits=("fired_bins", "size"))
    return out.reset_index()


GENE_KEY = ["pool", "fold", "layer", "seed", "gene", "chrom", "tss0"]
GENE_UNIT = ["pool", "layer", "modality", "fold", "gene", "chrom", "tss0"]


def dose_slopes(injection: pd.DataFrame) -> pd.DataFrame:
    """Per gene and seed: slope of paired log2 on log2(dose), and Spearman rho."""
    rows = []
    for key, g in injection.groupby(GENE_KEY + ["modality"]):
        if g.multiplier.nunique() < 2:
            continue
        x, y = np.log2(g.multiplier.to_numpy(float)), g.paired_log2.to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0])
        rho = float(sps.spearmanr(x, y).statistic) if np.ptp(y) > 0 else np.nan
        rows.append(dict(zip(GENE_KEY + ["modality"], key), slope=slope, spearman=rho))
    return pd.DataFrame(rows)


def analyze_injection(injection: pd.DataFrame, n_boot: int, seed: int, tost_margin: float,
                      primary_dose: str = "x1") -> dict[str, pd.DataFrame]:
    # Doses planned per configuration, counted before dropping unmeasurable ones,
    # so a dose nobody could measure does not silently leave the "common" set.
    planned = injection.groupby(["pool", "fold", "layer"]).dose.nunique().rename("planned")
    measured = injection[(injection.status == "measured") & injection.modality.notna()].copy()
    measured = measured.join(planned, on=["pool", "fold", "layer"])
    measured["every_dose"] = measured.groupby(GENE_KEY).dose.transform("nunique") == measured.planned
    groups = ["pool", "layer", "modality", "dose", "multiplier"]
    out = {}
    units = seed_average(measured, "paired_log2", ["dose", "multiplier"] + GENE_UNIT)
    out["injection_per_fold"], out["injection_across_folds"] = hierarchical(
        units, "paired_log2", groups, "tss0", n_boot=n_boot, seed=seed, family=["pool", "dose"])
    common = measured[measured.every_dose]
    common_units = seed_average(common, "paired_log2", ["dose", "multiplier"] + GENE_UNIT)
    out["injection_common_per_fold"], out["injection_common_across_folds"] = hierarchical(
        common_units, "paired_log2", groups, "tss0", n_boot=n_boot, seed=seed, family=["pool", "dose"])
    slopes = dose_slopes(common)
    if not slopes.empty:
        slope_units = (slopes.groupby(GENE_UNIT, sort=False)[["slope", "spearman"]].mean().reset_index())
        out["dose_slope_per_fold"], out["dose_slope_across_folds"] = hierarchical(
            slope_units, "slope", ["pool", "layer", "modality"], "tss0", n_boot=n_boot, seed=seed,
            family=["pool"])
        rho = slope_units.groupby(["pool", "layer", "modality", "fold"]).spearman.agg(
            spearman_mean="mean", monotone_share=lambda s: float(np.mean(s > 0.99)))
        out["dose_monotonicity"] = rho.reset_index()
    out.update(equivalence(units[units.dose == primary_dose], tost_margin, n_boot, seed))
    return out


def equivalence(units: pd.DataFrame, margin: float, n_boot: int, seed: int) -> dict[str, pd.DataFrame]:
    """Is the effect within +/- margin (log2)? Per fold and across folds.

    Per fold: the 90% genomic-block-bootstrap interval of the mean must lie
    inside the margin (the interval form of TOST, without assuming genes are
    independent). Across folds: TOST on the fold means.
    """
    rng = np.random.default_rng(seed)
    per_fold, across = [], []
    for key, g in units.groupby(["pool", "layer", "modality"], sort=False):
        label = dict(zip(["pool", "layer", "modality"], key))
        means = []
        for fold, part in g.groupby("fold", sort=True):
            x = part.paired_log2.to_numpy(float)
            reps = block_bootstrap(x, _blocks(part, "tss0"), rng=rng, n_boot=n_boot)
            low, high = _interval(reps, confidence=0.90)
            means.append(float(x.mean()))
            per_fold.append({**label, "fold": fold, "margin_log2": margin, "mean": means[-1],
                             "ci90_low": low, "ci90_high": high,
                             "equivalent": bool(-margin < low and high < margin), "n_genes": len(x)})
        across.append({**label, "margin_log2": margin, "mean_of_folds": float(np.mean(means)),
                       "n_folds": len(means), "p_tost_folds": tost_p(np.array(means), margin),
                       "all_folds_equivalent": all(r["equivalent"] for r in per_fold
                                                   if all(r[k] == v for k, v in label.items()))})
    return {"injection_equivalence_per_fold": pd.DataFrame(per_fold),
            "injection_equivalence_across_folds": pd.DataFrame(across)}


MOTIF_UNIT = ["pool", "layer", "head", "fold", "window", "chrom", "start"]


def analyze_motif(motif: pd.DataFrame, n_boot: int, seed: int) -> dict[str, pd.DataFrame]:
    units = motif.groupby(MOTIF_UNIT, as_index=False)[["delta_ism", "delta_abl", "delta_rescue"]].mean()
    out = {}
    for name, num in (("mediation", "delta_abl"), ("rescue", "delta_rescue")):
        per_fold, across = hierarchical(units, num, ["pool", "layer", "head"], "start", n_boot=n_boot,
                                        seed=seed, ratio_of=(num, "delta_ism"), family=["pool"])
        out[f"motif_{name}_per_fold"], out[f"motif_{name}_across_folds"] = per_fold, across
    return out


def seed_agreement(discovery: pd.DataFrame, sae_root: Path) -> pd.DataFrame:
    """Did every seed of a (pool, fold, layer) select the same decoder direction per concept."""
    from ag_sae.controls import concept_seed_agreement
    from ag_sae.sae import BorzoiSAE

    rows = []
    for (pool, fold, layer), g in discovery.groupby(["pool", "fold", "layer"]):
        decoders, best = {}, {}
        for seed_name, part in g.groupby("seed"):
            path = Path(sae_root) / pool / fold / f"{layer}_{seed_name}" / "best.pt"
            if not path.is_file():
                continue
            s = int(seed_name.removeprefix("seed"))
            decoders[s] = BorzoiSAE.from_checkpoint(path, device="cpu").core.decoder.weight.detach().T.numpy()
            best[s] = dict(zip(part.concept, part.feature_id))
        if len(decoders) >= 2:
            table = concept_seed_agreement(decoders, best)
            rows.append(table.assign(pool=pool, fold=fold, layer=layer))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _ordered(values, order):
    present = list(dict.fromkeys(values))
    return [v for v in order if v in present] + [v for v in present if v not in order]


def figure_triad_heatmap(across: pd.DataFrame, measure: str, pool: str, out: Path) -> list[Path]:
    """Concept x head, one panel per layer: across-fold mean relative change (%)."""
    import matplotlib.pyplot as plt
    from ag_sae.figures import INK_PRIMARY, save, use_style

    use_style()
    if across.empty:
        return []
    data = across[(across.measure == measure) & (across.pool == pool)]
    if data.empty:
        return []
    layers = _ordered(data.layer, LAYER_ORDER)
    concepts = _ordered(data.concept, CONCEPT_ORDER)
    heads = _ordered(data["head"], HEAD_ORDER)
    scale = 100 if data["value"].iloc[0] == "relative_change" else 1
    limit = max(float(np.nanmax(np.abs(data.estimate))) * scale, 1e-6)
    fig, axes = plt.subplots(1, len(layers), figsize=(2.4 * len(layers) + 0.8, 0.45 * len(concepts) + 1.2),
                             squeeze=False)
    for ax, layer in zip(axes[0], layers):
        part = data[data.layer == layer].set_index(["concept", "head"])
        grid = np.full((len(concepts), len(heads)), np.nan)
        for i, c in enumerate(concepts):
            for j, h in enumerate(heads):
                if (c, h) in part.index:
                    row = part.loc[(c, h)]
                    grid[i, j] = row.estimate * scale
                    ax.text(j, i, f"{grid[i, j]:.2g}" + ("*" if row.replicated else ""),
                            ha="center", va="center", fontsize=6, color=INK_PRIMARY)
        image = ax.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(len(heads)), heads, rotation=45, ha="right")
        ax.set_yticks(range(len(concepts)), concepts if ax is axes[0][0] else [])
        ax.set_title(layer)
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8,
                 label="relative change (%)" if scale == 100 else "delta")
    fig.suptitle(f"{pool}: {measure} (* = replicated: >= 2 folds, all agree in sign, "
                 "fold t-interval excludes 0)", fontsize=8)
    table = data[["pool", "measure", "layer", "concept", "head", "estimate", "ci_low", "ci_high",
                  "t_ci_low", "t_ci_high", "n_folds", "folds_same_sign", "replicated", "q_folds_ttest",
                  "value"]]
    written = save(fig, out, table)
    plt.close(fig)
    return written


def figure_dose_response(across: pd.DataFrame, pool: str, out: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from ag_sae.figures import SERIES, save, use_style

    use_style()
    data = across[across.pool == pool]
    if data.empty:
        return []
    layers = _ordered(data.layer, LAYER_ORDER)
    fig, axes = plt.subplots(1, len(layers), figsize=(2.6 * len(layers), 2.4), squeeze=False, sharey=True)
    for ax, layer in zip(axes[0], layers):
        colors = dict(zip(sorted(data.modality.unique()), SERIES))     # same colour in every panel
        for modality, g in data[data.layer == layer].groupby("modality"):
            color = colors[modality]
            g = g.sort_values("multiplier")
            x = np.log2(g.multiplier)
            ax.plot(x, g.estimate, marker="o", color=color, label=modality)
            ax.fill_between(x, g.ci_low, g.ci_high, color=color, alpha=0.18, linewidth=0)
        ax.axhline(0, color="#8a8983", linewidth=0.6)
        ax.set_xlabel("dose (log2 multiple of natural level)")
        ax.set_title(layer)
    axes[0][0].set_ylabel("target - control (log2)")
    axes[0][-1].legend()
    written = save(fig, out, data)
    plt.close(fig)
    return written


def figure_mediation_forest(per_fold: pd.DataFrame, across: pd.DataFrame, pool: str, head: str,
                            out: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from ag_sae.figures import CONTEXT, SERIES, save, use_style

    use_style()
    folds = per_fold[(per_fold.pool == pool) & (per_fold["head"] == head)]
    pooled = across[(across.pool == pool) & (across["head"] == head)]
    if folds.empty:
        return []
    rows = []
    for layer in _ordered(folds.layer, LAYER_ORDER):
        for _, r in folds[folds.layer == layer].sort_values("fold").iterrows():
            rows.append((f"{layer} {r.fold}", r.estimate, r.ci_low, r.ci_high, CONTEXT))
        for _, r in pooled[pooled.layer == layer].iterrows():
            rows.append((f"{layer} all folds", r.estimate, r.ci_low, r.ci_high, SERIES[0]))
    fig, ax = plt.subplots(figsize=(3.6, 0.22 * len(rows) + 0.8))
    for i, (_label, est, lo, hi, color) in enumerate(rows):
        ax.plot([lo, hi], [i, i], color=color, linewidth=1.2)
        ax.plot(est, i, "o", color=color, markersize=4)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color="#8a8983", linewidth=0.6)
    ax.set_xlabel(f"mediation (ratio of means), {head}")
    written = save(fig, out, pd.concat([folds, pooled], ignore_index=True))
    plt.close(fig)
    return written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _pairs(items: Sequence[str] | None) -> dict[str, Path]:
    out = {}
    for item in items or []:
        pool, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"expected POOL=DIR, got {item!r}")
        out[pool] = Path(path)
    return out


def run_analysis(tables: dict[str, pd.DataFrame], out_dir: Path, *, n_boot: int = 2000, seed: int = 0,
                 tost_margin: float = 0.1, figures: bool = True, sae_root: Path | None = None,
                 triad_value: str = "relative_change") -> dict:
    out_dir = Path(out_dir)
    stats_dir, fig_dir = out_dir / "stats", out_dir / "figures"
    stats_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    if not tables.get("triad", pd.DataFrame()).empty:
        results.update(analyze_triad(tables["triad"], n_boot, seed, triad_value))
    if not tables.get("injection", pd.DataFrame()).empty:
        results.update(analyze_injection(tables["injection"], n_boot, seed, tost_margin))
    if not tables.get("motif", pd.DataFrame()).empty:
        results.update(analyze_motif(tables["motif"], n_boot, seed))
    if sae_root is not None and not tables.get("discovery", pd.DataFrame()).empty:
        results["seed_agreement"] = seed_agreement(tables["discovery"], sae_root)
    for name, frame in results.items():
        frame.to_csv(stats_dir / f"{name}.csv", index=False)
    written = []
    if figures:
        if not results.get("triad_across_folds", pd.DataFrame()).empty:
            across = results["triad_across_folds"]
            for pool in across.pool.unique():
                for measure in ("feature_ep", "specificity_ep", "joint_ep"):
                    written += figure_triad_heatmap(across, measure, pool, fig_dir / f"triad_{measure}_{pool}")
        if not results.get("injection_common_across_folds", pd.DataFrame()).empty:
            across = results["injection_common_across_folds"]
            for pool in across.pool.unique():
                written += figure_dose_response(across, pool, fig_dir / f"dose_response_{pool}")
        if not results.get("motif_mediation_per_fold", pd.DataFrame()).empty:
            pf, ac = results["motif_mediation_per_fold"], results["motif_mediation_across_folds"]
            for pool in pf.pool.unique():
                for head in ("chip_tf", "atac"):
                    written += figure_mediation_forest(pf, ac, pool, head, fig_dir / f"mediation_{head}_{pool}")
    summary = {"tables": {k: len(v) for k, v in tables.items()},
               "stats": {k: len(v) for k, v in results.items()},
               "figures": [str(p) for p in written if p.suffix == ".pdf"],
               "settings": {"n_boot": n_boot, "seed": seed, "tost_margin_log2": tost_margin,
                            "block_bp": BLOCK_BP}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", nargs="*", help="POOL=DIR of triad/injection configuration folders")
    parser.add_argument("--motif", nargs="*", help="POOL=DIR of motif configuration folders")
    parser.add_argument("--tables", type=Path, default=None, help="reuse tables written by an earlier run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sae-root", type=Path, default=None,
                        help="folder holding <pool>/<fold>/<layer>_<seed>/best.pt, for seed agreement")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tost-margin", type=float, default=0.1, help="equivalence margin in log2 units")
    parser.add_argument("--triad-value", choices=("relative_change", "delta"), default="relative_change",
                        help="delta for runs made before the baseline signal was stored")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    if args.tables is not None:
        tables = read_tables(args.tables)
    else:
        tables = collect(_pairs(args.campaign), _pairs(args.motif))
        write_tables(tables, args.out / "tables")
    summary = run_analysis(tables, args.out, n_boot=args.n_boot, seed=args.seed,
                           tost_margin=args.tost_margin, figures=not args.no_figures,
                           sae_root=args.sae_root, triad_value=args.triad_value)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

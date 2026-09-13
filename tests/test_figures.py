"""Figure tests: contracts and the pooling arithmetic, not pixel appearance."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")   # no display on the cluster; must precede pyplot import

import numpy as np
import pandas as pd
import pytest

from ag_sae import figures as FIG


def concept_frame(seed: int = 0, concepts: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = ["PLS", "pELS", "dELS", "CTCF-only", "splice_donor", "GC_content"][:concepts]
    auc = rng.uniform(0.55, 0.95, concepts)
    return pd.DataFrame({
        "concept": names, "n_positive_bins": rng.integers(100, 9000, concepts),
        "prevalence": rng.uniform(.01, .2, concepts), "best_feature": rng.integers(0, 8192, concepts),
        "best_auroc": auc, "raw_best_auroc": auc - rng.uniform(.01, .1, concepts),
        "null_p95": np.full(concepts, 0.56), "recovered": auc > 0.56,
    })


def suffixes(paths) -> set[str]:
    return {p.suffix for p in paths}


class TestRecovery:
    def test_writes_vector_raster_and_the_source_table(self, tmp_path):
        written = FIG.figure_recovery(concept_frame(), tmp_path / "fig1", title="Fold 0")
        assert suffixes(written) == {".pdf", ".svg", ".png", ".csv"}
        assert all(p.exists() and p.stat().st_size > 0 for p in written)
        # The table view is the relief for palette slots below 3:1 contrast,
        # so it must actually carry the plotted numbers.
        table = pd.read_csv(tmp_path / "fig1.csv")
        assert {"concept", "best_auroc", "null_p95", "recovered"} <= set(table.columns)
        assert len(table) == 4

    def test_survives_a_single_concept_and_a_missing_baseline(self, tmp_path):
        frame = concept_frame(concepts=1).drop(columns=["raw_best_auroc"])
        assert FIG.figure_recovery(frame, tmp_path / "one")


class TestDepth:
    def test_orders_taps_canonically_regardless_of_dict_order(self, tmp_path):
        by_tap = {t: concept_frame(i) for i, t in
                  enumerate(("resid_pre_b8", "bin_size_4", "resid_pre_b0"))}
        written = FIG.figure_depth(by_tap, tmp_path / "fig2")
        table = pd.read_csv(tmp_path / "fig2.csv")
        assert set(table.tap) == set(by_tap)
        assert len(table) == 3 * 4
        assert suffixes(written) == {".pdf", ".svg", ".png", ".csv"}

    def test_rejects_unknown_taps_rather_than_plotting_them_anywhere(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown taps"):
            FIG.figure_depth({"layer7": concept_frame()}, tmp_path / "x")

    def test_one_tap_is_not_a_depth_figure(self, tmp_path):
        with pytest.raises(ValueError, match="at least two taps"):
            FIG.figure_depth({"bin_size_4": concept_frame()}, tmp_path / "x")


class TestLocus:
    def frame(self, n: int) -> pd.DataFrame:
        start = 1_000_000 + np.arange(n) * 128
        return pd.DataFrame({"chrom": "chr19", "bin_start": start, "bin_end": start + 128})

    def test_renders_lanes_and_records_the_profile(self, tmp_path):
        bins = self.frame(300)
        profile = np.abs(np.random.default_rng(0).normal(size=300))
        ccre = {"PLS": {"chr19": np.array([[1_000_100, 1_000_400]])},
                "dELS": {"chr19": np.array([[1_010_000, 1_010_300]])}}
        written = FIG.figure_locus(bins, profile, ccre, tmp_path / "fig3", feature=7)
        assert suffixes(written) == {".pdf", ".svg", ".png", ".csv"}
        table = pd.read_csv(tmp_path / "fig3.csv")
        assert len(table) == 300                      # not pooled at this size
        np.testing.assert_allclose(table.activation, profile)

    def test_pooling_keeps_peak_heights_and_covers_the_window(self, tmp_path):
        n = 4000
        bins = self.frame(n)
        profile = np.zeros(n)
        profile[1234] = 9.5                           # one tall isolated peak
        ccre = {"PLS": {"chr19": np.empty((0, 2), dtype=np.int64)}}
        FIG.figure_locus(bins, profile, ccre, tmp_path / "pooled", max_points=500)
        table = pd.read_csv(tmp_path / "pooled.csv")
        assert len(table) == 500
        # Max-pooling, not averaging: the peak must survive at full height.
        assert table.activation.max() == pytest.approx(9.5)
        assert table.bin_start.min() == bins.bin_start.min()
        assert table.bin_end.max() == bins.bin_end.max()

    def test_elements_outside_the_window_are_dropped_not_clipped(self, tmp_path):
        bins = self.frame(100)
        ccre = {"PLS": {"chr19": np.array([[500_000, 500_200], [1_000_100, 1_000_400]])}}
        assert FIG.figure_locus(bins, np.ones(100), ccre, tmp_path / "clip")

    def test_rejects_mismatched_profile_and_multiple_chromosomes(self, tmp_path):
        bins = self.frame(50)
        with pytest.raises(ValueError, match="one value per bin"):
            FIG.figure_locus(bins, np.ones(49), {}, tmp_path / "x")
        mixed = pd.concat([bins, bins.assign(chrom="chr20")], ignore_index=True)
        with pytest.raises(ValueError, match="single chromosome"):
            FIG.figure_locus(mixed, np.ones(100), {}, tmp_path / "x")

    def test_rejects_a_class_with_no_intervals(self, tmp_path):
        with pytest.raises(ValueError, match="No intervals"):
            FIG.figure_locus(self.frame(50), np.ones(50), {"PLS": {}},
                             tmp_path / "x", classes=["dELS"])


class TestSpecificity:
    def test_ecdf_is_monotone_and_starts_at_one(self, tmp_path):
        frame = pd.DataFrame({"feature": range(200), "n_top_bins": 20,
                              "dominant": "PLS",
                              "dominant_share": np.random.default_rng(1).uniform(0, 1, 200)})
        assert FIG.figure_specificity(frame, tmp_path / "fig4", chance=0.2)
        assert (tmp_path / "fig4.csv").exists()

    def test_no_features_is_an_error_not_an_empty_plot(self, tmp_path):
        empty = pd.DataFrame({"feature": [], "dominant_share": []})
        with pytest.raises(ValueError, match="No features"):
            FIG.figure_specificity(empty, tmp_path / "x")


def test_palette_slots_are_the_validated_ones():
    """The slot ORDER is the CVD-safety mechanism; re-ordering silently breaks it."""
    assert FIG.SERIES == ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
    assert len(FIG.TAP_ORDER) == len(FIG.TAP_LABEL) == 6
    assert set(FIG.TAP_ORDER) == set(FIG.TAP_LABEL)

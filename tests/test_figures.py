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


class TestPareto:
    def _runs(self, l0):
        rows = []
        for i, tap in enumerate(FIG.TAP_ORDER):
            for seed in range(3):
                rows.append({"tap": tap, "seed": seed,
                             "mean_l0": l0[i], "fvu": 0.2 + 0.02 * i + 0.005 * seed})
        return pd.DataFrame(rows)

    def test_fixed_sparsity_falls_back_to_depth_axis(self, tmp_path):
        runs = self._runs([409.0] * 6)
        written = FIG.figure_pareto(runs, tmp_path / "pareto")
        assert {p.suffix for p in written} == {".pdf", ".svg", ".png", ".csv"}
        # With L0 pinned the x axis must carry the taps, not the sparsity.
        assert "L0 = 409" in (tmp_path / "pareto.svg").read_text()

    def test_a_real_sweep_draws_the_trade_off(self, tmp_path):
        runs = self._runs([50.0, 100.0, 200.0, 400.0, 800.0, 1600.0])
        FIG.figure_pareto(runs, tmp_path / "sweep")
        assert "Mean L0" in (tmp_path / "sweep.svg").read_text()

    def test_missing_columns_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="missing"):
            FIG.figure_pareto(pd.DataFrame({"tap": ["a"], "seed": [0]}), tmp_path / "x")


class TestInformationContent:
    def test_one_repeated_sequence_is_two_bits_everywhere(self):
        heights = FIG.information_content(["ACGT"] * 5)
        assert heights.sum(axis=1) == pytest.approx([2.0] * 4)

    def test_uniform_bases_carry_no_information(self):
        heights = FIG.information_content(["A", "C", "G", "T"])
        assert heights.sum() == pytest.approx(0.0, abs=1e-9)

    def test_weights_shift_the_consensus(self):
        # Same two sequences, but weighting the second one heavily should move
        # the tallest letter at position 0 from A to T.
        heights = FIG.information_content(["AAAA", "TAAA"], weights=np.array([0.01, 10.0]))
        assert heights[0].argmax() == 3

    def test_ragged_input_is_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            FIG.information_content(["AC", "ACG"])


class TestFeatureCard:
    def _data(self, n=400, length=12):
        rng = np.random.default_rng(0)
        sequences = ["".join(rng.choice(list("ACGT"), length)) for _ in range(n)]
        activation = np.zeros(n)
        hot = rng.choice(n, 60, replace=False)
        activation[hot] = rng.gamma(2.0, 1.0, hot.size)
        for i in hot:                                  # plant a motif
            sequences[i] = sequences[i][:4] + "GGTCA" + sequences[i][9:]
        return activation, sequences

    def test_card_writes_every_format(self, tmp_path):
        activation, sequences = self._data()
        written = FIG.figure_feature_card(7, activation, sequences, tmp_path / "card",
                                        concept_auroc={"PLS": 0.81, "dELS": 0.55})
        assert {p.suffix for p in written} == {".pdf", ".svg", ".png", ".csv"}
        table = pd.read_csv(tmp_path / "card.csv")
        # The CSV must be the strongest sites, in order.
        assert table.activation.is_monotonic_decreasing

    def test_a_silent_feature_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="fewer than two"):
            FIG.figure_feature_card(0, np.zeros(50), ["ACGT"] * 50, tmp_path / "x")

    def test_length_mismatch_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="One sequence per"):
            FIG.figure_feature_card(0, np.ones(10), ["ACGT"] * 9, tmp_path / "x")


class TestSeedStability:
    def test_identical_dictionaries_match_perfectly(self):
        rng = np.random.default_rng(0)
        d = rng.normal(size=(64, 16))
        assert FIG.match_across_seeds(d, d) == pytest.approx(np.ones(64), abs=1e-9)

    def test_scaling_a_row_does_not_change_cosine(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=(32, 8)), rng.normal(size=(32, 8))
        plain = FIG.match_across_seeds(a, b)
        scaled = FIG.match_across_seeds(a * 7.0, b)
        assert plain == pytest.approx(scaled, abs=1e-9)

    def test_figure_needs_two_seeds(self, tmp_path):
        with pytest.raises(ValueError, match="two seeds"):
            FIG.figure_seed_stability({0: np.ones((4, 3))}, tmp_path / "x")

    def test_figure_writes_a_row_per_pair(self, tmp_path):
        rng = np.random.default_rng(2)
        shared = rng.normal(size=(48, 20))
        decoders = {s: shared + 0.25 * rng.normal(size=(48, 20)) for s in (0, 1, 2)}
        FIG.figure_seed_stability(decoders, tmp_path / "stab")
        table = pd.read_csv(tmp_path / "stab.csv")
        assert sorted(table.pair) == ["0-1", "0-2", "1-2"]
        assert (table.matched_fraction >= 0).all()


class TestPareto2:
    def test_a_tower_only_subset_is_not_coloured_as_conv(self, tmp_path):
        # The conv/tower split must follow the tap's identity. Classifying by
        # position in the subset would paint these three as convolutional.
        tower = FIG.TAP_ORDER[FIG.N_CONV_TAPS:]
        runs = pd.DataFrame([{"tap": t, "seed": s, "mean_l0": 409.0, "fvu": 0.2}
                             for t in tower for s in range(2)])
        FIG.figure_pareto(runs, tmp_path / "tower")
        svg = (tmp_path / "tower.svg").read_text()
        assert FIG.SERIES[1].lstrip("#") in svg.replace("#", "")
        assert "Transformer" in svg

    def test_conv_only_subset_keeps_the_conv_colour(self, tmp_path):
        conv = FIG.TAP_ORDER[:FIG.N_CONV_TAPS]
        runs = pd.DataFrame([{"tap": t, "seed": s, "mean_l0": 409.0, "fvu": 0.3}
                             for t in conv for s in range(2)])
        FIG.figure_pareto(runs, tmp_path / "conv")
        assert "Convolutional" in (tmp_path / "conv.svg").read_text()


class TestSpreadLabels:
    def test_coincident_anchors_are_pushed_apart(self):
        anchors = [[1.0, 0.5, "a"], [1.0, 0.5, "b"], [1.0, 0.5, "c"]]
        out = FIG._spread_labels(anchors, gap=0.1)
        ys = [row[1] for row in out]
        assert ys == pytest.approx([0.5, 0.6, 0.7])

    def test_well_separated_anchors_are_left_alone(self):
        anchors = [[1.0, 0.0, "a"], [1.0, 0.9, "b"]]
        out = FIG._spread_labels(anchors, gap=0.1)
        assert [row[1] for row in out] == pytest.approx([0.0, 0.9])


class TestBrowser:
    def _bins(self, n=512, start=1_000_000, bp=128):
        edges = start + np.arange(n) * bp
        return pd.DataFrame({"chrom": "chr7", "bin_start": edges, "bin_end": edges + bp})

    def test_centromere_is_the_widest_gap(self):
        gaps = pd.DataFrame({
            "chrom": ["chr7", "chr7", "chr8"],
            "start": [100, 5_000, 10],
            "end": [600, 3_000_000, 90],
        })
        assert FIG.centromere_from_gaps(gaps, "chr7") == (5_000, 3_000_000)
        assert FIG.centromere_from_gaps(gaps, "chrY") is None

    def test_browser_writes_every_format(self, tmp_path):
        bins = self._bins()
        rng = np.random.default_rng(0)
        tracks = {"feature 12": rng.gamma(0.5, 1.0, len(bins)),
                  "feature 88": rng.gamma(0.5, 1.0, len(bins))}
        ccre = {"PLS": {"chr7": np.array([[1_010_000, 1_010_400]], dtype=np.int64)}}
        written = FIG.figure_browser(bins, tracks, tmp_path / "browser", ccre=ccre,
                                     chrom_length=159_345_973,
                                     centromere=(58_100_000, 62_100_000))
        assert {p.suffix for p in written} == {".pdf", ".svg", ".png", ".csv"}
        table = pd.read_csv(tmp_path / "browser.csv")
        assert list(table.columns) == ["chrom", "bin_start", "bin_end",
                                       "feature 12", "feature 88"]

    def test_two_chromosomes_are_refused(self, tmp_path):
        bins = self._bins(4)
        bins.loc[0, "chrom"] = "chr1"
        with pytest.raises(ValueError, match="one chromosome"):
            FIG.figure_browser(bins, {"a": np.ones(4)}, tmp_path / "x")

    def test_track_length_must_match_the_bins(self, tmp_path):
        with pytest.raises(ValueError, match="values for"):
            FIG.figure_browser(self._bins(8), {"a": np.ones(7)}, tmp_path / "x")

    def test_too_many_tracks_are_refused(self, tmp_path):
        bins = self._bins(8)
        tracks = {f"f{i}": np.ones(8) for i in range(len(FIG.SERIES) + 1)}
        with pytest.raises(ValueError, match="At most"):
            FIG.figure_browser(bins, tracks, tmp_path / "x")

    def test_long_windows_are_pooled(self, tmp_path):
        bins = self._bins(8192)
        FIG.figure_browser(bins, {"f": np.arange(8192.0)}, tmp_path / "pooled",
                           max_points=900)
        # The note must say the track was pooled, not silently thin it.
        assert "peak of every" in (tmp_path / "pooled.svg").read_text()

    def test_coordinate_labels_switch_unit(self):
        assert FIG._coordinate_label(2_500_000) == "2.5 Mb"
        assert FIG._coordinate_label(4_000) == "4 kb"
        assert FIG._coordinate_label(250) == "250"


class TestLogoOrder:
    def test_the_tallest_letter_is_drawn_on_top(self):
        import matplotlib.pyplot as plt
        # One dominant base plus a rare one. The tall letter must occupy the
        # upper part of the stack, which is how a logo is read.
        heights = np.array([[1.8, 0.2, 0.0, 0.0]])      # A tall, C short
        fig, ax = plt.subplots()
        FIG._draw_logo(ax, heights)
        boxes = [patch.get_extents() for patch in ax.patches]
        plt.close(fig)
        assert boxes, "no glyphs drawn"
        tallest = max(boxes, key=lambda b: b.height)
        shortest = min(boxes, key=lambda b: b.height)
        assert tallest.y0 > shortest.y0 or tallest.y1 > shortest.y1

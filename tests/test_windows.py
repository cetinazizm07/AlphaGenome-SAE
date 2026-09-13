"""Interval algebra and manifest invariants. The algebra is where leakage hides."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ag_sae import windows as W


def iv(*pairs) -> np.ndarray:
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def frame(chrom: str, *pairs) -> pd.DataFrame:
    arr = iv(*pairs)
    return pd.DataFrame({"chrom": chrom, "start": arr[:, 0], "end": arr[:, 1]})


class TestMerge:
    def test_joins_overlapping_and_abutting_and_sorts(self):
        got = W.merge(frame("chr1", (30, 40), (0, 10), (10, 20), (15, 25)))
        # 0-10 abuts 10-20 and overlaps 15-25 -> one run; 30-40 stays apart.
        np.testing.assert_array_equal(got["chr1"], iv((0, 25), (30, 40)))

    def test_nested_intervals_collapse(self):
        np.testing.assert_array_equal(W.merge(frame("chr1", (0, 100), (20, 30)))["chr1"],
                                      iv((0, 100)))

    def test_rejects_empty_or_inverted(self):
        with pytest.raises(ValueError, match="end <= start"):
            W.merge(frame("chr1", (50, 50)))


class TestSubtract:
    @pytest.mark.parametrize("cut,expected", [
        ((200, 300), [(0, 100)]),                 # disjoint, nothing removed
        ((0, 100), []),                           # exact cover
        ((0, 200), []),                           # cut larger than the interval
        ((0, 40), [(40, 100)]),                   # trim the left
        ((60, 100), [(0, 60)]),                   # trim the right
        ((40, 60), [(0, 40), (60, 100)]),         # punch a hole
    ])
    def test_single_cut_cases(self, cut, expected):
        got = W.subtract({"chr1": iv((0, 100))}, {"chr1": iv(cut)})
        assert [tuple(x) for x in got.get("chr1", iv())] == expected

    def test_many_cuts_and_a_cut_spanning_several_intervals(self):
        left = {"chr1": iv((0, 100), (200, 300), (400, 500))}
        right = {"chr1": iv((10, 20), (50, 250), (460, 600))}
        got = W.subtract(left, right)
        assert [tuple(x) for x in got["chr1"]] == [(0, 10), (20, 50), (250, 300), (400, 460)]

    def test_other_chromosomes_are_untouched(self):
        got = W.subtract({"chr1": iv((0, 100)), "chr2": iv((0, 100))},
                         {"chr1": iv((0, 100))})
        assert "chr1" not in got
        np.testing.assert_array_equal(got["chr2"], iv((0, 100)))

    def test_subtracting_nothing_is_identity(self):
        got = W.subtract({"chr1": iv((0, 100))}, {})
        np.testing.assert_array_equal(got["chr1"], iv((0, 100)))


class TestPadAndTile:
    def test_pad_widens_clips_at_zero_and_remerges(self):
        got = W.pad({"chr1": iv((100, 200), (260, 300))}, 30)
        np.testing.assert_array_equal(got["chr1"], iv((70, 330)))   # the two now touch
        np.testing.assert_array_equal(W.pad({"chr1": iv((10, 20))}, 50)["chr1"], iv((0, 70)))

    def test_tile_emits_whole_windows_inside_the_interval_only(self):
        # chr1 has room for two and a 50 bp tail that is dropped, not shortened;
        # chr2 is one base short of a window and yields nothing at all.
        got = W.tile({"chr1": iv((0, 250)), "chr2": iv((1000, 1099)),
                      "chr3": iv((1000, 1100))}, 100)
        assert [tuple(x) for x in got[["win_start", "win_end"]].to_numpy()] == [
            (0, 100), (100, 200), (1000, 1100)]
        assert got.chrom.tolist() == ["chr1", "chr1", "chr3"]

    def test_tile_never_straddles_an_interval_edge(self):
        # Two intervals 100 apart; a naive tiler would emit 90-190 across the gap.
        got = W.tile({"chr1": iv((0, 90), (100, 190))}, 100)
        assert got.empty

    def test_tile_of_nothing(self):
        assert W.tile({}, 100).empty


# --- fold territory --------------------------------------------------------


def synthetic_bed(region_bp: int = 200_000, per_fold: int = 40) -> pd.DataFrame:
    """Eight folds interleaved along one chromosome, like the real assignment."""
    rows = []
    for i in range(8 * per_fold):
        start = i * region_bp
        rows.append({"chrom": "chr1", "start": start, "end": start + region_bp,
                     "fold": f"fold{i % 8}"})
    return pd.DataFrame(rows)


class TestTerritory:
    def test_heldout_and_trained_never_overlap(self):
        place = W.territory(synthetic_bed(), 0, context_bp=400_000, target_bp=200_000)
        overlap = W.subtract(place.heldout, W.subtract(place.heldout, place.trained))
        assert W.span(overlap) == 0

    def test_context_guard_costs_territory(self):
        bed = synthetic_bed()
        loose = W.territory(bed, 0, context_bp=200_000, target_bp=200_000)   # no margin
        strict = W.territory(bed, 0, context_bp=1_000_000, target_bp=200_000)
        assert W.span(strict.heldout) < W.span(loose.heldout)

    def test_uses_the_published_fold_mapping(self):
        place = W.territory(synthetic_bed(), 1)
        assert (place.valid_fold, place.test_fold) == ("fold3", "fold4")
        assert W.VALID_FOLD[0] == "fold0" and W.TEST_FOLD[0] == "fold1"

    def test_model_folds_hold_out_disjoint_territory(self):
        """The design property every fold-to-fold comparison rests on."""
        bed = synthetic_bed()
        places = {k: W.territory(bed, k, context_bp=200_000, target_bp=200_000)
                  for k in (0, 1, 2)}
        for a, b in ((0, 1), (0, 2), (1, 2)):
            shared = W.subtract(places[a].heldout,
                                W.subtract(places[a].heldout, places[b].heldout))
            assert W.span(shared) == 0, f"folds {a} and {b} share territory"

    def test_rejects_unknown_fold_and_impossible_context(self):
        bed = synthetic_bed()
        with pytest.raises(ValueError, match="model_fold must be"):
            W.territory(bed, 9)
        with pytest.raises(ValueError, match="context cannot be shorter"):
            W.territory(bed, 0, context_bp=100, target_bp=200)

    def test_exclude_removes_regions_from_both_sides(self):
        bed = synthetic_bed()
        block = {"chr1": iv((0, 10_000_000))}
        place = W.territory(bed, 0, context_bp=200_000, target_bp=200_000, exclude=block)
        for chrom, arr in {**place.heldout, **place.trained}.items():
            assert (arr[:, 0] >= 10_000_000).all()


class TestManifest:
    def build(self, **kwargs):
        options = dict(window_bp=100_000, test_windows=5, dev_windows=3,
                       train_windows=10, val_windows=2, context_bp=200_000,
                       target_bp=200_000)
        options.update(kwargs)
        return W.build_manifest(synthetic_bed(), 0, **options)

    def test_splits_are_disjoint_non_overlapping_and_right_sized(self):
        frame, record = self.build()
        assert frame.split.value_counts().to_dict() == {
            "train": 10, "test": 5, "test_trained": 5, "dev": 3, "val": 2}
        assert (frame.win_end - frame.win_start == 100_000).all()
        assert not frame.duplicated(["chrom", "win_start"]).any()
        edges = frame.sort_values("win_start")
        assert (edges.win_start.to_numpy()[1:] >= edges.win_end.to_numpy()[:-1]).all()
        # Evaluation splits must sit on unseen territory, controls on seen.
        label = frame.groupby("split").ag_label.unique()
        assert set(label["test"]) == {"heldout"} and set(label["dev"]) == {"heldout"}
        assert set(label["train"]) == {"trained"} and set(label["test_trained"]) == {"trained"}

    def test_test_trained_defaults_to_the_size_of_test(self):
        frame, _ = self.build(test_windows=7, test_trained_windows=None)
        counts = frame.split.value_counts()
        assert counts["test_trained"] == counts["test"] == 7

    def test_same_seed_reproduces_the_manifest(self):
        a, _ = self.build(seed=3)
        b, _ = self.build(seed=3)
        pd.testing.assert_frame_equal(a, b)
        c, _ = self.build(seed=4)
        assert not a.equals(c)

    def test_asking_for_more_windows_than_exist_is_an_error(self):
        with pytest.raises(ValueError, match="only"):
            self.build(test_windows=10**6)

    def test_record_carries_the_provenance_needed_to_rebuild_it(self):
        _, record = self.build()
        for key in ("model_fold", "valid_fold", "test_fold", "heldout_mb", "window_bp",
                    "seed", "context_margin_bp", "available_windows", "selected_windows"):
            assert key in record
        assert record["valid_fold"] == "fold0" and record["test_fold"] == "fold1"

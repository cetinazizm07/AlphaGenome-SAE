"""Controls: stratification arithmetic, the memorisation gap, seed agreement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ag_sae import controls as K


class TestStrata:
    def test_quantiles_split_evenly(self):
        strata = K.quantile_strata(np.arange(100.0), 4)
        assert sorted(np.bincount(strata)) == [25, 25, 25, 25]

    def test_missing_values_are_dropped_not_placed(self):
        values = np.array([1.0, 2.0, np.nan, 4.0])
        strata = K.quantile_strata(values, 2)
        assert strata[2] == -1
        assert (strata[[0, 1, 3]] >= 0).all()

    def test_one_stratum_is_refused(self):
        with pytest.raises(ValueError, match="at least two"):
            K.quantile_strata(np.arange(10.0), 1)


class TestStratifiedAuroc:
    def test_perfect_confounding_leaves_nothing_to_measure(self):
        # If the concept is exactly the covariate, no stratum contains both
        # classes and there is no within-stratum evidence at all. Saying so is
        # the right answer, not an error to paper over.
        gc = np.repeat(np.arange(4.0), 100)
        labels = gc >= 2
        with pytest.raises(ValueError, match="enough of both"):
            K.stratified_auroc(gc, labels, K.quantile_strata(gc, 4))

    def test_a_confound_only_feature_collapses_to_chance(self):
        # The concept mostly follows the covariate but not perfectly, so each
        # stratum holds both classes. The feature is a copy of the covariate,
        # so inside a stratum it carries nothing. Unadjusted it looks strong;
        # conditioned on the covariate it must fall to chance.
        rng = np.random.default_rng(0)
        gc = np.repeat(np.arange(4.0), 400)
        flip = rng.random(gc.size) < 0.25
        labels = (gc >= 2) ^ flip
        values = gc + rng.normal(0, 1e-6, gc.size)
        out = K.stratified_auroc(values, labels, K.quantile_strata(gc, 4))
        assert out["unadjusted_auroc"] > 0.70
        assert out["pooled_auroc"] == pytest.approx(0.5, abs=0.03)

    def test_a_real_signal_survives_stratification(self):
        rng = np.random.default_rng(1)
        gc = np.repeat(np.arange(4.0), 200)
        labels = rng.random(gc.size) < 0.3
        values = gc + labels * 3.0 + rng.normal(0, 0.1, gc.size)
        out = K.stratified_auroc(values, labels, K.quantile_strata(gc, 4))
        assert out["pooled_auroc"] > 0.95
        assert out["strata_used"] == 4

    def test_thin_strata_are_skipped_not_counted(self):
        values = np.arange(60.0)
        labels = np.zeros(60, dtype=bool)
        labels[:30] = True
        strata = np.zeros(60, dtype=np.int64)
        strata[55:] = 1                      # a stratum with almost nothing in it
        out = K.stratified_auroc(values, labels, strata, min_per_stratum=10)
        assert out["strata_skipped"] == [1]
        assert out["strata_used"] == 1

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="same bins"):
            K.stratified_auroc(np.ones(5), np.ones(5, dtype=bool), np.ones(4, dtype=np.int64))

    def test_no_usable_stratum_raises(self):
        values = np.arange(10.0)
        labels = np.zeros(10, dtype=bool)
        labels[:2] = True
        with pytest.raises(ValueError, match="enough of both"):
            K.stratified_auroc(values, labels, np.zeros(10, dtype=np.int64))


class TestMemorisationGap:
    def _frame(self, values):
        return pd.DataFrame({"concept": ["PLS", "dELS"], "sae_excess": values})

    def test_gap_is_unseen_minus_trained(self):
        out = K.memorisation_gap(self._frame([0.10, 0.05]), self._frame([0.12, 0.02]))
        gaps = dict(zip(out.concept, out.gap))
        assert gaps["PLS"] == pytest.approx(-0.02)
        assert gaps["dELS"] == pytest.approx(0.03)
        # Sorted worst first, so the memorisation suspects are at the top.
        assert out.concept.iloc[0] == "PLS"

    def test_disjoint_concepts_are_refused(self):
        left = pd.DataFrame({"concept": ["PLS"], "sae_excess": [0.1]})
        right = pd.DataFrame({"concept": ["CTCF"], "sae_excess": [0.1]})
        with pytest.raises(ValueError, match="share no concepts"):
            K.memorisation_gap(left, right)

    def test_a_missing_column_is_named(self):
        with pytest.raises(ValueError, match="sae_excess"):
            K.memorisation_gap(pd.DataFrame({"concept": ["PLS"]}), self._frame([0.1, 0.1]))


class TestSeedAgreement:
    def test_seeds_that_found_the_same_direction_agree(self):
        rng = np.random.default_rng(0)
        shared = rng.normal(size=64)
        a = rng.normal(size=(20, 64))
        b = rng.normal(size=(20, 64))
        a[3] = shared
        b[11] = shared + 0.01 * rng.normal(size=64)
        out = K.concept_seed_agreement({0: a, 1: b}, {0: {"PLS": 3}, 1: {"PLS": 11}})
        row = out.iloc[0]
        assert row.cosine > 0.95
        assert bool(row.agrees) and bool(row.nearest_is_best)

    def test_seeds_that_found_different_directions_disagree(self):
        rng = np.random.default_rng(1)
        a = rng.normal(size=(20, 64))
        b = rng.normal(size=(20, 64))
        out = K.concept_seed_agreement({0: a, 1: b}, {0: {"PLS": 0}, 1: {"PLS": 0}})
        assert not bool(out.iloc[0].agrees)

    def test_every_seed_pair_is_reported(self):
        rng = np.random.default_rng(2)
        decoders = {s: rng.normal(size=(12, 8)) for s in (0, 1, 2)}
        best = {s: {"PLS": s, "dELS": s + 1} for s in (0, 1, 2)}
        out = K.concept_seed_agreement(decoders, best)
        assert sorted(out.pair.unique()) == ["0-1", "0-2", "1-2"]
        assert len(out) == 6

    def test_one_seed_is_refused(self):
        with pytest.raises(ValueError, match="two seeds"):
            K.concept_seed_agreement({0: np.ones((3, 4))}, {0: {"PLS": 0}})
